# LeRobot 第 4 阶段学习文档：ACT 策略贯通学习

## 1. 本阶段目标

第 4 阶段的目标是第一次完整打通一条具体 policy 主线：

> 论文概念 -> 配置字段 -> dataset 时间窗口 -> processor -> 模型结构 -> loss -> 推理动作输出

在前 3 个阶段里，我们已经理解了：

- 训练主链如何组织
- dataset 如何产生时序样本
- processor 如何把数据转成 policy 所需形式

所以这一阶段终于可以把注意力集中到一个具体策略上。我选择的是：

- ACT（Action Chunking Transformer）

本次学习主要基于以下材料：

- `docs/source/policy_act_README.md`
- `src/lerobot/policies/act/configuration_act.py`
- `src/lerobot/policies/act/processor_act.py`
- `src/lerobot/policies/act/modeling_act.py`
- `tests/processor/test_act_processor.py`
- `tests/policies/test_policies.py`

---

## 2. 先给结论：ACT 在 LeRobot 里是什么样的策略

如果先用一句话概括：

> LeRobot 中的 ACT 是一个基于 action chunking 的时序控制策略，它每次预测一段未来动作序列，并通过动作队列或 temporal ensemble 在环境中逐步执行单步动作。

进一步说，它在 LeRobot 中体现为：

1. 一个带明确时间结构的 `ACTConfig`
2. 一套相对简单但标准的 pre/post processor
3. 一个以 transformer 为核心、可选带 VAE 目标的模型
4. 一个训练时做 chunk 级监督、推理时只输出单步动作的 policy 包装器

所以 ACT 非常适合作为第一个深入的 policy，因为它恰好把 LeRobot 架构中的多条主线串到一起：

- dataset 的未来 action 窗口
- processor 的 normalization 与 batch 处理
- policy 的 `forward()` / `select_action()`
- 训练与部署之间不同的数据流

---

## 3. 先从论文问题意识看 ACT

`docs/source/policy_act_README.md` 只提供了论文入口：

- `Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware`

虽然仓库里没有展开论文细节，但从代码和配置可以明确反推出 ACT 的几个核心思想：

### 3.1 不是一步一预测，而是一次预测一段动作

也就是所谓：

- action chunking

模型每次推理不只产生下一步动作，而是产生一个长度为 `chunk_size` 的动作序列。

### 3.2 推理时不一定执行完整 chunk

配置里明确区分了：

- `chunk_size`
- `n_action_steps`

这意味着：

- 模型能预测更长的未来动作
- 但环境中可以只执行其中一部分

### 3.3 可以做 temporal ensembling

如果开启 `temporal_ensemble_coeff`：

- 模型会把不同推理时刻得到的 chunk 中、对应到当前时刻的动作做在线加权平均

这对应论文中的 temporal ensemble 思路。

### 3.4 可选的 VAE 训练目标

ACT 在 LeRobot 中不仅仅是“transformer 预测动作”，还支持：

- 通过变分目标学习 latent
- loss = reconstruction + KL

这使它不只是简单的行为克隆头。

---

## 4. `ACTConfig`：论文设定如何映射到配置字段

配置文件位于：

- `src/lerobot/policies/act/configuration_act.py`

这是理解 ACT 在 LeRobot 中“长什么样”的第一站。

## 4.1 最关键的时间相关字段

### `n_obs_steps`

默认是：

- `1`

而且代码里明确限制：

- 当前不支持多 observation steps

也就是说，LeRobot 里的 ACT 当前默认只看当前时刻观测，而不是显式多帧历史。

### `chunk_size`

默认：

- `100`

表示模型每次预测 100 步动作序列。

### `n_action_steps`

默认：

- `100`

表示一次模型调用后，最多可执行多少步动作。

这两个字段合起来非常关键：

- `chunk_size` 是模型预测长度
- `n_action_steps` 是部署/rollout 执行长度

如果 `n_action_steps < chunk_size`，说明模型预测得更远，但只消费前半段。

## 4.2 temporal ensemble 相关字段

### `temporal_ensemble_coeff`

默认：

- `None`

如果启用，就要求：

- `n_action_steps == 1`

配置里直接强约束了这一点，因为 temporal ensemble 要每个环境 step 都重新推理一次，才能聚合多个 chunk 中指向当前时刻的动作。

## 4.3 训练相关字段

### `use_vae`

默认：

- `True`

### `latent_dim`

默认：

- `32`

### `kl_weight`

默认：

- `10.0`

这说明 LeRobot 中的 ACT 默认不是纯 deterministic transformer，而是带变分目标的版本。

## 4.4 视觉与 transformer 结构字段

配置还描述了：

- `vision_backbone = resnet18`
- `dim_model = 512`
- `n_heads = 8`
- `n_encoder_layers = 4`
- `n_decoder_layers = 1`

这里有一个特别值得注意的实现细节：

代码注释明确说：

- 原始 ACT 实现声称 decoder 有 7 层
- 但原始代码有 bug，实际上只用了第一层
- LeRobot 为了对齐原始实现，默认设置成 `1`

这说明 LeRobot 在这里采取的是“忠于原始行为而不是忠于论文文字”的工程选择。

---

## 5. ACT 与 dataset 的关系：为什么它天然依赖 action 窗口

`ACTConfig` 里有一个特别关键的属性：

- `action_delta_indices = list(range(self.chunk_size))`

而：

- `observation_delta_indices = None`
- `reward_delta_indices = None`

这意味着：

1. dataset 不会为 ACT 返回多帧 observation 历史
2. dataset 会为 ACT 返回未来 `chunk_size` 长度的 action 序列

也就是说，ACT 和 dataset 的契约非常清晰：

> ACT 监督信号的核心是未来动作 chunk，而不是历史 observation window。

这正好印证了前面第 2 阶段学到的东西：

- policy 配置会反向决定 dataset 的时间切片方式

对于 ACT 而言，这种反向约束主要体现在 action 端，而不是 observation 端。

---

## 6. ACT 的 processor：简单，但很标准

processor 文件位于：

- `src/lerobot/policies/act/processor_act.py`

## 6.1 preprocessor 做了什么

ACT 的 preprocessor steps 顺序是：

1. `RenameObservationsProcessorStep`
2. `AddBatchDimensionProcessorStep`
3. `DeviceProcessorStep`
4. `NormalizerProcessorStep`

这说明 ACT 在 LeRobot 中使用的是一套非常标准的 policy preprocessor 模板：

- 统一 observation key
- 加 batch 维
- 放到正确 device
- 用 dataset stats 做 normalization

## 6.2 postprocessor 做了什么

ACT 的 postprocessor steps 是：

1. `UnnormalizerProcessorStep`
2. `DeviceProcessorStep(device="cpu")`

这也很标准：

- 模型在归一化动作空间输出
- postprocessor 负责恢复真实尺度
- 再把动作移回 CPU，方便交给环境或机器人

## 6.3 这说明什么

ACT 的 processor 并没有特别复杂的专用逻辑，这很有启发性：

> 只要 policy 的核心差异主要在模型结构和时间建模，而不是输入输出格式上，LeRobot 的通用 processor 体系就已经足够承载它。

这也是为什么 ACT 很适合作为第一个 policy 学习对象。

---

## 7. 测试帮助我们确认 ACT processor 的契约

`tests/processor/test_act_processor.py` 很清楚地验证了几件事：

## 7.1 ACT processor 的结构

测试确认：

- preprocessor 名称为 `policy_preprocessor`
- postprocessor 名称为 `policy_postprocessor`
- preprocessor 正好是 4 个标准 step
- postprocessor 正好是 2 个标准 step

## 7.2 归一化与反归一化

测试验证了：

- 输入经过 preprocessor 后会被 batch 化和归一化
- 输出经过 postprocessor 后会被 unnormalize

## 7.3 设备放置行为

测试还明确检查：

- CPU 情况
- CUDA 情况
- Accelerate 场景下数据已在 GPU 上时，不应被不必要迁移
- 多 GPU 情况下，processor 应保持数据所在设备，而不是强行搬到 config.device

这说明 ACT processor 不只是“理论正确”，还特别注意和分布式训练配合时的设备语义。

## 7.4 可保存和加载

测试也验证了：

- preprocessor 可以 `save_pretrained`
- 之后可以 `DataProcessorPipeline.from_pretrained(...)`

这再次说明 processor 是 ACT 资产的一部分，而不是运行时临时对象。

---

## 8. `ACTPolicy`：LeRobot 对论文模型的 policy 封装

核心实现位于：

- `src/lerobot/policies/act/modeling_act.py`

这里最值得分清的是两个层次：

1. `ACTPolicy`
2. `ACT`

## 8.1 `ACTPolicy` 是训练/推理协议适配层

它负责：

- 挂接 `ACTConfig`
- 暴露 `forward()`
- 暴露 `predict_action_chunk()`
- 暴露 `select_action()`
- 管理 action queue / temporal ensembler

也就是说，`ACTPolicy` 是 LeRobot 风格的 policy 包装器。

## 8.2 `ACT` 是底层神经网络

它负责：

- backbone
- encoder
- decoder
- latent/VAE 路径
- action head

也就是说，`ACT` 更接近“论文模型本体”。

---

## 9. `select_action`：为什么 ACT 推理时不是直接返回完整 chunk

这是理解 ACT 的关键代码路径之一。

## 9.1 非 temporal ensemble 情况

默认情况下，如果没有 `temporal_ensemble_coeff`：

1. `select_action(batch)` 检查 action queue 是否为空
2. 如果为空，则调用 `predict_action_chunk(batch)`
3. 取前 `n_action_steps` 个动作
4. 转置后塞进 `_action_queue`
5. 每次只 `popleft()` 返回一个动作

这意味着：

> 虽然模型预测的是一个完整 chunk，但对环境接口来说，ACT 仍然表现为单步 action policy。

这个设计非常优雅：

- 模型保留 chunk-level 时序建模能力
- 环境仍然维持 step-level 接口

## 9.2 temporal ensemble 情况

如果启用 `temporal_ensemble_coeff`：

1. 不再使用 action queue
2. 每一步都重新 `predict_action_chunk`
3. 用 `ACTTemporalEnsembler.update(actions)` 在线聚合当前时刻动作

这对应论文中的 temporal ensembling。

## 9.3 为什么这点很重要

这正是 ACT 和普通单步 policy 最大的不同：

- `forward()` 学的是整段动作
- `select_action()` 却只交付一步动作

这也是 ACTION chunking 能融入通用 RL/robot step 接口的关键实现。

---

## 10. `predict_action_chunk`：policy 与底层网络的最直接接口

`predict_action_chunk(batch)` 做的事情相对简单：

1. 如果有 image features，就把多相机图像打包到 `batch[OBS_IMAGES]`
2. 调底层 `self.model(batch)`
3. 取返回值中的 action sequence

这一点说明：

- `ACTPolicy` 负责把 LeRobot 风格 batch 整理成底层模型更直接想要的形式
- 但绝大多数真正的建模逻辑仍然在 `ACT` 里

---

## 11. `forward`：ACT 的训练目标在 LeRobot 中如何实现

`ACTPolicy.forward(batch)` 是训练主链真正调用的入口。

## 11.1 核心输出

它会得到：

- `actions_hat`
- `(mu_hat, log_sigma_x2_hat)`

也就是：

- 预测动作序列
- latent 分布参数

## 11.2 reconstruction loss

动作重建损失是：

- `L1 loss`
- 对 `action_is_pad` 做 mask

这非常关键，因为 dataset 在 action chunk 超过 episode 边界时，可能会出现 padding。

所以：

- `action_is_pad` 用来避免把无效 padding 动作纳入损失

这再次说明 dataset 和 policy 的契约是紧密配合的。

## 11.3 KL loss

如果 `use_vae=True`：

- 会计算标准 VAE 形式的 KL divergence
- 再乘上 `kl_weight`

最终：

- `loss = l1_loss + kl_weight * mean_kld`

如果 `use_vae=False`：

- 只保留 `l1_loss`

## 11.4 输出日志

返回的 `loss_dict` 会包含：

- `l1_loss`
- 可选 `kld_loss`

这正好能被训练主链中的日志系统直接消费。

---

## 12. `ACTTemporalEnsembler`：在线版 temporal ensemble

这是 ACT 在 LeRobot 实现中一个很有意思的部件。

## 12.1 它做什么

当每个 step 都重新推理出一个 chunk 时，很多 chunk 都会包含“指向当前时刻”的动作候选。

`ACTTemporalEnsembler` 会：

- 用指数权重对这些候选动作做在线平均
- 每次消费掉当前时刻的一个动作

## 12.2 权重直觉

权重形式是：

- `exp(-m * i)`

其中：

- `m = temporal_ensemble_coeff`
- `i` 越大表示越新的动作

代码注释也明确说了：

- 正值会更偏向旧动作
- 负值会更偏向新动作

## 12.3 测试验证

`tests/policies/test_policies.py::test_act_temporal_ensembler` 做了一个非常好的校验：

- 用 offline 方式算加权平均
- 用 online ensembler 一步步更新
- 检查二者数值一致

这说明 temporal ensembler 不是经验性代码，而是经过明确数学契约验证的实现。

---

## 13. `ACT` 模型本体：底层结构怎么拼起来

底层网络类是：

- `class ACT(nn.Module)`

## 13.1 如果开启 VAE

模型会额外包含：

- `vae_encoder`
- `cls token`
- robot state 输入投影
- action 输入投影
- latent 参数输出层

训练时会：

1. 把 `[cls, robot_state, action_sequence]` 喂入 VAE encoder
2. 得到 `mu` 和 `log_sigma_x2`
3. 用重参数化采样 latent

## 13.2 图像 backbone

如果有 image features：

- 用 `torchvision` 的 ResNet backbone
- 取 `layer4` feature map
- 再投影到 transformer hidden dim

## 13.3 主 transformer 路径

主路径大致是：

1. latent token
2. 可选 robot state token
3. 可选 env state token
4. image feature map tokens

这些 token 经过：

- `ACTEncoder`
- `ACTDecoder`

最后：

- `action_head` 回归出 `(B, chunk_size, action_dim)`

## 13.4 positional encoding

实现里同时用了：

- 1D sinusoidal position embedding
- 2D sinusoidal position embedding（图像 feature map）
- decoder query embedding

所以 ACT 在 LeRobot 里的结构不是“直接把图像 flatten 后塞 transformer”，而是相对规范地保留了视觉特征图与位置编码结构。

---

## 14. `get_optim_params`：为什么 ACT 的 backbone 可以单独设置学习率

`ACTPolicy.get_optim_params()` 会返回两个 param groups：

1. 非 backbone 参数
2. backbone 参数，使用 `optimizer_lr_backbone`

这和配置中的：

- `optimizer_lr`
- `optimizer_lr_backbone`

正好对应。

`tests/policies/test_policies.py::test_act_backbone_lr` 也验证了：

- optimizer 里确实会出现两个 group

这说明 LeRobot 并没有把 optimizer 参数组逻辑写死在训练脚本里，而是让 policy 自己声明更合适的优化器分组。

---

## 15. 从测试看，ACT 在 LeRobot 里遵守了哪些重要契约

ACT 相关测试让我们能看到几个很重要的约束。

## 15.1 `forward()` 不能破坏输入 batch

`tests/policies/test_policies.py` 明确验证：

- 调完 `policy.forward(batch)` 后
- batch 的 key 和 value 都不应被篡改

这说明 LeRobot 对 policy 有很明确的工程要求：

- 不能偷偷原地修改上游输入

## 15.2 `select_action()` 也不能破坏 observation

测试同样检查：

- 推理时 observation 不应被 policy 原地改写

## 15.3 policy 必须能无缝接入 env step

测试路径大致是：

- dataset 取 batch
- policy 做前向
- env reset
- preprocess observation
- policy.select_action
- env.step(action)

这说明 ACT 在 LeRobot 中不是“单元测试里能跑”，而是被当作完整工作流组件验证。

---

## 16. 用一条完整数据流总结 ACT 在 LeRobot 中怎么跑

现在可以把前面学过的 dataset / processor / policy / train loop 串起来看。

## 16.1 训练时

1. `TrainPipelineConfig` 选择 `policy=act`
2. `ACTConfig.action_delta_indices = range(chunk_size)`
3. dataset 工厂据此返回未来动作 chunk
4. preprocessor：
   - rename
   - add batch dim
   - move to device
   - normalize
5. `ACTPolicy.forward(batch)`：
   - 整理图像输入
   - 底层 ACT 预测 chunk
   - 用 `action_is_pad` mask L1
   - 可选加 KL
6. train loop 用 loss 做反向传播

## 16.2 推理时

1. 当前 observation 经过 ACT preprocessor
2. `ACTPolicy.select_action(batch)`
3. 若 queue 为空，则先 `predict_action_chunk`
4. 从 chunk 中取前 `n_action_steps` 填入队列
5. 每次只返回一个动作
6. postprocessor 反归一化并回到 CPU
7. env/robot 执行该动作

这条链很好地体现了 LeRobot 的架构价值：

- dataset 提供 chunk supervision
- processor 保证输入输出格式
- policy 只专注时序动作建模与动作消费策略

---

## 17. 第 4 阶段最重要的设计认识

这一阶段最值得记住的几个结论如下。

## 认识 1：ACT 的关键不只是 transformer，而是 action chunking

它最核心的系统意义在于：

- 模型预测 chunk
- 环境只消费单步 action

这个桥接逻辑是 `select_action()` 的核心价值。

## 认识 2：ACT 的 dataset 契约主要体现在 action 端

通过 `action_delta_indices = range(chunk_size)`，ACT 把“未来动作段监督”这个需求传递给了 dataset。

## 认识 3：ACT 的 processor 很标准，说明平台抽象已经足够强

ACT 并不需要复杂专用 processor 才能落地，通用 normalization/batching/device pipeline 已经足够支撑。

## 认识 4：LeRobot 把论文里的 temporal ensemble 变成了一个清晰的工程组件

`ACTTemporalEnsembler` 是一个很好的例子：

- 数学思路明确
- 接口清晰
- 有单独测试验证

## 认识 5：LeRobot 中的 `ACTPolicy` 是“协议适配层 + 模型封装层”

它一头对接 train loop / processor / env，另一头对接底层 `ACT` 网络。

这也是 LeRobot 所有 policy 统一风格的一部分。

---

## 18. 本阶段结束后，你应该已经能回答的问题

完成第 4 阶段后，你应该能比较清楚地回答：

1. `chunk_size` 和 `n_action_steps` 的区别是什么？
2. 为什么 ACT 的 dataset 需要未来 action window？
3. 为什么 `select_action()` 不直接返回整个 chunk？
4. `action_is_pad` 为什么会影响 loss？
5. `use_vae=True` 时 ACT 的 loss 如何组成？
6. ACT 的 pre/post processor 在 LeRobot 中各做什么？
7. `temporal_ensemble_coeff` 为什么要求 `n_action_steps == 1`？
8. 为什么 ACT 能比较自然地融入 LeRobot 的统一训练/推理框架？

如果这些问题已经能比较顺畅地回答，说明第 4 阶段达标。

---

## 19. 下一阶段建议

完成 ACT 后，下一步最适合进入第 5 阶段，学习第二个具体 policy，推荐：

- Diffusion Policy

原因很简单：

1. 它和 ACT 同样是 imitation learning
2. 但它的时间建模方式完全不同
3. 这样最适合做第一次 policy 对照学习

推荐顺序：

1. `docs/source/policy_diffusion_README.md`
2. `src/lerobot/policies/diffusion/configuration_diffusion.py`
3. `src/lerobot/policies/diffusion/modeling_diffusion.py`
4. `src/lerobot/policies/diffusion/processor_diffusion.py`
5. `tests/processor/test_diffusion_processor.py`

到那时你就可以真正比较：

- ACT 的 chunk queue
- Diffusion 的 iterative denoising

在 LeRobot 平台里是如何被统一组织的。

---

## 20. 本阶段个人学习结论

如果用一句话总结第 4 阶段：

> LeRobot 中的 ACT 通过 `ACTConfig + 标准 policy processor + ACTPolicy + ACT` 这套分层结构，把论文中的 action chunking、可选变分目标和 temporal ensembling 落成了一条既符合训练主链、又兼容单步环境接口的工程实现。

它是理解 LeRobot “如何承载具体 policy”的第一块非常理想的样板。
