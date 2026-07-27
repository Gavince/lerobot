# LeRobot 第 5 阶段学习文档：Diffusion Policy 对照学习

## 1. 本阶段目标

第 5 阶段的目标是完成第一次真正的策略对照学习：

> 用和 ACT 相同的分析框架，读懂 Diffusion Policy，并明确两者在 LeRobot 中的共同点与差异。

本次学习重点包括：

1. Diffusion Policy 的时间窗口设定
2. 它如何依赖 dataset 提供 observation history 和 action horizon
3. 它的 processor 与 ACT 有哪些相同和不同
4. 它的训练目标为什么是扩散式 denoising，而不是动作重建
5. 它在推理时如何缓存 observation 和 action

本次学习主要基于以下材料：

- `docs/source/policy_diffusion_README.md`
- `src/lerobot/policies/diffusion/configuration_diffusion.py`
- `src/lerobot/policies/diffusion/processor_diffusion.py`
- `src/lerobot/policies/diffusion/modeling_diffusion.py`
- `tests/processor/test_diffusion_processor.py`
- `tests/policies/test_policies.py`

---

## 2. 先给结论：Diffusion Policy 在 LeRobot 里是什么样的策略

如果先用一句话概括：

> LeRobot 中的 Diffusion Policy 是一个条件动作生成模型，它基于最近若干步 observation 历史，对整段未来动作轨迹做扩散式建模，并在 rollout 时从生成出的动作段中取出当前应该执行的连续若干步动作。

和 ACT 相比，它在 LeRobot 里的整体感觉是：

1. 时间结构更复杂
2. observation history 更重要
3. 训练目标从 action regression 变成 noise/sample prediction
4. 推理机制从 transformer chunk prediction 变成 iterative denoising

但与此同时，两者在平台层面又非常一致：

- 都有 `PreTrainedConfig`
- 都有标准 pre/post processor
- 都通过 `forward()` 接训练主链
- 都通过 `select_action()` 兼容 step-level 环境接口

这正是 LeRobot 架构设计的价值所在。

---

## 3. 从论文问题意识看 Diffusion Policy

`docs/source/policy_diffusion_README.md` 给出的论文入口是：

- `Diffusion Policy: Visuomotor Policy Learning via Action Diffusion`

结合代码可以看出，这个策略的核心思路是：

### 3.1 不是直接回归动作，而是对动作轨迹做扩散建模

训练时：

- 对真实动作轨迹加噪
- 模型学习预测噪声或原始轨迹

推理时：

- 从随机噪声开始
- 逐步去噪
- 得到完整动作段

### 3.2 不是只看当前 observation，而是看最近若干步 observation

与 ACT 不同，Diffusion 默认设置里：

- `n_obs_steps = 2`

这意味着 dataset 和 rollout 缓存都要维护 observation history。

### 3.3 生成的是一段更长 horizon 的动作

默认：

- `horizon = 16`
- `n_action_steps = 8`

也就是说，模型生成 16 步动作，但 rollout 只消费其中一部分。

---

## 4. `DiffusionConfig`：论文设定如何映射到配置字段

配置文件位于：

- `src/lerobot/policies/diffusion/configuration_diffusion.py`

## 4.1 最关键的时间字段

### `n_obs_steps`

默认：

- `2`

表示模型输入不是单步 observation，而是最近两步 observation。

### `horizon`

默认：

- `16`

表示扩散模型训练和推理时的完整动作序列长度。

### `n_action_steps`

默认：

- `8`

表示 rollout 中从生成轨迹里真正消费多少步动作。

### `drop_n_last_frames`

默认：

- `7`

注释里解释得很清楚：

- 原始实现不采样最后几帧，以减少 padding
- 它的值正好与 `horizon - n_action_steps - n_obs_steps + 1` 对应

这说明 Diffusion 比 ACT 更明确地把“episode 边界上的采样质量”编码进了配置。

## 4.2 时间窗口属性如何影响 dataset

`DiffusionConfig` 定义：

- `observation_delta_indices = range(1 - n_obs_steps, 1)`
- `action_delta_indices = range(1 - n_obs_steps, 1 - n_obs_steps + horizon)`

用默认值展开就是：

- observation offsets: `[-1, 0]`
- action offsets: `[-1, 0, 1, ..., 14]`

这点非常关键：

> Diffusion 的 action horizon 是相对于“最早 observation”来定义的，而不是简单从当前时刻开始。

因此在 rollout 时，真正用于执行的动作片段不是从序列头部直接拿，而要从 `n_obs_steps - 1` 开始切。

## 4.3 视觉相关字段

DiffusionConfig 比 ACT 多了很多图像处理相关参数：

- `resize_shape`
- `crop_ratio`
- `crop_shape`
- `crop_is_random`
- `use_group_norm`
- `spatial_softmax_num_keypoints`
- `use_separate_rgb_encoder_per_camera`

这说明 Diffusion 在 LeRobot 中对视觉编码路径的工程控制更细。

## 4.4 噪声调度与扩散参数

DiffusionConfig 还集中描述了扩散核心超参数：

- `noise_scheduler_type = DDPM / DDIM`
- `num_train_timesteps`
- `beta_schedule`
- `beta_start`
- `beta_end`
- `prediction_type = epsilon / sample`
- `clip_sample`
- `clip_sample_range`
- `num_inference_steps`

这说明与 ACT 不同，Diffusion 的“训练配置”很大一部分就是“生成过程配置”。

---

## 5. Diffusion 与 dataset 的关系：比 ACT 更依赖 observation history

这是第 5 阶段最值得抓住的差异之一。

## 5.1 ACT vs Diffusion 在 dataset 契约上的区别

### ACT

- 主要依赖 action chunk
- observation 仍然是单步

### Diffusion

- 需要 observation history
- 也需要 action horizon

所以：

> Diffusion 在 dataset 层面比 ACT 更强依赖时序切片。

## 5.2 为什么 `drop_n_last_frames` 很重要

Diffusion 的 action 序列更长，而且 observation history 也要往前看。

这会导致：

- episode 尾部更容易出现大量 padding

因此配置里显式定义 `drop_n_last_frames`，训练 dataloader 会据此使用 `EpisodeAwareSampler` 跳过这些尾帧。

这一点体现了一个很 LeRobot 的特点：

- policy 的时间建模需求会直接反馈到 dataset sampling 策略

---

## 6. Diffusion 的 processor：结构上和 ACT 很像

processor 文件位于：

- `src/lerobot/policies/diffusion/processor_diffusion.py`

## 6.1 preprocessor

Diffusion 的 preprocessor 也是：

1. `RenameObservationsProcessorStep`
2. `AddBatchDimensionProcessorStep`
3. `DeviceProcessorStep`
4. `NormalizerProcessorStep`

## 6.2 postprocessor

postprocessor 也是：

1. `UnnormalizerProcessorStep`
2. `DeviceProcessorStep(device="cpu")`

## 6.3 ACT 和 Diffusion 在 processor 上为什么几乎一样

这是一个很重要的观察：

虽然 ACT 和 Diffusion 的算法完全不同，但它们的 policy pre/post processor 结构几乎一样。

这说明：

> LeRobot 已经把“模型前后公共数据整理逻辑”抽成了强通用层，策略差异更多体现在 config 和 model，而不是 processor 模板本身。

## 6.4 唯一更值得注意的地方

Diffusion 的默认 normalization mapping 和 ACT 不同：

- ACT 常用 `STATE/ACTION/VISUAL -> MEAN_STD`
- Diffusion 默认是：
  - `VISUAL -> MEAN_STD`
  - `STATE -> MIN_MAX`
  - `ACTION -> MIN_MAX`

这说明不同策略虽然共用同一 processor 骨架，但可以通过 config 精细控制数值空间。

---

## 7. Diffusion processor 测试说明了哪些契约

`tests/processor/test_diffusion_processor.py` 的价值在于，它说明 Diffusion processor 依然遵守 LeRobot 的标准契约，但在视觉 batch 维度上更敏感。

## 7.1 图像输入是正式主线输入

测试明确检查：

- `observation.state`
- `observation.image`
- `action`

经过 preprocessor 后的 shape 是否正确。

## 7.2 device 语义与 ACT 一样严谨

测试覆盖：

- CUDA
- Accelerate 场景
- 多 GPU
- bfloat16 / float32 混合情况

这说明平台层面对 processor 的设备行为要求是一致的。

## 7.3 `IDENTITY` normalization

一个很有意思的测试是：

- 图像 feature 如果设置 `IDENTITY` normalization，就不应被改动

这表明 normalization step 不只是“默认套公式”，而是严格服从 feature-specific normalization mode。

---

## 8. `DiffusionPolicy`：LeRobot 风格的扩散策略封装

核心实现位于：

- `src/lerobot/policies/diffusion/modeling_diffusion.py`

和 ACT 一样，我们也需要分清两个层次：

1. `DiffusionPolicy`
2. `DiffusionModel`

## 8.1 `DiffusionPolicy` 的职责

它负责：

- 挂接 `DiffusionConfig`
- 管理 observation/action queues
- 暴露 `predict_action_chunk()`
- 暴露 `select_action()`
- 暴露 `forward()`

## 8.2 `DiffusionModel` 的职责

它负责：

- 视觉与状态条件编码
- 条件扩散采样
- loss 计算
- U-Net 主体

和 ACT 的分层方式类似：

- policy 层负责适配 LeRobot 协议
- model 层负责真正算法建模

---

## 9. `select_action`：Diffusion 如何在 rollout 中消费 observation history

这是理解 Diffusion 的关键入口。

## 9.1 它会缓存 observation

`reset()` 时会初始化队列：

- `OBS_STATE` 队列，长度 `n_obs_steps`
- `OBS_IMAGES` 队列（如果有图像）
- `OBS_ENV_STATE` 队列（如果有 env state）
- `ACTION` 队列，长度 `n_action_steps`

这说明 Diffusion 在 rollout 时是显式维护时序上下文的。

## 9.2 `select_action` 的工作流

大致流程是：

1. 如果 batch 里有 `action`，先弹掉
2. 如果有多相机图像，把它们堆成 `OBS_IMAGES`
3. 用 `populate_queues(...)` 把当前 observation 填入缓存
4. 如果 action queue 为空：
   - 调 `predict_action_chunk`
   - 生成当前轨迹的动作段
   - 放入 action queue
5. 每次只从 queue 里 `popleft()` 一个动作返回

所以 Diffusion 和 ACT 在 rollout 层有一个共性：

- 模型都生成一段动作
- 环境接口仍然每次只消费一步

但二者前面的生成机制完全不同。

## 9.3 它和 ACT 的关键区别

### ACT

- 当前 observation -> transformer -> 直接预测动作 chunk

### Diffusion

- 最近 `n_obs_steps` observation -> 条件扩散采样 -> 得到动作 horizon -> 切出可执行片段

因此，Diffusion 更像：

- “基于上下文条件，生成整段未来动作轨迹”

而 ACT 更像：

- “基于当前 observation，直接回归一段未来动作”

---

## 10. `predict_action_chunk`：Diffusion 不是直接拿 batch，而是先堆 observation history

`predict_action_chunk(batch)` 里最重要的一步是：

- 从 `_queues` 中把最近的 observation 历史堆叠成 `(B, n_obs_steps, ...)`

然后再调用：

- `self.diffusion.generate_actions(batch, noise=noise)`

这意味着：

> rollout 时真正喂给 DiffusionModel 的不是单个 observation，而是一个经过缓存堆叠的 observation 序列。

这也是为什么它必须有 observation queue，而 ACT 不需要。

---

## 11. `forward`：Diffusion 的训练目标为什么和 ACT 完全不同

`DiffusionPolicy.forward(batch)` 做的事情很短，但背后的含义很大。

它会：

1. 整理多相机图像到 `OBS_IMAGES`
2. 确保 observation 的时序维度存在
3. 调 `self.diffusion.compute_loss(batch)`

这意味着 policy 层本身并不直接实现 loss，而把核心训练目标交给底层扩散模型。

---

## 12. `DiffusionModel.compute_loss`：训练目标的本体

这是理解 Diffusion 的核心。

## 12.1 输入要求

它明确要求 batch 至少包含：

- `observation.state`: `(B, n_obs_steps, state_dim)`
- 图像或 env state
- `action`: `(B, horizon, action_dim)`
- `action_is_pad`: `(B, horizon)`

所以 Diffusion 对 dataset 输出的契约比 ACT 更严格、更时序化。

## 12.2 训练流程

大致步骤是：

1. 编码 observation history，得到全局条件 `global_cond`
2. 取真实动作轨迹 `trajectory = batch[ACTION]`
3. 采样噪声 `eps`
4. 为每个 batch item 采样随机 timestep
5. 用 scheduler 对真实轨迹加噪，得到 `noisy_trajectory`
6. U-Net 预测：
   - 噪声 `eps`
   - 或原始 sample
7. 依据 `prediction_type` 选 target
8. 计算 `MSE loss`

## 12.3 padding mask

如果 `do_mask_loss_for_padding=True`：

- 会用 `action_is_pad` mask 掉 padding 部分 loss

与 ACT 一样，dataset 和 policy 在 episode 边界处理上是深度配合的；
只不过：

- ACT 默认用 masked L1
- Diffusion 默认用 unmasked MSE（除非显式打开）

---

## 13. `generate_actions`：为什么生成出的 horizon 还要再切一刀

`DiffusionModel.generate_actions(batch)` 里有一个很关键的步骤：

1. 先基于条件做完整扩散采样，得到 `(B, horizon, action_dim)`
2. 再取：
   - `start = n_obs_steps - 1`
   - `end = start + n_action_steps`
3. 返回 `actions[:, start:end]`

这点非常重要，因为它解释了前面配置中 action delta indices 的含义。

Diffusion 的 action horizon 是从“最早 observation”算起的；
真正和“当前时刻”对齐的动作，是从第 `n_obs_steps - 1` 个位置开始。

所以：

> Diffusion 的 horizon 不是简单“从现在往后 horizon 步”，而是包含了和 observation history 对齐后的动作轨迹窗口。

这是它和 ACT 在时间语义上的根本差异之一。

---

## 14. `conditional_sample`：Diffusion 推理的本体

推理阶段，DiffusionModel 使用：

- `conditional_sample(...)`

其核心流程是标准扩散采样：

1. 从噪声轨迹开始
2. scheduler 设定推理 timestep
3. 循环：
   - U-Net 根据当前 sample、timestep 和全局条件预测输出
   - scheduler 计算上一步 sample
4. 最终得到动作轨迹 sample

相比 ACT，这里的关键差异是：

- ACT 一次前向得到动作 chunk
- Diffusion 要经历多步 denoising 才得到最终动作段

所以 Diffusion 的推理成本和结构复杂度都更高。

---

## 15. 条件编码：Diffusion 如何把多模态 observation 变成 global condition

`_prepare_global_conditioning(batch)` 是一个很值得注意的函数。

它会把以下内容拼在一起：

- `observation.state`
- 图像编码特征
- `observation.environment_state`（如果有）

最后 flatten 成：

- `(B, global_cond_dim)`

这说明 Diffusion U-Net 本身不直接“看图像张量”，而是通过条件向量接收观测信息。

图像编码部分还支持：

- 共享 RGB encoder
- 每个 camera 一个独立 encoder

这让它在多相机场景下更灵活。

---

## 16. DiffusionModel 的主干：条件 1D U-Net

底层生成网络核心是：

- `DiffusionConditionalUnet1d`

这和 ACT 的 transformer 主体形成鲜明对比。

## 16.1 输入是什么

输入是：

- `(B, T, action_dim)` 的动作轨迹

这里的时间维就是动作 horizon。

## 16.2 条件来自哪里

条件由两部分组成：

1. diffusion timestep embedding
2. global observation conditioning

然后通过 FiLM 风格调制进入各层残差块。

## 16.3 这意味着什么

与 ACT 相比：

- ACT 更像 seq2seq transformer
- Diffusion 更像条件生成模型，U-Net 是动作轨迹空间中的 denoiser

LeRobot 把这两种完全不同的建模范式都装进了同一个 policy 抽象中。

---

## 17. ACT vs Diffusion：第一次系统对照

这一阶段最重要的价值之一，就是完成第一次真正的策略对照。

## 17.1 输入时间结构

### ACT

- 默认 `n_obs_steps = 1`
- 主要依赖未来 action chunk

### Diffusion

- 默认 `n_obs_steps = 2`
- 同时依赖 observation history 和 action horizon

## 17.2 训练目标

### ACT

- 直接动作重建
- `L1 + 可选 KL`

### Diffusion

- 对动作轨迹做前向扩散
- 训练 U-Net 预测噪声或 sample
- `MSE`

## 17.3 推理方式

### ACT

- 一次前向直接得到动作 chunk

### Diffusion

- 多步 denoising 后得到动作轨迹

## 17.4 rollout 缓存逻辑

### ACT

- 只缓存生成好的 action queue

### Diffusion

- 同时缓存 observation history 和 action queue

## 17.5 processor

二者 surprisingly 相似：

- rename
- batch
- device
- normalize
- unnormalize

这说明 processor 模板的共性很强，而策略差异主要集中在 model/config/time semantics。

---

## 18. 从测试看 Diffusion 在 LeRobot 中遵守了哪些重要契约

虽然没有专门的大型 Diffusion policy 测试块像 ACT temporal ensembler 那样醒目，但现有测试仍然很有价值。

## 18.1 通用 policy 契约

`tests/policies/test_policies.py` 会验证：

- policy 可实例化
- `forward()` 不应破坏 batch
- `select_action()` 输出可直接喂 env.step

所以 Diffusion 和 ACT 一样，必须符合统一 policy 协议。

## 18.2 backward compatibility

测试里还保留了 Diffusion 的 backward compatibility artifact 检查。

这说明：

- Diffusion 在 LeRobot 中也是受稳定性约束的重要策略
- 更改 processor 或模型实现时，需要关注输出一致性

---

## 19. 用一条完整数据流总结 Diffusion 在 LeRobot 中怎么跑

## 19.1 训练时

1. `DiffusionConfig` 定义：
   - observation history
   - action horizon
2. dataset 按 delta indices 返回：
   - `(B, n_obs_steps, ...)` observation
   - `(B, horizon, action_dim)` action
3. preprocessor：
   - rename
   - batch
   - device
   - normalize
4. `DiffusionPolicy.forward(batch)`
5. `DiffusionModel.compute_loss(batch)`：
   - encode conditions
   - add noise
   - U-Net denoise
   - MSE to target

## 19.2 推理时

1. 每步 observation 进入 queue
2. queue 满足 `n_obs_steps` 后，若 action queue 空：
   - 根据 observation history 生成整段动作 horizon
   - 切出当前应该执行的 `n_action_steps`
   - 放进 action queue
3. 每次只取一个 action 返回
4. postprocessor 把动作恢复到原尺度并搬回 CPU

这条链说明 Diffusion 虽然算法上比 ACT 更复杂，但在 LeRobot 中仍然服从相同的“训练 chunk / rollout 单步动作”接口哲学。

---

## 20. 第 5 阶段最重要的设计认识

这一阶段最值得记住的几个结论如下。

## 认识 1：Diffusion 的核心复杂度来自时间语义，而不仅是扩散本身

最容易忽略的其实不是 U-Net，而是：

- `n_obs_steps`
- `horizon`
- `n_action_steps`
- action 截取起点与 observation history 的对齐关系

## 认识 2：Diffusion 比 ACT 更依赖 dataset 的时序切片能力

因为它不仅要未来 action，还要过去 observation。

## 认识 3：LeRobot 用统一 policy 抽象承载了完全不同的生成机制

ACT 是直接 chunk regression；
Diffusion 是 iterative denoising。

但两者依然都通过：

- `forward()`
- `select_action()`
- pre/post processors

接入主链。

## 认识 4：processor 的强通用性在这里被再次验证

两种算法完全不同，但 processor 模板高度一致，这说明平台中间层设计确实成功。

## 认识 5：Diffusion rollout 的 observation queue 是理解它部署行为的关键

如果忽略 queue，只看 U-Net，很容易误解这个策略在环境中到底怎么运行。

---

## 21. 本阶段结束后，你应该已经能回答的问题

完成第 5 阶段后，你应该能比较清楚地回答：

1. Diffusion 的 `n_obs_steps / horizon / n_action_steps` 分别是什么意思？
2. 为什么它的 action horizon 要从 `n_obs_steps - 1` 位置开始截取？
3. 为什么 Diffusion 比 ACT 更依赖 observation history？
4. Diffusion 的训练目标为什么是 MSE 而不是动作回归 L1？
5. `prediction_type = epsilon / sample` 会影响什么？
6. rollout 时为什么既要 observation queue，也要 action queue？
7. Diffusion 和 ACT 在 processor 层为什么几乎一样？
8. ACT 和 Diffusion 在 LeRobot 中最大的系统差异是什么？

如果这些问题已经能比较顺畅地回答，说明第 5 阶段达标。

---

## 22. 下一阶段建议

现在为止，你已经看过两个具体 policy：

1. ACT
2. Diffusion

下一步最自然的是进入第 6 阶段，把视角从“模型内部”再切回“模型如何被评估”，也就是：

- `lerobot_eval.py`
- `envs/factory.py`
- rollout 逻辑

因为到这里你已经能理解：

- policy 想吃什么输入
- policy 会吐什么动作

下一步就该搞清楚：

> 这些策略在仿真环境里到底是怎么被跑起来的？

---

## 23. 本阶段个人学习结论

如果用一句话总结第 5 阶段：

> LeRobot 中的 Diffusion Policy 通过 observation history 缓存、动作 horizon 切片、条件扩散采样和统一 pre/post processor，把一个生成式动作模型稳稳地装进了与 ACT 相同的训练与 rollout 框架里。

这让我们第一次真正看清了 LeRobot 的平台价值：它不是围绕某一种策略设计的，而是能用一致接口承载不同时间建模范式。
