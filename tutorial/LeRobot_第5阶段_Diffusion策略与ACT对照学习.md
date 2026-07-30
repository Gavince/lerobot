# LeRobot 第5阶段：Diffusion策略与ACT对照学习

## 一、阶段目标

这一阶段的目标不是只看懂 `DiffusionPolicy` 这个类，而是完成一次更重要的学习跃迁：

1. 理解 LeRobot 如何在同一套 `dataset + processor + policy` 平台上承载与 ACT 完全不同的策略范式。
2. 把 Diffusion Policy 的论文概念映射到 LeRobot 中的配置、时间窗口、模型结构、loss 和推理逻辑。
3. 建立第一份真正有用的“ACT vs Diffusion”对照表，为后续继续学习 SmolVLA、Pi0 或 RL 策略打基础。

这一阶段对应的核心文件：

- [docs/source/policy_diffusion_README.md](/Users/zwy/Macin/lerobot/docs/source/policy_diffusion_README.md)
- [src/lerobot/policies/diffusion/configuration_diffusion.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py)
- [src/lerobot/policies/diffusion/modeling_diffusion.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py)
- [src/lerobot/policies/diffusion/processor_diffusion.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/diffusion/processor_diffusion.py)
- [tests/processor/test_diffusion_processor.py](/Users/zwy/Macin/lerobot/tests/processor/test_diffusion_processor.py)
- [tests/policies/test_policies.py](/Users/zwy/Macin/lerobot/tests/policies/test_policies.py)

---

## 二、先用一句话理解 Diffusion Policy

如果说 ACT 是“给定当前观测，直接预测未来一段动作 chunk”，那么 Diffusion Policy 则是：

给定最近几步观测，把未来整段动作轨迹当作一个需要逐步去噪生成的序列，用条件扩散模型从噪声中采样出动作，再从中截取当前应该执行的那一段动作。

这个差异会直接影响 5 个层面：

1. 数据窗口更长。
2. 配置项之间的约束更强。
3. 模型主体从 Transformer/VAE 风格切换到 Conditional U-Net。
4. 训练目标从直接回归动作变成预测噪声或干净样本。
5. 推理时要显式运行多步 denoising，而不是一次前向直接得到动作。

---

## 三、论文视角下应该先抓住什么

`docs/source/policy_diffusion_README.md` 本身很短，主要提供了原论文链接。真正需要你从论文中带着问题去读的是下面这几个点：

1. 输入是什么  
   不是单帧观测，而是一个短历史窗口。

2. 输出是什么  
   不是一步动作，而是一整段动作轨迹。

3. 训练目标是什么  
   对真实动作序列加噪，再学习预测噪声或恢复原始动作。

4. 推理时做什么  
   从高斯噪声开始，经过若干步逆扩散，逐步得到一条动作轨迹。

5. 为什么适合机器人控制  
   它天然适合多峰动作分布，比单点回归更容易表达“同一观测下可能存在多种合理动作”。

建议你的论文阅读方式不要急着看全部数学推导，而是先建立下面这张映射表：

| 论文概念 | LeRobot 对应位置 |
| --- | --- |
| observation history | `n_obs_steps` 与 `observation_delta_indices` |
| action horizon | `horizon` 与 `action_delta_indices` |
| denoising steps | `num_train_timesteps`、`num_inference_steps` |
| conditional generation | `global_cond` 与 `DiffusionConditionalUnet1d` |
| noise prediction / sample prediction | `prediction_type` |
| reverse diffusion sampling | `conditional_sample()` |

---

## 四、配置系统：Diffusion 的第一关键层

Diffusion 的配置定义在 [src/lerobot/policies/diffusion/configuration_diffusion.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py)。

这是这一阶段最值得先吃透的文件，因为它几乎直接决定了 dataset 时间窗口、模型输入输出形状和推理行为。

### 4.1 默认时间配置

默认配置里最重要的 3 个字段是：

- `n_obs_steps = 2`
- `horizon = 16`
- `n_action_steps = 8`

这 3 个数决定了 Diffusion 在 LeRobot 中的基本时序语义：

1. 模型每次会读取最近 2 步观测。
2. 模型内部会生成长度为 16 的整段动作轨迹。
3. 最终只会执行其中 8 步动作，然后再重新感知、重新生成。

这一点和 ACT 很不一样。ACT 更像“直接吐出未来 chunk”；Diffusion 更像“先生成一整条轨迹，再从中切一段拿去执行”。

### 4.2 observation/action 时间索引

`DiffusionConfig` 中最重要的两个属性是：

- `observation_delta_indices`
- `action_delta_indices`

其中：

```python
observation_delta_indices = list(range(1 - self.n_obs_steps, 1))
```

当 `n_obs_steps = 2` 时，对应 `[-1, 0]`，表示样本会取“前一帧 + 当前帧”。

```python
action_delta_indices = list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))
```

默认情况下对应 `[-1, 0, 1, ..., 14]`。

这件事非常值得认真理解，因为它体现了 Diffusion 在 LeRobot 里的一个核心设计：

动作序列的时间轴不是简单地从“当前之后”开始，而是要和 observation history 对齐。模型要看到一段观察历史，再生成覆盖这段历史末端及未来的一整条动作轨迹。

后面 `generate_actions()` 会从这条长度为 `horizon` 的轨迹中，取出从 `start = n_obs_steps - 1` 开始的那一段作为真正执行的动作。

### 4.3 `drop_n_last_frames` 的意义

默认配置里还有一个容易忽略但非常关键的字段：

```python
drop_n_last_frames = 7
```

注释说明它等于：

```python
horizon - n_action_steps - n_obs_steps + 1
```

这背后的含义是：

靠近 episode 尾部时，数据很容易不够支撑完整的 observation history 和 action horizon。如果仍然保留这些样本，就会带来大量 padding，弱化训练信号。

所以 Diffusion 默认会丢掉 episode 末尾的一部分帧，以减少“标签主要由 padding 构成”的情况。这说明它比 ACT 更依赖完整时序窗口，时间边界处理也更重要。

### 4.4 特征与归一化约束

配置中还规定了默认归一化方式：

- 图像 `VISUAL` 用 `MEAN_STD`
- 状态 `STATE` 用 `MIN_MAX`
- 动作 `ACTION` 用 `MIN_MAX`

`validate_features()` 会检查：

1. 至少要有图像或环境状态之一。
2. 所有图像 shape 必须一致。
3. 如果使用 crop，裁剪尺寸必须不超过输入图像尺寸。

这说明 Diffusion 的输入编码对视觉 shape 一致性要求很明确，和后面 `DiffusionRgbEncoder` 的批量编码逻辑直接相关。

---

## 五、dataset 视角：Diffusion 为什么比 ACT 更依赖时间窗口

这一阶段最好把配置文件和第 2 阶段的 dataset 学习结果连起来看。

Diffusion 对 dataset 的要求可以概括成一句话：

它不是在学“当前观测到当前动作”的映射，而是在学“短历史观测到未来整段动作轨迹”的条件生成。

所以数据样本里至少要稳定提供两类信息：

1. 观测历史窗口
2. 动作 horizon 窗口

结合默认配置，dataset 给 Diffusion 提供的 supervision 近似可以理解为：

- 观测：`t-1, t`
- 动作：`t-1, t, t+1, ..., t+14`

然后在推理时，真正执行的是这段动作中的一部分。

这也是为什么 Diffusion 比 ACT 更强调下面这些字段之间的配合：

- `n_obs_steps`
- `horizon`
- `n_action_steps`
- `drop_n_last_frames`
- `action_is_pad`

在 ACT 中，你也需要时序窗口；但在 Diffusion 中，这套窗口关系几乎就是算法本体的一部分。

---

## 六、processor 层：Diffusion 其实高度复用了平台公共能力

这一阶段一个很重要的收获是：

虽然 Diffusion 算法本身和 ACT 差异很大，但在 LeRobot 平台里，它们在 processor 层反而高度相似。

`processor_diffusion.py` 中的 `make_diffusion_pre_post_processors(...)` 基本延续了平台的标准 policy processor 方案。

### 6.1 pre-processor

Diffusion 的前处理管线主要包含：

1. `RenameObservationsProcessorStep`
2. `AddBatchDimensionProcessorStep`
3. `DeviceProcessorStep`
4. `NormalizerProcessorStep`

这意味着：

- 输入命名仍然先统一到平台约定的 feature key。
- 单样本推理时仍然先补 batch 维。
- device 放置仍然在 processor 中做，而不是散落到 policy 内部。
- 归一化依然基于 dataset stats。

### 6.2 post-processor

后处理管线主要包含：

1. `UnnormalizerProcessorStep`
2. `DeviceProcessorStep(device="cpu")`

也就是说，动作输出在送给 env / robot 之前，仍然先反归一化，再转回 CPU。

### 6.3 这一点为什么重要

这正是 LeRobot 平台价值最直观的体现：

ACT 和 Diffusion 虽然模型不同、loss 不同、推理不同，但对外仍然遵守同一套 processor 契约。所以 env、robot、checkpoint、from_pretrained 等平台能力几乎不需要为某个策略重写一遍。

`tests/processor/test_diffusion_processor.py` 也正是在验证这层契约，包括：

1. pre/post processor 的 step 顺序。
2. 图像和状态输入的 batch 形状。
3. 设备迁移是否正确。
4. identity normalization 情况下数据是否保持不变。
5. processor save/load 后是否还能复原行为。

从学习角度说，这说明你在第 3 阶段理解的 processor 体系已经可以复用到具体策略学习中，而不是每个 policy 都要重新建立一套心智模型。

---

## 七、策略外壳：`DiffusionPolicy` 在平台中的职责

`DiffusionPolicy` 定义在 [src/lerobot/policies/diffusion/modeling_diffusion.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py)。

它不是完整算法本体，而是策略层的“平台适配外壳”，主要负责：

1. 维护推理时的 observation/action 队列。
2. 把 batch 中的多模态特征整理为模型需要的统一格式。
3. 调用内部 `DiffusionModel` 做训练或生成。
4. 向上层暴露统一的 `forward()` 和 `select_action()` 接口。

### 7.1 `reset()`

`reset()` 会初始化若干 deque：

- 观测状态队列
- 动作队列
- 图像队列
- 可选的环境状态队列

这表明 Diffusion 的在线推理不是无状态的一步前向，而是显式依赖短历史缓存。

### 7.2 `forward(batch)`

训练时的 `forward()` 主要做两件事：

1. 把多路图像特征整理到统一的 `OBS_IMAGES` 键下。
2. 调用 `self.diffusion.compute_loss(batch)` 返回 loss。

所以训练入口非常“薄”，真正复杂的逻辑都下沉到了 `DiffusionModel`。

### 7.3 `select_action(batch)`

这部分是理解 Diffusion 推理流程的关键。

高层逻辑可以概括为：

1. 把当前观测写入 observation queues。
2. 如果 action queue 为空，就根据最近 `n_obs_steps` 观测生成一段动作。
3. 把这段动作拆开放入队列。
4. 每次只 `popleft()` 返回一步动作。

这说明 Diffusion 虽然每次生成的是一段动作，但对 env / robot 暴露的仍是统一的一步 action 接口。平台上层完全不需要知道底层策略到底是一步预测、chunk 预测还是 diffusion sampling。

---

## 八、模型本体：`DiffusionModel` 在做什么

真正的算法主体在同一文件中的 `DiffusionModel`。

这一层建议你按“编码条件 -> 加噪训练 -> 去噪采样”三个部分来理解。

### 8.1 条件编码：把观测历史变成 `global_cond`

`_prepare_global_conditioning()` 的职责是把历史观测编码成全局条件向量。

可能进入条件编码的内容包括：

1. 机器人状态 `observation.state`
2. 图像特征 `observation.images`
3. 环境状态 `observation.environment_state`

图像部分会先经过 `DiffusionRgbEncoder`：

- 可选 resize
- 可选 random crop
- ResNet backbone
- 可选用 GroupNorm 替换 BatchNorm
- `SpatialSoftmax`
- 线性层输出固定维度特征

最后，多个时间步的观测特征会被拼接并展平，形成 `global_cond_dim * n_obs_steps` 大小的条件向量。

这一点很关键：  
Diffusion 在 LeRobot 中不是把每个时间步单独喂给 U-Net，而是先把 observation history 编码成一个统一的全局条件，再条件化动作轨迹生成。

### 8.2 动作生成器：`DiffusionConditionalUnet1d`

动作模型主体是一个一维条件 U-Net。

它的输入不是图像，而是“动作序列张量”：

- 形状大致为 `(batch, horizon, action_dim)`

训练时，这个动作序列会先被加噪；
推理时，它从纯高斯噪声开始。

U-Net 的条件信息来自两部分：

1. diffusion timestep embedding
2. `global_cond`

模型通过 FiLM 风格的调制把这些条件注入卷积块，从而学会“在给定观测历史时，如何把噪声逐步变成合理动作轨迹”。

### 8.3 `conditional_sample()`：逆扩散采样过程

`conditional_sample(...)` 负责从噪声开始运行整个逆扩散过程。

它的思路是：

1. 初始化动作序列噪声 `x_T`
2. 按 scheduler 的时间步从大到小迭代
3. 每一步用 U-Net 预测当前应去掉的噪声或重建目标
4. 用 scheduler 更新到下一步更干净的样本
5. 最终得到长度为 `horizon` 的动作轨迹

这部分就是 Diffusion 和 ACT 最本质的推理差异：

- ACT：一次前向直接输出动作 chunk
- Diffusion：多步迭代采样后得到动作 chunk

### 8.4 `generate_actions()`：从整条轨迹切出可执行片段

`generate_actions()` 在生成完整动作轨迹后，不会直接全部执行，而是做一次切片：

```python
start = n_obs_steps - 1
end = start + n_action_steps
actions[:, start:end]
```

默认配置下就是从长度 16 的轨迹中，取索引 `1:9` 的 8 步动作。

这一步非常值得单独记住，因为它解释了前面那些时间索引为什么那样设计：

模型内部生成的是一个和 observation history 对齐的完整 horizon，而不是只生成“当前之后的未来动作”。真正送给控制器执行的是其中和当前时刻对齐的那一小段。

---

## 九、训练目标：Diffusion 的 loss 和 ACT 完全不同

`compute_loss(batch)` 是这个策略最值得精读的训练逻辑。

它大致分为 6 步：

1. 读取 batch 中的 `OBS_STATE`、`ACTION` 和 `action_is_pad`
2. 准备条件向量 `global_cond`
3. 对真实动作序列采样高斯噪声
4. 随机采样 diffusion timestep
5. 用 scheduler 将噪声加入到干净动作序列中
6. 让 U-Net 预测噪声或样本，并用 MSE 计算损失

其中关键变量是 `prediction_type`：

- 如果是 `epsilon`，模型学习预测噪声
- 如果是 `sample`，模型学习预测干净动作样本

loss 形式是逐元素 MSE，最后再求平均。

### 9.1 `action_is_pad` 的意义

如果 `do_mask_loss_for_padding` 为真，loss 会只在非 padding 位置上计算。

这说明 Diffusion 和 ACT 一样，也必须显式处理 episode 边界处的不完整监督；只是 Diffusion 因为 horizon 更长，对 padding 问题更敏感。

### 9.2 和 ACT 的 loss 对比

ACT 中更像是：

- 直接回归目标动作
- 主要损失是 L1
- 还可能叠加 latent KL 项

Diffusion 中则是：

- 对动作轨迹加噪
- 学习逆扩散目标
- 主要损失是 MSE

所以从训练目标上看，这两者已经不是“两个实现不同的时序策略”，而是“两个学习范式不同的策略家族”。

---

## 十、推理流程：从当前观测到一步动作

这一部分建议你能自己复述出来，因为它最能体现 DiffusionPolicy 的平台接口设计。

一次 `select_action()` 的简化流程如下：

1. 当前观测进入 pre-processor，完成重命名、加 batch、归一化和上设备。
2. `DiffusionPolicy` 把新观测写入 observation queues。
3. 如果 action queue 为空，则收集最近 `n_obs_steps` 的观测历史。
4. `DiffusionModel.generate_actions()` 从噪声出发采样出整段动作轨迹。
5. 从整段轨迹中切出 `n_action_steps` 段可执行动作。
6. 这些动作被逐个压入 action queue。
7. 本次调用返回队首的一步动作。
8. post-processor 对动作反归一化并转回 CPU。

你可以把它理解成：

Diffusion 是“批量生成，逐步消费”的控制策略。

这一点和 ACT 的 chunk queue 有相似之处，但内部动作来源不同：

- ACT 的 queue 里放的是一次前向直接预测的 chunk
- Diffusion 的 queue 里放的是多步采样得到的去噪轨迹切片

---

## 十一、ACT vs Diffusion：这一阶段最重要的对照表

下面这张表是本阶段最核心的学习结果。

| 维度 | ACT | Diffusion |
| --- | --- | --- |
| 策略范式 | 直接动作 chunk 预测 | 条件扩散动作轨迹生成 |
| 默认观测步数 | 通常 1 | 默认 2 |
| 输出对象 | 未来一段动作 | 完整 horizon 动作轨迹 |
| 真正执行 | chunk 中的动作 | 从 horizon 切出的 `n_action_steps` |
| 训练目标 | 动作回归，常见为 L1 | 噪声/样本预测，MSE |
| 模型主体 | Transformer/VAE 风格 | Conditional 1D U-Net |
| 推理复杂度 | 一次前向即可 | 需要多步 reverse diffusion |
| 队列机制 | action queue，可配 temporal ensemble | observation queue + action queue |
| 对时间窗口依赖 | 强 | 更强 |
| 对 padding 敏感度 | 高 | 更高 |

### 11.1 两者的共同点

尽管算法差异很大，但在 LeRobot 里它们仍然共享：

1. 同一套 dataset 时间索引机制。
2. 同一套 processor pipeline 思想。
3. 同一个 `PreTrainedPolicy` 抽象。
4. 同样的 `forward()` / `select_action()` 平台接口。
5. 同样依赖 dataset stats 做标准化。

### 11.2 两者的真正区别

真正的区别不在“代码文件夹不一样”，而在这三点：

1. 如何定义动作学习目标  
   ACT 直接学动作；Diffusion 学去噪过程。

2. 如何利用时间  
   ACT 更像直接预测未来片段；Diffusion 更像对整段轨迹建模。

3. 如何推理  
   ACT 是前向回归；Diffusion 是迭代采样。

当你把这三点想清楚后，再看后续策略就不会只停留在“不同 policy 目录长得不一样”这种表层印象。

---

## 十二、测试文件告诉了我们什么

虽然这一阶段主角是源码，但测试依然很重要。

### 12.1 `tests/processor/test_diffusion_processor.py`

这个文件主要帮助确认：

1. Diffusion 的 processor 仍然遵守平台统一契约。
2. 归一化与反归一化确实围绕 dataset stats 工作。
3. 设备切换和 batch shape 管理是稳定的。
4. save/load 后的 processor 仍然可恢复。

### 12.2 `tests/policies/test_policies.py`

这个文件更像是平台层验证：

1. Diffusion 是否能像其他 policy 一样实例化。
2. `forward()` 是否能跑通。
3. `select_action()` 是否满足统一接口。
4. batch 是否被策略意外修改。
5. 策略是否能参与统一的 env rollout 测试。

这再次说明：  
LeRobot 不是把 Diffusion 当成一个“特殊脚本”，而是把它纳入了完整的统一策略协议中。

---

## 十三、这一阶段你应该真正掌握的 6 个问题

完成这一阶段后，你最好能不看代码就回答下面这些问题：

1. `n_obs_steps`、`horizon`、`n_action_steps` 三者分别控制什么？
2. 为什么 `action_delta_indices` 要和 observation history 对齐，而不是只取未来动作？
3. `drop_n_last_frames` 为什么对 Diffusion 特别重要？
4. `DiffusionPolicy` 和 `DiffusionModel` 的职责边界在哪里？
5. `generate_actions()` 为什么要先生成完整 horizon，再切片输出？
6. Diffusion 在 LeRobot 中与 ACT 最大的共同点和最大差异分别是什么？

如果这 6 个问题都能答清楚，你对 LeRobot 中“第二类时序策略”的理解就已经建立起来了。

---

## 十四、建议的巩固练习

这一阶段建议做 3 个纸面练习，不需要跑代码也很有帮助。

### 练习 1：手动画时间轴

假设：

- `n_obs_steps = 2`
- `horizon = 16`
- `n_action_steps = 8`

自己画出：

- observation 取哪些时刻
- action horizon 覆盖哪些时刻
- 最终执行的是哪一段动作

只要这张图画清楚，Diffusion 的很多实现细节都会自然顺下来。

### 练习 2：写一张 ACT vs Diffusion 映射表

建议列出这些列：

- 论文概念
- ACT 对应实现
- Diffusion 对应实现
- 所在文件

这张表会在第 7 阶段学习 VLA 时非常有用。

### 练习 3：自己复述一次 `select_action()`

要求不用源码，直接口头或笔记复述：

“从一个新观测进入系统，到最终吐出一步动作，中间经过了哪些缓存、编码、采样和切片过程？”

如果复述顺畅，说明你已经真正理解推理链路，而不是只记住了几个函数名。

---

## 十五、本阶段总结

第 5 阶段最大的学习收获，不是“我又读完了一个 policy”，而是开始真正理解 LeRobot 作为策略平台的设计能力：

1. dataset 与 processor 负责统一输入输出协议。
2. policy 外壳负责统一训练/推理接口。
3. 不同算法只需要把自己的时序建模与 loss 逻辑嵌入进来。

在这个框架下，ACT 和 Diffusion 虽然学习范式完全不同，但都能以相同方式被训练、保存、加载、评估和接入机器人控制链路。

如果说第 4 阶段帮你建立了“第一个具体策略”的理解，那么第 5 阶段真正帮你建立的是：

“我开始能比较不同策略在 LeRobot 里的共性与个性了。”

---

## 十六、下一阶段建议

下一步最自然的是进入第 6 阶段：`env / eval` 主线学习。

因为到目前为止，我们已经理解了：

1. train 主链
2. dataset 主链
3. processor 主链
4. ACT 与 Diffusion 两类代表性 policy

接下来最值得补上的就是：

策略是如何真正进入 rollout、如何与 env 交互、以及 LeRobot 怎样把仿真评测和真实控制统一到同一条上层链路中。

