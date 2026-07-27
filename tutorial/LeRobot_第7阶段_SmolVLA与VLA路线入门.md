# LeRobot 第7阶段：SmolVLA与VLA路线入门

## 一、阶段目标

第 7 阶段的目标，不再是继续比较经典 imitation policy，而是开始进入 LeRobot 的 VLA 路线，重点回答两个问题：

1. 语言任务、视觉观测、机器人状态和动作序列，如何在同一套平台里汇合？
2. LeRobot 是如何把一个更接近 foundation model 的策略，纳入已有的 `config + processor + policy + eval` 框架中的？

这一阶段的核心文件：

- [docs/source/policy_smolvla_README.md](/Users/zwy/Macin/lerobot/docs/source/policy_smolvla_README.md)
- [docs/source/smolvla.mdx](/Users/zwy/Macin/lerobot/docs/source/smolvla.mdx)
- [src/lerobot/policies/smolvla/configuration_smolvla.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py)
- [src/lerobot/policies/smolvla/processor_smolvla.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/smolvla/processor_smolvla.py)
- [src/lerobot/policies/smolvla/modeling_smolvla.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py)
- [src/lerobot/policies/smolvla/smolvlm_with_expert.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py)
- [tests/processor/test_smolvla_processor.py](/Users/zwy/Macin/lerobot/tests/processor/test_smolvla_processor.py)

这一阶段最重要的收获可以先提前说出来：

SmolVLA 不是“加了文本输入的 ACT”，也不是“语言版 Diffusion”。它在 LeRobot 中对应的是一条新的策略路线：

`多相机图像 + 机器人状态 + 自然语言任务 -> VLM前缀条件 -> action expert -> 动作chunk`

---

## 二、先建立高层认识：SmolVLA 在项目里扮演什么角色

`docs/source/smolvla.mdx` 对 SmolVLA 的定位非常明确：

1. 它是 Hugging Face 面向机器人场景的轻量级 foundation model。
2. 它不是开箱即用的“万能机器人大模型”，而是一个适合在 LeRobot dataset 上 fine-tune 的 base model。
3. 它的输入天然是多模态的：
   - 多个 camera views
   - 当前 sensorimotor state
   - 自然语言 instruction
4. 它的输出不是单步动作，而是一段 action chunk。

这个定位很重要，因为它决定了你读代码时的正确期待：

不要把它当成一个“更大的 policy 子目录”，而要把它看成 LeRobot 把 VLM 主干接入机器人控制的第一个系统性例子。

---

## 三、论文与文档视角下应该先抓住什么

`docs/source/policy_smolvla_README.md` 提供了论文链接，`docs/source/smolvla.mdx` 则给了工程使用路径。

如果从学习代码的角度去读论文和文档，最值得先抓住 5 个概念：

1. 多模态输入  
   图像、语言和状态都不是附属输入，而是策略本体的一部分。

2. action chunking  
   SmolVLA 输出的是一段动作，而不是一步动作。

3. pretrained VLM backbone  
   不是从零实现视觉和语言理解，而是建立在已有 VLM 之上。

4. action expert  
   机器人动作预测不是直接让 VLM 头来做，而是接了一个专门的动作专家模块。

5. fine-tuning friendly  
   整个设计强调可以在 LeRobot 数据集上高效微调，而不是只适合超大规模预训练。

你可以先建立一张最关键的映射表：

| 概念 | LeRobot 对应位置 |
| --- | --- |
| 图像/状态/语言联合输入 | `processor_smolvla.py` + `prepare_images()` + `prepare_state()` |
| 语言 tokenizer | `TokenizerProcessorStep` |
| VLM 主干 | `SmolVLMWithExpertModel.vlm` |
| action expert | `SmolVLMWithExpertModel.lm_expert` |
| 动作 chunk 生成 | `sample_actions()` |
| 训练目标 | `VLAFlowMatching.forward()` |

---

## 四、配置层：SmolVLA 首先是一个多模态时序策略

SmolVLA 的配置定义在 [src/lerobot/policies/smolvla/configuration_smolvla.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py)。

这个文件最值得注意的，不是参数数量多，而是它同时控制了：

1. 时序结构
2. 多模态输入尺寸
3. 微调策略
4. VLM 主干行为
5. 动作生成方式

### 4.1 时序结构

最关键的时间字段是：

- `n_obs_steps = 1`
- `chunk_size = 50`
- `n_action_steps = 50`

这说明在默认设置下：

1. SmolVLA 主要依赖当前观测，而不是较长 observation history。
2. 每次生成长度为 50 的动作 chunk。
3. 默认情况下，一次生成出来的 chunk 会完整作为执行窗口。

这一点和前面的策略很有意思：

- 它和 ACT 一样有明显的 chunking 语义
- 但模型内部不是 ACT 的回归形式
- 也不像 Diffusion 那样强依赖 observation history

### 4.2 特征与归一化

默认归一化策略是：

- `VISUAL -> IDENTITY`
- `STATE -> MEAN_STD`
- `ACTION -> MEAN_STD`

图像保持 identity 非常值得注意，因为 SmolVLA 自己在模型内部会再做符合 VLM 预期的图像预处理，例如把 `[0,1]` 变成 `[-1,1]`。

也就是说：

- 平台 processor 只负责基础标准化契约
- 模型内部还会根据 backbone 需求做更专门的预处理

### 4.3 维度上限

配置里还有两个很重要的字段：

- `max_state_dim = 32`
- `max_action_dim = 32`

它们的意义不是“真实状态一定有 32 维”，而是：

SmolVLA 在平台中希望把不同机器人状态/动作空间统一 pad 到一个固定上界，以便动作专家和投影层使用固定宽度。

这体现了 foundation-style policy 常见的一种设计：  
输入输出空间要先被投影到统一 latent interface。

### 4.4 VLM 与微调相关配置

配置还显式控制：

- `vlm_model_name`
- `load_vlm_weights`
- `freeze_vision_encoder`
- `train_expert_only`
- `train_state_proj`

这些字段一起说明了 SmolVLA 不是简单的 end-to-end 全量训练模型，而是明确支持分层微调策略：

1. 只训练 action expert
2. 冻结视觉编码器
3. 只训练状态投影层
4. 或加载已有 SmolVLA 权重继续微调

这对理解 VLA 实战很重要，因为 foundation model 训练的难点很多时候不在模型结构，而在“哪些部分该冻、哪些部分该调”。

---

## 五、processor 层：语言第一次成为显式的一等输入

SmolVLA 的 processor 定义在 [src/lerobot/policies/smolvla/processor_smolvla.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/smolvla/processor_smolvla.py)。

这是 VLA 路线里最值得关注的第一个代码点。

### 5.1 preprocessor 步骤

SmolVLA 的前处理链包括：

1. `RenameObservationsProcessorStep`
2. `AddBatchDimensionProcessorStep`
3. `NewLineTaskProcessorStep`
4. `TokenizerProcessorStep`
5. `DeviceProcessorStep`
6. `NormalizerProcessorStep`

这和前几阶段最大的区别是：

这里第一次把“任务文本处理”明确写进了 processor pipeline。

### 5.2 `NewLineTaskProcessorStep`

从 [tests/processor/test_smolvla_processor.py](/Users/zwy/Macin/lerobot/tests/processor/test_smolvla_processor.py) 可以很清楚看到，这个 step 的职责很简单但很关键：

- 确保 task 字符串结尾带换行符

它支持：

1. 单个 task 字符串
2. task 列表
3. 空 task / 无 task 情况

这说明在 SmolVLA 中，语言输入不是“有就带上”，而是有稳定文本格式要求的正式输入通道。

### 5.3 `TokenizerProcessorStep`

接下来 processor 会直接把任务文本送入 tokenizer，生成：

- 语言 token
- attention mask

这一步的意义非常大：

从这一阶段开始，policy 的输入已经不再只是张量观测，而是“观测张量 + 文本 token 序列”的组合 batch。

换句话说，SmolVLA 把语言理解纳入了 policy 输入契约，而不只是某个下游 hack。

### 5.4 postprocessor

后处理仍然是平台熟悉的模式：

1. `UnnormalizerProcessorStep`
2. `DeviceProcessorStep(device="cpu")`

这说明虽然输入变复杂了，但动作输出仍然遵守 LeRobot 的统一动作后处理协议。

---

## 六、从测试文件看 processor 契约

`tests/processor/test_smolvla_processor.py` 很值得认真看，因为它几乎就是 SmolVLA processor 的契约说明。

它帮助我们确认了几件事：

1. preprocessor 确实包含 6 个步骤，且顺序稳定。
2. 语言任务 newline 处理是明确且被测试覆盖的。
3. tokenizer step 可以被替换或 mock，说明 processor 设计是可组合的。
4. device 切换和归一化依然遵守平台公共逻辑。

从学习角度说，这个测试文件告诉我们：

SmolVLA 并没有逃离前面学过的 processor 平台，相反，它是在这个平台上第一次系统引入语言模态。

---

## 七、策略外壳：`SmolVLAPolicy` 的职责

`SmolVLAPolicy` 定义在 [src/lerobot/policies/smolvla/modeling_smolvla.py](/Users/zwy/Macin/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py)。

和前面的策略一样，它仍然是 `PreTrainedPolicy` 的一个子类，所以对平台上层仍然暴露统一接口。

但内部职责已经明显更复杂：

1. 组织图像、状态和语言输入
2. 调用 VLA 模型主体
3. 管理 action queue
4. 处理状态/动作 padding
5. 可选进行 Aloha 特定空间变换
6. 可选接入 RTC 实时 chunking 逻辑

### 7.1 `reset()`

`reset()` 会初始化 action queue。

这说明 SmolVLA 在执行层面依然属于 chunk-based policy：  
一次预测多个动作，逐步从队列中消费。

### 7.2 `select_action()`

`select_action()` 的高层逻辑和 ACT 有亲缘关系：

1. 如果 action queue 为空，就生成一段 action chunk
2. 把 chunk 压入队列
3. 每次只弹出一步动作

但它的生成来源与 ACT 完全不同，因为这里的动作 chunk 来自多模态 VLM + expert 的生成过程。

### 7.3 `predict_action_chunk()`

这是理解 SmolVLA 的更好入口，因为它直接暴露了“给定一批观测，生成整段动作”的能力。

高层流程是：

1. 准备 batch
2. `prepare_images()`
3. `prepare_state()`
4. 读取 tokenizer 生成的语言 token 和 mask
5. `model.sample_actions(...)`
6. 去掉 pad 出来的额外 action 维度

所以在策略层的角度，SmolVLA 的动作生成接口已经非常清楚：

`多模态输入 -> sample_actions -> 动作chunk`

### 7.4 `forward()`

训练时的 `forward()` 会：

1. 处理图像、状态、语言 token 和目标动作
2. 调用 `self.model.forward(...)`
3. 根据真实 action 维度裁掉 pad 出来的额外部分
4. 用 `action_is_pad` 掩掉 episode 边界外的无效监督

这说明虽然 SmolVLA 是 foundation-style policy，但在训练入口形式上仍然兼容 LeRobot 统一的：

- `forward(batch) -> loss`

---

## 八、输入准备：图像、状态、动作如何被统一

这一部分是 SmolVLA 真正“平台化”的关键。

### 8.1 `prepare_images()`

这个函数做了几件很重要的事情：

1. 从 batch 中取出所有图像特征
2. 如果是时序图像，只取最后一个时间步
3. resize 并 padding 到固定尺寸
4. 把像素范围从 `[0,1]` 转成 `[-1,1]`
5. 为每路图像创建对应 mask
6. 对缺失相机可以补空图像

这里能看出两个重要思想：

1. VLM backbone 需要自己的视觉预处理规范
2. 多相机输入必须在进入 backbone 前被整理成稳定列表和 mask

所以 SmolVLA 的图像处理不只是“把图片送进去”，而是在为一个 foundation backbone 构建多视角前缀。

### 8.2 `prepare_state()`

状态处理更简单但同样重要：

1. 取当前状态
2. pad 到 `max_state_dim`

### 8.3 `prepare_action()`

训练目标动作会 pad 到 `max_action_dim`。

所以这里的状态和动作都经过了一个“真实机器人空间 -> 统一模型空间”的对齐过程。

这一步是 VLA policy 与前面较小策略的一个关键区别：

模型不是围绕单一机器人定制固定维度，而是围绕“统一 latent interface”来设计输入输出。

---

## 九、模型主体：`VLAFlowMatching`

SmolVLA 的真正动作建模核心在 `VLAFlowMatching`。

这个名字非常重要，因为它提醒你：

SmolVLA 在 LeRobot 里并不是 diffusion policy，而是 flow matching 风格的动作生成模型。

### 9.1 高层结构

`VLAFlowMatching` 包含几个关键模块：

1. `SmolVLMWithExpertModel`
2. `state_proj`
3. `action_in_proj`
4. `action_out_proj`
5. 时间相关 MLP

可以把它理解成：

`VLM负责理解多模态上下文，expert 负责在这个上下文中生成动作流场`

### 9.2 `embed_prefix()`

这个函数负责构建“前缀条件”，包括：

1. 图像 embedding
2. 语言 token embedding
3. 状态 embedding

然后还会构建：

1. pad masks
2. attention masks
3. position ids 所需的序列结构

所以 prefix 的本质是：

把图像、文本和状态拼成一个统一的条件 token 序列，交给 VLM 侧处理。

### 9.3 `embed_suffix()`

suffix 部分对应动作生成侧。

它会把：

1. noisy actions
2. timestep embedding

融合成动作 expert 的输入 token。

这就把整个模型的结构分成了非常清晰的两半：

1. prefix：环境与任务条件
2. suffix：待生成动作序列

---

## 十、训练目标：SmolVLA 学的是 flow matching

`VLAFlowMatching.forward()` 的逻辑非常值得单独整理。

它大致做这些事：

1. 对真实动作采样噪声 `noise`
2. 采样一个连续时间 `time`
3. 构造插值状态：
   - `x_t = t * noise + (1 - t) * actions`
4. 构造目标速度场：
   - `u_t = noise - actions`
5. 用 prefix 条件和 suffix 动作 token 做前向
6. 输出 `v_t`
7. 用 MSE 让 `v_t` 拟合 `u_t`

这意味着：

SmolVLA 训练时不是直接回归动作，也不是像 Diffusion 那样走离散去噪 scheduler，而是在学习连续时间下的动作流场。

这是这一阶段最重要的算法层理解点。

### 10.1 和前面策略的区别

你可以这样对比：

- ACT：直接动作回归
- Diffusion：离散时间去噪
- SmolVLA：连续时间 flow matching

所以到第 7 阶段，你已经见到了 LeRobot 平台承载三种不同动作学习范式的能力。

---

## 十一、推理：`sample_actions()` 如何生成动作chunk

推理逻辑在 `VLAFlowMatching.sample_actions()` 中。

它的思路可以总结为：

1. 先对 prefix 条件做一次编码
2. 缓存 prefix 的 key/value
3. 从噪声动作序列开始
4. 用多个时间步迭代更新 `x_t`
5. 最终得到动作 chunk

### 11.1 prefix cache

推理时会先把图像、语言和状态构成的 prefix 送进 `vlm_with_expert.forward(...)`，并生成 `past_key_values`。

这个 cache 的意义是：

后续每一步动作去噪或流场更新时，不需要重复重新编码整段多模态上下文。

所以这里你会看到一种典型的 foundation model 推理优化：

把静态条件缓存起来，把计算集中在动作 suffix 的多步迭代上。

### 11.2 多步更新

然后模型会从初始噪声 `x_t` 出发，循环 `num_steps` 次。

每一步：

1. 根据当前时间 `t`
2. 调用 `denoise_step(...)`
3. 得到当前速度场 `v_t`
4. 更新 `x_t = x_t + dt * v_t`

这和 Diffusion 的“逆扩散多步采样”在形式上有一点相似，但本质上依然是另一种连续时间建模方法。

### 11.3 输出形式

最终输出是：

- `(batch, chunk_size, action_dim)`

然后策略层再按需要：

1. 去掉 pad 的 action 维度
2. 压入 action queue
3. 逐步弹出执行

---

## 十二、`SmolVLMWithExpertModel`：为什么叫 VLM with Expert

这个文件是理解 SmolVLA 架构最关键的一层。

`SmolVLMWithExpertModel` 的核心思想可以概括成一句话：

不是直接拿现成 VLM 输出动作，而是在 VLM 旁边接入一个专门的动作 expert，并在注意力层面让两者协作。

### 12.1 组成部分

它内部至少有两大块：

1. `vlm`
   来自 `AutoModelForImageTextToText` 的主干

2. `lm_expert`
   一个更小的、面向动作建模的 expert 网络

### 12.2 为什么要有 expert

这说明项目作者并不认为“通用 VLM 自己就足够做动作控制头”。

相反，SmolVLA 的设计哲学是：

1. VLM 负责视觉和语言条件理解
2. 专门的动作 expert 负责动作序列建模

这是一种非常值得学习的架构分工，因为它保留了 foundation backbone 的表达能力，同时让动作控制部分保持专门化。

### 12.3 attention 交互方式

在 `smolvlm_with_expert.py` 中可以看到：

- self attention
- cross attention
- KV cache
- RoPE

这些逻辑说明 expert 不是简单接在 VLM 输出后面做 MLP，而是更深地参与了 token-level 表示交互。

从学习角度，你不一定需要一次吃透每个 attention 细节，但必须理解这个边界：

SmolVLA 的创新重点之一就在于“如何把 VLM 条件和动作 expert 在 token 层连接起来”。

---

## 十三、SmolVLA 在 LeRobot 平台中的真正价值

到这里，你应该能看出 SmolVLA 在 LeRobot 中的意义不只是“又多了一个模型”。

它实际上证明了 LeRobot 这套平台抽象可以承载：

1. 多相机输入
2. 文本任务输入
3. foundation backbone
4. 专门动作 expert
5. chunk-based control
6. 与现有 processor / policy / eval 接口兼容

这非常关键，因为它把 LeRobot 从“机器人 imitation learning 库”进一步推向了“机器人 foundation model 实验平台”。

---

## 十四、与前面策略的对照

这一阶段最值得做的事之一，是把 SmolVLA 与前面的 ACT、Diffusion 放在一起看。

| 维度 | ACT | Diffusion | SmolVLA |
| --- | --- | --- | --- |
| 输入模态 | 图像 + 状态 | 图像/状态 | 图像 + 状态 + 语言 |
| 主体结构 | Transformer/VAE 风格 | Conditional 1D U-Net | VLM + action expert |
| 动作建模 | 直接回归 chunk | 生成动作轨迹 | flow matching 生成动作 chunk |
| 训练目标 | L1 / KL | MSE 去噪 | MSE 拟合速度场 |
| 推理方式 | 一次前向出 chunk | 多步采样 | 多步 flow 更新 |
| 语言任务 | 通常无 | 通常无 | 一等输入 |
| 平台接口 | 统一 | 统一 | 统一 |

这张表能帮你真正看到：

LeRobot 的平台抽象不是围绕某一个算法做的，而是围绕“如何统一承载不同策略家族”做的。

---

## 十五、这一阶段你应该真正掌握的 7 个问题

完成第 7 阶段后，你最好能回答下面这些问题：

1. 为什么 SmolVLA 的 processor 里必须显式加入 tokenizer？
2. `NewLineTaskProcessorStep` 为什么存在？
3. `max_state_dim` 和 `max_action_dim` 的作用是什么？
4. `SmolVLAPolicy` 和 `VLAFlowMatching` 的职责边界在哪里？
5. prefix 和 suffix 在 SmolVLA 中分别代表什么？
6. 为什么 SmolVLA 需要 `VLM + action expert` 的分工，而不是只靠 VLM？
7. SmolVLA 的训练目标为什么更接近 flow matching，而不是 Diffusion？

如果这 7 个问题能说清楚，你就已经真正入门了 LeRobot 的 VLA 路线。

---

## 十六、建议的巩固练习

### 练习 1：画一张 SmolVLA 数据流图

建议你自己手画下面这条链：

`task text -> tokenizer -> language tokens`

`images -> image embeddings`

`state -> state projection`

然后把它们一起并入 prefix，再接到 action suffix 和 expert 上。

### 练习 2：做一个三策略对照表

把 ACT、Diffusion、SmolVLA 的：

- 输入模态
- 模型主体
- loss
- 推理方式
- 输出形式

整理成一张表。

这张表会直接帮助你后续读 Pi0 或其他 VLA 策略。

### 练习 3：复述一次 `sample_actions()`

尝试不看代码，自己解释：

1. 为什么先算 prefix cache
2. 为什么从 noise 开始
3. 为什么要多步更新
4. 最后为什么输出的是 action chunk

如果这条链能顺畅复述，说明你已经真正看懂了 SmolVLA 的推理主线。

---

## 十七、本阶段总结

第 7 阶段最大的意义，是你第一次看到了 LeRobot 如何把 foundation-style 机器人模型接入原有平台。

SmolVLA 告诉我们：

1. processor 可以扩展到处理文本任务
2. config 可以同时管理多模态、微调策略和动作建模参数
3. policy 外壳依然能保持统一接口
4. 更复杂的模型主体仍然能被纳入 `forward()` 和 `select_action()` 这套统一协议

所以这一阶段真正建立起来的不是“我读懂了 SmolVLA 的所有 attention 实现”，而是：

“我开始理解 LeRobot 如何承载 VLA 这类更大、更复杂、更多模态的策略。”

---

## 十八、下一阶段建议

原始学习计划中，第 8 阶段是按兴趣分支：

1. RL 路线
2. 真实机器人 / record / teleop 路线

如果你想继续把平台闭环补完整，我更推荐先走真实机器人分支。因为到目前为止我们已经理解了：

- 训练
- dataset
- processor
- policy
- eval
- VLA

接下来再看 `lerobot_record`、`lerobot_teleoperate`、硬件接入和 robot processor，会更容易形成“从数据采集到部署控制”的完整闭环。

这份文档基于源码与项目文档阅读整理，没有做本地运行验证，重点是帮助你建立第一个面向 VLA 的代码理解框架。

