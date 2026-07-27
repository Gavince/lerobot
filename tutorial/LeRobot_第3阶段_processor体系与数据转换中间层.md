# LeRobot 第 3 阶段学习文档：Processor 体系与数据转换中间层

## 1. 本阶段目标

第 3 阶段的核心问题是：

> LeRobot 是如何把“机器人世界的数据”和“模型世界的数据”连接起来的？

前两个阶段里我们已经知道：

- 训练主链会在 dataset 和 policy 之间调用 `preprocessor`
- eval 时既有 env processor，也有 policy processor
- checkpoint 会保存 processors

所以这一阶段要真正搞清楚：

1. processor 是什么
2. 它为什么是 LeRobot 中间层的核心
3. 它如何把 observation/action/reward 等不同表示统一到一个抽象里
4. 它为什么不仅处理数据值，还要处理 feature contract 和可序列化状态

本次学习主要基于以下材料：

- `docs/source/introduction_processors.mdx`
- `src/lerobot/processor/pipeline.py`
- `src/lerobot/processor/converters.py`
- `src/lerobot/processor/__init__.py`
- `tests/processor/test_pipeline.py`
- `tests/processor/test_policy_robot_bridge.py`
- `tests/processor/test_normalize_processor.py`
- `tests/processor/test_converters.py`
- `tests/processor/test_batch_conversion.py`

---

## 2. 先给结论：Processor 在 LeRobot 中扮演什么角色

如果先用一句话概括：

> Processor 是 LeRobot 的“通用翻译层”，它把 dataset、robot、teleoperator、env、policy 之间不兼容的数据形式统一到一条可组合、可序列化、可复用的数据流里。

再具体一点，它解决的是这几个根本问题：

1. 机器人原始观测如何变成模型可消费输入
2. 模型输出如何变成机器人/环境可执行动作
3. 文本、图像、状态、动作等多模态数据如何在统一流水线中变换
4. 不同策略/机器人/环境之间如何保持松耦合
5. 数据结构发生变化时，dataset schema 和 policy feature 定义如何同步更新

所以 processor 并不是“预处理函数集合”，而是一层真正的系统抽象。

---

## 3. 为什么 LeRobot 需要 processor

`docs/source/introduction_processors.mdx` 对这个问题解释得很清楚。

机器人学习里天然存在几类错配：

### 3.1 数据形态错配

机器人输出的是：

- 图像
- 关节状态
- 末端执行器位姿
- 文本任务描述

模型希望看到的是：

- batched tensors
- 标准化后的数值范围
- tokenized 文本
- 放到正确 device 上的数据

### 3.2 动作表示错配

模型可能输出：

- 归一化动作
- tensor 形式动作
- 相对动作
- 末端执行器动作

但机器人/环境可能需要的是：

- 未归一化动作
- dict 形式动作
- 绝对动作
- 关节空间命令

### 3.3 跨域错配

还会有更多工程层面的不一致：

- 训练数据中的 observation key 和部署时不同
- 一个 policy 训练在某种 camera 配置上，但部署到另一种布局
- 不同机器人使用不同命名约定

LeRobot 的回答是：

> 不把这些适配逻辑写死在模型里，而是把它们拆成可组合的 processor steps。

---

## 4. `EnvTransition`：所有 processor 的统一中间表示

processor 文档里首先强调的是：

- `EnvTransition`

这是整个 processor 体系最重要的概念。

## 4.1 为什么需要统一中间表示

因为 processor 接到的原始输入可能是很多种：

- dataset batch
- robot observation dict
- robot action dict
- policy 输出 tensor

如果每个 processor step 都直接处理这些异构输入，系统会非常难扩展。

所以 LeRobot 先把所有东西转换成一个统一结构：

- `OBSERVATION`
- `ACTION`
- `REWARD`
- `DONE`
- `TRUNCATED`
- `INFO`
- `COMPLEMENTARY_DATA`

这样后续 processor steps 就不用关心“我面对的是 dataset 还是 robot”，而只需要关心自己处理 transition 的哪个部分。

## 4.2 `create_transition`

在 `converters.py` 中，`create_transition(...)` 提供了一个统一构造函数，给所有字段设置了合理默认值。

这说明 LeRobot 并不把 transition 当成临时 dict，而是当成 processor 层的正式数据结构。

---

## 5. converter：把异构输入统一映射到 transition，再映射回去

`src/lerobot/processor/converters.py` 里的 converter 是 processor 体系的第一道门。

## 5.1 converter 的核心思想

processor pipeline 自身只想处理 `EnvTransition`；
但 pipeline 的输入输出并不总是 transition。

所以系统需要两类函数：

1. `to_transition`
2. `to_output`

这也是 `DataProcessorPipeline` 构造参数中的关键项。

## 5.2 常见 converter 类型

文档和 `__init__.py` 里列出的最常见几类是：

### Robot / hardware 侧

- `observation_to_transition`
- `robot_action_to_transition`
- `robot_action_observation_to_transition`
- `transition_to_robot_action`
- `transition_to_observation`

### Policy / batch 侧

- `batch_to_transition`
- `transition_to_batch`
- `policy_action_to_transition`
- `transition_to_policy_action`

### 通用工具

- `create_transition`
- `identity_transition`
- `to_tensor`

## 5.3 `batch_to_transition` 为什么重要

dataset 返回的是 batch dict，例如：

- `observation.state`
- `observation.images.left`
- `action`
- `reward`
- `task`
- `index`

而 processor steps 想看到的是统一 transition。

`batch_to_transition` 做的核心工作包括：

1. 把 `observation.*` 键收拢到 `TransitionKey.OBSERVATION`
2. 把 `action` 放到 `TransitionKey.ACTION`
3. 把 `reward/done/truncated/info` 映射到标准字段
4. 把 `task/index/task_index/episode_index/...` 收到 `COMPLEMENTARY_DATA`

这一点非常重要，因为它说明 complementary_data 是专门用来容纳“不是 observation/action 本体，但训练和部署仍然需要”的辅助信息。

## 5.4 测试验证了什么

`tests/processor/test_batch_conversion.py` 和 `tests/processor/test_converters.py` 清晰地说明：

- `observation.*` 键会被正确分组与展开
- round-trip (`batch -> transition -> batch`) 应保持契约
- `index/task_index` 应该进入 `COMPLEMENTARY_DATA`
- 没有 observation 时应该返回 `None`

这表明 converter 不只是便利函数，而是 processor 体系稳定的数据协议层。

---

## 6. `to_tensor`：数值转换基础设施

`to_tensor` 使用 `singledispatch` 实现，对不同输入类型做统一转换。

它支持：

- `torch.Tensor`
- `numpy.ndarray`
- numpy 标量
- Python 标量
- list / tuple
- 嵌套 dict

它的价值在于：

1. 统一 processor 内部的数据类型入口
2. 让很多 processor 不必反复处理 dtype/device 细节
3. 支持嵌套统计量结构的批量张量化

`tests/processor/test_converters.py` 对这一点覆盖很全，也间接说明 normalization 这类 processor 会大量依赖这个函数。

---

## 7. `ProcessorStep`：最小可组合变换单元

在 `pipeline.py` 中，`ProcessorStep` 是所有 processor 的抽象基类。

## 7.1 它要求的两个核心方法

每个 step 至少要实现：

1. `__call__(transition) -> transition`
2. `transform_features(features) -> features`

这里的设计非常值得注意。

### `__call__`

负责真正处理数据值。

### `transform_features`

负责声明：

- 这个 step 会如何改变 feature 的 shape/type/modality

也就是说，processor step 不只是“改数据”，还要“声明自己如何改数据结构”。

这对 dataset 创建和 policy 输入输出定义非常重要。

## 7.2 可选的状态与配置接口

`ProcessorStep` 还提供了可选方法：

- `get_config()`
- `state_dict()`
- `load_state_dict()`
- `reset()`

这说明 step 可以是：

- 纯函数式的
- 也可以是有状态、可保存、可恢复的

例如 normalization step 这种组件，就非常适合有状态。

## 7.3 `transition` 属性

step 内部还能通过 `self.transition` 访问最近处理的 transition。

这使得某些 step 可以在处理一个字段时参考其他字段，而不用额外改函数签名。

---

## 8. `DataProcessorPipeline`：统一流水线执行器

`DataProcessorPipeline[TInput, TOutput]` 是 processor 体系的核心 orchestrator。

## 8.1 它做了什么

一条 pipeline 的执行过程很清楚：

1. 用 `to_transition` 把输入转成 `EnvTransition`
2. 按顺序执行所有 `ProcessorStep`
3. 用 `to_output` 把最终 transition 转回目标输出格式

也就是说，processor pipeline 本身是一个“外面类型自由、里面 transition 统一”的结构。

## 8.2 为什么这种设计很强

因为它天然支持：

- dataset batch -> transition -> batch
- robot observation -> transition -> robot observation / policy batch
- policy tensor -> transition -> robot action

统一点在中间，灵活点在两头。

## 8.3 `step_through`

`step_through(data)` 可以逐步产出每个 step 后的 transition。

这是很好的调试接口，也说明这个框架从设计上考虑了 pipeline 可观测性。

## 8.4 hooks

pipeline 还支持：

- `before_step_hooks`
- `after_step_hooks`

这类扩展点很工程化，适合调试、记录中间态或做额外监控。

---

## 9. pipeline 的另一个关键职责：可序列化

`DataProcessorPipeline` 一个特别重要的能力是：

- `save_pretrained(...)`
- `from_pretrained(...)`

## 9.1 为什么 processor 要能保存和恢复

因为在 LeRobot 里：

- processor 是训练资产的一部分
- checkpoint 不只是模型，也包括 processor
- 训练和部署需要共享同一套数据变换逻辑

如果 processor 不能序列化，训练好的模型很难稳定复现推理行为。

## 9.2 保存时会保存什么

`save_pretrained` / `_save_pretrained` 会保存：

1. pipeline 的 JSON 配置
2. 每个 step 的 class 或 registry_name
3. step 的 config
4. step 的 tensor state（如果有）

其中 state 会被写成 `.safetensors` 文件。

这说明 pipeline 被正式当成一个“模型伴随资产”对待。

## 9.3 registry 与动态导入

加载时支持两种 step 定位方式：

1. 通过 `registry_name`
2. 通过完整 `class` 路径动态导入

优先 registry name，是为了更强的可移植性。

## 9.4 overrides

`from_pretrained(...)` 支持对某些 step 配置做 override。

这非常关键，因为训练好的 processor 在部署时经常需要调整：

- device
- rename map
- stats

训练主链里创建 processor 时就用了这个能力。

---

## 10. `ProcessorStepRegistry`：让 step 成为真正可发现组件

registry 的作用很明确：

- 用字符串名注册 step class
- 方便序列化后恢复
- 降低硬编码 class path 的依赖

这意味着 LeRobot 的 processor steps 不是散落函数，而是形成了一个真正可发现、可加载的组件生态。

而且从 `src/lerobot/processor/__init__.py` 可以看出，项目已经内建了大量 steps，例如：

- `NormalizerProcessorStep`
- `UnnormalizerProcessorStep`
- `DeviceProcessorStep`
- `RenameObservationsProcessorStep`
- `TokenizerProcessorStep`
- `AddBatchDimensionProcessorStep`
- `RelativeActionsProcessorStep`
- `AbsoluteActionsProcessorStep`
- `PolicyActionToRobotActionProcessorStep`
- `RobotActionToPolicyActionProcessorStep`

这再次说明 processor 不是一个理论框架，而是已经成为项目中的实际中间层基础设施。

---

## 11. RobotProcessorPipeline vs PolicyProcessorPipeline

文档里对这两个概念解释得很清楚。

## 11.1 共性

二者本质上都继承自统一的 pipeline 思路：

- 中间都走 `EnvTransition`
- 都由一组 `ProcessorStep` 组成

## 11.2 区别

### RobotProcessorPipeline

处理的是：

- 非 batched
- 更接近硬件接口
- 通常是 dict 风格 observation/action

### PolicyProcessorPipeline

处理的是：

- batched tensors
- 更接近模型输入输出
- 常用于训练和推理

这也是为什么 LeRobot 没有试图只用一个“超大万能 preprocessor”，而是区分了 robot 侧和 policy 侧的典型使用场景。

---

## 12. feature contract：processor 为什么还要管 shape/type

这一点是第 3 阶段最容易被忽略、但最重要的部分之一。

## 12.1 `transform_features` 的意义

processor 会改变：

- key 名称
- shape
- 数据模态

如果系统只在“值级别”上处理，而不维护 feature contract，就会出现：

- dataset.create 不知道该为哪些字段分配 schema
- policy config 不知道最终输入 shape
- normalization/unnormalization 不知道目标 feature 定义

所以 processor step 不只要改数据，还必须声明“改完之后数据长什么样”。

## 12.2 文档里的例子

文档举的 velocity processor 就非常有代表性：

- 它会把 state shape 改成原来的两倍
- `transform_features` 需要同步反映这个变化

这使 processor 体系不仅能运行在训练时，还能服务 dataset schema 构建。

## 12.3 这和 dataset 创建的关系

文档中还提到：

- `create_initial_features()`
- `aggregate_pipeline_dataset_features()`

这说明 processor 不只是推理/训练阶段组件，也会参与 dataset 录制前的 feature 规划。

换句话说：

> processor 是数据 contract 的一部分，而不仅是 runtime 变换器。

---

## 13. 从测试理解 pipeline 的契约

`tests/processor/test_pipeline.py` 是学习这套体系的捷径。

它非常清楚地展示了几个关键原则。

## 13.1 step config 和 tensor state 要分离

测试中的 `MockStep` / `MockStepWithTensorState` 很明确地表达了：

- JSON 可序列化配置放在 `get_config()`
- tensor state 放在 `state_dict()`

这个约束很重要，因为：

- 配置和状态的生命周期不同
- 配置更稳定
- 状态通常需要 safetensors 保存

## 13.2 pipeline 支持空流水线与多 step 串联

测试验证了：

- 空 pipeline 应直接返回原对象
- 多 step 顺序执行
- `step_through` 能逐步观察中间态

## 13.3 输入最终都应被校验成 transition 契约

测试还会验证：

- 传错类型会抛出错误

这说明 pipeline 不希望 silently 接受错误数据形态，而是倾向于尽早失败。

---

## 14. normalization processor：一个很典型的 processor 家族

`tests/processor/test_normalize_processor.py` 帮助我们理解 processor 的实际价值。

这里可以看到：

- stats 支持 numpy/tensor/scalar/list 多种来源
- 不同 feature type 可绑定不同 normalization mode
- 支持：
  - mean/std
  - min/max
  - quantile
  - quantile10

而且还能：

- normalize
- unnormalize
- round-trip 恢复

这说明 processor 的典型场景不是“做点小变换”，而是承担真正关键的数值空间适配。

---

## 15. policy 与 robot 之间的桥接 processor

`tests/processor/test_policy_robot_bridge.py` 是另一个非常重要的测试文件。

它展示了两个典型 bridge：

- `RobotActionToPolicyActionProcessorStep`
- `PolicyActionToRobotActionProcessorStep`

## 15.1 它们解决的问题

policy 常常输出的是：

- 一维 tensor

而 robot 需要的是：

- 按 motor name 命名的 dict，例如 `joint1.pos`

bridge step 负责：

- tensor 和 dict 的互转
- 保证电机顺序一致
- 校验长度
- 更新 feature contract

## 15.2 这说明了什么

这再次证明：

> processor 不只是训练时数据清洗工具，而是模型动作空间与真实执行空间之间的接口层。

---

## 16. 用一个完整视角理解 processor 在系统中的位置

现在可以把 processor 放回主链中重新看。

### 16.1 训练时

`dataset sample -> policy preprocessor -> policy.forward/select_action -> postprocessor`

### 16.2 eval 时

`env observation -> env preprocessor -> policy preprocessor -> policy -> policy postprocessor -> env postprocessor`

### 16.3 机器人部署时

`robot observation -> robot/policy bridge -> policy -> policy/robot bridge -> robot action`

这三条链说明：

- dataset 不直接绑定某个 policy
- env 不直接绑定某个 policy
- robot 也不直接绑定某个 policy

它们是通过 processor 松耦合接起来的。

这也是 LeRobot 架构可扩展的关键原因之一。

---

## 17. 第 3 阶段最重要的设计认识

这一阶段最值得记住的几个结论如下。

## 认识 1：processor 是 LeRobot 的真正中间层

dataset、robot、env、policy 之间之所以能保持相对独立，是因为它们不直接对接，而是通过 processor 层转换。

## 认识 2：`EnvTransition` 是统一协议

processor 体系能成立的前提，是所有异构输入最终都被映射到统一 transition 表示。

## 认识 3：converter 负责异构输入输出边界

pipeline 内部统一，边界通过 `to_transition` / `to_output` 保持灵活，这个设计非常关键。

## 认识 4：processor 不只变换数据值，还维护 feature contract

这是它能参与 dataset schema 推导和 policy feature 规划的根本原因。

## 认识 5：processor 是可保存、可恢复、可复用的训练资产

这就是为什么 checkpoint 要保存 processor，为什么 `from_pretrained(...)` 要支持 overrides。

---

## 18. 本阶段结束后，你应该已经能回答的问题

完成第 3 阶段后，你应该能比较清楚地回答：

1. 为什么 LeRobot 需要 processor，而不是把逻辑写死在 policy 里？
2. `EnvTransition` 在 processor 体系里扮演什么角色？
3. `batch_to_transition` 和 `transition_to_batch` 在主链中的作用是什么？
4. `ProcessorStep` 和 `DataProcessorPipeline` 的职责边界是什么？
5. 为什么 processor step 要实现 `transform_features()`？
6. 为什么 processor 需要可序列化？
7. `RobotProcessorPipeline` 和 `PolicyProcessorPipeline` 的区别是什么？
8. processor 如何帮助 policy 和 robot/action space 对接？

如果这些问题已经能比较清楚地回答，说明第 3 阶段达标。

---

## 19. 下一阶段建议

到这里为止，平台基础设施三件套已经比较完整了：

1. 训练主链
2. dataset 主线
3. processor 中间层

下一步最适合进入第 4 阶段，开始读第一个具体 policy，推荐 ACT。

推荐顺序：

1. `docs/source/policy_act_README.md`
2. `src/lerobot/policies/act/configuration_act.py`
3. `src/lerobot/policies/act/modeling_act.py`
4. `src/lerobot/policies/act/processor_act.py`
5. `tests/processor/test_act_processor.py`

因为现在你已经有足够上下文去理解：

- dataset 给 policy 什么
- processor 如何适配输入输出
- train loop 如何调用 `forward()` 和 `select_action()`

这时再进入 ACT，会比一开始直接看模型代码清晰很多。

---

## 20. 本阶段个人学习结论

如果用一句话总结第 3 阶段：

> Processor 是 LeRobot 的通用翻译层和结构化中间层，它通过 `EnvTransition + converter + ProcessorStep + DataProcessorPipeline` 的组合，把异构的机器人、环境、数据集和模型输入输出统一成了一套可组合、可保存、可复用的系统接口。

理解这一层之后，后面阅读具体 policy 时，注意力就可以真正放在“算法本身”上，而不是反复被数据格式转换问题打断。
