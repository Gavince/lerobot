# LeRobot 代码库学习流程报告

## 1. 学习目标

这份学习流程的目标不是“尽快看完所有代码”，而是循序渐进地建立对 LeRobot 的整体理解，最终能够回答下面几个问题：

1. LeRobot 解决的核心问题是什么？
2. 训练、评估、数据集、处理器、策略、环境、硬件控制之间是如何衔接的？
3. 一个 policy 是如何从论文概念落到配置、数据、processor、模型和训练逻辑中的？
4. 如果后续要自己加一个 policy、一个 processor，或者接入自己的机器人，需要改哪些层？

建议整体采用“平台基础设施优先，再进入具体算法”的方式学习。这个仓库不是单纯的模型仓库，而是一个完整的机器人学习平台。

---

## 2. 代码库整体认知

LeRobot 可以先抽象成下面这条主线：

`CLI脚本 -> Config -> Factory -> Dataset/Env -> Processor -> Policy -> Train/Eval -> Hub/Robot`

从仓库结构上看，建议优先理解这些目录：

- `src/lerobot/scripts/`：训练、评估、数据录制、遥操作等入口脚本
- `src/lerobot/configs/`：配置系统，决定整个系统如何被装配
- `src/lerobot/datasets/`：LeRobotDataset 数据格式、读取、写入、元数据管理
- `src/lerobot/processor/`：数据预处理/后处理流水线，是平台中枢
- `src/lerobot/policies/`：各类 policy 的配置、模型、processor
- `src/lerobot/envs/`：仿真环境适配
- `src/lerobot/robots/`、`src/lerobot/teleoperators/`：真实机器人与遥操作设备
- `tests/`：理解模块输入输出契约的重要材料

### 建议先记住的关键文件

- `README.md`
- `pyproject.toml`
- `src/lerobot/scripts/lerobot_train.py`
- `src/lerobot/scripts/lerobot_eval.py`
- `src/lerobot/configs/train.py`
- `src/lerobot/configs/policies.py`
- `src/lerobot/datasets/factory.py`
- `src/lerobot/datasets/lerobot_dataset.py`
- `src/lerobot/datasets/dataset_metadata.py`
- `src/lerobot/processor/pipeline.py`
- `src/lerobot/policies/factory.py`
- `src/lerobot/policies/pretrained.py`
- `src/lerobot/envs/factory.py`

---

## 3. 推荐学习顺序总览

建议分 8 个阶段来学：

1. 项目总览与安装
2. 训练主链与配置系统
3. 数据集系统
4. Processor 体系
5. 第一个经典 policy：ACT
6. 第二个 policy：Diffusion Policy
7. 评估与环境系统
8. 深入方向：VLA / RL / 真实机器人

整体原则：

- 先理解平台骨架，再深入具体算法
- 先学一个最典型的 IL policy，再看更复杂模型
- 先用文档和测试建立输入输出直觉，再回到大文件读实现

---

## 4. 详细学习流程

## 阶段 0：项目预热（1-2 天）

### 目标

建立全局地图，知道这个仓库在做什么，不急着进入细节实现。

### 建议阅读

- `README.md`
- `docs/source/index.mdx`
- `docs/source/installation.mdx`
- `pyproject.toml`

### 重点问题

1. LeRobot 支持哪些任务形态：IL、RL、VLA、真实机器人控制
2. 这个仓库的核心抽象有哪些
3. 哪些功能是基础能力，哪些是 optional extras
4. 为什么它要统一 dataset / policy / env / robot 接口

### 建议实践

先完成基础环境安装，并至少跑一个最轻量的测试：

```bash
uv sync --locked --extra test --extra dev
uv run pytest tests/processor/test_pipeline.py -q
```

### 阶段产出

写一页笔记，回答：

- LeRobot 是“模型库”还是“平台”
- 仓库中哪些模块属于通用基础设施
- 哪些模块以后可以按兴趣选学

---

## 阶段 1：训练主链与配置系统（3-4 天）

### 目标

读懂一次 `lerobot-train` 从启动到训练循环的完整过程。

### 核心阅读

- `src/lerobot/scripts/lerobot_train.py`
- `src/lerobot/configs/train.py`
- `src/lerobot/configs/policies.py`
- `src/lerobot/optim/factory.py`
- `src/lerobot/common/train_utils.py`

### 重点理解内容

1. `TrainPipelineConfig` 如何组织 dataset、env、policy、optimizer、scheduler、wandb 等配置
2. `validate()` 如何补全默认值与预设
3. `lerobot_train.py` 如何创建 dataset、policy、processor、optimizer 和 dataloader
4. train loop 中 `policy.forward(batch)`、反向传播、checkpoint、eval 的位置
5. resume / checkpoint / push_to_hub 是如何串起来的

### 推荐带着问题读

- 为什么配置层放了这么多逻辑，而不是全堆到脚本里？
- 为什么 `policy` 配置既能从命令行构造，也能从 pretrained path 加载？
- 训练逻辑里哪些部分是 policy 通用的，哪些部分依赖具体算法？

### 阶段练习

手动画一张流程图：

`CLI参数 -> TrainPipelineConfig -> make_dataset -> make_policy -> make_pre_post_processors -> optimizer -> training loop`

### 阶段产出

如果你能不看代码解释一次 `lerobot-train` 的执行路径，这一阶段就算完成。

---

## 阶段 2：数据集系统（1 周）

### 目标

理解 LeRobotDataset 的数据组织方式、元数据结构和训练时的数据读取机制。

### 核心阅读

- `src/lerobot/datasets/factory.py`
- `src/lerobot/datasets/lerobot_dataset.py`
- `src/lerobot/datasets/dataset_metadata.py`
- `docs/source/lerobot-dataset-v3.mdx`

### 配套测试

- `tests/datasets/test_lerobot_dataset.py`
- `tests/datasets/test_dataset_metadata.py`
- `tests/datasets/test_delta_timestamps.py`
- `tests/datasets/test_streaming.py`

### 必须搞懂的点

1. LeRobotDataset v3.0 为什么采用 `Parquet + MP4 + meta/`
2. `meta/info.json`、`meta/stats.json`、`meta/tasks`、`meta/episodes` 分别做什么
3. `LeRobotDatasetMetadata` 如何在本地和 Hugging Face Hub 之间同步元数据
4. `delta_timestamps` 如何从单帧样本构造时序窗口
5. `dataset.meta.stats` 如何为后续 normalization 服务
6. `StreamingLeRobotDataset` 的意义是什么

### 推荐阅读顺序

1. 先看文档 `lerobot-dataset-v3.mdx`
2. 再看 `dataset_metadata.py`
3. 再看 `lerobot_dataset.py`
4. 最后看测试来补契约

### 阶段练习

做一个表格：

`字段/文件 -> 作用 -> 被谁读取 -> 与训练流程的关系`

例如：

- `meta/info.json` -> 描述 schema 和路径模板 -> dataset reader / metadata -> 决定如何恢复样本
- `meta/stats.json` -> 归一化统计量 -> processor / policy -> 决定训练输入输出尺度

### 阶段产出

你应该能解释：

- 为什么这个仓库不把机器人数据简单地存成一个个 episode 文件夹
- 为什么训练时 dataset 还需要知道 policy 的 delta indices

---

## 阶段 3：Processor 体系（1 周）

### 目标

真正理解 LeRobot 平台最核心的中间层：processor。

### 核心阅读

- `docs/source/introduction_processors.mdx`
- `src/lerobot/processor/pipeline.py`
- `src/lerobot/processor/converters.py`
- `src/lerobot/processor/factory.py`

### 配套测试

- `tests/processor/test_pipeline.py`
- `tests/processor/test_batch_conversion.py`
- `tests/processor/test_converters.py`
- `tests/processor/test_normalize_processor.py`
- `tests/processor/test_policy_robot_bridge.py`

### 必须理解的核心概念

1. `EnvTransition` 是统一中间表示
2. `ProcessorStep` 是一个最小变换单元
3. `DataProcessorPipeline` 是如何顺序组合多个 step 的
4. `RobotProcessorPipeline` 和 `PolicyProcessorPipeline` 的区别
5. 为什么 processor 可以把 robot / dataset / env / policy 解耦

### 关键理解

LeRobot 里很多“看起来耦合”的事情，本质上都是通过 processor 被解开了：

- 原始观测如何变成模型输入
- 模型输出如何恢复成真实动作
- 不同机器人、不同策略、不同环境之间如何对接

如果这一层没看懂，后面看 policy 会觉得零散；一旦这一层看懂，整个仓库就会顺很多。

### 阶段练习

自己尝试实现一个最简单的 `ProcessorStep`，例如：

- 给 observation 加一个附加字段
- 对 action 做简单缩放
- 在 complementary_data 里记录经过的 step

### 阶段产出

你应该能用自己的话解释：

“为什么 LeRobot 把 processor 放在系统中间，而不是把预处理逻辑写死在模型里？”

---

## 阶段 4：第一个 policy，推荐 ACT（1 周）

### 目标

第一次完整打通“论文 -> 配置 -> processor -> 模型 -> 损失函数 -> 推理动作”。

### 建议阅读顺序

#### 先读论文

- `docs/source/policy_act_README.md`

论文主题：

- ACT（Action Chunking Transformer）
- 核心思想：一次预测多个未来动作，用 action chunking 改善控制稳定性

#### 再读代码

- `src/lerobot/policies/act/configuration_act.py`
- `src/lerobot/policies/act/modeling_act.py`
- `src/lerobot/policies/act/processor_act.py`
- `tests/processor/test_act_processor.py`

### 阅读重点

1. `chunk_size`、`n_action_steps`、`temporal_ensemble_coeff` 的关系
2. 为什么 `select_action()` 里要维护 action queue
3. 为什么 `forward()` 里使用 `action_is_pad`
4. 图像输入和状态输入是如何组织到 batch 里的
5. `predict_action_chunk()` 和 `select_action()` 的职责区别

### 为什么先学 ACT

1. 它是经典 imitation learning policy，结构相对清晰
2. 很适合作为 action chunking 思想的第一站
3. LeRobot 对时序动作输出的抽象，在 ACT 里体现得很明显

### 阶段练习

自己写一份 ACT 的数据流总结：

`dataset sample -> preprocessor -> model.forward/select_action -> postprocessor -> env/robot action`

### 阶段产出

如果你能解释 ACT 在 LeRobot 中：

- 输入是什么
- 输出是什么
- 训练损失是什么
- 推理时为什么不是每步都重新预测全部动作

说明这一阶段基本掌握。

---

## 阶段 5：第二个 policy，推荐 Diffusion Policy（1 周）

### 目标

通过和 ACT 对照，理解 LeRobot 平台如何承载不同风格的 policy。

### 先读论文

- `docs/source/policy_diffusion_README.md`

### 再读代码

- `src/lerobot/policies/diffusion/configuration_diffusion.py`
- `src/lerobot/policies/diffusion/modeling_diffusion.py`
- `src/lerobot/policies/diffusion/processor_diffusion.py`
- `tests/processor/test_diffusion_processor.py`

### 核心对照问题

1. ACT 和 Diffusion 的 observation / action 时间窗口有什么不同
2. 为什么 Diffusion 的 `n_obs_steps`、`horizon`、`n_action_steps` 配得更复杂
3. dataset 是如何根据 policy 配置给出时间切片的
4. processor 在两种 policy 里承担了哪些共同职责
5. 一种是 chunk prediction，另一种是 diffusion denoising，它们如何都被统一放进 `PreTrainedPolicy` 框架

### 阶段练习

写一张 ACT vs Diffusion 对照表：

- 输入时序长度
- 输出形式
- 推理方式
- 训练目标
- processor 重点
- 适合先读哪些测试

### 阶段产出

如果你能从平台视角而不是算法视角比较 ACT 和 Diffusion，就说明你开始真正理解 LeRobot 的架构了。

---

## 阶段 6：评估与环境系统（3-4 天）

### 目标

理解 policy 如何被放入仿真环境中进行 rollout 和评估。

### 核心阅读

- `src/lerobot/scripts/lerobot_eval.py`
- `src/lerobot/envs/factory.py`
- `src/lerobot/envs/configs.py`
- `tests/envs/test_envs.py`
- `tests/policies/test_policies.py`

### 重点理解内容

1. `make_env()` 如何构建 vectorized environments
2. 为什么环境返回的是 `{suite_name: {task_id: vec_env}}`
3. eval 中 env pre/post processor 与 policy pre/post processor 如何拼接
4. rollout 时 observation 到 action 的整条数据流
5. 仿真环境和 policy 的接口是如何统一的

### 推荐带着问题读

- 为什么 `lerobot_eval.py` 不直接把 env observation 喂给 policy？
- env processor 和 policy processor 的职责边界在哪里？
- 为什么环境适配要保留 suite/task 层次？

### 阶段练习

手画一张 rollout 图：

`env.reset -> preprocess_observation -> env_preprocessor -> policy_preprocessor -> policy.select_action -> policy_postprocessor -> env_postprocessor -> env.step`

### 阶段产出

你应该能独立解释一次 rollout 的完整执行过程。

---

## 阶段 7：VLA 方向（1 周，可选主线）

### 目标

理解更复杂的视觉-语言-动作模型如何接入 LeRobot 平台。

### 推荐优先级

1. SmolVLA
2. Pi0 / Pi0.5
3. XVLA / GR00T / Wall-X

### 推荐阅读

- `docs/source/policy_smolvla_README.md`
- `docs/source/smolvla.mdx`
- `tests/processor/test_smolvla_processor.py`

### 阅读重点

1. 文本任务是如何进入模型输入的
2. tokenizer、视觉 backbone、动作头之间如何衔接
3. VLA policy 中哪些部分仍然复用了通用 processor 体系
4. 平台公共抽象和模型私有实现的边界在哪里

### 学习建议

这一阶段不要一口气把所有大模型代码都啃完，优先看：

- 配置
- processor
- 输入输出定义
- 测试

最后再看模型内部结构。

---

## 阶段 8：RL 与真实机器人方向（按兴趣选）

## 方向 A：RL

### 推荐阅读

- `docs/source/policy_tdmpc_README.md`
- `src/lerobot/rl/learner.py`
- `src/lerobot/rl/actor.py`
- `src/lerobot/rl/buffer.py`
- `tests/rl/test_actor.py`
- `tests/rl/test_actor_learner.py`

### 重点问题

1. RL 部分和 imitation learning 的训练链路有什么不同
2. actor / learner 是如何分工的
3. buffer 和 policy 更新是如何衔接的

## 方向 B：真实机器人与遥操作

### 推荐阅读

- `src/lerobot/scripts/lerobot_record.py`
- `src/lerobot/scripts/lerobot_teleoperate.py`
- `docs/source/integrate_hardware.mdx`
- `docs/source/processors_robots_teleop.mdx`

### 重点问题

1. 真实机器人的 observation/action 如何进入 processor 流水线
2. 录制数据、遥操作和训练之间是如何接起来的
3. 机器人抽象层和 teleoperator 抽象层的职责分别是什么

---

## 5. 论文阅读计划

建议论文阅读不要脱离代码，最好和阶段学习同步。

## 第一组：建立主线

1. LeRobot 论文
2. ACT
3. Diffusion Policy

### 目标

建立对平台目标、模仿学习主流策略和 action modeling 方式的基本认识。

## 第二组：分方向深入

### 如果偏 VLA

1. SmolVLA
2. Pi0 / Pi0.5 相关文档与论文

### 如果偏 RL

1. TDMPC

## 推荐阅读方法

每篇论文建议分三轮：

### 第一轮

只看：

- 任务定义
- 输入输出
- 训练目标
- 推理方式

### 第二轮

尝试做“论文到代码映射”：

- 论文概念 -> 配置字段
- 论文模块 -> 模型类
- 训练目标 -> `forward()` 中的 loss
- 推理流程 -> `select_action()`

### 第三轮

再看细节：

- cache / queue / normalization / masking
- processor 如何弥补论文和真实工程之间的差距

---

## 6. 核心代码理解计划

下面这条顺序比“随机跳着看”更高效：

1. `README.md`
2. `src/lerobot/scripts/lerobot_train.py`
3. `src/lerobot/configs/train.py`
4. `src/lerobot/datasets/factory.py`
5. `src/lerobot/datasets/lerobot_dataset.py`
6. `src/lerobot/datasets/dataset_metadata.py`
7. `docs/source/introduction_processors.mdx`
8. `src/lerobot/processor/pipeline.py`
9. `src/lerobot/processor/converters.py`
10. `src/lerobot/policies/pretrained.py`
11. `src/lerobot/policies/factory.py`
12. `ACT 配置 + 模型 + processor`
13. `Diffusion 配置 + 模型 + processor`
14. `src/lerobot/scripts/lerobot_eval.py`
15. `src/lerobot/envs/factory.py`
16. `tests/processor/*`
17. `tests/datasets/*`
18. `tests/policies/*`

这个顺序的核心思想是：

- 先搞懂“系统怎么跑”
- 再搞懂“数据怎么流”
- 再搞懂“模型怎么接”
- 最后再看更复杂的特定算法

---

## 7. 每阶段建议输出物

为了避免“看了很多，但没沉淀下来”，建议每个阶段都留一个明确输出。

## 阶段输出建议

### 阶段 1

一张训练主链流程图

### 阶段 2

一张 LeRobotDataset 元数据结构表

### 阶段 3

一张 processor 数据流图

### 阶段 4

一份 ACT 训练与推理流程笔记

### 阶段 5

一张 ACT vs Diffusion 对照表

### 阶段 6

一张 rollout 图

### 阶段 7/8

一篇专题总结：VLA / RL / 真实机器人三选一

---

## 8. 一个推荐的 5 周执行节奏

如果你希望更落地一点，可以参考下面的时间安排。

## 第 1 周

- 项目总览
- 安装与基础测试
- 训练主链
- 配置系统

## 第 2 周

- 数据集系统
- Dataset v3 文档
- dataset 相关测试

## 第 3 周

- processor 体系
- converters
- pipeline 测试

## 第 4 周

- ACT
- Diffusion Policy
- policy 相关测试

## 第 5 周

- eval / env
- VLA 或 RL 或 硬件方向选一个深入

---

## 9. 学完后的自测标准

如果你能比较顺畅地回答下面这些问题，说明你已经真正入门 LeRobot：

1. `lerobot-train` 从命令行到训练循环的执行路径是什么？
2. `delta_timestamps` 为什么重要？
3. LeRobotDataset v3.0 为什么要用 `Parquet + MP4 + meta/`？
4. 为什么 processor 是 LeRobot 的中枢层？
5. `PreTrainedConfig`、`PreTrainedPolicy`、`make_pre_post_processors` 三者是什么关系？
6. ACT 和 Diffusion 在 LeRobot 平台中的共同点和不同点是什么？
7. env preprocessor 和 policy preprocessor 分别负责什么？
8. 如果自己加一个 policy，需要至少实现哪些文件？

---

## 10. 最后的学习建议

这个仓库不适合“从头到尾硬啃所有源码”。更有效的方式是：

1. 先抓主线
2. 再抓平台中枢（processor）
3. 再挑典型 policy 深挖
4. 最后根据兴趣分方向深入

尤其要重视 `tests/`。这个仓库的测试写得很有教学价值，很多时候比主实现更适合第二轮阅读，因为它们更短、更明确地表达了模块契约。

如果后续继续深入，最值得优先掌握的能力不是“背会某个模型结构”，而是能把下面这条链完整串起来：

`论文概念 -> 配置字段 -> 数据切片 -> processor步骤 -> 模型输入输出 -> 训练损失 -> 推理动作`

一旦这条链打通，你不仅能更快看懂 LeRobot，也更容易在这个平台上继续做自己的实验、接自己的数据、加自己的 policy。
