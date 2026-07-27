# LeRobot 第8阶段：RL与真实机器人闭环学习

## 一、阶段目标

第 8 阶段是整套学习路线的收尾阶段。

前面 0 到 7 阶段，我们已经依次理解了：

1. 项目定位与安装结构
2. 训练主链
3. dataset 主线
4. processor 中间层
5. ACT
6. Diffusion 与 eval/env
7. SmolVLA 与 VLA 路线

最后这一阶段要补上两条剩余主线：

1. RL 分支  
   看 LeRobot 如何支持在线、分布式、真实机器人上的强化学习训练。

2. 真实机器人 / 数据采集 / teleop 分支  
   看 LeRobot 如何把 teleoperator、robot、processor、dataset 连接起来，形成从采集到部署的完整闭环。

核心文件与文档：

- [docs/source/policy_tdmpc_README.md](/Users/zwy/Macin/lerobot/docs/source/policy_tdmpc_README.md)
- [src/lerobot/rl/learner.py](/Users/zwy/Macin/lerobot/src/lerobot/rl/learner.py)
- [src/lerobot/rl/actor.py](/Users/zwy/Macin/lerobot/src/lerobot/rl/actor.py)
- [tests/rl/test_actor.py](/Users/zwy/Macin/lerobot/tests/rl/test_actor.py)
- [src/lerobot/scripts/lerobot_record.py](/Users/zwy/Macin/lerobot/src/lerobot/scripts/lerobot_record.py)
- [src/lerobot/scripts/lerobot_teleoperate.py](/Users/zwy/Macin/lerobot/src/lerobot/scripts/lerobot_teleoperate.py)
- [docs/source/integrate_hardware.mdx](/Users/zwy/Macin/lerobot/docs/source/integrate_hardware.mdx)
- [docs/source/processors_robots_teleop.mdx](/Users/zwy/Macin/lerobot/docs/source/processors_robots_teleop.mdx)

这一阶段最重要的收获可以先提前说出来：

LeRobot 的平台价值，不只在于“训练已有 dataset 上的策略”，而在于它能把：

`硬件接入 -> teleop控制 -> 数据采集 -> processor对齐 -> policy训练/推理 -> 在线RL更新`

连成一条完整链路。

---

## 二、RL 分支先看什么：不要先钻算法细节

`docs/source/policy_tdmpc_README.md` 只给了论文链接，但真正从代码里看，LeRobot 的 RL 分支最值得先理解的不是某个 loss，而是整体架构：

它采用的是一种分布式 actor-learner 结构。

也就是说：

1. actor 负责在环境或真实机器人上执行策略、采集 transition
2. learner 负责维护 replay buffer、更新策略参数
3. actor 与 learner 通过 gRPC 和队列通信

这比前面 offline IL 的训练主线明显更复杂，因为系统现在必须同时处理：

1. 在线交互
2. 实时控制
3. 参数同步
4. 数据传输
5. 回放缓冲区采样

所以 RL 分支的正确学习顺序应该是：

`actor/learner 架构 -> 数据流 -> replay/training loop -> 再回头看具体 RL policy`

---

## 三、RL 主线：`learner.py` 在做什么

[src/lerobot/rl/learner.py](/Users/zwy/Macin/lerobot/src/lerobot/rl/learner.py) 是 RL 分支的中枢。

### 3.1 高层职责

learner 的职责主要有：

1. 初始化 RL policy
2. 创建 replay buffer
3. 接收 actor 发来的 transitions
4. 采样 batch 做参数更新
5. 周期性把新参数推送给 actor
6. 负责 checkpoint 和日志

也就是说，它既是训练器，也是在线训练系统的协调中心。

### 3.2 为什么是 server

文件注释里明确说明 learner 是 server 端，这一点很关键。

这意味着：

1. actor 不直接调用 learner 的 Python 函数
2. learner 是一个长期运行的服务
3. actor 与 learner 之间默认是松耦合、可分布部署的

从系统设计角度看，这和 offline train loop 完全不同：

- offline train：一个进程里读 dataset、算梯度、存 checkpoint
- online RL：至少两个角色协作，且需要实时通信

### 3.3 `start_learner_threads()`

这个函数很值得关注，因为它把 learner 结构显式拆成多个并发单元：

1. transition queue
2. interaction message queue
3. parameters queue
4. learner communication process/thread
5. training 主循环

这说明 learner 并不是一个单纯的“训练函数”，而是一套并发系统。

### 3.4 `add_actor_information_and_train()`

这是 learner 中最重要的核心函数之一。

它大致做的事情包括：

1. 从 transition queue 持续读取 actor 传来的经验
2. 放进 replay buffer
3. 当数据足够时启动训练
4. 按 `utd_ratio` 等配置进行多次更新
5. 周期性把 actor 需要的新参数推送出去
6. 同时记录 interaction 和训练指标

这里你能看到一个非常典型的在线 RL 系统思路：

采样、训练、日志、参数同步并不是完全分离的批处理阶段，而是持续交织在一起。

---

## 四、RL 主线：`actor.py` 在做什么

[src/lerobot/rl/actor.py](/Users/zwy/Macin/lerobot/src/lerobot/rl/actor.py) 是另一半。

### 4.1 actor 的高层职责

actor 负责：

1. 连接 learner
2. 在真实环境里执行策略
3. 采集 transition
4. 把 transition 发送给 learner
5. 接收 learner 新发回来的 policy 参数

所以 actor 的角色不是“本地推理脚本”，而是在线数据采集与控制执行端。

### 4.2 `act_with_policy()`

这是 actor 中最关键的主循环。

高层流程大致是：

1. `make_robot_env(cfg.env)` 创建在线机器人环境
2. `make_processors(...)` 创建 env/action processors
3. 本地实例化 policy
4. `env.reset()`
5. 对初始 observation 做 processor 处理
6. 循环：
   - 从 transition 里抽 observation
   - `policy.select_action(...)`
   - 执行动作
   - 得到新 transition
   - 提取 executed action / reward / done 等信息
   - 累积并发送给 learner

这条链说明：

RL actor 在 LeRobot 里并没有绕开前面学过的 processor 思想，而是继续复用了“先处理 observation/action 再进入策略”的中间层设计。

### 4.3 actor 与 learner 为什么都实例化 policy

代码里有一个非常重要的注释：

actor 和 learner 都会各自实例化一个 policy，而不是把 learner 里的 policy 对象直接传过去。

原因很实际：

1. 跨进程/跨机器传递完整模型对象不方便
2. 更合理的做法是：
   - 两边都有同结构 policy
   - learner 只传 state_dict 参数

这正是分布式训练系统里很典型的工程选择。

---

## 五、RL 分支的关键理解：LeRobot 的 RL 更像系统工程

如果把第 8 阶段 RL 分支压缩成一句话：

LeRobot 的 RL 路线，核心不在于把 SAC/TDMPC 某个公式写进仓库，而在于它把真实机器人在线训练所需的 actor-learner 基础设施搭起来了。

这意味着：

1. policy 只是其中一部分
2. replay buffer、消息传输、参数同步同样重要
3. 真实机器人和人类干预可以纳入在线训练回路

从学习顺序上说，这也是为什么这个分支我建议放到最后：  
如果前面不先理解 processor、policy、eval 和 robot 接口，这里会显得特别碎。

---

## 六、测试文件告诉了我们什么：`tests/rl/test_actor.py`

这个测试文件虽然没有覆盖完整 RL 系统，但它揭示了几个很重要的接口契约。

### 6.1 learner 连接是正式接口

测试会检查：

1. `establish_learner_connection()` 成功时返回 True
2. 服务不可用时返回 False

这说明 actor 与 learner 的握手机制不是随意写的实现细节，而是一个被明确测试的对外协议。

### 6.2 transitions / interactions 是可序列化消息

测试还验证了：

1. transition 能被序列化成 bytes
2. 再从 queue/gRPC stream 中恢复回来

这进一步说明：

RL 分支的“数据”不是 DataLoader batch，而是跨进程传输的 transport payload。

这是和 offline train 最大的系统差别之一。

---

## 七、真实机器人分支的入口：`lerobot-teleoperate`

接下来转向真实机器人主线。

[src/lerobot/scripts/lerobot_teleoperate.py](/Users/zwy/Macin/lerobot/src/lerobot/scripts/lerobot_teleoperate.py) 是最容易上手的硬件入口。

### 7.1 它解决什么问题

这个脚本的目标很直接：

把 teleoperator 的动作经过 processor 处理后发给 robot，让人类可以实时控制机器人。

所以它本质上是硬件链路中最简版的闭环：

`teleop -> processor -> robot`

### 7.2 `teleop_loop()` 数据流

这个函数非常值得记，因为它是理解 robot processor 的最佳入口。

每一轮循环大致做：

1. `robot.get_observation()`
2. `teleop.get_action()`
3. `teleop_action_processor((raw_action, obs))`
4. `robot_action_processor((teleop_action, obs))`
5. `robot.send_action(robot_action_to_send)`
6. 可选显示观察和动作

你会发现，这条链虽然没有 policy，但其结构和我们前面学习过的很多链是同构的：

- 有 observation
- 有 action
- 有 processor
- 有执行端

所以 `teleoperate.py` 可以被看成“去掉了 policy 的在线控制主线”。

### 7.3 `make_default_processors()`

默认处理器是关键，因为它说明：

LeRobot 并不假设 teleoperator 输出动作空间一定和机器人控制空间完全一致。

processor 的作用，就是在这两者之间做可插拔对齐。

---

## 八、真实机器人采集主线：`lerobot-record`

[src/lerobot/scripts/lerobot_record.py](/Users/zwy/Macin/lerobot/src/lerobot/scripts/lerobot_record.py) 是真实机器人工作流里最关键的文件之一。

### 8.1 它解决什么问题

`lerobot-record` 不是简单录视频，而是在做一件更系统的事：

把机器人执行过程组织成符合 LeRobotDataset 规范的 episode 数据。

换句话说，它要同时打通：

1. robot
2. teleop 或 policy
3. processor
4. dataset feature contract
5. 视频编码
6. Hub 上传

### 8.2 `RecordConfig`

这个配置把 4 类对象放在一起：

1. `robot`
2. `dataset`
3. `teleop`
4. `policy`

这说明在 LeRobot 中，recording 并不是一个只属于 dataset 子系统的行为，而是一个跨硬件与学习系统的“汇合点”。

### 8.3 teleop 和 policy 可以同时存在

`RecordConfig.__post_init__()` 中明确要求：

- 至少要有 teleop 或 policy 之一

这背后对应着两个很重要的 recording 场景：

1. 用 teleop 采集 demonstration
2. 用 policy 执行、并允许 teleop 介入或接管

这和前面 RL actor 中“human intervention”思路是相通的，说明 LeRobot 对真实机器人工作流一直把“人类在环”视为正式能力，而不是临时调试方式。

---

## 九、record 脚本真正展示了什么：平台闭环

如果你把 `record.py` 放到整个学习路线的末尾再回看，会发现它几乎浓缩了整个代码库的核心思想。

它需要同时处理：

1. 硬件连接
2. 观测/动作处理
3. dataset feature 聚合
4. policy pre/post processing
5. episode 构建
6. 视频写入和编码

也就是说，`record.py` 是一个真正把 platform 各模块串起来的工程入口。

---

## 十、Bring Your Own Hardware：LeRobot 如何抽象机器人

[docs/source/integrate_hardware.mdx](/Users/zwy/Macin/lerobot/docs/source/integrate_hardware.mdx) 是理解硬件抽象最重要的文档。

它的核心思想可以压缩成一句话：

只要你的机器人实现了 `Robot` 抽象契约，它就可以接入 LeRobot 的采集、控制、训练与推理工具链。

### 10.1 最关键的接口不是电机，而是契约

文档最值得记住的是这几类接口：

1. `RobotConfig`
2. `Robot`
3. `observation_features`
4. `action_features`
5. `connect() / disconnect()`
6. `get_observation() / send_action()`
7. `calibrate()`

这说明 LeRobot 的硬件抽象不是围绕某种具体机械臂写死的，而是围绕“特征接口 + 连接生命周期 + 控制/观测方法”设计的。

### 10.2 feature contract 再次出现

文档特别强调：

- `observation_features`
- `action_features`

必须在机器人还未连接时也能定义出来。

这一点和前面 dataset feature、env feature、policy feature 完全呼应：

LeRobot 里真正稳定的中心不是某个设备对象，而是 feature contract。

### 10.3 为什么这一点重要

因为这样一来：

1. 采集工具知道该存什么
2. processor 知道该如何转换
3. policy 知道该接收什么
4. dataset 知道该如何描述元数据

这也是为什么我们一路学到现在，会发现“feature contract”在几乎每条主线里都出现。

---

## 十一、Robots / Teleoperators / Processors：三管线思想

[docs/source/processors_robots_teleop.mdx](/Users/zwy/Macin/lerobot/docs/source/processors_robots_teleop.mdx) 对真实机器人路线特别关键。

它明确提出了一个非常值得记住的结构：

### 三条 pipeline

1. Teleop action space -> dataset action space
2. Dataset action space -> robot command space
3. Robot observation space -> dataset observation space

这几乎就是把前面几阶段 processor 抽象在硬件世界里重新讲了一遍。

### 11.1 为什么是三条

因为真实机器人系统里至少有三个“空间”：

1. 人类控制空间
2. 学习空间
3. 机器人执行空间

它们并不天然相同。

例如：

1. 人类可能用手机或 leader arm 给 EE pose
2. dataset 可能想存相对 EE delta
3. 机器人底层真正执行的是 joint commands

processor 的价值，就是把这些空间拆开、再有组织地连接起来。

### 11.2 `to_transition / to_output`

文档还特别强调 pipeline adapters：

- `to_transition`
- `to_output`

这和我们在 processor 阶段学到的 `EnvTransition` 完全一致，说明：

真实机器人路线并没有另外发明一套中间表示，而是继续复用了整个仓库统一的数据转换思想。

### 11.3 这份文档真正要你理解什么

不是“某个 phone-to-SO100 示例里用了哪些 step”，而是下面这件事：

LeRobot 鼓励你把 action/observation space 的设计和机器人底层命令空间解耦。

这一点对真实机器人特别重要，因为：

1. 你想学的表示未必是机器人最自然的控制表示
2. 你想存的 dataset 表示也未必等于 teleop 原始表示

processor 正是这三者之间的翻译层。

---

## 十二、把第 8 阶段两条分支放在一起看

现在把 RL 分支和真实机器人分支并排看，会发现它们的底层思想其实很一致。

| 维度 | RL 分支 | 真实机器人分支 |
| --- | --- | --- |
| 核心问题 | 在线训练与参数同步 | 采集、控制与数据对齐 |
| 关键角色 | actor / learner | teleop / robot / dataset / policy |
| 关键中间层 | transition / replay buffer / transport | processor / feature contract / dataset frame |
| 数据来源 | 在线交互 | 人控或策略执行 |
| 目标 | 持续更新策略 | 形成可训练、可回放、可部署的数据与控制链 |

所以这两条线并不是互不相关的“附加功能”，而是 LeRobot 面向真实机器人工作流的两种补充能力：

1. 一条偏在线学习
2. 一条偏数据采集与部署

---

## 十三、整套学习路线走完后，应该如何总结这个仓库

如果你现在要用一句话概括 LeRobot，可以不再说：

“这是一个机器人 policy 库。”

更准确的说法应该是：

LeRobot 是一个围绕 feature contract 和 processor 中间层构建的机器人学习平台，它把 dataset、policy、env、robot、teleop、eval 和在线 RL 都接进了一套统一接口。

### 13.1 这个平台的真正中心

整条学习路线走到这里，可以看出真正的中心不是某个单独模型，而是这几层抽象：

1. feature contract
2. processor pipeline
3. policy 统一接口
4. env / robot / teleop 的标准适配

### 13.2 为什么这个结论重要

因为这样你之后再读新策略、新硬件、新环境时，不会再迷失在文件数量里，而会自然问这几个问题：

1. 它的输入输出 feature contract 是什么？
2. 它在哪一层用 processor 做转换？
3. 它是否遵守统一的 policy / robot / env 接口？
4. 它接入的是哪条主线：train、eval、record，还是 RL actor-learner？

这就是“从读代码到形成平台视角”的真正跨越。

---

## 十四、你现在应该掌握的最终 8 个问题

走完整套学习路线后，建议你检查自己是否能清楚回答下面这些问题：

1. `lerobot-train` 和 `lerobot-eval` 的主链各是什么？
2. `LeRobotDataset` 与 `delta_timestamps` 在平台里承担什么角色？
3. 为什么 processor 是这个仓库最核心的中间层？
4. ACT、Diffusion、SmolVLA 分别代表哪三种动作学习范式？
5. env processor 和 policy processor 为什么要分层？
6. robot / teleop / dataset 三个 action-observation 空间为什么要显式解耦？
7. RL actor 和 learner 分别负责什么？
8. 为什么说 feature contract 比具体模型更接近这个仓库的“系统中心”？

如果这 8 个问题都能答得比较清楚，说明你已经不只是“看过这套代码”，而是基本建立了 LeRobot 的系统性认知。

---

## 十五、后续深入建议

整个路线走完后，后面就不再建议“平均读完所有目录”，而是按兴趣专题深入。

### 方向 1：策略研究

如果你更想走模型和算法方向，建议继续：

1. 精读 SmolVLA / Pi0 / 更大的 VLA 策略
2. 对比 ACT / Diffusion / Flow Matching 的时序建模差异
3. 研究 PEFT、frozen backbone 和不同 fine-tuning 方案

### 方向 2：真实机器人系统

如果你更想走硬件系统与部署方向，建议继续：

1. 精读 `robots/` 与 `teleoperators/`
2. 按文档尝试自己设计一套 robot processor
3. 读实际某个机器人实现，例如 SO100 / SO101

### 方向 3：在线学习与人类在环

如果你更想走 RL / HIL / 在线更新方向，建议继续：

1. 精读 `rl/` 子目录
2. 读 replay buffer、learner service、transport
3. 再回头结合具体 RL policy 看更新逻辑

---

## 十六、整套路线总结

这套学习流程到这里就算完整走完了。

如果把 0 到 8 阶段压缩成一条主线，可以写成：

`README/Config -> Train -> Dataset -> Processor -> Policy -> Eval/Env -> VLA -> RL/Robot`

这条顺序的价值在于：

1. 每一阶段都建立在前一阶段的抽象之上
2. 不会一上来就陷在模型细节里
3. 最后能够形成完整的平台视角

到这里，你对 LeRobot 的理解应该已经从“知道它能训练一些机器人模型”，升级到：

“我知道它如何把数据、模型、环境、硬件和在线学习组织成一个统一系统。”

这份文档基于源码和项目文档整理，没有做本地运行验证，重点是完成第 8 阶段的系统性收尾与闭环总结。

