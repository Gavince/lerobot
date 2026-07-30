# LeRobot 第6阶段：env与eval评测主线学习

## 一、阶段目标

这一阶段的目标，是把前面已经理解的 `dataset / processor / policy` 放到真正的 rollout 场景中，回答下面这个关键问题：

一个策略在 LeRobot 里，到底是如何进入环境、逐步产生动作、收集 reward 和 success，并最终形成评测结果的？

对应的核心文件：

- [src/lerobot/scripts/lerobot_eval.py](/Users/zwy/Macin/lerobot/src/lerobot/scripts/lerobot_eval.py)
- [src/lerobot/configs/eval.py](/Users/zwy/Macin/lerobot/src/lerobot/configs/eval.py)
- [src/lerobot/configs/default.py](/Users/zwy/Macin/lerobot/src/lerobot/configs/default.py)
- [src/lerobot/envs/factory.py](/Users/zwy/Macin/lerobot/src/lerobot/envs/factory.py)
- [src/lerobot/envs/configs.py](/Users/zwy/Macin/lerobot/src/lerobot/envs/configs.py)
- [src/lerobot/envs/utils.py](/Users/zwy/Macin/lerobot/src/lerobot/envs/utils.py)
- [tests/envs/test_envs.py](/Users/zwy/Macin/lerobot/tests/envs/test_envs.py)
- [tests/policies/test_policies.py](/Users/zwy/Macin/lerobot/tests/policies/test_policies.py)

这一阶段的核心收获可以先提前说一句：

LeRobot 的 eval 主线并不是“把 policy 丢进 env 跑一下”那么简单，而是通过 env processor 和 policy processor 把环境世界、策略世界和统计世界连接成了一条统一数据链。

---

## 二、先建立总流程图

如果把 `lerobot-eval` 用一句话概括，可以写成：

`EvalPipelineConfig -> make_env -> make_policy -> make_pre_post_processors -> rollout -> metrics aggregation -> save eval_info`

更完整一点的主链是：

1. 读取 `EvalPipelineConfig`
2. 从 Hub 或本地 checkpoint 加载 policy config
3. 创建一个或多个 vectorized env
4. 创建 policy 的 pre/post processors
5. 创建 env 专属的 pre/post processors
6. 在 rollout 中循环：
   - reset env 和 policy
   - 把 env observation 变成 LeRobot observation
   - 先走 env preprocessor
   - 再走 policy preprocessor
   - `policy.select_action(...)`
   - 走 policy postprocessor
   - 走 env postprocessor
   - `env.step(...)`
7. 汇总 reward / success / done / video
8. 形成 per-task、per-group 和 overall 三级评测结果
9. 写入 `eval_info.json`

这一阶段最重要的不是记住函数名，而是记住这个顺序。因为它说明了 processor 在评测阶段不是附属品，而是控制链的中间枢纽。

---

## 三、配置入口：`EvalPipelineConfig`

评测配置定义在 [src/lerobot/configs/eval.py](/Users/zwy/Macin/lerobot/src/lerobot/configs/eval.py)。

### 3.1 这个配置解决了什么问题

`EvalPipelineConfig` 主要负责三件事：

1. 组合 env、eval 和 policy 这三类配置。
2. 允许通过 `--policy.path=...` 从 checkpoint 或 Hub repo 直接恢复 policy config。
3. 自动补齐 `job_name` 和 `output_dir`。

这和训练配置很像，但有一个明显区别：

训练时 top-level config 的中心是 dataset；  
评测时 top-level config 的中心是 env。

因为在 eval 主链里，policy 不再从 dataset 学习，而是要与 environment 做闭环交互。

### 3.2 `policy.path` 的特殊处理

`EvalPipelineConfig.__post_init__()` 中有一个关键逻辑：

- 通过 `parser.get_path_arg("policy")` 拿到 CLI 中传入的 policy 路径
- 再调用 `PreTrainedConfig.from_pretrained(...)` 恢复 policy 配置
- 并把 `pretrained_path` 写回配置对象

这意味着 `lerobot-eval` 的真正入口通常不是“手写一大段 policy 配置”，而是“给它一个已训练好的 policy”。

这个设计和训练主线的 checkpoint 机制自然衔接上了。

### 3.3 `EvalConfig`

在 [src/lerobot/configs/default.py](/Users/zwy/Macin/lerobot/src/lerobot/configs/default.py) 中，`EvalConfig` 定义了几个很关键的运行参数：

- `n_episodes`
- `batch_size`
- `use_async_envs`

其中值得注意的是：

1. `batch_size=0` 时会自动根据 CPU 核数推断。
2. `batch_size` 不会超过 `n_episodes`。
3. `use_async_envs=True` 只是默认偏好；当 batch size 为 1 时实际会退化到同步环境。

这说明 LeRobot 的 eval 配置已经把“评测吞吐”和“资源约束”纳入了默认设计，而不是只关心算法逻辑。

---

## 四、环境构造：`make_env()` 为什么返回嵌套字典

`make_env()` 定义在 [src/lerobot/envs/factory.py](/Users/zwy/Macin/lerobot/src/lerobot/envs/factory.py)。

它的返回值不是单个 env，而是：

```python
dict[str, dict[int, gym.vector.VectorEnv]]
```

也就是：

`{suite_name: {task_id: vec_env}}`

### 4.1 为什么不是直接返回一个 env

这是因为 LeRobot 不只支持单任务环境，还支持多任务 benchmark。

所以这个结构同时兼容两种情况：

1. 单任务环境  
   比如 `pusht`，可以视为只有一个 suite，内部只有 `task_id=0`

2. 多任务基准  
   比如某些 benchmark 可能一个 suite 下有多个任务，每个任务对应自己的 vector env

也就是说，LeRobot 的 env 抽象不是“一个环境实例”，而是“一个可被统一评测的任务集合”。

### 4.2 本地 env 与 Hub env

`make_env()` 支持两类来源：

1. 本地注册的 `EnvConfig`
2. Hub 上携带 `env.py` 的远程环境

如果是 Hub env：

- 会解析 Hub URL
- 下载远程 `env.py`
- 动态导入模块
- 调用远程 `make_env`
- 再统一标准化为 `{suite: {task_id: vec_env}}`

而且这里必须显式传入 `trust_remote_code=True`。

这个点非常重要：  
LeRobot 在功能上支持远程环境扩展，但在接口上明确要求显式信任，避免默认执行远程代码。

### 4.3 `EnvConfig.create_envs()`

`EnvConfig` 基类在 [src/lerobot/envs/configs.py](/Users/zwy/Macin/lerobot/src/lerobot/envs/configs.py) 中定义。

默认的 `create_envs()` 会：

1. 判断是 `SyncVectorEnv` 还是 `AsyncVectorEnv`
2. 检查目标 gym id 是否已经注册
3. 如未注册，则尝试导入对应 `gym_<env_type>` 包
4. 创建 `n_envs` 个子环境
5. 返回 `{self.type: {0: vec_env}}`

所以 env config 在 LeRobot 中不只是“保存参数”，还负责实际的环境工厂逻辑。

---

## 五、环境配置里的另一个重点：feature contract

`EnvConfig` 里有两个很关键的字段：

- `features`
- `features_map`

这两个字段其实就是 env 与 policy 之间的输入契约。

以 `PushtEnv` 为例，它会声明：

1. 哪些原始环境键存在  
   比如 `pixels`、`agent_pos`、`environment_state`

2. 这些键在策略世界里该映射成什么  
   比如：
   - `pixels -> observation.image`
   - `agent_pos -> observation.state`
   - `environment_state -> observation.environment_state`

这说明 env 不是直接把原始 gym observation 生硬交给 policy，而是通过 feature contract 明确告诉平台：

“我的观测有哪些组成部分，它们在 LeRobot 语义里分别属于哪种 feature。”

这和前几阶段学到的 policy feature、dataset feature 是完全呼应的。

---

## 六、从环境观测到策略输入：`preprocess_observation()`

这一步是 env/eval 主线里最值得认真看的函数之一，位于 [src/lerobot/envs/utils.py](/Users/zwy/Macin/lerobot/src/lerobot/envs/utils.py)。

### 6.1 它做了什么

`preprocess_observation()` 的作用是把环境返回的 numpy observation 转成 LeRobot policy 熟悉的 tensor 格式。

它主要做这些事情：

1. 图像：
   - 支持单路 `pixels`
   - 也支持多路 camera dict
   - 从 `uint8` 转成 `float32`
   - 从 `[0,255]` 归一到 `[0,1]`
   - 从 channel-last 转成 channel-first

2. 状态：
   - `agent_pos -> observation.state`
   - `environment_state -> observation.environment_state`

3. 其他嵌套结构：
   - `robot_state`
   - IsaacLab 的 `policy` / `camera_obs`

### 6.2 这一步为什么不能省

因为环境世界和策略世界默认并不说同一种“数据语言”。

例如：

1. gym env 常用 numpy
2. policy 常用 torch tensor
3. 图像在 env 中通常是 HWC uint8
4. policy 更常用 CHW float32

所以 `preprocess_observation()` 的地位有点像 dataset 阶段的基础样本转换层。它负责把原始 env 输出推进到后续 processor 能稳定处理的格式。

### 6.3 与 dataset 的关系

这一点很值得记：

- dataset 阶段负责把离线数据变成 policy 可训练的 batch
- env 阶段负责把在线观测变成 policy 可推理的 batch

本质上它们在做同一类工作：  
把“世界中的数据”翻译成“策略中的数据”。

---

## 七、processor 在评测主线里的双层作用

到这一阶段，processor 的核心价值终于完整显现出来了。

在 `eval_main()` 中，实际上会创建两类 processor：

1. policy 的 pre/post processors
2. env 的 pre/post processors

### 7.1 policy processors

通过 `make_pre_post_processors(...)` 创建。

它们负责的事情包括：

1. 观察键重命名
2. 加 batch 维
3. 放到正确 device
4. 归一化
5. 动作反归一化
6. 转回 CPU

这和训练、推理阶段的 processor 逻辑是统一的。

### 7.2 env processors

通过 `make_env_pre_post_processors(...)` 创建。

默认情况下，env processor 可能是 identity；  
但对某些环境或特定 policy 组合，例如 LIBERO / XVLA，这一层会负责更专门的适配逻辑。

### 7.3 为什么是两层而不是一层

因为 env 适配和 policy 适配不是一回事：

1. env processor 负责环境语义层面的转换
2. policy processor 负责模型输入输出层面的转换

这个分层非常重要。它保证：

- 换一个 policy，不一定要重写 env 适配
- 换一个 env，不一定要重写 policy 归一化逻辑

这正是“平台化解耦”的体现。

---

## 八、rollout：评测主循环真正发生了什么

`rollout()` 定义在 [src/lerobot/scripts/lerobot_eval.py](/Users/zwy/Macin/lerobot/src/lerobot/scripts/lerobot_eval.py)。

这是第 6 阶段最核心的函数。

### 8.1 初始化阶段

一开始会做两件事：

1. `policy.reset()`
2. `env.reset(seed=...)`

这一步非常关键，因为像 ACT、Diffusion 这类带 action queue / observation history 的策略，必须在 episode 开头清空内部状态。

### 8.2 每一步循环的顺序

rollout 主循环的顺序非常值得背下来：

1. `preprocess_observation(observation)`
2. 可选缓存 observation
3. 从 env 里推断 `task_description` 或 `task`
4. `env_preprocessor(observation)`
5. `preprocessor(observation)`
6. `policy.select_action(observation)`
7. `postprocessor(action)`
8. `env_postprocessor({ACTION: action})`
9. `env.step(action_numpy)`
10. 收集 `reward / done / success`

也就是说，真正的链路是：

`env raw obs -> env utils -> env preprocessor -> policy preprocessor -> policy -> policy postprocessor -> env postprocessor -> env.step`

这是整个代码库里最能体现“模块拼装顺序”的地方。

### 8.3 `task` 为什么会被动态注入

rollout 会优先尝试从子环境里读：

- `task_description`
- 或 `task`

然后把它放进 observation 中。

这说明 LeRobot 在评测阶段已经考虑到某些策略需要自然语言任务描述，而不只是数值观测。这为后续 VLA 策略和多任务环境打下了统一接口。

### 8.4 `done` 的处理方式

rollout 中的 `done` 不是单步 done，而是 cumulative done：

- 一旦某个环境实例 done，后续时间步都保持 True

这样做的好处是后续聚合时可以方便地构造 mask，把 episode 真正结束后的“多余时间步”统一屏蔽掉。

---

## 九、`eval_policy()`：从 rollout 到单环境评测结果

如果说 `rollout()` 负责“跑一遍”，那么 `eval_policy()` 负责“把多次 rollout 组织成评测结果”。

### 9.1 它的主要工作

1. 根据 `n_episodes` 和 `env.num_envs` 计算需要多少个 batched rollouts
2. 每次调用一次 `rollout()`
3. 根据 `done_indices` 找到每个 episode 真正结束的位置
4. 构造 mask，去掉 done 之后的无效数据
5. 统计：
   - `sum_reward`
   - `max_reward`
   - `success`
   - `seed`
6. 可选保存视频
7. 可选返回完整 episode data

### 9.2 `done_indices` 和 mask 是关键理解点

`eval_policy()` 不直接相信 rollout 序列的全部时间步，而是先找：

- 每个 batch 元素第一次 done 出现在哪里

然后构造 mask，仅保留到 done 那一刻为止的有效片段。

这个设计非常重要，因为 vector env 的 batch 中，不同环境的 episode 长度可能不完全一致。如果不做 mask，后续 reward/success 聚合就会混入无效尾部。

### 9.3 视频保存

如果设置了 `max_episodes_rendered`，评测会：

1. 在 rollout 过程中缓存渲染帧
2. 在 episode 完成后异步写入 mp4

这说明评测输出不只是数值指标，还可以带有可视化证据，方便做 qualitative 检查。

---

## 十、`eval_policy_all()`：从单任务评测扩展到多任务基准

这是第 6 阶段另一个必须理解的函数。

因为 `make_env()` 返回的是 `{suite: {task_id: env}}`，所以评测不能只考虑一个 env，而要考虑一个任务集合。

### 10.1 它做了什么

`eval_policy_all()` 的流程是：

1. 把嵌套 env dict 拉平成 `(task_group, task_id, env)` 列表
2. 逐个任务调用 `eval_one()` / `run_one()`
3. 把结果累计到：
   - per-group
   - overall
4. 最后返回：
   - `per_task`
   - `per_group`
   - `overall`

### 10.2 三级统计结构

最终输出最值得记住的是这三层：

1. `per_task`
   每个具体任务的明细

2. `per_group`
   同一 suite / benchmark 组内的聚合指标

3. `overall`
   跨全部任务的整体指标

这使得 LeRobot 的 eval 能同时支持：

1. 单任务调试
2. benchmark 内比较
3. 跨任务总体报告

### 10.3 并行任务执行

`eval_policy_all()` 允许 `max_parallel_tasks > 1`，会使用 `ThreadPoolExecutor` 并行执行多个任务。

但它也保留了串行路径，并在串行路径里做了一点很细致的优化：

- 当前 env 关闭后，再预热下一个 lazy async env，避免 GPU/EGL 资源重叠

这说明评测系统设计时，不只是算法正确性，还考虑了资源峰值控制。

---

## 十一、env 工具层还有哪些重要设计

### 11.1 `_LazyAsyncVectorEnv`

`envs/utils.py` 中的 `_LazyAsyncVectorEnv` 很值得留意。

它不是立刻创建所有 async worker，而是在第一次 `reset()/step()/call()` 时才真正创建底层 `AsyncVectorEnv`。

这个设计特别适合多任务 benchmark：

如果一开始就把所有 task 的 async env 都拉起来，会产生大量 worker 进程和 GPU/EGL 资源占用；但很多任务其实是顺序评测的。

所以 lazy creation 的好处是：

只在任务真正开始评测时才消耗这部分资源。

### 11.2 `close_envs()`

LeRobot 没有把 env close 当成一次简单的 `env.close()`，而是提供了能处理 mapping / sequence 等嵌套结构的 `close_envs()`。

这和前面 env 返回嵌套字典的设计是匹配的。

### 11.3 `check_env_attributes_and_types()`

rollout 开头会检查环境是否提供 `task_description` 或 `task`。

这其实是在提醒你：

从平台角度看，一个现代机器人环境不仅要给 observation 和 reward，最好还要给任务语义信息。

---

## 十二、测试文件帮我们确认了什么

### 12.1 `tests/envs/test_envs.py`

这个文件帮助确认：

1. 常见 env 能正确注册并通过 gym checker
2. `make_env_config()` 和 `make_env()` 能正常构造 env
3. `preprocess_observation()` 输出的图像已经是 `float32` 且范围在 `[0,1]`
4. Hub env 的 URL 解析、结果标准化和 `trust_remote_code` 机制都是生效的

从学习角度上说，这个测试文件就是 env factory 契约的最短说明书。

### 12.2 `tests/policies/test_policies.py`

这个文件虽然主要测试 policy，但也顺带证明了 eval 主线里的一个重要事实：

任意 policy 只要遵守 `PreTrainedPolicy` 契约，就可以：

1. 被实例化
2. 执行 `forward()`
3. 接收 `preprocess_observation()` 后的 env observation
4. 通过 `select_action()` 输出 action
5. 把 action 真正喂回 env.step()

这就从测试层面验证了“policy 与 env 的统一接口确实成立”。

---

## 十三、这一阶段最重要的理解：LeRobot 如何统一仿真评测

经过这一阶段，你应该能把 LeRobot 的评测统一性总结成下面 4 点。

### 13.1 环境统一

无论是本地 gym env，还是 Hub 远程 env，最终都会被整理成：

`{suite_name: {task_id: vec_env}}`

### 13.2 观测统一

无论原始环境观测长什么样，都会先经过 `preprocess_observation()` 转成 LeRobot 约定的 observation 格式。

### 13.3 策略接口统一

无论底层是 ACT、Diffusion 还是后续更复杂的策略，对上层统一暴露的仍然是：

- `reset()`
- `select_action()`

### 13.4 评测结果统一

无论是单任务还是多任务，最终都会归结成可比较的：

- `avg_sum_reward`
- `avg_max_reward`
- `pc_success`
- `eval_s`
- `eval_ep_s`

这就是 LeRobot 能把不同任务、不同策略、不同环境纳入同一评测框架的原因。

---

## 十四、建议你在纸面上完成的 3 个练习

### 练习 1：手画 rollout 数据流

建议从 `env.reset()` 开始，一直画到 `env.step(action)`，把中间每一步标出来：

- 原始 observation
- `preprocess_observation`
- env preprocessor
- policy preprocessor
- `policy.select_action`
- policy postprocessor
- env postprocessor
- numpy action

只要这张图画出来，第 6 阶段就基本算吃透了。

### 练习 2：解释为什么需要两套 processor

试着不用看代码，自己说明：

“为什么 LeRobot 不把 env 适配和 policy 适配混在一个 processor 里做？”

如果你能答出“语义适配”和“模型适配”分层的区别，这一层就通了。

### 练习 3：解释 `done_indices + mask`

自己举一个 vector env 的例子：

- 环境 A 第 20 步结束
- 环境 B 第 27 步结束

然后解释 `eval_policy()` 为什么一定要先找 done 索引再做 mask。

这个练习会帮你真正理解 batched rollout 和 episode-level metrics 之间的差别。

---

## 十五、本阶段总结

第 6 阶段最重要的收获，是把前几阶段那些看起来分散的模块真正串成了闭环：

1. env 提供可交互世界
2. `preprocess_observation()` 把在线观测翻译成策略可理解的格式
3. env processor 和 policy processor 分层完成适配
4. policy 通过统一接口输出动作
5. rollout 记录 reward / success / done
6. eval 层再把 rollout 压缩成可比较的评测指标

从这里开始，你对 LeRobot 的理解就不再只是“能训练某个 policy”，而是开始进入：

“我知道这个平台如何把训练、推理、评测和多任务环境统一到一条标准化链路里。”

---

## 十六、下一阶段建议

接下来最自然的方向有两个：

1. 继续按原计划进入第 7 阶段，学习 VLA 路线，例如 SmolVLA
2. 如果你更想先闭合平台认知，也可以先补一个“训练主线 vs 评测主线”对照总结

如果继续按既定学习路线走，第 7 阶段会很合适，因为你现在已经具备了读更复杂策略接入平台时所需要的基础视角：

- 看输入输出契约
- 看 processor 适配
- 看策略内部建模
- 看它如何进入 eval / rollout

这份文档基于源码阅读整理，当前阶段没有做本地运行验证，重点在于梳理架构主线与代码理解路径。

