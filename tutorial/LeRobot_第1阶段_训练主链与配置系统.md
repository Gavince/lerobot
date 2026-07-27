# LeRobot 第 1 阶段学习文档：训练主链与配置系统

## 1. 本阶段目标

第 1 阶段的核心目标是回答一个非常具体的问题：

> `lerobot-train` 到底是如何把配置、数据、策略、处理器、优化器、评估和 checkpoint 串起来的？

这一阶段不追求理解某个具体 policy 的内部算法，而是优先搞清楚训练主链的系统组织方式。

本次学习主要基于以下文件：

- `src/lerobot/scripts/lerobot_train.py`
- `src/lerobot/configs/train.py`
- `src/lerobot/configs/parser.py`
- `src/lerobot/configs/default.py`
- `src/lerobot/optim/factory.py`
- `src/lerobot/common/train_utils.py`
- `src/lerobot/utils/logging_utils.py`
- `tests/utils/test_train_utils.py`
- `tests/training/test_multi_gpu.py`

说明：

- 本阶段以源码阅读和结构分析为主。
- 这份文档关注训练主链和配置系统，不展开 dataset、processor、policy 细节实现。

---

## 2. 第 1 阶段先给出的总结

如果先用一句话概括 LeRobot 的训练主链：

> `lerobot-train` 的本质是：先通过配置系统把训练任务“装配”出来，再通过统一训练循环把 dataset、policy、processor、optimizer、eval 和 checkpoint 串起来执行。

更具体一点，它的训练主链可以概括成：

`CLI -> parser.wrap -> TrainPipelineConfig -> cfg.validate() -> make_dataset -> make_env -> make_policy -> make_pre_post_processors -> make_optimizer_and_scheduler -> accelerator.prepare -> training loop -> checkpoint/eval -> push_to_hub`

这里最值得注意的是：

1. 配置系统不只是“存参数”，还负责相当多的装配逻辑
2. 训练脚本本身更像 orchestration layer，而不是把所有细节都写死
3. 分布式训练是主链中的一等公民，不是补丁式支持

---

## 3. 从入口开始：`lerobot_train.py` 在做什么

训练入口位于：

- `src/lerobot/scripts/lerobot_train.py`

这个文件承担两个角色：

1. 训练 orchestrator
2. 统一训练循环的实现位置

从结构上看，它主要由三部分构成：

### 3.1 `update_policy`

这是单步参数更新函数，负责：

- 调 `policy.forward(batch)` 计算 loss
- 用 `accelerator.backward(loss)` 反向传播
- 做 gradient clipping
- optimizer step
- scheduler step
- 记录训练指标

这一层说明 LeRobot 对 policy 的要求是明确的：

- policy 必须能在统一接口下输出 loss 和日志信息
- 训练脚本不关心 policy 内部结构，只关心 `forward()` 的协议

### 3.2 `train(cfg, accelerator=None)`

这是主训练函数，真正把系统装起来并运行。

它负责：

- 校验配置
- 创建 Accelerator
- 初始化日志和随机种子
- 创建 dataset / env / policy / processors / optimizer / scheduler
- 处理 resume
- 构建 dataloader
- 执行训练循环
- 定期记录日志、保存 checkpoint、运行 eval
- 结束后 push 到 Hub

### 3.3 `main()`

`main()` 先执行插件注册，再调用 `train()`。

这说明训练入口不是纯粹的静态代码路径，它允许第三方插件先参与注册扩展，再进入训练。

---

## 4. 配置系统：`TrainPipelineConfig` 才是训练装配中心

训练主配置位于：

- `src/lerobot/configs/train.py`

这个类非常关键。它不是被动参数容器，而是训练任务的“装配前置层”。

## 4.1 它包含哪些配置块

`TrainPipelineConfig` 主要由这些部分组成：

- `dataset`
- `env`
- `policy`
- `output_dir`
- `resume`
- `seed`
- `num_workers`
- `batch_size`
- `steps`
- `eval_freq`
- `log_freq`
- `save_freq`
- `optimizer`
- `scheduler`
- `eval`
- `wandb`
- `peft`
- `rename_map`
- RA-BC 相关参数

这说明 LeRobot 的训练配置是“顶层总配置”，不是把每个功能散落在不同地方。

## 4.2 `validate()` 才是理解训练主链的关键

`validate()` 做了很多决定训练行为的工作：

### 4.2.1 处理 `policy.path`

如果命令行里提供了 `--policy.path=...`，它不会简单地保留字符串，而是：

- 通过 `PreTrainedConfig.from_pretrained(...)` 加载 policy 配置
- 合并对应的 CLI overrides
- 设置 `policy.pretrained_path`

这说明：

- policy 可以从本地目录或 Hub 预训练配置中恢复
- 配置加载和 checkpoint 恢复是统一思路的一部分

### 4.2.2 处理 `resume`

如果 `resume=True`：

- 会读取 `config_path`
- 推断 policy/checkpoint 的目录
- 准备后续恢复 training state

这说明 LeRobot 的 resume 不是“重新指定一堆参数再拼回去”，而是明确依赖保存下来的训练配置和训练状态。

### 4.2.3 自动推导 `job_name` 和 `output_dir`

如果用户没手动指定：

- `job_name` 默认基于 `env.type` 和 `policy.type`
- `output_dir` 默认写到 `outputs/train/YYYY-MM-DD/...`

这说明输出目录命名本身也是系统设计的一部分，而不是临时拼接。

### 4.2.4 自动使用 policy 预设的 optimizer/scheduler

如果 `use_policy_training_preset=True`：

- `optimizer = policy.get_optimizer_preset()`
- `scheduler = policy.get_scheduler_preset()`

这是一个很重要的设计点：

- 训练配置允许统一覆盖
- 但 policy 仍然可以携带“默认最合理训练预设”

也就是说，LeRobot 把“算法推荐训练配置”和“训练脚本主循环”分离了。

### 4.2.5 push_to_hub 的合法性检查

如果要 push 模型，但没给 `policy.repo_id`，会直接报错。

这说明 Hub 并不是后置附加能力，而是训练流程明确考虑的一部分。

### 4.2.6 RA-BC 参数自动推导

如果启用 `use_rabc`，但没给 `rabc_progress_path`，它会尝试：

- 从本地数据路径推断
- 或构造 Hugging Face 数据路径

这说明配置系统也在服务更高级的训练模式，而不只是基础训练。

---

## 5. `parser.wrap`：为什么入口函数不是普通 argparse

配置包装逻辑在：

- `src/lerobot/configs/parser.py`

这里的 `@parser.wrap()` 非常关键。

## 5.1 它比普通配置解析多做了什么

相比直接 `draccus.parse(...)`，`wrap()` 额外做了三件事：

1. 过滤 `.path` 参数，后续再处理
2. 支持从 `config_path` / `from_pretrained` 直接恢复完整配置
3. 支持插件参数和插件自动加载

这说明 LeRobot 的 parser 并不是单纯“把命令行转成 dataclass”，而是在做运行时装配准备。

## 5.2 为什么要特别处理 `.path`

像 `--policy.path=...` 这种参数，不能简单和 `--policy.type=...` 一样处理，因为：

- 它代表“从已有预训练目录恢复配置”
- 而不是“根据 CLI 从零构造配置”

所以 parser 先把 path 参数拿掉，再交给 `TrainPipelineConfig.validate()` 处理。

这是一种很工程化的做法：

- CLI 解析逻辑尽量保持统一
- 特殊恢复逻辑留给具体配置类去决定

## 5.3 插件机制是训练主链的前置扩展点

`wrap()` 会在配置解析前尝试加载插件。

这意味着：

- 第三方包可以在导入时注册自己的 config subclass
- 然后 `draccus` 才能识别对应的 `.type`

所以 LeRobot 的训练链并不封闭，它允许外部扩展接入统一训练入口。

---

## 6. 训练主链的真实执行顺序

下面按照 `train()` 的执行顺序，把一次训练如何跑起来梳理出来。

## 6.1 第一步：校验配置

进入 `train(cfg)` 后，第一件事是：

- `cfg.validate()`

这一步的意义是把“用户输入的半成品配置”补全成“真正可执行配置”。

所以第 1 阶段最重要的认识之一是：

> 训练任务在进入主循环前，已经被配置系统预装配了一次。

## 6.2 第二步：创建 `Accelerator`

如果没显式传入 accelerator，则创建新的 `Accelerator`，并设置：

- `step_scheduler_with_optimizer=False`
- `find_unused_parameters=True`
- 如果 policy 指定 CPU，则强制 CPU

这说明：

- LeRobot 默认把分布式训练和 mixed precision 纳入主链
- 它不是后续才“适配” accelerate，而是训练主流程从一开始就围绕 accelerate 来写

## 6.3 第三步：日志、主进程判断、wandb、随机种子

这一步会：

- 初始化日志系统
- 判断 `is_main_process`
- 主进程打印配置
- 主进程初始化 wandb
- 统一设定 seed

这里可以看出一个统一原则：

- 多进程训练时，副进程尽量不做重复日志和重复初始化

## 6.4 第四步：创建 dataset

训练中 dataset 的创建有一个很有代表性的分布式设计：

1. 主进程先创建 dataset
2. `accelerator.wait_for_everyone()`
3. 其他进程再创建 dataset

这样做是为了避免多个进程同时下载或构建数据引发竞争条件。

这说明 LeRobot 已经考虑到：

- 多进程训练时的数据初始化 race condition

## 6.5 第五步：可选创建 eval env

如果：

- `cfg.eval_freq > 0`
- `cfg.env is not None`
- 当前是主进程

就会创建评估环境。

这说明训练中 checkpoint 评估被视为主训练流程的一部分，而不是完全独立的外部脚本行为。

## 6.6 第六步：创建 policy

policy 通过统一工厂构建：

- `make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)`

这个调用说明训练脚本认为 policy 构建至少依赖：

- policy config
- dataset metadata
- 观测字段重命名规则

所以 policy 不是只依赖一个 config，它和 dataset schema 是有绑定关系的。

## 6.7 第七步：可选套上 PEFT

如果启用 `cfg.peft`：

- 会调用 `policy.wrap_with_peft(...)`

这说明 LeRobot 把 PEFT 视为 policy 构建后的包装层，而不是另一套独立训练路径。

## 6.8 第八步：创建 processors

processors 的创建逻辑非常值得注意：

- 如果从 pretrained processor 恢复，会传入 overrides
- 如果不是从已有 processor 恢复，会传入 `dataset.meta.stats`
- 对 SARM 还会传入 `dataset_meta`

这说明 processor 在训练主链里不是可有可无的前处理，而是：

- 一个可以单独保存和恢复的对象
- 一个依赖 dataset stats 的状态化组件
- 一个需要随着 checkpoint 一起管理的训练资产

## 6.9 第九步：创建 optimizer 和 scheduler

逻辑在：

- `src/lerobot/optim/factory.py`

实现很短，但意义明确：

- 如果 `use_policy_training_preset=True`，就用 `policy.get_optim_params()`
- 否则用 `policy.parameters()`
- optimizer/scheduler 的具体构造来自配置对象自身的 `build(...)`

这说明优化器层有两层控制：

1. 顶层决定是否使用 policy 预设参数分组
2. optimizer config 决定具体实例化方式

## 6.10 第十步：如果 resume，则恢复 training state

resume 逻辑依赖：

- `load_training_state(...)`

恢复内容包括：

- step
- optimizer state
- scheduler state
- RNG state

这很重要，因为它说明 resume 并不仅仅恢复模型参数，而是尽量恢复训练进程的完整状态。

---

## 7. DataLoader 与训练循环是如何组织的

## 7.1 DataLoader 构建

训练中的 dataloader 使用：

- `torch.utils.data.DataLoader`

但它并不总是简单 `shuffle=True`。

如果 policy 有 `drop_n_last_frames`：

- 会改用 `EpisodeAwareSampler`
- 并关闭普通 shuffle

这说明 dataloader 行为会被 policy 的时间窗口需求影响。

换句话说：

> 训练主链中，policy 配置会反过来影响 dataset sampling 策略。

这也是后续理解时序模型时很关键的一点。

## 7.2 `accelerator.prepare(...)`

在进入训练循环前，LeRobot 会统一对以下对象做 prepare：

- policy
- optimizer
- dataloader
- lr_scheduler

这一步是 distributed / mixed-precision 支持的主入口。

所以从训练主链视角看，真正进入“可训练状态”的时刻是：

`accelerator.prepare(...)` 之后。

## 7.3 训练循环每一步做什么

每个 step 的主逻辑是：

1. 从 dataloader 拿一个 batch
2. 把 uint8 图像转成 float32 / 255
3. 应用 `preprocessor(batch)`
4. 记录 dataloading 时间
5. 调 `update_policy(...)`
6. 更新 step 和 tracker
7. 按频率执行 log / save / eval

这说明训练循环非常明确地分成三段：

- 数据准备
- 参数更新
- 周期性副作用（日志/保存/评估）

---

## 8. 日志与指标系统

指标跟踪逻辑在：

- `src/lerobot/utils/logging_utils.py`

这里有两个核心类：

- `AverageMeter`
- `MetricsTracker`

## 8.1 `AverageMeter`

负责单个指标的累计平均，比如：

- loss
- grad_norm
- lr
- dataloading_s
- update_s

## 8.2 `MetricsTracker`

负责把训练过程中的更高层统计也一起维护起来，例如：

- `steps`
- `samples`
- `episodes`
- `epochs`

这个设计很有意思，因为它说明 LeRobot 的训练日志不是只输出 loss，而是把“在整个数据集尺度上训练到了哪里”也显式表达出来。

这对于机器人数据训练尤其重要，因为很多训练不是按传统 epoch 驱动来理解的。

---

## 9. checkpoint 机制是如何设计的

checkpoint 工具在：

- `src/lerobot/common/train_utils.py`

## 9.1 step 标识与目录结构

每个 checkpoint 使用统一 step id，例如：

- `000005`
- `000123`

目录大致结构是：

- `checkpoints/<step>/pretrained_model/`
- `checkpoints/<step>/training_state/`

## 9.2 checkpoint 里保存什么

`save_checkpoint(...)` 会保存：

### `pretrained_model/`

- policy config
- model weights
- train config
- preprocessor
- postprocessor

### `training_state/`

- training step
- RNG state
- optimizer state
- scheduler state

这个设计非常关键，因为它说明 LeRobot checkpoint 不是只保存模型，而是保存：

> 模型 + 训练配置 + processor + 训练状态

这和很多只存 `state_dict` 的简化训练脚本完全不同。

## 9.3 `last` 软链接

`update_last_checkpoint(...)` 会维护一个 `last` 符号链接指向最近 checkpoint。

这类设计很小，但很实用，说明项目已经考虑到工程使用体验。

## 9.4 从测试看 checkpoint 契约

`tests/utils/test_train_utils.py` 验证了：

- step identifier 的格式
- checkpoint 路径生成
- last symlink 更新
- training step 的保存与恢复
- optimizer/scheduler/RNG state 的保存与恢复

这说明 checkpoint 体系是稳定接口，不是临时实现。

---

## 10. eval 在训练主链中的位置

LeRobot 的训练脚本会在 step 满足 `eval_freq` 时执行 eval。

eval 逻辑的大致路径是：

1. 调 `eval_policy_all(...)`
2. 传入：
   - envs
   - env_preprocessor
   - env_postprocessor
   - policy preprocessor
   - policy postprocessor
3. 收集整体指标
4. 写日志 / wandb / 视频

这说明训练中的 eval 不是“只拿模型去跑一下”，而是完整复用了：

- env processor
- policy processor
- rollout 逻辑

这也是为什么 processor 在这个平台里地位很高。

---

## 11. 多 GPU 支持说明了什么

`tests/training/test_multi_gpu.py` 非常有价值，因为它让我们看到：

- 这个训练入口是被设计成可以直接由 `accelerate launch` 驱动的
- 数据预下载是为了避免多进程竞争
- 只有主进程负责 checkpoint 保存

这几个事实能帮助我们确认：

1. accelerate 支持不是装饰性功能
2. 训练脚本的主流程从一开始就考虑了多进程语义
3. 主进程/副进程职责边界是明确设计过的

---

## 12. 第 1 阶段最关键的设计认识

通过这一阶段，可以总结出几个特别重要的认识。

## 认识 1：配置系统不只是参数容器

`TrainPipelineConfig.validate()` 实际承担了：

- 恢复 pretrained policy config
- 推导 output_dir
- 生成 optimizer/scheduler 预设
- resume 参数协调
- Hub 参数合法性检查

所以在 LeRobot 中：

> 配置层就是训练任务装配逻辑的一部分。

## 认识 2：训练脚本是 orchestration layer

`lerobot_train.py` 自己不实现具体模型算法，也不深入数据细节，而是统一调度：

- dataset
- env
- policy
- processor
- optimizer
- eval
- checkpoint

这意味着后续读代码时，要把它看成“系统调度器”，不是“大而全实现文件”。

## 认识 3：processor 已经在训练主链中占据关键位置

即使这一阶段还没深入 processor 实现，也已经能看到：

- batch 进入训练前要先过 preprocessor
- eval 时 env 和 policy 两侧都要过 processor
- checkpoint 要保存 processors

这说明 processor 不是附属工具，而是主链核心组件之一。

## 认识 4：resume 设计比普通训练脚本更完整

LeRobot 恢复的不仅是模型权重，还有：

- optimizer
- scheduler
- RNG
- step
- train config
- processor

这使得它更接近真正的工程训练平台，而不是只为单次实验服务的脚本。

## 认识 5：多进程语义是内建的

无论是 dataset 初始化、日志、checkpoint、eval，主进程和副进程职责都分得很明确。

这意味着：

- 后续阅读任何训练相关代码，都要默认它可能运行在 distributed 环境里

---

## 13. 本阶段结束后，你应该已经能回答的问题

完成第 1 阶段后，你应该能比较清楚地回答：

1. `lerobot-train` 的执行入口在哪里？
2. `@parser.wrap()` 比普通 CLI 解析多做了什么？
3. `TrainPipelineConfig.validate()` 为什么是训练主链中的关键步骤？
4. dataset、policy、processor、optimizer 是按什么顺序创建的？
5. `accelerator.prepare(...)` 在训练主链中的作用是什么？
6. checkpoint 里为什么不仅保存模型，还保存 processor 和 training state？
7. eval 为什么是训练主链中的一部分？

如果这些问题已经能回答清楚，说明第 1 阶段达标。

---

## 14. 下一阶段建议

第 1 阶段之后，最自然的下一步是进入 dataset 主线。

推荐顺序：

1. `src/lerobot/datasets/factory.py`
2. `src/lerobot/datasets/lerobot_dataset.py`
3. `src/lerobot/datasets/dataset_metadata.py`
4. `docs/source/lerobot-dataset-v3.mdx`

因为到这里为止，我们已经知道训练主链会调用 `make_dataset(cfg)`，下一阶段就应该搞清楚：

> 训练看到的那个 dataset，到底是怎么被构造出来的？

---

## 15. 本阶段个人学习结论

如果用一句话总结第 1 阶段：

> LeRobot 的训练脚本本质上是一个围绕配置系统和 accelerate 构建的统一调度器，它把 dataset、policy、processor、optimizer、eval 和 checkpoint 组织成一条可恢复、可扩展、可分布式执行的训练主链。

这条主链看懂后，后续再学 dataset、processor 和具体 policy，就不会迷失在零散文件里。
