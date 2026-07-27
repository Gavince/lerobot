# LeRobot 第 2 阶段学习文档：数据集系统与 LeRobotDataset

## 1. 本阶段目标

第 2 阶段要解决的问题是：

> 训练主链里看到的 dataset，到底是怎么被构造出来的？它为什么是 LeRobot 平台中的核心对象？

这一阶段的重点不是某个模型如何消费数据，而是先理解 LeRobot 自己是如何组织、存储、读取、流式访问和写入机器人数据的。

本次学习主要基于以下材料：

- `src/lerobot/datasets/factory.py`
- `src/lerobot/datasets/lerobot_dataset.py`
- `src/lerobot/datasets/dataset_metadata.py`
- `src/lerobot/datasets/streaming_dataset.py`
- `docs/source/lerobot-dataset-v3.mdx`
- `tests/datasets/test_lerobot_dataset.py`
- `tests/datasets/test_dataset_metadata.py`
- `tests/datasets/test_delta_timestamps.py`
- `tests/datasets/test_streaming.py`

---

## 2. 先给结论：LeRobotDataset 为什么是平台核心

如果先用一句话总结：

> LeRobotDataset 不是一个普通的 `torch.utils.data.Dataset` 包装器，而是 LeRobot 把机器人多模态时序数据、视频、元数据、Hub 同步和训练采样规则统一起来的核心抽象。

更具体地说，它同时承担了几类职责：

1. 统一数据格式
2. 把 episode 级机器人数据映射成 frame/sample 级训练样本
3. 通过 metadata 恢复多 episode 的真实边界与索引关系
4. 支持本地读取、Hub 下载和流式读取
5. 支持数据录制、追加写入和上传到 Hub
6. 为时序 policy 提供 `delta_timestamps` 采样能力

这也是为什么第 2 阶段非常关键：后面的 processor、policy、train loop 都默认 dataset 这层已经把复杂的数据组织问题解决掉了。

---

## 3. 数据主线从哪里开始：`make_dataset(cfg)`

训练主链真正创建 dataset 的入口在：

- `src/lerobot/datasets/factory.py`

它的作用非常明确：

1. 从 policy config 推导 `delta_timestamps`
2. 根据 dataset config 构建 `LeRobotDataset` 或 `StreamingLeRobotDataset`
3. 在需要时给视觉模态打上 ImageNet 统计量

### 3.1 `resolve_delta_timestamps`

这个函数很重要。它会读取 policy config 中的：

- `observation_delta_indices`
- `action_delta_indices`
- `reward_delta_indices`

然后结合数据集的 `fps`，把离散 index 变成真实时间偏移：

- `index / fps -> delta_timestamps`

这意味着一个非常关键的架构事实：

> dataset 的取样窗口不是完全由 dataset 自己决定，而是会被 policy 的时序需求反向约束。

例如：

- 单帧 policy 可能不需要历史窗口
- 动作 chunk policy 需要未来 action 窗口
- 多帧视觉 policy 需要过去若干帧观察

所以 LeRobot 的 dataset 工厂本质上是数据和策略的接口层之一。

### 3.2 `make_dataset` 的主要决策

这个函数会先：

- 构建 `image_transforms`
- 加载 `LeRobotDatasetMetadata`
- 推导 `delta_timestamps`

然后根据配置：

- 非 streaming：创建 `LeRobotDataset`
- streaming：创建 `StreamingLeRobotDataset`

最后可选地对 camera keys 写入 ImageNet 统计量。

这个逻辑说明：

1. dataset 创建前先看 metadata
2. dataset 的返回形式与训练方式有关
3. 视觉统计量可以由系统自动注入

---

## 4. LeRobotDataset 的角色：一个“读写双模态”的数据门面

核心类在：

- `src/lerobot/datasets/lerobot_dataset.py`

这个类最值得注意的一点是：

> 它同时支持读模式和写模式。

## 4.1 它不是单纯的读数据对象

从构造和类方法可以看出，`LeRobotDataset` 支持三种主要使用方式：

1. `__init__()`：读已有数据集
2. `create()`：创建一个新的可写数据集
3. `resume()`：在已有数据集上继续追加录制

这意味着 LeRobotDataset 不是训练专用类，而是一个统一数据 facade。

## 4.2 读模式

通过 `LeRobotDataset(...)` 初始化时：

- 它会先创建 `LeRobotDatasetMetadata`
- 再创建 `DatasetReader`
- 再尝试从本地或 Hub 加载实际数据

此时：

- `reader` 存在
- `writer` 为 `None`

`tests/datasets/test_lerobot_dataset.py` 也明确验证了这一契约。

## 4.3 写模式

通过 `create()` 或 `resume()` 创建时：

- 会创建 metadata
- 不会立刻加载 reader
- 会创建 `DatasetWriter`

这时 dataset 进入“录制/写入模式”。

这也是为什么 `LeRobotDataset` 里有一组很明确的 writer guard：

- `add_frame`
- `save_episode`
- `clear_episode_buffer`
- `finalize`

并且在未 `finalize()` 之前，不允许读 `__getitem__`。

这是一种很好的状态机设计：

- 读和写模式互相隔离
- 不允许混乱使用

---

## 5. v3 数据格式：为什么要从“按 episode 文件夹”转向“按文件分片 + metadata 恢复”

LeRobotDataset v3 的说明在：

- `docs/source/lerobot-dataset-v3.mdx`

这是第 2 阶段必须认真读的一份文档。

## 5.1 v3 的三个支柱

文档把 v3 概括为三个支柱：

1. **Tabular data**
   - 状态、动作、时间戳等低维信号存在 Parquet 中
2. **Visual data**
   - 视频帧按 camera 编码为 MP4
3. **Metadata**
   - 通过 info/stats/tasks/episodes 等元数据恢复 episode 边界和索引

## 5.2 为什么不是每个 episode 一个文件

v3 的核心思想是：

> 存储层和用户 API 解耦。

对用户来说，你感觉像是在访问 episode 或 frame；
对存储层来说，数据会被聚合成更大的文件块。

这样做的原因很现实：

1. 减少文件系统压力
2. 提升大规模数据集加载效率
3. 更适合 Hub 存储与流式读取
4. 避免百万级 episode 带来的小文件灾难

所以：

- episode 边界不再通过文件名表达
- episode 边界通过 metadata 恢复

这也是 LeRobotDataset 和普通“从文件夹读图像”的 dataset 最大的不同之一。

---

## 6. `LeRobotDatasetMetadata`：真正描述数据集结构的对象

元数据核心类在：

- `src/lerobot/datasets/dataset_metadata.py`

如果说 `LeRobotDataset` 是外部门面，那么 `LeRobotDatasetMetadata` 更像是“数据集目录与结构索引中心”。

## 6.1 它管理哪些东西

这个类负责管理：

- `info.json`
- `stats.json`
- `tasks`
- `subtasks`
- `episodes`

这些内容共同定义了：

- 数据集 schema
- 数据模态类型
- 统计量
- 任务集合
- episode 与数据文件/视频文件的映射关系

## 6.2 `info.json` 是最关键的结构文件

`info.json` 里保存了：

- codebase version
- fps
- features
- data_path 模板
- video_path 模板
- total_episodes / total_frames / total_tasks
- chunk/file size 配置

这意味着：

> 真实数据文件只是内容载体，而 `info.json` 才是“如何理解这些内容”的入口。

## 6.3 `stats.json` 的作用

这里保存每个 feature 的：

- mean
- std
- min
- max
- count

后续会被：

- processor
- normalization/unnormalization
- 某些 policy 初始化

直接使用。

所以 `stats` 不只是分析信息，而是训练用的真正输入。

## 6.4 `tasks` 与 `episodes`

### `tasks`

把自然语言 task 映射到 `task_index`。

### `episodes`

保存每个 episode：

- 长度
- 任务
- 在共享 parquet/video 文件中的位置
- dataset 索引范围
- chunk/file 索引

这说明 episode 不再是“一个文件”，而是“metadata 定义的一段连续范围”。

---

## 7. metadata 的读路径：本地优先，其次 Hub

`LeRobotDatasetMetadata.__init__()` 的核心行为是：

1. 先尝试从本地加载 metadata
2. 若缺失或 `force_cache_sync=True`，则从 Hub 下载 `meta/`
3. 然后加载 metadata 到内存

## 7.1 当给了 `root`

如果提供了 `root`：

- Hub 下载会直接 materialize 到该目录

## 7.2 当没给 `root`

如果没提供 `root`：

- 本地默认查 `$HF_LEROBOT_HOME/{repo_id}`
- Hub 下载使用 revision-safe snapshot cache：
  - `$HF_LEROBOT_HOME/hub`

这个细节在 `tests/datasets/test_lerobot_dataset.py` 里有明确契约验证。

它说明 LeRobot 不只是“缓存一下数据”，而是认真考虑了：

- revision 隔离
- 本地目录与 Hub snapshot 的角色区别

---

## 8. `LeRobotDataset.__getitem__` 真正做了什么

`__getitem__` 是这条数据主线里非常关键的一步。

它并不是简单返回 HF dataset 的一行，而是：

1. 检查是否还处于未 finalize 的写模式
2. 确保 reader 存在并已加载
3. 委托给 `reader.get_item(idx)`

而从类注释和外围逻辑可以知道，这个过程会完成：

- 读取底层 HF dataset row
- 展开 delta timestamp 窗口
- 解码视频帧
- 应用图像变换

也就是说：

> 训练循环看到的 batch，已经不是原始存储行，而是被 dataset 层加工过的“模型可消费样本”。

这也是为什么 dataset 在 LeRobot 中地位很高：

- 它已经承担了大量“预采样”和“多模态拼接”的工作

---

## 9. `delta_timestamps`：dataset 如何为时序模型服务

这是第 2 阶段最值得重点理解的机制之一。

## 9.1 它是什么

`delta_timestamps` 是一个按 key 定义的时间偏移字典，例如：

- `observation.images.front = [-0.2, -0.1, 0.0]`
- `action = [0.0, 0.02, 0.04, ...]`

这表示：

- 对 observation，要返回过去若干时刻的堆叠结果
- 对 action，要返回未来若干步动作窗口

## 9.2 它为什么重要

很多 policy 不是消费单帧输入：

- 有的需要多帧视觉历史
- 有的需要动作 chunk supervision
- 有的需要 reward/history 窗口

LeRobot 的设计不是让每个 policy 自己去手写对时逻辑，而是：

- 由 policy 提供 delta indices
- 由 dataset 工厂转换成 delta_timestamps
- 由 dataset 统一产出堆叠样本

这是一种非常干净的职责划分。

## 9.3 测试揭示的契约

`tests/datasets/test_delta_timestamps.py` 说明：

- delta_timestamps 必须和 `fps` 对齐
- 允许在 `tolerance_s` 内有微小误差
- 超出容忍范围会抛错

这说明时间窗口机制不是“粗略近似”，而是有明确数值契约的。

---

## 10. StreamingLeRobotDataset：为什么要有流式读取版本

流式数据集在：

- `src/lerobot/datasets/streaming_dataset.py`

## 10.1 设计目标

`StreamingLeRobotDataset` 是为大规模数据集准备的：

- 不必把整个数据集下载到本地
- 不必一次性全部加载到内存
- 仍然支持 delta timestamp 风格的时序访问

## 10.2 它难的地方在哪里

普通 iterable dataset 最大的问题是：

- 你很难回看过去帧
- 也很难预看未来帧

为了解决这个问题，LeRobot 在 streaming 里引入了 `Backtrackable`：

- 能够 bounded look-back
- 能够 bounded look-ahead

这正是 streaming 下实现 delta windows 的关键技巧。

## 10.3 streaming 与普通 dataset 的关系

`tests/datasets/test_streaming.py` 做了非常重要的验证：

1. streaming 样本和普通 dataset 对应 frame 一致
2. shuffle / shard 行为有明确规则
3. 带 delta_timestamps 时，两者输出也应一致

所以 streaming 不是“另一套近似数据接口”，而是尽量保持和普通 dataset 契约一致的等价读取实现。

---

## 11. 写路径：创建、录制、保存 episode、finalize

LeRobotDataset 不只是读数据，也负责写数据。

## 11.1 `create()`

`LeRobotDataset.create(...)` 会：

1. 调 `LeRobotDatasetMetadata.create(...)`
2. 初始化空数据集结构
3. 创建 `DatasetWriter`
4. 返回一个 write-mode dataset

这一步是录制数据的入口。

## 11.2 `resume()`

`resume()` 用于：

- 加载已有 metadata
- 在现有数据集后追加新的 episode

这说明 LeRobot 把“继续录制”当作正式受支持工作流。

## 11.3 写入 episode 的主流程

写路径大致是：

1. `add_frame(frame)`
2. `save_episode()`
3. metadata 更新 total_episodes / total_frames / stats / tasks
4. 最后 `finalize()`

## 11.4 `finalize()` 为什么重要

文档和代码都非常强调：

- 不调用 `finalize()`，parquet footer 可能不会写完
- buffered metadata 可能不会 flush
- 数据集可能无法正确加载

这是 v3 数据格式下一个很重要的工程约束。

所以要牢记：

> 写完 dataset 不等于数据已经有效，必须 finalize 才算完成。

---

## 12. metadata 写路径：为什么它不是“顺手记几行 JSON”

`LeRobotDatasetMetadata.create(...)` 和 `save_episode(...)` 表现出很明显的工程化设计。

## 12.1 `create(...)`

它会：

- 创建 root 目录
- 校验 feature names
- 自动把 `DEFAULT_FEATURES` 合并进 features
- 生成空的 `info.json`

这说明 metadata 创建本身就是 schema 建立过程，而不是数据录制之后的附带动作。

## 12.2 `_save_episode_metadata(...)`

这一层做了几件很关键的事：

- 给 episode 分配 chunk/file 索引
- 计算 `dataset_from_index` / `dataset_to_index`
- 根据文件大小决定是否切换到新的 chunk/file
- 用 buffer 批量写 metadata parquet

这说明 v3 不是简单把每个 episode 单独落盘，而是在写入时就已经考虑：

- 存储效率
- chunk 切分
- 后续读取路径恢复

## 12.3 `save_episode(...)`

它不仅写 episode metadata，还会更新：

- `total_episodes`
- `total_frames`
- `total_tasks`
- `splits`
- 聚合后的 `stats`

所以 metadata 层在写入路径里也承担了“数据集全局状态维护”的职责。

---

## 13. 从测试里能学到什么

这一阶段里，测试文件非常有教学价值。

## 13.1 `test_lerobot_dataset.py`

它明确表达了：

- `__init__()` 是读模式
- `writer` 在读模式下应为 `None`
- `__getitem__`、`__len__` 的基本契约
- root / snapshot cache 的行为

## 13.2 `test_dataset_metadata.py`

它帮助理解：

- `create()` 之后 metadata 应该长什么样
- `info.json` 应如何落盘
- video/image feature 情况下 property 应怎么表现
- task management 的预期行为

## 13.3 `test_delta_timestamps.py`

它告诉我们：

- 时间偏移并不是“看着差不多就行”
- 必须严格受 `fps` 和 `tolerance_s` 约束

## 13.4 `test_streaming.py`

它最有价值的点是：

- streaming 版本必须与普通 dataset 输出一致

这使我们对 `StreamingLeRobotDataset` 的地位有一个明确判断：

- 它是正式主线实现，不是实验性附属模块

---

## 14. 第 2 阶段最重要的设计认识

这一阶段最值得记住的几个结论如下。

## 认识 1：LeRobotDataset 是“数据平台层”，不是普通数据读取器

它把：

- schema
- 统计量
- 多模态视觉/状态/动作
- 本地与 Hub 加载
- 流式读取
- 数据录制

统一到了一个抽象里。

## 认识 2：metadata 是数据系统的真正骨架

数据文件本身只存内容；
真正定义“这些内容如何被组织和解释”的，是 metadata。

## 认识 3：policy 会反向约束 dataset 的采样方式

通过 `delta_timestamps` 机制，dataset 不再是固定样本返回器，而是会根据 policy 的时序需求调整样本组织方式。

## 认识 4：v3 格式的核心思想是“文件聚合 + metadata 恢复”

这是它相比传统 episode-per-file 数据集设计最重要的工程升级。

## 认识 5：写路径和读路径是同一个 facade 管理的

这让 LeRobotDataset 成为贯穿：

- 录制
- 转换
- 加载
- 训练
- 上传

的统一数据接口。

---

## 15. 本阶段结束后，你应该已经能回答的问题

完成第 2 阶段后，你应该能比较顺畅地回答：

1. `make_dataset(cfg)` 到底做了哪些装配工作？
2. 为什么 dataset 的时间窗口会受到 policy 配置影响？
3. LeRobotDataset v3 为什么采用 `Parquet + MP4 + metadata`？
4. `LeRobotDatasetMetadata` 和 `LeRobotDataset` 的职责边界是什么？
5. 为什么 episode 边界不再由文件名表达？
6. `delta_timestamps` 在训练时有什么作用？
7. 为什么 `finalize()` 是必须调用的？
8. `StreamingLeRobotDataset` 与普通 dataset 的关系是什么？

如果这些问题已经能回答清楚，第 2 阶段就算达标。

---

## 16. 下一阶段建议

第 2 阶段之后，最自然的下一步就是进入 processor 体系。

推荐顺序：

1. `docs/source/introduction_processors.mdx`
2. `src/lerobot/processor/pipeline.py`
3. `src/lerobot/processor/converters.py`
4. `tests/processor/test_pipeline.py`
5. `tests/processor/test_policy_robot_bridge.py`

因为到这里为止，我们已经知道 dataset 能产出什么样的样本，下一步就应该搞清楚：

> 这些样本是如何进一步变成 policy 真正的输入输出格式的？

---

## 17. 本阶段个人学习结论

如果用一句话总结第 2 阶段：

> LeRobotDataset 是 LeRobot 平台中真正承上启下的数据层，它用 metadata 驱动的方式把机器人多模态时序数据、Hub 同步、流式读取和训练窗口采样统一成了一个可以被训练主链直接消费的抽象。

理解这一层之后，再去看 processor 和 policy，就更容易明白为什么 LeRobot 能把那么多不同模型和机器人接在同一套系统上。
