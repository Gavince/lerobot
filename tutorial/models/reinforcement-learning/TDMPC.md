# TDMPC


## 1. 概览

**TD-MPC** 将时间差分学习与模型预测控制结合：学潜空间动力学与价值，用 MPC 在线规划动作。LeRobot 提供 `tdmpc` policy 包以便在统一接口下训练/评测。

| 项 | 内容 |
|----|------|
| 类别 | Reinforcement Learning |
| 论文 | [Temporal Difference Learning for Model Predictive Control (TD-MPC)](https://arxiv.org/abs/2203.04955) (2022 (ICML)) |
| 作者（常见引用） | Nicklas Hansen, Xiaolong Wang, Hao Su |
| 项目页 | https://www.nicklashansen.com/td-mpc/ |
| 本仓库代码 | `src/lerobot/policies/tdmpc/` |
| 官方文档 | `docs/source/policy_tdmpc_README.md` |

---

## 2. 论文精读

### 2.1 要解决的问题

纯 model-free 样本效率有限；纯 MPC 需要准模型。需要在潜空间里把 TD 与规划结合起来。

### 2.2 方法核心

编码器映射观测到潜状态；学转移与奖励/价值；规划时在潜空间做轨迹优化（细节见论文与 `modeling_tdmpc.py`）。

### 2.3 实验与结论（论文报告）

论文在 DeepMind Control 等基准上展示相对强 model-free/model-based 基线的样本效率与性能（见表）。

### 2.4 你可以带走的设计要点

- 规划发生在潜空间而非像素；\n- 训练目标含一致性/价值多项；\n- 与 IL chunk 策略的数据假设不同（常需在线交互）。

---

## 3. LeRobot 中的实现地图

### 3.1 关键文件

- `configuration_tdmpc.py` — 规划视界、潜空间维等
- `modeling_tdmpc.py` — 模型与规划
- `processor_tdmpc.py` — 观测处理

### 3.2 数据 / 控制流

```mermaid
flowchart LR
  O[obs] --> E[latent encode]
  E --> P[latent MPC planning]
  P --> A[action]
  A --> ENV[env]
  ENV --> TD[TD / model update]
```

### 3.3 与平台其它层的关系

- 配置注册：`PreTrainedConfig.register_subclass` / reward 配置（以具体模块为准）
- 训练入口：IL/VLA 多用 `lerobot-train`；HIL-SERL 走 actor/learner
- 文档交叉：`docs/source/` 英文页 + 本中文详解

---

## 4. 示例：如何跑起来

```bash
# policy.type 以 register_subclass 为准
uv run lerobot-train --policy.type=tdmpc --env.type=<supported_env> ...
```
先读：`docs/source/policy_tdmpc_README.md` 与项目页。


---

## 5. 关键代码详解

阅读顺序：`configuration_tdmpc.py` → `modeling_tdmpc.py` 中 `forward` / 规划函数 → `select_action`。
关注：潜状态维度、规划 horizon、是否用 CEEMC/MPPI 类采样（以源码命名为准）。


---

## 6. 社区优秀讨论

> 只列可公开核对的入口；楼层内容会变化。

- 项目页：https://www.nicklashansen.com/td-mpc/
- 后续 TD-MPC2 等工作常被社区拿来对比（阅读时注意版本差异）
- LeRobot Issues：搜 `tdmpc`

- Discord：https://discord.gg/q8Dzzpym3f
- Issues：https://github.com/huggingface/lerobot/issues

---

## 7. 调优与常见坑

| 场景 | 可能原因 | 建议 |
|------|----------|------|
| 规划太慢 | horizon/采样数过大 | 减规划步数 |
| 不稳定 | 潜空间坍塌 | 调损失权重与学习率 |

---

## 8. 延伸阅读

- https://www.nicklashansen.com/td-mpc/

- 返回：[`../README.md`](../README.md)
