# LeRobot 中文学习教程

面向「想把主链跑通、把关键代码读懂」的阶段课。
每课统一采用：**可运行示例 → 图示 → 代码解析 → 设计思路 → 运行时调优**。

---

## 先选入口

| 你的目标 | 建议入口 |
|----------|----------|
| 第一次读 LeRobot 源码 | 本页 → 总览 → 第 0–3 阶段 |
| 想训练经典模仿学习策略 | 第 0–3 阶段 → ACT → Diffusion |
| 想理解或微调 VLA | 第 0–3 阶段 → 第 7 阶段 → [模型百科 / VLA](./models/vla/README.md) |
| 想做仿真评测 | 第 0–3 阶段 → 第 6 阶段 |
| 想做实机采集或在线 RL | 先读 `AGENT_GUIDE.md` 的安全与硬件流程，再学第 8 阶段 |
| 只想查某个模型 | 直接进入 [模型百科](./models/README.md) |

阶段课讲平台主链，模型百科讲单个模型；二者互相链接，但不重复维护大段内容。

---

## 目录结构

```text
tutorial/
├── README.md                 # 总入口
├── stages/                  # 0–8 阶段课与学习地图
├── notebooks/               # 可交互练习
├── models/                  # 模型百科
└── _templates/              # 教程维护模板，不是学习必读
```

---

## 怎么学（固定节奏）

1. **先跑示例**（或走文档里的降级路径）——建立体感
2. **看图**——记住数据/控制流形状
3. **对着标注代码读**——搞清输入输出与调用点
4. **读设计思路**——知道为什么不「全写在脚本里」
5. **看调优表**——下次 OOM / 抖动 / 成功率掉时有抓手
6. **做练习 + 自测**——确认不是「看过了」而是「讲得出」

---

## 课程目录

| 阶段 | 文档 | 本课锚点示例 |
|------|------|----------------|
| 总览 | [阶段学习地图](./stages/README.md) | 地图与验收标准 |
| 0 | [预热与安装](./stages/00-项目预热.md) | `uv sync` / `pytest` / `lerobot-info` |
| 1 | [训练主链](./stages/01-训练主链与配置.md) | `lerobot-train` + `update_policy` |
| 2 | [数据集](./stages/02-数据集.md) | [数据学习.ipynb](./notebooks/数据学习.ipynb) + `resolve_delta_timestamps` |
| 3 | [Processor](./stages/03-Processor.md) | `make_pre_post_processors` |
| 4 | [ACT](./stages/04-ACT.md) | `examples/tutorial/act/` |
| 5 | [Diffusion](./stages/05-Diffusion.md) | `examples/tutorial/diffusion/` |
| 6 | [Env/Eval](./stages/06-Env与Eval.md) | `lerobot-eval` + `rollout` |
| 7 | [VLA](./stages/07-SmolVLA与VLA.md) | `examples/tutorial/smolvla/` |
| 8 | [RL/硬件](./stages/08-RL与真实机器人.md) | teleop/record + actor/learner |

## 模型百科（按 README SoTA 表展开）

每个模型的论文精读、代码图示、关键实现与社区入口：

→ **[`models/README.md`](./models/README.md)**

相关但不在阶段课里维护的内容：

- 官方文档：`docs/source/`
- 可运行脚本：`examples/tutorial/`（本教程引用解析，默认不改脚本）
- 用户速查：`AGENT_GUIDE.md`
- 教程写作模板：[`_templates/model-page.md`](./_templates/model-page.md)

---

## 版本与校验原则

本教程以当前工作区源码为准，不把行号或某个历史版本写成永久事实。代码更新后优先核对：

1. `pyproject.toml`：CLI 与 optional extras
2. `src/lerobot/configs/`：CLI 字段和注册名
3. `src/lerobot/scripts/`：train / eval / record 实际调用链
4. `src/lerobot/policies/`、`src/lerobot/rewards/`、`src/lerobot/rl/`：模型与算法实现
5. `tests/`：输入输出契约

当前源码已将旧的内部 `types` 模块改名为
`src/lerobot/lerobot_types.py`。教程代码仍优先使用稳定公开入口，例如
`from lerobot.configs import FeatureType`，不要依赖内部文件名。

---

## 环境与最小验证

```bash
uv sync --locked --extra test --extra dev
uv run pytest tests/processor/test_pipeline.py -q
uv run lerobot-info
```

按课程再加：`--extra training` / `feetech` / `smolvla` / `pusht` 等。

如果 `uv sync --locked` 提示 lockfile 与本地环境不同，先完成同步再运行教程；
不要用系统 Python 绕过项目依赖。
