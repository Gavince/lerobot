# LeRobot 根目录 `tutorial/` 教学内容刷新设计

**日期**：2026-07-27  
**状态**：已批准并完成实现（2026-07-27）  
**范围**：仅根目录 `tutorial/`（不含 `examples/tutorial/`、不含官方 `docs/source/`）

---

## 1. 背景与目标

根目录 `tutorial/` 已有一套中文阶段教程（总览 + 第 0–8 阶段 + `数据学习.ipynb`），整体骨架可用，但存在：

- 第 5 阶段两份重叠文档
- 部分文档使用本机绝对路径（`/Users/zwy/Macin/lerobot/...`）
- 内容偏“源码阅读笔记”，与最新 public API / CLI / 可跑练习衔接不足
- 文档风格不统一（有的用“一、二、三”，有的用“1. 2. 3.”）

**目标**：在保持 0–8 阶段骨架不变的前提下，把教学内容校正到最新代码，并加深为更适合系统学习的混合型教程。

**成功标准**：

1. 新人按 README → 总览 → 第 0…8 阶段顺序学习时，路径、命令、API 与当前仓库一致
2. 每阶段具备可检验产出、原理图、动手练习、自测题
3. 第 5 阶段只保留一份正本；绝对路径清零
4. notebook 可按公开 API 打开跑通最小流程（不要求覆盖全部数据集能力）

---

## 2. 约束（已确认）

| 项 | 选择 |
|----|------|
| 范围 | 只整理根目录 `tutorial/` |
| 形态 | 混合型：目标 → 原理图 → 关键路径 → 动手练习 → 自测 |
| 阶段结构 | 保持 0–8 骨架，校正 / 去重 / 加深 |
| 实施方案 | 方案 2：统一模板 + 按阶段加深 |
| 不改动 | `examples/tutorial/`、`docs/source/`、训练/策略源码（除非发现教程引用错误需在教程侧纠正） |

---

## 3. 文件清单与动作

| 文件 | 动作 |
|------|------|
| `tutorial/README.md` | **新增**：目录、怎么学、与官方 docs / `examples/tutorial` 的关系 |
| `tutorial/LeRobot_代码库学习流程报告.md` | **更新**：总导航、阶段入口、预计产出；对齐最新代码表述 |
| `tutorial/LeRobot_第0阶段_项目预热与安装认知.md` | **按模板加深并校对** |
| `tutorial/LeRobot_第1阶段_训练主链与配置系统.md` | **按模板加深并校对** |
| `tutorial/LeRobot_第2阶段_数据集系统与LeRobotDataset.md` | **按模板加深并校对**；挂 notebook |
| `tutorial/LeRobot_第3阶段_processor体系与数据转换中间层.md` | **按模板加深并校对** |
| `tutorial/LeRobot_第4阶段_ACT策略贯通学习.md` | **按模板加深并校对**；引用（不改）`examples/tutorial/act` |
| `tutorial/LeRobot_第5阶段_DiffusionPolicy对照学习.md` | **正本**：合并另一份精华后加深 |
| `tutorial/LeRobot_第5阶段_Diffusion策略与ACT对照学习.md` | **删除**（内容并入正本） |
| `tutorial/LeRobot_第6阶段_env与eval评测主线学习.md` | **按模板加深并校对**；绝对路径改相对路径 |
| `tutorial/LeRobot_第7阶段_SmolVLA与VLA路线入门.md` | **按模板加深并校对**；绝对路径改相对路径 |
| `tutorial/LeRobot_第8阶段_RL与真实机器人闭环学习.md` | **按模板加深并校对**；绝对路径改相对路径 |
| `tutorial/数据学习.ipynb` | **轻量修正**：公开 API import、去掉重复 cell；不新增大阶段 |

---

## 4. 统一文档模板

每个阶段文档按此顺序组织：

1. **本阶段目标**（3–5 条可检验产出）
2. **你会用到的材料**（相对路径：源码 / 测试 / 官方 docs / 可选 `examples/tutorial`）
3. **主线原理图**（1–2 张 mermaid；语法自检；暗色主题可读：避免依赖浅色背景的样式）
4. **先抓结论**（白话说明这一层在系统中的职责）
5. **关键代码路径**（按阅读顺序走调用链，标文件与函数名；少贴大段源码，多解释“为什么这样接”）
6. **核心概念拆解**（每个概念：是什么 → 为什么存在 → 落在哪）
7. **动手练习**（≥2：轻量阅读/画图 + 可运行命令或 notebook）
8. **自测问题**（5–8 题；附简短参考要点）
9. **常见坑**
10. **下一阶段预告**

### 全局一致性规则

- 路径一律相对仓库根：`src/lerobot/...`、`docs/source/...`、`examples/tutorial/...`
- 命令优先：`uv run ...` 与 `lerobot-*` CLI
- Public API 对齐当前导出，例如：
  - `from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata`
  - `from lerobot.configs import FeatureType`
  - `from lerobot.policies import make_pre_post_processors`
- 每阶段给出“最小验证命令”；硬件相关练习标明“无硬件可跳过”
- 引用 `examples/tutorial` 仅作对照，不在本次修改那些脚本

---

## 5. 各阶段加深要点

学习主线不变：

`0 预热 → 1 训练/配置 → 2 数据集 → 3 Processor → 4 ACT → 5 Diffusion → 6 Env/Eval → 7 VLA → 8 RL/硬件`

| 阶段 | 加深重点 |
|------|----------|
| 0 | 平台 vs 模型库；extras；安装；仓库地图；最小 pytest |
| 1 | `TrainPipelineConfig` → factory → train loop；draccus / `ChoiceRegistry`；checkpoint / resume / hub |
| 2 | Dataset v3；`delta_timestamps`；metadata / stats；挂修过的 notebook |
| 3 | `EnvTransition` / step / pipeline；pre/post；normalize 与 stats；解耦原因 |
| 4 | chunking、action queue、`forward` vs `select_action`；config↔dataset 窗口；对照 ACT 示例 |
| 5 | 合并双份文档；`n_obs_steps` / `horizon` / `n_action_steps`；denoising vs ACT 对照表 |
| 6 | `lerobot-eval`、`make_env`、rollout；obs→policy→action |
| 7 | SmolVLA 为主，点出 π0 等；语言条件；processor 差异；预训练加载 |
| 8 | actor/learner；record/teleop；robot 抽象；与 IL 路径差异 |

### 第 5 阶段合并策略

1. 以 `LeRobot_第5阶段_DiffusionPolicy对照学习.md` 为正本
2. 从 `LeRobot_第5阶段_Diffusion策略与ACT对照学习.md` 吸收：一句话定义、论文要点清单、ACT 对照表中更清晰的表述
3. 统一进模板后删除后者

### Notebook 策略

- 修重复 cell 与过时 import
- 在第 2 阶段增加“打开并跑通 notebook”练习入口
- 不拆成独立阶段

---

## 6. 执行顺序（实现时）

1. 新增 `tutorial/README.md` + 更新总览报告（导航先对齐）
2. 第 0 → 1 → 2（含 notebook）→ 3
3. 第 4 → 合并并刷新第 5 → 删除重复文件
4. 第 6 → 7 → 8
5. 全量自检：绝对路径、死链路径、相对时间措辞、模板章节齐全

每完成一批阶段，应用当前代码核对引用路径是否存在。

---

## 7. 验收清单

- [ ] `tutorial/` 内无 `/Users/zwy/...` 绝对路径
- [ ] 第 5 阶段仅 1 份文档
- [ ] 每个阶段 md 具备模板 10 节（总览与 README 除外，但需有清晰导航）
- [ ] 文档中引用的关键源码路径抽样存在
- [ ] 每阶段至少 2 个动手练习 + 自测题
- [ ] 至少一张主线 mermaid 图（阶段文档）
- [ ] notebook import 使用公开 API，无明显重复前置 cell
- [ ] README 说明学习顺序与边界（本目录 vs 官方 docs vs examples）

---

## 8. 非目标

- 不翻译/重写官方 Hugging Face docs
- 不保证教程内所有命令在无 GPU/无硬件机器上全绿（需标注前置条件）
- 不新增第 9 阶段或拆分 RL/VLA 为新册
- 不在本次改动中提交 git commit（除非用户明确要求）

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 源码边读边变导致表述漂移 | 以仓库当前文件为准；引用标到目录/函数级，少依赖行号 |
| 内容过长难读 | 控制每阶段核心概念数量；大段 API 表放到“材料清单”而非正文堆砌 |
| 合并第 5 阶段丢信息 | 合并前列出两份文档的独有要点再并入 |

---

## 10. 后续步骤

1. 用户审阅并批准本 spec  
2. 使用 writing-plans 产出实现计划  
3. 按执行顺序改 `tutorial/`  
4. neat-freak 自检后给出变更摘要  
