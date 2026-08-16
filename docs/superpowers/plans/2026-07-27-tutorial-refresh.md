# Tutorial Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh root `tutorial/` Chinese learning materials to match current LeRobot code, using a unified hybrid template.

**Architecture:** Keep stages 0–8; add `tutorial/README.md`; rewrite each stage doc to a fixed 10-section template; merge duplicate stage-5 docs; lightly fix `数据学习.ipynb`. Verify with path existence checks and absolute-path greps (no pytest for prose).

**Tech Stack:** Markdown + mermaid; citations to `src/lerobot/**`, `docs/source/**`, `examples/tutorial/**`; commands via `uv run` / `lerobot-*`.

## Global Constraints

- Scope: only `tutorial/` (plus this plan/spec under `docs/superpowers/`)
- Do not modify `examples/tutorial/` or `docs/source/`
- Relative paths only; no `/Users/zwy/...`
- Prefer public imports: `from lerobot.datasets import ...`, `from lerobot.configs import FeatureType`
- No git commit unless user explicitly asks
- Mermaid must render on dark theme (use default node styles; avoid white fills)
- Each stage doc must include the 10 template sections from the spec

**Spec:** `docs/superpowers/specs/2026-07-27-tutorial-refresh-design.md`

---

### Task 1: Navigation — README + overview

**Files:**
- Create: `tutorial/README.md`
- Modify: `tutorial/LeRobot_代码库学习流程报告.md`

- [ ] **Step 1: Write `tutorial/README.md`**

Include: what this folder is, learning order table linking all stage files, relationship to official docs and `examples/tutorial/`, how to verify environment (`uv sync`, one pytest), note that stage 5 has a single canonical file.

- [ ] **Step 2: Rewrite overview report**

Keep learning goals; update critical file list to current paths (`lerobot_train.py` uses `make_train_eval_datasets`, `common/train_utils.py`, etc.); point each stage to its md file; remove references to the deleted stage-5 duplicate name if present; add mermaid of the platform mainline.

- [ ] **Step 3: Verify links**

Run: `rg -n '/Users/zwy' tutorial/README.md tutorial/LeRobot_代码库学习流程报告.md`  
Expected: no matches. Confirm every linked `tutorial/*.md` exists.

---

### Task 2: Stages 0–1

**Files:**
- Modify: `tutorial/LeRobot_第0阶段_项目预热与安装认知.md`
- Modify: `tutorial/LeRobot_第1阶段_训练主链与配置系统.md`

- [ ] **Step 1: Rewrite stage 0** with full template. Cite `README.md`, `docs/source/installation.mdx`, `pyproject.toml`. Practice: `uv sync --locked --extra test --extra dev` and `uv run pytest tests/processor/test_pipeline.py -q`.

- [ ] **Step 2: Rewrite stage 1** with full template. Cite `src/lerobot/scripts/lerobot_train.py` (`train`, `update_policy`), `src/lerobot/configs/train.py` (`TrainPipelineConfig`), `src/lerobot/configs/policies.py`, `src/lerobot/optim/factory.py`, `src/lerobot/common/train_utils.py`. Mermaid: CLI → config → factories → loop.

- [ ] **Step 3: Path check**

Run: `test -f src/lerobot/scripts/lerobot_train.py && test -f src/lerobot/configs/train.py`

---

### Task 3: Stage 2 + notebook

**Files:**
- Modify: `tutorial/LeRobot_第2阶段_数据集系统与LeRobotDataset.md`
- Modify: `tutorial/数据学习.ipynb`

- [ ] **Step 1: Fix notebook imports** to `from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata` (and any other public exports used). Remove duplicated early cells that repeat the same setup.

- [ ] **Step 2: Rewrite stage 2** with template; link notebook; cover v3 layout, `delta_timestamps`, metadata/stats, `StreamingLeRobotDataset`. Cite `docs/source/lerobot-dataset-v3.mdx`, `dataset_metadata.py`, `lerobot_dataset.py`, `factory.py`.

- [ ] **Step 3: Verify** notebook JSON parses and has no `from lerobot.datasets.dataset_metadata import`.

---

### Task 4: Stage 3

**Files:**
- Modify: `tutorial/LeRobot_第3阶段_processor体系与数据转换中间层.md`

- [ ] **Step 1: Rewrite** covering `ProcessorStep`, `DataProcessorPipeline`, converters, normalize, `make_pre_post_processors`. Cite `docs/source/introduction_processors.mdx`, `processor/pipeline.py`, `processor/converters.py`, `policies/factory.py`.

- [ ] **Step 2: Mermaid** showing robot/dataset → preprocessor → policy → postprocessor → action.

---

### Task 5: Stages 4–5 (merge)

**Files:**
- Modify: `tutorial/LeRobot_第4阶段_ACT策略贯通学习.md`
- Modify: `tutorial/LeRobot_第5阶段_DiffusionPolicy对照学习.md`
- Delete: `tutorial/LeRobot_第5阶段_Diffusion策略与ACT对照学习.md`

- [ ] **Step 1: Rewrite stage 4** — ACT chunking, queue, forward vs select_action; reference `examples/tutorial/act/*.py` without editing them.

- [ ] **Step 2: Merge stage 5** — use DiffusionPolicy对照学习 as canonical; absorb one-liner definition and ACT comparison strengths from the duplicate; delete duplicate file.

- [ ] **Step 3: Confirm** only one stage-5 md remains: `ls tutorial/*第5阶段*`

---

### Task 6: Stages 6–8

**Files:**
- Modify: `tutorial/LeRobot_第6阶段_env与eval评测主线学习.md`
- Modify: `tutorial/LeRobot_第7阶段_SmolVLA与VLA路线入门.md`
- Modify: `tutorial/LeRobot_第8阶段_RL与真实机器人闭环学习.md`

- [ ] **Step 1: Rewrite stage 6** — `lerobot_eval.py`, `EvalPipelineConfig`/`EvalConfig`, `make_env`, rollout. Strip absolute paths.

- [ ] **Step 2: Rewrite stage 7** — SmolVLA primary; briefly mention π0 / `examples/tutorial/pi0`. Strip absolute paths.

- [ ] **Step 3: Rewrite stage 8** — `rl/learner.py`, `rl/actor.py`, record/teleop, robot abstraction, HIL-SERL pointers (`docs/source/hilserl.mdx`). Strip absolute paths.

---

### Task 7: Final verification (neat-freak checklist)

**Files:** all under `tutorial/`

- [ ] **Step 1: Absolute path scan**

```bash
rg -n '/Users/zwy' tutorial/ || true
```

Expected: no matches.

- [ ] **Step 2: Duplicate stage-5 scan**

```bash
ls tutorial/*第5阶段*
```

Expected: only `LeRobot_第5阶段_DiffusionPolicy对照学习.md`.

- [ ] **Step 3: Template section spot-check**

Each stage md contains headings equivalent to: 本阶段目标, 材料, 原理图/mermaid, 结论, 代码路径, 核心概念, 动手练习, 自测, 常见坑, 下一阶段.

- [ ] **Step 4: Write sync summary** for the user (files changed / deleted / not touched).

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| README + overview | Task 1 |
| Stages 0–1 | Task 2 |
| Stage 2 + notebook | Task 3 |
| Stage 3 | Task 4 |
| Stages 4–5 merge | Task 5 |
| Stages 6–8 | Task 6 |
| Acceptance greps | Task 7 |
| No examples/docs/source edits | Global Constraints |
| No commit unless asked | Global Constraints |
