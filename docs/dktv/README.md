# DKTV 文件导航

DKTV 已并入仓库现有的五目录工作流，不再维护独立的
`src/koopman_control/dktv/`、`experiments/dktv/` 和 `configs/dktv/`
并行体系。

## 当前边界

```text
traj_data/dktv_data.py
  时变扰动数据生成、质量检查与训练流切分

prediction/
  dktv_foundation_prediction.py     Plan 01 基础模型入口
  dktv_accumulative_prediction.py   Hao 累积式更新入口
  dktv_accumulative_aggregate.py    Hao 多随机种子汇总入口
  dktv_window_prediction.py         Zhang 滑动窗口比较入口
  dktv_window_aggregate.py          Zhang 多随机种子汇总入口
  dktv_base_config.json             Plan 01 稳定配置
  dktv_accumulative_config.json     Hao 方法稳定配置
  dktv_window_config.json           Zhang 方法稳定配置
  dktv/                              三个入口共享的内部算法包

tests/dktv/
  DKTV 单元测试与小型端到端测试

docs/dktv/
  plans/                             计划与总体路线
  formula_mapping/                   论文公式到代码的映射
  reviews/                           执行报告、审查与整改反馈
```

`prediction/dktv/` 是预测阶段的内部算法包，不是新的顶层工作流。
其中累积式更新、滑动窗口、选择性接受、最小二乘和统一回放评估均由
上面的独立入口调用。

## 独立运行入口

所有命令从仓库根目录运行，并使用项目指定的 Python：

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m prediction.dktv_foundation_prediction --help

env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m prediction.dktv_accumulative_prediction --help

env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m prediction.dktv_window_prediction --help
```

Plan 02 和 Plan 03 都通过 `--plan01-run` 复用 Plan 01 冻结的模型，不会
各自重新训练编码器。结果继续保存在根目录 `outputs/` 中，并遵守仓库统一的
artifact 契约。

## 与控制目录的关系

当前 DKTV 实现只完成预测模型的在线更新和比较，因此全部位于
`prediction/`。现有 `control/lqr_control.py` 与 `control/mpc_control.py`
保持不动。后续真正实现 Zhang 文献中的稳定性保证控制时，再新增
`control/dktv_mpc_control.py`，并通过模型 artifact 接入在线更新结果。

## 历史文档说明

`reviews/` 中的报告记录了整理前的开发过程，正文可能提到当时使用过的
Plan 编号或旧目录。这些内容作为审计历史保留；当前有效路径以本文件和
`FIVE_FOLDER_RUN_GUIDE.md` 为准。
