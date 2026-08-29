# Hao-DKTV 审查整改执行反馈 02

> 执行日期：2026-08-28  
> 对照文档：`docs/plan_review_01.md`  
> 当前结论：P0 科研比较串扰已修复；P1 的模块入口、坐标一致性、best-state 和 trial 隔离已完成；阶段 2 与阶段 5 的自动化证据得到扩充。正式数据安全验收、完整 checkpoint 身份绑定和阶段 6 正式实验仍未完成。

## 1. 本轮范围

本轮只整改审查文档指出的 DKTV 算法、时变数据入口、测试和评估产物问题。没有修改 OTVDKL、控制方法、已有历史产物，也没有把 smoke 指标解释为论文性能结论。

## 2. 已完成整改

### 2.1 fixed DKUC 基线完全冻结

- 回放开始时创建独立的 fixed network 副本，冻结全部参数。
- fixed one-step 始终使用该副本 lifting，并继续使用 artifact 原始 `A/B`。
- 在线 DKTV encoder 的任何训练不再影响 fixed DKUC。
- 新增高学习率、多批次测试，逐点对照原始网络重算结果。

新 smoke 的 fixed one-step RMSE 为 `0.12030496805744631`，与审查时重新加载原 artifact 得到的 `0.1203049680574463` 一致。

### 2.2 full 模式的 `A/B/C/theta` 坐标一致性

每批执行顺序现为：

1. 用批次到达前的状态预测；
2. 观察完整批次并执行矩阵更新；
3. 固定矩阵优化 encoder，选择可复现最低 loss 的 `theta*`；
4. 用 `theta*` 对初始历史和当前 trial 已消费的全部物理转移重新 lifting；
5. 重新建立累计 ridge 充分统计量和 `A/B/C`，原子提交同一坐标下的状态。

该实现优先保证论文对象的一致性，代价是 full 模式每批需要重 lifting 全部已消费历史。更新记录新增 `coordinate_consistent=true` 和 `coordinate_refit_sample_count`。

### 2.3 在线训练 best-state 修复

- 每次 `optimizer.step()` 后重新计算 L1/L2/L，再判断是否保存权重。
- 返回 `best_loss`，测试验证最终加载权重可复现该值。
- `train_online_encoder` 和 `full` replay 均拒绝 `online_epochs <= 0`。
- 训练失败仍恢复 encoder；replay 同时回滚该批矩阵状态。

### 2.4 时变采集模块入口

采集器加入包内相对导入与直接文件执行回退。以下命令已成功：

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m traj_data.collect_data_time_varying --help
```

### 2.5 递推诊断和测试

- 诊断新增累计 ridge 正则系统的 rank 和 condition number。
- 非有限输入测试现检查全部矩阵、统计量、逆矩阵、计数和版本均不发生部分修改。
- 新增不同 batch 划分最终状态等价测试。
- 新增保存/加载后继续更新与不中断运行等价测试。
- 新增多 trial 独立性测试。
- 新增 full 在线训练时 fixed 基线不受污染测试。

### 2.6 DKTV rollout 产物

新增严格快照式 rollout：窗口起点保存当时的 lift、`A/B/C`，窗口内固定该快照递推，不消费未来真实状态或未来模型更新。现保存：

- one-step；
- batch-size horizon rollout；
- CLI 指定的多个 fixed-horizon rollout；
- 每项对应 truth/prediction 原始数组和 JSON 指标。

## 3. 验证结果

指定解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
Linux
```

检查结果：

```text
py_compile：通过
模块 --help：通过
全仓库 pytest：9 passed in 1.89s
```

重新生成的 smoke：

```text
/tmp/dktv_review02b/dktv/20260828_113944_dktv_review02/
```

关键指标（1 trial、8 steps，仅用于程序验收）：

| 指标 | RMSE |
|---|---:|
| fixed DKUC one-step | 0.1203049681 |
| full DKTV one-step | 0.0503477608 |
| full DKTV horizon-2 snapshot rollout | 0.0469070396 |
| full DKTV horizon-4 snapshot rollout | 0.0460337869 |
| full DKTV batch/horizon-4 rollout | 0.0503456146 |

两次矩阵/encoder 批次均记录 `coordinate_consistent=true`。累计正则系统 condition number 约为 `1e7`，说明 ridge 使计算可行，但该数据仍具有明显病态性，正式报告必须保留这一诊断。

## 4. 尚未完成及原因

以下审查项没有在本轮虚报完成：

- 时变数据的执行器力矩和缆索张力安全阈值尚未从 XML/执行器物理约束中正式确定；无扰动/有扰动成对数据及确定性可视化验收尚未生成。
- CLI 已强制 `resume_state` 与 `resume_model` 成对提供，但尚未把 state、model、normalizer fingerprint、配置和数据位置封装为单一 checkpoint manifest；错误配对的 fingerprint 拒绝仍待实现。
- 多 trial 目前评估数组和运行过程相互隔离，但入口仍只写一个最终 state/model（最后一条 trial），尚未改为逐 trial checkpoint。
- 尚缺专门篡改未来观测的 causality 测试；当前因果性由执行顺序和快照 rollout 契约保证，但仍应补测试证据。
- 正式三模式、多 trial 均值/标准差、实时性/内存测量和阶段 6 参数扫描未执行。

## 5. 文件变化

- `prediction/dktv_prediction.py`：冻结基线、坐标重建、best-state、累计诊断、快照 rollout 和恢复参数配对校验。
- `traj_data/collect_data_time_varying.py`：兼容模块与文件路径入口。
- `tests/time_varying/test_dktv_algorithm.py`：新增递推、恢复、回滚、冻结基线和 trial 隔离测试。
- `docs/plan_feedback_02.md`：本报告。

## 6. 建议下一步

下一轮应优先完成统一 checkpoint manifest 与逐 trial checkpoint，然后从 MuJoCo actuator/cable 配置推导并固化安全阈值，生成同 seed 的无扰动/有扰动成对数据。完成 causality 和数据确定性集成测试后，再进入 full、frozen_encoder、fixed DKUC 的多 seed 正式实验。
