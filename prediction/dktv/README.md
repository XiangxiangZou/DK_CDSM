# DKTV 内部算法包

本目录只存放 DKTV 预测入口共享的算法实现，不直接保存实验结果：

- `foundation.py`：冻结并评估公共 fixed-DKO artifact；
- `least_squares.py`：统一 ridge 最小二乘原语；
- `accumulative_update.py`：Hao 风格累积式在线更新；
- `window_update.py`：Zhang 风格滑动窗口在线更新；
- `selective_update.py`：候选窗口模型的选择性接受；
- `online_model.py`：因果回放、统一预测评价和结果汇总；
- `config.py`：公共配置校验。

可运行入口位于上一级 `prediction/dktv_*_prediction.py`，不要从本目录
新增另一套实验入口。
