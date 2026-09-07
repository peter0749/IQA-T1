# 論文 IQA-T1：KonIQ 官方 test PLCC / SRCC

2026-09-08。官方 test **2015/2015** 有分數。這是域內 KonIQ，**不是**表 1 的七庫平均。

## 協定

- 權重：`scratch/IQA-T1-checkpoint`，官方 `inference/infer.py`，greedy，`max_tool_calls=6`
- 切分：`koniq10k_distributions_sets.csv` 的 `set=test`
- MOS：CSV `MOS`（約 1–100）。線性 PLCC / SRCC 對 MOS 仿射縮放不變
- 工具圖：既有 `repo/dataset/KONIQ/tools`，未刪除、未重算
- 論文表 1 KonIQ：PLCC **0.942** / SRCC **0.925**

## 結果

| | 本機 | 論文表 1 KonIQ |
| --- | ---: | ---: |
| n / coverage | 2015 / 1.00 | — |
| **SRCC** | **0.927** | 0.925 |
| **PLCC（線性）** | **0.942** | 0.942 |
| PLCC（4 參數 logistic，Adam 重擬合） | 0.937 | 未註明是否擬合 |

第一次 LBFGS logistic 發散（NaN）。改用 MOS 尺度初始化後 logistic PLCC 略低於線性，表示分數與 MOS 已接近線性。與表 1 對齊時以線性 PLCC 與 SRCC 為準。

第一工具：1681/2015（83%）為 `GradientMagnitudeHistogram`；其餘為 Colorfulness、Luminance、色偏、梯度圖等。2–3 個工具為主。這與 D16 盤點一致：常有預設開場，但不是 L5 那種整段鎖死。

## 限制

- 只驗證公開權重在官方 KonIQ test 上的分數相關，不驗證另外六庫。
- 未重跑論文的多 seed。
- MAE（pred vs MOS/20）0.59 只是尺度對齊後的誤差描述，論文未報。

產物：`scratch/e1/koniq_test_plcc/run1/`（traces、analysis.json、manifest）。約 6.7 小時。
