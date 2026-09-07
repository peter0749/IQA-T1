# 論文釋出 IQA-T1：D16 第一工具／套式盤點

2026-09-07。**不是 PLCC、不是 H32、不是 E1 品質檢定。** 只問：官方權重在已曝光的 D16 上，是不是也像本機 L5 那樣走固定劇本。

## 設定

- 權重：`scratch/IQA-T1-checkpoint`（公開 `zibuyu-02/IQA-T1`，full SFT+GRPO，不是本機 LoRA）
- 官方 `inference/infer.py`：system prompt + `What is your overall rating on the quality of this picture?`，`max_tool_calls=6`，greedy
- 圖：凍結 D16 原圖；工具圖由官方腳本現算，不是 E1 的 cached KONIQ PNG
- 未載入 H32，未訓練

## 結果（N=16）

| | 論文 IQA-T1 | 本機 L5（同 16 張，E1 harness） |
| --- | --- | --- |
| 無呼叫 | 0/16 | 0/16 |
| 第一工具 | 14×梯度直方圖、1×亮度直方圖、1×梯度圖 | 16/16 梯度直方圖 |
| 完整序列種類 | 8 種 | 幾乎只有 Histogram→Colorfulness→Noise |
| 與 L5 整段序列相同 | 0/16 | — |
| 分數 | 16 個不同值，無 3.07 塌縮 | 常打 3.07 |
| 對 gt MAE（描述） | 0.159 | 0.372 |

論文權重**常常**以 `GradientMagnitudeHistogram` 開場（14/16），所以「有預設第一工具」這件事，官方模型在這批圖上也有。但它不是 L5 那種整段三件套：第二、第三個工具會換成亮度／極端亮度／色偏／色彩，分數也會跟著標註高低走（約 2.08–4.28）。

## 可以支持／不可以支持

- 可以支持：本機 L5 的「整段鎖死 + 分數塌在 3」不是論文釋出權重在 D16 上的行為。
- 可以支持：官方權重在這 16 張上也偏好同一個第一工具，套式沒有完全消失，只是弱很多、而且停在開場，不是整條鏈。
- 不可以支持：論文方法已被驗證、PLCC/SRCC、或官方權重已學會依圖選「正確」工具。沒有標註唯一正解，也沒有在 H 或 OOD benchmark 上比。
- 設定不同：prompt、token、工具圖來源、max tokens／tool 上限都與 E1 LoRA 不同。MAE 對照只是同圖描述，不是成對實驗。

## 產物

`scratch/e1/paper_d16/run1/`：manifest、rows.json、traces、analysis.json。約 3.5 分鐘。
