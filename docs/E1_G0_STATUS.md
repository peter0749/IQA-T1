# E1 G0：實際管線驗證

2026-09-07。G0 **PASS**。這是器材／可解讀性 gate，不是品質或工具策略結論。未查看 H32，未跑 D16 正式對照，未訓練。

## 範圍

- 舊 development8 固定首兩張：`7617916556`、`6835007518`（`scratch/l2_validation/run1/validation8.json`）。
- 共同 prompt、greedy、512 tokens、最多 4 個不重複工具、同一 BASE processor。
- 三個 checkpoints 各以 **fresh process** 載入：base、L2 adapter-micro80、L5 adapter-micro80。
- 12 次 rollout（3×2×2），等於協定上限。

## 通過項目（皆有 artifact，非 source-string 搜尋）

| 項目 | 結果 |
| --- | --- |
| 原圖／evidence 路徑 → pixel tensor | 單圖 encode 的 slice SHA 與多圖對應 slice 一致 |
| 工具前後 token IDs、image grid、label mask | vision start/pad/end 全 mask；pad 數與 `image_grid_thw / merge²` 相符 |
| 無未來 evidence 提前送入 | step 0 僅原圖；之後只含已 dispatch 的工具圖 |
| 停止標記／continuation | 18 次工具邊界均可對齊；無丟棄後重送、無重複 continuation |
| Fresh reload active adapter | base 無 LoRA；L2/L5 僅 `default`，16,515,072 參數，eval frozen |
| 重放兩次 | 6/6 對 replay **token 完全相同**；分數差 0。此為噪聲基準，零差異不是科學門檻 |

## 產物

- Harness：`scratch/e1/harness.py`
- Runner：`scratch/e1/g0.py`
- 正式 run：`scratch/e1/g0/run2/`（manifest、static_audit、adapter_*、12 份 traces、decision.json）
- 失敗的 run1 保留：相對 `--out` 路徑在寫完第一份 rollout 後崩潰；未覆蓋，未據此調參。

## 預算外推

run2 三個 fresh process 合計約 182 秒（base 76 / L2 46 / L5 60）。12 rollout ≈ 3 分鐘。依此粗估正式矩陣 288 continuations 約 **1.2 小時**，低於 4 小時上限。這是 G0 開發影像的耗時，不是 H 的結果。

## 明確不是結論

- base 兩張均 `eos`、無工具、無分數：只說明這兩張在此 harness 下 base 沒走工具契約，不是「無工具基準已建立」。
- L2/L5 有工具呼叫與分數，重放一致：只證明 dispatcher 與 adapter 載入可解讀。
- 不得把 G0 分數或工具名單當成 E1a/b/c 結果。

## 下一步

D 已完成：見 [E1_D_STATUS.md](E1_D_STATUS.md)。H 仍鎖定，需另一次 Outcome。
