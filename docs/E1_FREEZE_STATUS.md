# E1 資料與 checkpoint 凍結紀錄

2026-09-07。資料/權重清單已凍結，runtime inventory 已記錄；實驗 runner 與 G0 尚未完成。E1 保持進行中，未執行新模型推論。

- 來源：`repo/dataset/KONIQ/metas/koniq_training-multimodal-all.json`，11,315 rows / 3,540 unique original images。這是本機 SFT metadata 所覆蓋的池，不等於全部 10,373 張原圖；推論範圍限此池。
- 64 張保守曝光排除，其中 8 張確認用於本機優化。候選 slice、腳本和報告內 ID 也保守排除；ID 出現不必然代表曾生成。僅能涵蓋可取得的本機紀錄，base 預訓練曝光未知。
- D16、H32、共同 prompt、split 內循環 donor 已凍結。未用模型輸出選樣；選用 row 的 `gt_score`，未宣稱 MOS。H 未進行推論或誤差分析。
- 48 × 16 = 768 個原圖/工具圖 asset 驗證可讀並雜湊。完整池 3,540 張原圖可讀且均有15種工具檔。
- 原圖 byte SHA256 + grayscale 64-bit dHash，距離 <=6 視為疑似近重複，對曝光及先前入選圖篩選。本次前48張無觸發。此啟發式可能漏掉裁切/變形，不是全面去重保證。
- base/L2/L5 權重及設定已 SHA256；Python/library versions 已記錄。active adapters、runtime tensor 與 token gate 尚待 G0。

## 產物

Mac project root 下 `scratch/e1/freeze_v2/`：exposure_ledger.json、asset_inventory.json、split_manifest.json、checkpoint_manifest.json、runtime_inventory.json、summary.json。Runner：`scratch/e1/freeze.py`。

| Manifest | SHA256 |
| --- | --- |
| split_manifest.json | b6ff066cb93bcb637335378c4cd9cddce80f68f9a0e677f5d31f791e73d0e33b |
| checkpoint_manifest.json | d972fb23358d2c0f037e4e58c9c1b85c8a19020b3fee106e0f475c5601aac72d |
| runtime_inventory.json | 6a53dfcf46fa36fbc0bdeb5ad2c1f63789f5f8362dc9634fbf0cd5a96e96eec3 |
| exposure_ledger.json | bb66c3b549292a2d5e942a9238b4a359455ed314c20b72791d21820577f7eade |

## 偏離與驗證

首次 freeze_v1 將下載/解壓日誌誤作曝光，排除全池後停止，未產生 split。保留其失敗產物；freeze_v2 排除 `scratch/d1_logs` 的傳輸紀錄，其他曝光規則未放寬。近重複規則是生成前的實作細化，沒有查看 D/H 模型結果。
獨立核對通過：48個唯一ID；D/H/曝光互斥；donor 同 split 無自配且為雙射；每列16個 asset。資料階段完成不代表 G0 PASS。

下一步：依 E1 G0 實作並凍結同一個推論 harness，在舊 development 首兩張驗證實際 token/tensor/mask、工具邊界與 fresh reload，通過後才跑 D/H。
