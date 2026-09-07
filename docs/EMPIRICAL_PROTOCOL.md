# E1：影像條件化工具策略與證據效益 — 實驗協定 v1

日期：2026-09-07。狀態：研究設計已登記；執行前須完成資料與 runtime manifest 凍結。L1–L5 均為探索性前導結果。
本階段只用本機既有模型與資料，不訓練、不合成、不用 11k、不租雲、不跑 GRPO。未執行任何新模型實驗。

## 研究問題與推論範圍

RQ：現有 LoRA adapter 是否依原圖選擇工具、利用回傳證據，以及在新影像上改善品質評分？
這是三個不同命題；合法格式只是必要的工程條件。觀察到輸出改變不等於證據有益，未觀察到改變也不等於模型完全不使用影像。
本階段比較固定 checkpoints，不估計訓練方法的跨 seed 效果，不宣稱論文復現。

## 前導證據與更正

| 證據 | 可以支持 | 不可以支持 |
| --- | --- | --- |
| L3 baseline 工具契約 3/8；長 contract 0/8 | 這批開發影像對這兩種提示有不同表現 | 所有長提示均無效；SFT 是唯一原因 |
| L4 靜態 prefix/template 檢查 | 尚未找到特定程式結構差異 | 已排除 token、tensor、reload 或 runtime bug |
| L5 工具契約 8/8；MAE 0.3625，baseline 0.2863 | 固定 checkpoint 在此開發集的行為不同 | 純粹工具位置的因果效應；評分品質提升 |
| L5 只用三種工具，標註序列相符 0/8 | 選擇集中、值得診斷 | 工具必定選錯，或完全不看影像 |

L5 同時改變工具位置、文字指涉、證據可見時機與後續 token 條件，視為複合表示處理。8 張 train 才參與優化，另 8 張只參與評估；先前「16 張 SFT 訓練樣本」的表述不精確。標註工具清單不是唯一正解。

## 競爭假設與可區分證據

| 假設 | 預期觀察 | 對照與限制 |
| --- | --- | --- |
| H0 管線或 adapter 載入錯誤 | 重放、mask、tensor 對應或 active adapter 檢查失敗 | G0 通過後才解讀機制實驗 |
| H1 模型主要學到固定工具套路 | 固定提示下第一工具高度集中、原圖替換很少改變第一工具 | E1a；集中可能也是合理通用策略，不能單獨判錯 |
| H2 會選工具，但回傳證據影響很小 | 首次 evidence 同類錯配後，分數與後續路徑變化接近重放差異 | E1b；只涵蓋首次呼叫與此次干預 |
| H3 證據影響生成但未改善評分 | 錯配改變輸出，真實證據 MAE 無優勢或 coverage 下降 | E1b + E1c；敏感性與效益分開 |
| H4 證據有實用效益 | 新保留集真實 evidence 的成對誤差較低且格式可靠 | E1c；限本資料池與固定 checkpoints |

## 資料與模型凍結

- 模型：base；L2 adapter-micro80；L5 adapter-micro80。各自 fresh process 載入，記錄權重/config/tokenizer/processor/template 與 runner SHA-256、active adapters、dtype、device、library versions。
- 舊 train8、development8 以及所有既有生成或訓練曾使用的影像，列入 exposure ledger。按原圖 ID 去重，並檢查 byte hash 與近重複；不只比較 row ID。
- 診斷集 D：從未用於訓練/既有生成的資料取 16 張。保留集 H：另外取 32 張，與 D/所有 exposed images 互斥。這是有界 pilot，未經 power analysis，不保證檢出小效果。
- 選樣：既有完整 asset 池，排除 exposure 後按 SHA-256("E1-v1:" + image_id) 排序，前 16 為 D、後 32 為 H。只依事前 asset 完整性篩選，不依模型輸出或誤差挑樣。少於 48 張則停止凍結並記錄限制，不偷換舊 development8。
- 每 image 的多個標註：以 row ID 字典序選一列，記錄 score 來源及尺度；不把此值自動稱為 MOS。共同 prompt 為 train8 已出現的 "Examine the image's quality and provide an evaluation based on your observations.\n<image>"。
- 首輪凍結 hash 後不得因結果修改模型、提示、decoder、donor 或 H。修改協定須記錄版本、理由、已查看的結果；H 一旦查看即不能再作新設計的未見保留集。

## G0：實際管線驗證

在舊 development8 中固定首兩張完成：原圖/evidence 路徑到 pixel tensor 的對應；工具前後實際 token IDs、image grid、label mask；無未來 evidence 被提前送入；停止標記不丟失/重複 continuation token；fresh reload active adapter 唯一。
重放同一輸入兩次記錄分數、tokens、工具路徑與數值差異。若 greedy 不完全重現，保留差異作數值噪聲基準。所有 G0 項目均須有 artifact，不能用 source-string 搜尋替代。
失敗只修正影響可解讀性的問題，記錄版本，重做受影響 gate；不根據品質結果調參。

## 預先固定的實驗矩陣

三個 checkpoints 使用同一 harness；greedy、512 generated tokens、最多 4 個不重複工具、相同 processor。工具圖僅在模型呼叫後注入。

| 實驗 | 資料/條件 | 主要估計量 |
| --- | --- | --- |
| E1a 原圖條件化 | D16 × 3 models，共同 prompt，真實 evidence | 第一工具分布/entropy、無呼叫率；按 manifest 循環配對原圖的第一工具改變率 |
| E1b 首次 evidence 干預 | D16 × 3，從各自真實軌跡的第一呼叫邊界分岔：同圖同工具 vs donor 同工具 | 成對分數絕對變化、後續工具序列變化、真實相對錯配的誤差差 |
| E1c 保留集效益 | H32 × 3，真實 evidence vs 首次同類錯配 | L5 真實相對錯配的成對 MAE 差為主要品質對照；base/L2 為次要比較 |

原圖輪換在固定 prompt 下等價於重用 E1a 的其他 image 軌跡，不另作重複 inference；只用第一呼叫作原圖效應指標，後續路徑混合了 evidence 影響。
donor 在各自 D/H 排序循環取下一影像，無自配對、不依分數或視覺差異挑 donor。同工具類型；固定 resize 至 receiver evidence 的尺寸並使用一致處理。記錄尺寸/分布差異，不能完全排除錯配帶來分布外效應。
E1b 分岔前 token prefix 完全相同；只換第一張回傳 evidence，後續呼叫依各分支自由生成並回填 receiver 的真實工具圖。不得強迫未自然呼叫工具的模型呼叫；無工具個案記為不適用，同時報全體母數與條件子集。
真實分支重用自然 rollout，錯配分支在同一 boundary 恢復。零 token 重放差異只是重現檢查，不作科學效果門檻。

## 無工具基準的處理

base 的自然 rollout 不是「保證無工具」基準。中止工具後截掉失敗輸出也不是公平基準。
直接評分比較列為後續次要實驗：需共同的 direct-score prompt 或明確的 constrained decoder，會同時改變生成條件。此版不把它混入主要矩陣，也不宣稱已估計工具相對完全無工具的淨效益。

## 指標、失敗與不確定性

- 主要機制量：E1b 在自然呼叫子集中的 mean absolute score change；同時報分支雙有效比例及全體無呼叫率。
- 主要品質量：H 上 L5 的 mean(|error_true| - |error_swapped|)，負值較佳。只在雙有效交集計算，另報全體 coverage、每種停止原因與全體失敗敏感性分析。
- 敏感性分析：score 尺度確認為 1–5 後，無效輸出分別賦予 0 和 4 的絕對誤差界限；報可能範圍，不隱藏失敗。
- 次要量：格式、合法工具執行、延遲/tokens、工具 entropy/序列多樣性、L5 vs L2/base 真實 rollout 的 paired MAE。標註清單 overlap 只作描述。
- 以 image 為單位 paired bootstrap 10,000 次、seed 20260907，報 95% interval 和逐例數據。checkpoint 與分支保持配對；不把每次工具呼叫當獨立樣本。
- 提升訊號：主要品質差上界低於 0、點估計改善至少 0.05 分，且真實分支 coverage 不低於錯配。0.05 是本計畫的工程實用門檻，非文獻標準。
- interval 跨 0 ⇒ inconclusive，不宣稱等效或「沒使用 evidence」。次要多重比較僅探索性報告，不以最好的那組事後替換主要假設。

## 預算與停止規則

上限：D/H 各 2 個條件 × 3 模型，共最多 288 rollout/continuations；G0 額外最多 12。每條件 512 tokens、4 tools；單一程序順序執行。
正式 H 前以舊 development G0 的實際耗時估算，總模型執行上限 4 小時。若估計超過，先版本化縮小範圍，不查看 H 結果後縮樣本。
OOM/非有限數值/錯誤 evidence mapping 立即停止受影響比較；保留 log。不得為湊成功率補抽樣本。時間截止保留未完成狀態，不宣稱 gate pass，不自動加訓練。

## 決策與交付

1. G0 不過：科學結論暫停，修管線。
2. G0 過、選擇集中但 evidence 有益：保留通用工具策略作候選，不因 overlap 低判失敗。
3. 原圖/證據敏感性低：優先檢查監督與工具分布，提出下一版受控訓練假設；不自動擴訓。
4. evidence 有影響但品質無改善：檢查 evidence/標註/評分目標；不把敏感性當效益。
5. H 支持實用效益：下一階段才設計新保留集與多 seed replication。
6. 陰性或區間寬：保留 inconclusive 結論，依效果/成本決定是否值得增加樣本。

交付：exposure ledger、split/donor/checkpoint/runtime manifest、G0 tensor/token audit、全矩陣原始 traces、逐 image 結果與失敗清單、效果與區間、協定偏離表、假設證據決策報告。
看板只設一張 E1 階段主任務，以下作同卡 checklist：
- [x] 協定與前導證據更正
- [ ] exposure inventory + D/H/model/runtime manifest 凍結
- [ ] G0 實際管線 gate
- [ ] D 機制對照
- [ ] H 保留集評估
- [ ] 統一分析、決策與交接

此版凍結設計規則；資料 manifest 與執行 gate 尚未完成。下一個 action 是 exposure inventory 與 manifest，之後依 gate 順序執行，不逐個結果另開臨時實驗。

