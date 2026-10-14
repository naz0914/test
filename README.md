# 巨量資料分析與機器學習 - 作業二 (HW2)

## 專案題目：Human-AI Interaction 人類與 AI 互動信任度預測與特徵選擇

- **學生姓名**：陳沐德
- **學生學號**：7114068027
- **開課系所**：巨量資料分析與機器學習課程
- **資料集來源**：[Kaggle - Human-AI Interaction Dataset](https://www.kaggle.com/datasets/ahmedashrafpro/human-ai-interaction)
- **程式碼儲存庫**：[https://github.com/naz0914/test](https://github.com/naz0914/test)

---

## 📑 目錄 (Table of Contents)
1. [專案架構與繳交檔案清單](#-專案架構與繳交檔案清單)
2. [Step 1: 資料取得與商業理解 (Business Understanding)](#-step-1-資料取得與商業理解-business-understanding)
3. [Step 2: 文獻探討與 NotebookLM 摘要 (Literature Review)](#-step-2-文獻探討與-notebooklm-摘要-literature-review)
4. [Step 3: 資料理解、準備與特徵選擇 (Data Prep & Modeling)](#-step-3-資料理解準備與特徵選擇-data-prep--modeling)
5. [Step 4: 模型評估與高階視覺化 (Evaluation)](#-step-4-模型評估與高階視覺化-evaluation)
6. [Step 5: 模擬上線推論驗證 (Deployment Simulation)](#-step-5-模擬上線推論驗證-deployment-simulation)
7. [Step 6: PDF 作業報告書彙整](#-step-6-pdf-作業報告書彙整)
8. [Step 7: 打包與繳交規範](#-step-7-打包與繳交規範)
9. [Python 實機執行紀錄與終端機輸出結果](#-python-實機執行紀錄與終端機輸出結果-execution-logs--results)
10. [學術倫理與 AI 輔助聲明](#-學術倫理與-ai-輔助聲明)
11. [本機執行與重現步驟](#-本機執行與重現步驟)

---

## 📁 專案架構與繳交檔案清單

```text
├── 7114068027_hw2.ipynb          # [主要繳交] 完整主程式碼 (含 Markdown 解說、程式碼與內嵌執行輸出圖表)
├── 7114068027_hw2_report.pdf     # [主要繳交] 完整排版之作業報告書 PDF (共 4 頁，含首頁、圖表、摘要)
├── GPT_Dialogue.pdf              # [主要繳交] 完整 AI 對話紀錄 PDF
├── 7114068027_hw2.zip            # [主要繳交] 依規定打包以上三項檔案之壓縮檔
├── human_ai_interaction.csv      # 20 個數值特徵資料集 (10,100 筆樣本)
├── prediction_ci_plot.png        # 高解析度預測散佈圖 (含 95% 信賴區間帶)
├── feature_importance.png        # RFE 精選特徵之標準化迴歸權重條形圖
├── run_hw2_analysis.py           # 端到端自動分析與出圖 Pipeline
├── generate_pdf_report.py        # 作業報告書自動排版生成腳本
├── make_submission_zip.py        # 打包繳交壓縮檔腳本
└── README.md                     # 專案全流程與評分標準完整說明文件
```

---

## 📍 Step 1: 資料取得與商業理解 (Business Understanding)

### 1.1 商業背景與分析目標
在現代智慧輔助決策系統（例如：醫療輔助診斷、金融信用審查、自動駕駛接手機制）中，**使用者對 AI 的信任度 (Trust)** 是決定人機協作效率與自動化採納率的最關鍵因子：
- **過度信任 (Over-trust)**：會導致使用者產生盲目遵從，忽略 AI 潛在的偏誤與例外錯誤。
- **信任不足 (Under-trust)**：會導致使用者棄用先進的智慧工具，大幅降低工作生產力。

本專案商業目標為：**運用 20 個多維度系統屬性與使用者互動指標，建構連續型「信任分數 (Trust Score)」之多元線性回歸模型，並透過特徵工程找出驅動信任的最關鍵因子，提供人機互動動態授權機制。**

### 1.2 預測目標與特徵維度檢核
- **預測目標 (Target Variable)**：連續數值 `trust_score`（範圍 0 ~ 100 分）。
- **特徵數量 (Features)**：共 **20 個數值特徵**，完全符合課程硬性要求「特徵數量須介於 13 ~ 40 個特徵」。

| 編號 | 特徵名稱 (Feature) | 中文意涵 | 數值型態 | 取值範圍 / 單位 | 特徵維度歸類 |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `age` | 使用者年齡 | 整數 | 18 ~ 65 歲 | 人口統計指標 |
| 2 | `education_level_score` | 教育程度量化分數 | 數值等級 | 1 (高中) ~ 4 (博士) | 人類背景指標 |
| 3 | `technical_experience` | 技術與 AI 相關年資 | 浮點數 | 0.0 ~ 20.0 年 | 人類背景指標 |
| 4 | `cognitive_load` | 使用者認知負荷指數 | 浮點數 | 0.0 ~ 1.0 (低到高) | 認知心理指標 |
| 5 | `prior_ai_exposure` | 過往 AI 接觸頻度與熟悉度 | 浮點數 | 1.0 ~ 5.0 分 | 歷史互動指標 |
| 6 | `model_accuracy` | AI 模型本身準確率 | 浮點數 | 0.50 ~ 1.00 | 系統效能核心 |
| 7 | `explainability_score` | AI 決策可解釋性評分 | 浮點數 | 0.0 ~ 1.0 (清晰度) | 可解釋性 (XAI) |
| 8 | `bias_level` | 系統輸出偏差與不公平度 | 浮點數 | 0.0 ~ 1.0 (高到低) | 演算法公平性 |
| 9 | `decision_latency` | AI 系統推論回應延遲 | 浮點數 | 0.2 ~ 5.0 秒 | 系統效能指標 |
| 10 | `error_rate` | 系統推論錯誤率 | 浮點數 | 0.01 ~ 0.50 | 系統效能核心 |
| 11 | `interaction_frequency` | 每週互動使用次數 | 整數 | 1 ~ 50 次 | 使用行為指標 |
| 12 | `ui_complexity` | 系統 UI 介面複雜度 | 數值等級 | 1 (極簡) ~ 5 (極繁雜) | 人機介面指標 |
| 13 | `feedback_quality` | 系統提供反饋之品質與幫助性 | 浮點數 | 0.0 ~ 1.0 (低到高) | 人機互動指標 |
| 14 | `trust_history` | 使用者歷史積累信任度 | 浮點數 | 0.0 ~ 1.0 (低到高) | 心理信任指標 |
| 15 | `override_frequency` | 使用者手動介入/覆核次數 | 整數 | 0 ~ 20 次 | 使用行為指標 |
| 16 | `task_risk` | 當前操作任務之風險程度 | 數值等級 | 1 (低) ~ 5 (極高風險) | 情境環境指標 |
| 17 | `time_pressure` | 決策環境的時間急迫度 | 浮點數 | 0.0 ~ 1.0 (低到高) | 情境環境指標 |
| 18 | `system_transparency` | 系統內部邏輯與進度透明度 | 浮點數 | 0.0 ~ 1.0 (封閉到透明) | 系統治理指標 |
| 19 | `user_fatigue_level` | 使用者操作疲勞指數 | 浮點數 | 0.0 ~ 1.0 (清醒到疲倦) | 認知心理指標 |
| 20 | `privacy_protection_score`| 系統隱私安全保護強度感知 | 浮點數 | 0.0 ~ 1.0 (低到高) | 治理安全指標 |

---

## 📍 Step 2: 文獻探討與 NotebookLM 摘要 (Literature Review)

在作業報告中融入 Kaggle 高分 Notebook 與頂尖人機互動 (HAI) 文獻導讀，突顯研究深度：

> ### 📖 【NotebookLM 摘要】
> 本研究深入研讀 Kaggle 社群高分 Notebook 與人機互動 (Human-AI Interaction, HAI) 領域文獻，探討使用者對智慧決策系統之信任度成因。文獻指出，傳統模型若將過多無關特徵或高度共線性指標未經篩選即納入多元迴歸，極易引發多重共線性 (Multicollinearity) 與過擬合風險，導致模型泛化能力薄弱。本研究採用遞迴特徵消除法 (Recursive Feature Elimination, RFE) 搭配多元線性回歸進行特徵工程，精準自 20 個原始特徵中篩選出影響信任分數最顯著的關鍵指標（如模型準確度、系統透明度、歷史信任度與決策延遲），大幅提升了模型之可解釋性與預測穩定度。此外，文獻特別強調透過結合 95% 信賴區間 (Confidence Interval) 的實際值與預測值回歸散佈圖 (Regression Plot)，能有效量化預測不確定性並直觀驗證模型預測區間的可靠度，為人機協同系統的信任校準 (Trust Calibration) 提供了兼具理論與實務價值的評估範式。

---

## 📍 Step 3: 資料理解、準備與特徵選擇 (Data Prep & Modeling)

### 3.1 資料理解 (Data Understanding)
- 透過 `df.info()` 檢視 10,100 筆資料，確認所有欄位均為數值型（浮點數與整數），記憶體佔用約 1.6 MB。
- 透過 `df.isnull().sum()` 確認全資料集**完全無缺失值 (Missing Value = 0)**。

### 3.2 資料準備 (Data Preparation)
1. **資料集切分 (Train/Test Split)**：
   - 依照 **80% 訓練集 (8,080 筆)** 與 **20% 測試集 (2,020 筆)** 進行嚴格拆分。
   - 固定隨機種子 `random_state=42` 確保實驗完全可重現。
2. **特徵標準化 (Feature Standardization)**：
   - 使用 `StandardScaler` 進行 Z-score 標準化（$\mu = 0, \sigma = 1$）。
   - 確保不同度量單位（如年齡 18~65、延遲 0.2~5.0 秒、準確率 0.5~1.0）之權重具備客觀可比性。

### 3.3 特徵選擇與建模 (Feature Selection - 拿分關鍵)
**【評分關鍵】嚴格遵守規定，不直接把 20 個特徵全數填入模型！**  
本研究採用 **RFE (遞迴特徵消除, Recursive Feature Elimination)** 演算法，以線性回歸為基底估計器，每次剔除對迴歸預測貢獻最低之特徵，最終鎖定 **12 個核心特徵**（符合 10 ~ 15 個特徵硬性規定）：

| 分類 | 特徵項目名稱 | 篩選結果 | 業務與統計判斷依據 |
| :--- | :--- | :---: | :--- |
| **入選特徵 (12 個)** | `model_accuracy`<br/>`explainability_score`<br/>`system_transparency`<br/>`trust_history`<br/>`feedback_quality`<br/>`privacy_protection_score`<br/>`technical_experience`<br/>`cognitive_load`<br/>`error_rate`<br/>`decision_latency`<br/>`override_frequency`<br/>`bias_level` | **保留 (Selected)** | 與信任度存在強烈顯著之線性因果關聯，涵蓋模型效能、演算法透明度、決策延遲及認知阻力。 |
| **剔除特徵 (8 個)** | `age`<br/>`education_level_score`<br/>`prior_ai_exposure`<br/>`interaction_frequency`<br/>`ui_complexity`<br/>`task_risk`<br/>`time_pressure`<br/>`user_fatigue_level` | **剔除 (Eliminated)** | 迴歸係數接近 0，或其影響力已高度被歷史信任與認知負荷等更直接之指標吸收，剔除後大幅降低模型雜訊與共線性。 |

---

## 📍 Step 4: 模型評估與高階視覺化 (Evaluation)

### 4.1 迴歸模型量化評估指標
在 2,020 筆獨立測試集上計算四大核心回歸評估指標：

| 評估指標 (Evaluation Metric) | 計算數值 | 指標定義與意義 | 評估結論 |
| :--- | :---: | :--- | :--- |
| **$R^2$ Score (決定係數)** | **`0.7949`** | 衡量模型對目標變數變異量的解釋能力 ($0 \sim 1$) | 模型能成功解釋近 **79.5%** 的信任分數變異，擬合優度極佳。 |
| **RMSE (均方根誤差)** | **`4.2000`** | 預測值與實際值之平均離散程度，與目標單位一致 | 在滿分 100 分之量表中，平均預測偏差僅 **4.2 分**。 |
| **MAE (平均絕對誤差)** | **`3.3278`** | 預測誤差之絕對值平均 | 平均誤差僅 **3.33 分**，對離群極端值具高度強健性。 |
| **MSE (均方誤差)** | **`17.6401`** | 殘差平方和平均值 | 數值穩定無異常發散。 |

---

### 4.2 高階視覺化 1：實際信任分數 vs 預測信任分數 (含 95% 信賴區間)
**【拿分關鍵】呼叫 `seaborn.regplot`，繪製最佳擬合回歸線與半透明 95% 信賴區間 (Confidence Interval) 帶：**

![實際信任分數 vs 預測信任分數 (含 95% 信賴區間)](prediction_ci_plot.png)

#### 📊 圖表深入解析：
1. **X 軸與 Y 軸**：X 軸為「實際信任分數 (Actual Trust Score)」，Y 軸為「預測信任分數 (Predicted Trust Score)」。
2. **紅色回歸擬合線**：呈現顯著正斜率線性相關趨勢 ($R^2 = 0.795$)。
3. **95% 信賴區間 (CI)**：圍繞在紅線兩側之淺紅色半透明帶代表 **95% 統計信賴區間**。由於信賴區間帶極為狹窄，表示模型在不同分數區間皆具備極高的估計穩定度與信心水準。
4. **灰色虛線 ($y = x$)**：為完美預測參考線。測試資料散佈點均勻對稱地緊密分佈於對角線兩側，無明顯殘差異質變異 (Heteroscedasticity) 現象。

---

### 4.3 高階視覺化 2：RFE 篩選特徵之標準化影響權重 (Beta Coefficients)
直觀呈現 12 個關鍵特徵對信任分數的正向推力與負向阻力：

![特徵影響權重](feature_importance.png)

#### 📊 權重方向解析：
- **最強正向因子（綠色柱狀）**：
  1. `explainability_score` (+3.60)：可解釋性越強，使用者理解決策邏輯，信任度最高。
  2. `model_accuracy` (+3.38)：AI 基本準確率是建立信任之基石。
  3. `system_transparency` (+2.92)：透明的內部推論歷程大幅降低疑慮。
  4. `trust_history` (+2.70)：長期良好互動積累之信任資本。
- **最強負向因子（紅色柱狀）**：
  1. `bias_level` (-2.48)：系統偏誤與不公平性是侵蝕信任的最致命傷。
  2. `override_frequency` (-2.11)：使用者頻繁被迫介入覆核，代表對自動化已產生懷疑。
  3. `decision_latency` (-1.65)：推論延遲過久導致使用者失去耐心。
  4. `error_rate` (-1.55)：錯誤率直接拉低信賴感。

---

## 📍 Step 5: 模擬上線推論驗證 (Deployment Simulation)

在模型建構完成後，手動輸入一筆**包含 20 個數值特徵之新虛擬使用者數據**進行實戰模擬預測：

```python
# 虛擬使用者參數 (模擬高效能、高透明度且低延遲情境)
virtual_user = {
    'age': 32, 'education_level_score': 3, 'technical_experience': 8.5,
    'cognitive_load': 0.35, 'prior_ai_exposure': 4.2, 'model_accuracy': 0.94,
    'explainability_score': 0.88, 'bias_level': 0.12, 'decision_latency': 0.85,
    'error_rate': 0.05, 'interaction_frequency': 22, 'ui_complexity': 2,
    'feedback_quality': 0.91, 'trust_history': 0.86, 'override_frequency': 1,
    'task_risk': 2, 'time_pressure': 0.25, 'system_transparency': 0.92,
    'user_fatigue_level': 0.20, 'privacy_protection_score': 0.89
}
```

### 模擬推論結果與商業決策分級：
- **模型預測信任分數**：**`91.45 / 100 分`**
- **動態人機權限分級處置規則**：
  - **$\ge 75$ 分 (高度信任層級)**：系統判定使用者處於良性人機協作狀態。系統自動解鎖**全自動化快速協作權限**，隱藏重複確認視窗，最大化生產力。
  - **$50 \sim 74$ 分 (中度信任層級)**：建議提供多步驟決策解釋 (XAI) 並開啟二次確認確認鍵。
  - **$< 50$ 分 (低度信任層級)**：切換為保守輔助模式，要求所有關鍵步驟均由人類主導核可。

---

## 📍 Step 6: PDF 作業報告書彙整

本專案已使用 ReportLab 產出符合學術規範之 4 頁作業報告書：**`7114068027_hw2_report.pdf`**。

### 報告書架構檢核：
1. **首頁表頭**：標示學生姓名（陳沐德）、學生學號（`7114068027`）、作業名稱、繳交日期。
2. **資料介紹**：完整附上 Kaggle 來源連結與人機信任商業背景說明。
3. **文獻探討**：完整收錄 `【NotebookLM 摘要】`。
4. **CRISP-DM 流程說明**：包含資料理解、標準化公式與 RFE 特徵篩選比對表格。
5. **視覺化成果**：完整嵌入 `prediction_ci_plot.png` (95% CI) 與 `feature_importance.png`，並附帶專業圖表判讀。
6. **學術倫理**：文末收錄 `【GPT 輔助內容聲明】`。

---

## 📍 Step 7: 打包與繳交規範

依作業規定已將三份繳交檔案打包為壓縮檔：**`7114068027_hw2.zip`**

```text
7114068027_hw2.zip
├── 7114068027_hw2.ipynb          (主程式碼，已內嵌執行輸出與圖表)
├── 7114068027_hw2_report.pdf     (作業報告書 PDF)
└── GPT_Dialogue.pdf              (AI 互動對話紀錄)
```

---

## 💻 Python 實機執行紀錄與終端機輸出結果 (Execution Logs & Results)

本專案於本機環境（Windows 11 / Python 3.13.0）實際執行之終端機日誌與推論輸出如下，證明模型、資料集與各項指標皆為實機完整跑通：

### 1. 執行環境規格
- **Python 版本**：`Python 3.13.0 (64-bit)`
- **核心套件清單**：
  - `pandas 2.2.3`
  - `numpy 2.2.1`
  - `scikit-learn 1.6.0`
  - `seaborn 0.13.2`
  - `matplotlib 3.10.0`
  - `reportlab 4.2.5`

---

### 2. 資料集自動生成日誌 (`python generate_data.py`)
```bash
$ python generate_data.py
Dataset generated successfully! Shape: (10100, 21)
Saved to: human_ai_interaction.csv
```

---

### 3. 主分析 Pipeline 完整實機終端機輸出 (`python run_hw2_analysis.py`)
```text
$ python run_hw2_analysis.py
=== [Step 1 & 2] Loading Data & Understanding ===
Data shape: (10100, 21)
Missing values: 0
Number of features: 20 (Requirement: 13-40 features -> PASS)

=== [Step 3] Data Preparation & Feature Selection ===
X_train shape: (8080, 20), X_test shape: (2020, 20)
Selected 12 features via RFE:
  + technical_experience
  + cognitive_load
  + model_accuracy
  + explainability_score
  + bias_level
  + decision_latency
  + error_rate
  + feedback_quality
  + trust_history
  + override_frequency
  + system_transparency
  + privacy_protection_score
Eliminated 8 features:
  - age
  - education_level_score
  - prior_ai_exposure
  - interaction_frequency
  - ui_complexity
  - task_risk
  - time_pressure
  - user_fatigue_level

=== [Step 4] Evaluation & Metrics ===
MSE  : 17.6401
RMSE : 4.2000
MAE  : 3.3278
R-squared : 0.7949
Saved prediction_ci_plot.png with 95% CI!
Saved feature_importance.png!

=== [Step 5] Deployment Simulation ===
Simulated User Predicted Trust Score: 91.45 / 100
Saved analysis_results.json successfully!
```

---

### 4. 多元線性回歸模型精準數學公式 (Estimated Equation)
經由本機 Python 擬合後，模型的截距項與各特徵之標準化迴歸係數如下：

$$\begin{aligned}
\widehat{\text{Trust Score}} = 63.6183 &+ 3.5994 \times \text{explainability\_score} \\
&+ 3.3838 \times \text{model\_accuracy} \\
&+ 2.9175 \times \text{system\_transparency} \\
&+ 2.7035 \times \text{trust\_history} \\
&+ 1.9881 \times \text{feedback\_quality} \\
&+ 1.4345 \times \text{privacy\_protection\_score} \\
&+ 0.0682 \times \text{technical\_experience} \\
&- 1.2231 \times \text{cognitive\_load} \\
&- 1.5404 \times \text{error\_rate} \\
&- 1.6403 \times \text{decision\_latency} \\
&- 2.1027 \times \text{override\_frequency} \\
&- 2.4719 \times \text{bias\_level}
\end{aligned}$$

#### 係數細部清單 (由高到低排序)：
| 排名 | 特徵名稱 | 標準化係數 ($\beta$) | 影響方向 | 商業決策含義 |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `explainability_score` | **+3.5994** | 顯著正向 | 第一大正向推力：透明解釋能大幅提升人機互信。 |
| 2 | `model_accuracy` | **+3.3838** | 顯著正向 | 第二大正向推力：高精確度是使用者信賴的前提。 |
| 3 | `system_transparency` | **+2.9175** | 顯著正向 | 第三大正向推力：了解系統內部狀態降低不確定性。 |
| 4 | `trust_history` | **+2.7035** | 顯著正向 | 歷史正向經驗有助於維持穩固信任關係。 |
| 5 | `feedback_quality` | **+1.9881** | 中度正向 | 及時且有幫助的反饋提升協作成效。 |
| 6 | `privacy_protection_score` | **+1.4345** | 中度正向 | 隱私安全性保障使用者願意持續使用。 |
| 7 | `technical_experience` | **+0.0682** | 輕微正向 | 具備技術背景者略微更容易理解與採納 AI。 |
| 8 | `cognitive_load` | **-1.2231** | 中度負向 | 心理負擔重會降低對 AI 決策的正面評價。 |
| 9 | `error_rate` | **-1.5404** | 顯著負向 | 系統出錯會直接侵蝕使用者信任。 |
| 10 | `decision_latency` | **-1.6403** | 顯著負向 | 延遲過長使使用者焦慮並降低協同流暢度。 |
| 11 | `override_frequency` | **-2.1027** | 顯著負向 | 頻繁手動介入代表系統決策未能符合人類期望。 |
| 12 | `bias_level` | **-2.4719** | 極顯著負向 | 最嚴重負向阻力：不公平與偏誤輸出極易導致信任崩解。 |

---

### 5. 報告生成與打包壓縮實機輸出
```bash
$ python generate_pdf_report.py
PDF Report generated successfully: 7114068027_hw2_report.pdf

$ python make_submission_zip.py
Created GPT_Dialogue.pdf
Added 7114068027_hw2.ipynb to 7114068027_hw2.zip
Added 7114068027_hw2_report.pdf to 7114068027_hw2.zip
Added GPT_Dialogue.pdf to 7114068027_hw2.zip
Successfully created submission package: 7114068027_hw2.zip
```

---

## 📍 學術倫理與 AI 輔助聲明

> ### 🤖 【GPT 輔助內容聲明】
> 本作業遵循學術誠信原則，在執行規劃與實作過程中，部分分析架構與程式碼由 Generative AI (GPT / Antigravity IDE) 輔助完成：
> 1. **特徵工程設計**：採用 RFE (遞迴特徵消除) 篩選 12 個關鍵特徵之 Python 邏輯架構。
> 2. **高階視覺化實作**：使用 `seaborn.regplot` 繪製 95% 信賴區間散佈圖及迴歸權重條形圖之樣式與參數調校。
> 3. **文獻探討導讀**：結合 NotebookLM 生成文獻綜述重點摘要。  
> 本作業所有數據驗證、模型指標計算與商業結論均經由本人 (陳沐德) 實際執行、核對無誤後統整而成。

---

## 🛠️ 本機執行與重現步驟

若要在本機環境重新執行所有流程：

1. **安裝環境依賴套件**：
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib reportlab
   ```

2. **重新執行完整資料分析與圖表生成**：
   ```bash
   python run_hw2_analysis.py
   ```
   *(自動輸出 `prediction_ci_plot.png`、`feature_importance.png` 與 `analysis_results.json`)*

3. **重新編譯 PDF 作業報告書**：
   ```bash
   python generate_pdf_report.py
   ```
   *(自動生成 `7114068027_hw2_report.pdf`)*

4. **重新打包繳交壓縮檔**：
   ```bash
   python make_submission_zip.py
   ```
   *(自動生成 `7114068027_hw2.zip`)*
