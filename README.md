# Human-AI Interaction (HW2)

專案題目：**Human-AI Interaction 人類與 AI 互動信任度預測與特徵選擇**  
作者：**陳沐德 (學號：7114068027)**  
資料集來源：[Kaggle Human-AI Interaction Dataset](https://www.kaggle.com/datasets/ahmedashrafpro/human-ai-interaction)

---

## 📁 專案檔案清單

- 📓 **`7114068027_hw2.ipynb`**：完整作業主程式碼（已包含執行結果、95% 信賴區間視覺化、特徵權重圖）。
- 📄 **`7114068027_hw2_report.pdf`**：依照規定產出之作業報告書（包含封面資訊、Kaggle 介紹、【NotebookLM 摘要】、CRISP-DM 流程、視覺化圖表與【GPT 輔助內容聲明】）。
- 🗜️ **`7114068027_hw2.zip`**：繳交打包壓縮檔。
- 📊 **`human_ai_interaction.csv`**：符合規定之 20 個數值特徵資料集（10,100 筆樣本）。
- 📈 **`prediction_ci_plot.png`**：`seaborn.regplot` 繪製之實際值 vs 預測值 95% 信賴區間圖。
- 📊 **`feature_importance.png`**：RFE 篩選之 12 個關鍵特徵標準化迴歸權重條形圖。
- ⚙️ **`run_hw2_analysis.py`**：完整的模型訓練、特徵消除與指標計算 Pipeline。
- 📝 **`generate_pdf_report.py`**：自動生成作業 PDF 報告書之腳本。

---

## 🚀 執行方式

1. 安裝所需套件：
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib reportlab
   ```
2. 執行分析與產出圖表：
   ```bash
   python run_hw2_analysis.py
   ```
3. 重新產出 PDF 報告：
   ```bash
   python generate_pdf_report.py
   ```
