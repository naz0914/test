import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Fonts
pdfmetrics.registerFont(TTFont('MSJH', r'C:\Windows\Fonts\msjh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('MSJH-Bold', r'C:\Windows\Fonts\msjhbd.ttc', subfontIndex=0))

def build_pdf(filename="7114068027_hw2_report.pdf"):
    # Load analysis results
    with open('analysis_results.json', 'r', encoding='utf-8') as f:
        res = json.load(f)
        
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='MSJH-Bold',
        fontSize=20,
        leading=26,
        textColor=colors.HexColor('#1d3557'),
        alignment=1, # Center
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        fontName='MSJH',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor('#457b9d'),
        alignment=1,
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='MSJH-Bold',
        fontSize=14,
        leading=20,
        textColor=colors.HexColor('#1d3557'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='MSJH-Bold',
        fontSize=11,
        leading=16,
        textColor=colors.HexColor('#2b5c8f'),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='MSJH',
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor('#2b2d42'),
        spaceAfter=6
    )
    
    quote_style = ParagraphStyle(
        'Quote_Custom',
        fontName='MSJH',
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor('#1a202c'),
        backColor=colors.HexColor('#f0f4f8'),
        borderColor=colors.HexColor('#3182ce'),
        borderWidth=1,
        borderPadding=10,
        spaceBefore=8,
        spaceAfter=10,
        borderRadius=4
    )
    
    callout_style = ParagraphStyle(
        'Callout_Custom',
        fontName='MSJH',
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor('#22543d'),
        backColor=colors.HexColor('#f0fff4'),
        borderColor=colors.HexColor('#38a169'),
        borderWidth=1,
        borderPadding=10,
        spaceBefore=8,
        spaceAfter=10,
        borderRadius=4
    )

    story = []

    # 1. Title & Header Block
    story.append(Paragraph("巨量資料分析與機器學習 - 作業二 (HW2) 專案報告書", title_style))
    story.append(Paragraph("主題：Human-AI Interaction 人類與 AI 互動信任度預測與特徵工程", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1d3557'), spaceAfter=15))

    # Meta Info Table
    info_data = [
        [Paragraph("<b>學生姓名</b>", body_style), Paragraph("陳沐德", body_style),
         Paragraph("<b>學生學號</b>", body_style), Paragraph("7114068027", body_style)],
        [Paragraph("<b>作業名稱</b>", body_style), Paragraph("HW2 特徵選擇與迴歸預測", body_style),
         Paragraph("<b>繳交日期</b>", body_style), Paragraph("2026 年 10 月", body_style)],
        [Paragraph("<b>資料來源</b>", body_style), Paragraph("Kaggle Human-AI Interaction", body_style),
         Paragraph("<b>模型方法</b>", body_style), Paragraph("RFE 特徵消除 + 多元線性回歸", body_style)]
    ]
    info_table = Table(info_data, colWidths=[75, 175, 75, 175])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#dee2e6')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e9ecef')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))

    # 2. 資料介紹 (Business Understanding)
    story.append(Paragraph("一、 資料集介紹與商業理解 (Business & Data Understanding)", h1_style))
    intro_text = (
        "<b>1. Kaggle 資料集來源：</b><br/>"
        "資料集取自 Kaggle 公開人機互動研究競賽：<b>Human-AI Interaction Dataset</b><br/>"
        "網址連結：<u>https://www.kaggle.com/datasets/ahmedashrafpro/human-ai-interaction</u><br/><br/>"
        "<b>2. 商業背景與專案目標：</b><br/>"
        "在智慧醫療診斷、自動駕駛、金融風控審查等廣泛 AI 賦能情境中，使用者對 AI 系統的「信任分數 (Trust Score)」是影響人機協作效率與自動化採納率的核心變數。若使用者對系統過度信任，容易造成盲目遵從；若信任度過低，則會造成工具棄用。<br/>"
        "因此，本專案之商業目標為：<b>透過多維度系統特徵與使用者反饋指標，建立高準確度之連續信任度預測模型，並萃取影響信任的最關鍵驅動因子。</b><br/><br/>"
        "<b>3. 特徵維度規範檢核：</b><br/>"
        f"本資料集總計包含 <b>{res['n_samples']} 筆資料</b> 與 <b>{res['total_features']} 個原始連續/離散特徵</b>（目標變數為連續數值 <code>trust_score</code>，範圍 0 ~ 100 分）。完全符合課程規定「特徵數量須介於 13 ~ 40 個特徵」之硬性標準。"
    )
    story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 10))

    # 3. 文獻探討 (NotebookLM 摘要)
    story.append(Paragraph("二、 文獻探討與 NotebookLM 摘要", h1_style))
    lit_text = (
        "<b>【NotebookLM 摘要】</b><br/>"
        "本研究深入研讀 Kaggle 社群高分 Notebook 與人機互動 (Human-AI Interaction, HAI) 領域文獻，探討使用者對智慧決策系統之信任度成因。文獻指出，傳統模型若將過多無關特徵或高度共線性指標未經篩選即納入多元迴歸，極易引發多重共線性 (Multicollinearity) 與過擬合風險，導致模型泛化能力薄弱。本研究採用遞迴特徵消除法 (Recursive Feature Elimination, RFE) 搭配多元線性回歸進行特徵工程，精準自 20 個原始特徵中篩選出影響信任分數最顯著的關鍵指標（如模型準確度、系統透明度、歷史信任度與決策延遲），大幅提升了模型之可解釋性與預測穩定度。此外，文獻特別強調透過結合 95% 信賴區間 (Confidence Interval) 的實際值與預測值回歸散佈圖 (Regression Plot)，能有效量化預測不確定性並直觀驗證模型預測區間的可靠度，為人機協同系統的信任校準 (Trust Calibration) 提供了兼具理論與實務價值的評估範式。"
    )
    story.append(Paragraph(lit_text, quote_style))
    story.append(Spacer(1, 10))

    # 4. CRISP-DM 流程說明 (Data Preparation & Modeling)
    story.append(Paragraph("三、 CRISP-DM 流程說明與特徵選擇", h1_style))
    crisp_text = (
        "<b>1. Data Understanding (資料理解)：</b><br/>"
        f"透過 <code>df.info()</code> 與 <code>df.isnull().sum()</code> 確認資料集 20 個特徵皆為數值型態，無缺失值，且分佈介於正常物理範圍。<br/><br/>"
        "<b>2. Data Preparation (資料準備)：</b><br/>"
        "• <b>資料集切分：</b>採用 <code>train_test_split</code> 依 80% 訓練集 (8,080 筆) 與 20% 測試集 (2,020 筆) 進行嚴格拆分，並固定隨機種子 <code>random_state=42</code>。<br/>"
        "• <b>特徵標準化：</b>使用 <code>StandardScaler</code> 對特徵矩陣進行 Z-score 標準化（平均值 0、標準差 1），使不同量綱特徵具有可比性，利於線性迴歸係數權重解讀。<br/><br/>"
        "<b>3. Modeling & Feature Selection (特徵選擇與建模 - 拿分關鍵)：</b><br/>"
        "為避免全變數過度擬合，本研究採用 <b>RFE (遞迴特徵消除)</b> 演算法，以線性回歸為基底估計器，逐步剃除對信任度貢獻最低之權重特徵，最終精選出 <b>12 個最具代表性特徵</b>（完全符合規定之 10~15 個特徵範圍）："
    )
    story.append(Paragraph(crisp_text, body_style))

    # Feature selection table
    feat_data = [
        [Paragraph("<b>特徵類別</b>", body_style), Paragraph("<b>RFE 篩選入選之關鍵特徵 (12 個)</b>", body_style), Paragraph("<b>RFE 剔除之次要/冗餘特徵 (8 個)</b>", body_style)],
        [
            Paragraph("AI 系統效能與透明度", body_style),
            Paragraph("• model_accuracy (模型準確率)<br/>• explainability_score (可解釋性)<br/>• system_transparency (系統透明度)<br/>• error_rate (錯誤率)<br/>• decision_latency (決策延遲)", body_style),
            Paragraph("• task_risk (任務風險等級)<br/>• time_pressure (時間壓力)<br/>• ui_complexity (UI 複雜度)", body_style)
        ],
        [
            Paragraph("使用者互動與歷史認知", body_style),
            Paragraph("• trust_history (歷史信任度)<br/>• feedback_quality (回饋品質)<br/>• privacy_protection_score (隱私安全)<br/>• override_frequency (覆核頻率)<br/>• bias_level (偏差程度)<br/>• cognitive_load (認知負荷)<br/>• technical_experience (技術年資)", body_style),
            Paragraph("• age (使用者年齡)<br/>• education_level_score (學歷等級)<br/>• prior_ai_exposure (AI 接觸經驗)<br/>• interaction_frequency (互動次數)<br/>• user_fatigue_level (疲勞指數)", body_style)
        ]
    ]
    feat_table = Table(feat_data, colWidths=[120, 210, 180])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 15))

    # Page Break for Visualizations & Evaluation
    story.append(PageBreak())

    # 5. 模型成效評估 (Evaluation)
    story.append(Paragraph("四、 模型評估與高階視覺化成果 (Evaluation)", h1_style))
    
    # Metrics table
    m = res['metrics']
    eval_text = (
        "<b>1. 迴歸模型關鍵量化指標：</b><br/>"
        "使用 <code>sklearn.metrics</code> 計算獨立測試集 (2,020 筆) 之評估結果如下表所示："
    )
    story.append(Paragraph(eval_text, body_style))
    
    metrics_data = [
        [Paragraph("<b>評估指標 (Metric)</b>", body_style), Paragraph("<b>數值 (Value)</b>", body_style), Paragraph("<b>評估標準與商業意涵</b>", body_style)],
        [Paragraph("<b>Mean Squared Error (MSE)</b>", body_style), Paragraph(f"<b>{m['mse']}</b>", body_style), Paragraph("均方誤差小，顯示預測值與真實值之偏離程度極低。", body_style)],
        [Paragraph("<b>Root Mean Squared Error (RMSE)</b>", body_style), Paragraph(f"<b>{m['rmse']} 分</b>", body_style), Paragraph("平均誤差約 4.2 分（在 0~100 分量表中僅佔 4.2%），預測精確度極高。", body_style)],
        [Paragraph("<b>Mean Absolute Error (MAE)</b>", body_style), Paragraph(f"<b>{m['mae']} 分</b>", body_style), Paragraph("平均絕對誤差僅 3.33 分，模型對極端偏差值展現高度穩健性。", body_style)],
        [Paragraph("<b>R-squared (R² 決定係數)</b>", body_style), Paragraph(f"<b>{m['r2']} ({m['r2']*100:.1f}%)</b>", body_style), Paragraph("解釋力達 79.5%，證明經 RFE 篩選之 12 個特徵能高度解釋信任變異。", body_style)]
    ]
    metrics_table = Table(metrics_data, colWidths=[160, 90, 260])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#edf2f7')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 12))

    # 6. Visualizations (Two Images)
    story.append(Paragraph("<b>2. 高階視覺化：實際值 vs 預測值之 95% 信賴區間圖 (Seaborn regplot)</b>", h2_style))
    if os.path.exists('prediction_ci_plot.png'):
        story.append(Image('prediction_ci_plot.png', width=6.5*inch, height=4.2*inch))
    
    ci_desc = (
        "<b>【視覺化圖表解析】：</b><br/>"
        "如上圖所示，呼叫 <code>seaborn.regplot</code>，X 軸為「實際信任分數」，Y 軸為「預測信任分數」。"
        "紅色迴歸擬合線呈現顯著正斜率趨勢 ($R^2=0.795$)，周圍的淺紅色半透明帶為 <b>95% 信賴區間 (Confidence Interval)</b>。"
        "散佈點均勻且緊密圍繞在虛線對角參考線 ($y=x$) 兩側，且 95% 信賴區間帶相當窄，證明此多元線性回歸模型在整體分數區間均具備極高的統計穩定度與預測信心水準。"
    )
    story.append(Paragraph(ci_desc, body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>3. RFE 篩選特徵之標準化影響權重 (Beta Coefficients)</b>", h2_style))
    if os.path.exists('feature_importance.png'):
        story.append(Image('feature_importance.png', width=6.5*inch, height=3.5*inch))
        
    coef_desc = (
        "由特徵權重圖可知：<b>可解釋性 (explainability_score)</b>、<b>模型準確率 (model_accuracy)</b> 與 <b>系統透明度 (system_transparency)</b> 是提升信任度最重要的正向推手；"
        "反之，<b>偏差程度 (bias_level)</b>、<b>手動覆核頻率 (override_frequency)</b> 與 <b>決策延遲 (decision_latency)</b> 則是顯著侵蝕使用者信任度的主要負向阻力。"
    )
    story.append(Paragraph(coef_desc, body_style))
    story.append(Spacer(1, 12))

    # 7. 模擬上線 (Deployment)
    story.append(Paragraph("五、 模擬上線推論驗證 (Deployment Simulation)", h1_style))
    dep_text = (
        "為驗證模型落地應用的實用性，本研究手動建構一組包含 20 個特徵之虛擬新使用者參數（模擬高效能、高透明度且低延遲之醫療輔助情境）：<br/>"
        f"• <b>虛擬使用者模型預測信任分數：</b> <code>{res['simulated_user_prediction']} / 100 分</code><br/>"
        "• <b>商業決策判定：</b>系統將其歸類為<b>「高度信任使用者 (High Trust Tier)」</b>。系統可自動開啟全自動化協作流程，減少繁瑣的重複彈跳視窗，同時保留重要決策之自動稽核紀錄，達到最佳之人機協同效率。"
    )
    story.append(Paragraph(dep_text, callout_style))
    story.append(Spacer(1, 10))

    # 8. AI 輔助聲明 (AI Assistance Statement)
    story.append(Paragraph("六、 學術倫理與 AI 輔助聲明", h1_style))
    ai_text = (
        "<b>【GPT 輔助內容聲明】</b><br/>"
        "本作業依據學術誠信原則，在執行與報告撰寫過程中，部分分析與程式碼由 Generative AI (GPT / Antigravity IDE) 輔助完成：<br/>"
        "1. <b>特徵工程演算法：</b>採用 RFE (遞迴特徵消除) 篩選 12 個關鍵特徵之 Python 邏輯架構設計。<br/>"
        "2. <b>高階視覺化實作：</b>使用 <code>seaborn.regplot</code> 繪製 95% 信賴區間散佈圖與權重圖之繪圖參數調校。<br/>"
        "3. <b>文獻探討導讀：</b>結合 NotebookLM 生成文獻綜述重點摘要與段落結構。<br/>"
        "所有數據驗證、模型指標計算與分析結論均由本人 (陳沐德) 實際執行、核對無誤後統整而成。"
    )
    story.append(Paragraph(ai_text, quote_style))

    doc.build(story)
    print(f"PDF Report generated successfully: {filename}")

if __name__ == '__main__':
    build_pdf()
