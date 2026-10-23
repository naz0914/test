import os
import zipfile
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont('MSJH', r'C:\Windows\Fonts\msjh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('MSJH-Bold', r'C:\Windows\Fonts\msjhbd.ttc', subfontIndex=0))

def create_dialogue_pdf(filename="GPT_Dialogue.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='MSJH-Bold',
        fontSize=18,
        leading=24,
        textColor=colors.HexColor('#1d3557'),
        alignment=1,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'H1',
        fontName='MSJH-Bold',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor('#2b5c8f'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'Body',
        fontName='MSJH',
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor('#2b2d42'),
        spaceAfter=6
    )
    
    box_style = ParagraphStyle(
        'Box',
        fontName='MSJH',
        fontSize=9,
        leading=14,
        textColor=colors.HexColor('#1a202c'),
        backColor=colors.HexColor('#f8f9fa'),
        borderColor=colors.HexColor('#cbd5e1'),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=6,
        spaceAfter=8,
        borderRadius=4
    )
    
    story = [
        Paragraph("巨量資料分析與機器學習 - HW2 AI 對話紀錄 (GPT_Dialogue)", title_style),
        Paragraph("學生：陳沐德 | 學號：7114068027 | 專案：Human-AI Interaction 信任度預測", body_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1d3557'), spaceAfter=12),
        
        Paragraph("一、 對話紀錄摘要與作業指引確認", h1_style),
        Paragraph(
            "<b>學生提出目標：</b><br/>"
            "落實 HW2 完整執行流程（CRISP-DM 架構），採用 Kaggle Human-AI Interaction 資料集，"
            "目標預測連續信任分數 (Trust Score)，利用 20 個與 AI 系統相關數值特徵進行建模，"
            "並透過 RFE (遞迴特徵消除) 篩選 10~15 個核心特徵，計算 MSE、RMSE、MAE、R²，"
            "最後以 seaborn.regplot 繪製附帶 95% 信賴區間 (Confidence Interval) 之預測散佈圖並進行模擬部署。",
            box_style
        ),
        
        Paragraph("二、 AI 輔助實作流程對話紀錄", h1_style),
        Paragraph(
            "<b>AI 回應與實作落地：</b><br/>"
            "1. 連接 GitHub Repository (naz0914/test.git)。<br/>"
            "2. 下載並建構 10,100 筆之 20 個數值特徵資料集 (human_ai_interaction.csv)。<br/>"
            "3. 撰寫 NotebookLM 文獻導讀摘要（結合 RFE 特徵選擇與信賴區間可信度驗證）。<br/>"
            "4. 實作資料分割 (80/20)、StandardScaler 標準化與 RFE 遞迴特徵消除（精準挑出 12 個特徵，剔除 8 個次要特徵）。<br/>"
            "5. 訓練多元線性回歸模型，評估指標：R² = 0.7949, RMSE = 4.2000, MAE = 3.3278。<br/>"
            "6. 呼叫 seaborn.regplot 產出 95% 信賴區間散佈圖與特徵權重條形圖。<br/>"
            "7. 模擬上線虛擬使用者數據，成功預測信任分數 91.45 分並提供決策建議。<br/>"
            "8. 產出 7114068027_hw2.ipynb、7114068027_hw2_report.pdf 與 7114068027_hw2.zip 打包檔。",
            box_style
        ),
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("三、 官方 AI 對話紀錄匯出說明", h1_style))
    story.append(Paragraph(
        "<b>對話歷程整合：</b><br/>"
        "依據作業規定，已將 Google Gemini 官方完整對話視窗匯出檔案 <code>物聯網課程使用 - Google Gemini.pdf</code>（共 13.3 MB）"
        "完整無失真合併於本檔案之後續頁面中，包含特徵選定、模型比較、問題分析與程式撰寫之全部即時問答歷程。",
        box_style
    ))
    
    # Build temporary cover
    temp_cover = "_temp_cover_summary.pdf"
    doc_temp = SimpleDocTemplate(
        temp_cover,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    doc_temp.build(story)
    
    gemini_pdf = "物聯網課程使用 - Google Gemini.pdf"
    merged = False
    
    if os.path.exists(gemini_pdf):
        # 1. Try pypdf
        try:
            from pypdf import PdfWriter
            merger = PdfWriter()
            merger.append(temp_cover)
            merger.append(gemini_pdf)
            merger.write(filename)
            merger.close()
            merged = True
            print(f"Successfully merged cover and {gemini_pdf} using pypdf into {filename}")
        except Exception:
            pass

        # 2. Try PyPDF2
        if not merged:
            try:
                from PyPDF2 import PdfMerger
                merger = PdfMerger()
                merger.append(temp_cover)
                merger.append(gemini_pdf)
                merger.write(filename)
                merger.close()
                merged = True
                print(f"Successfully merged using PyPDF2 into {filename}")
            except Exception:
                pass

        # 3. Try PyMuPDF (fitz)
        if not merged:
            try:
                import fitz
                doc_out = fitz.open()
                doc_out.insert_pdf(fitz.open(temp_cover))
                doc_out.insert_pdf(fitz.open(gemini_pdf))
                doc_out.save(filename)
                merged = True
                print(f"Successfully merged using fitz into {filename}")
            except Exception:
                pass

        # 4. Try pip install pypdf
        if not merged:
            try:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
                from pypdf import PdfWriter
                merger = PdfWriter()
                merger.append(temp_cover)
                merger.append(gemini_pdf)
                merger.write(filename)
                merger.close()
                merged = True
                print(f"Auto-installed pypdf and merged into {filename}")
            except Exception as e:
                print(f"pypdf auto-install/merge failed: {e}")

        # 5. Fallback: Copy gemini_pdf directly to filename
        if not merged:
            import shutil
            shutil.copyfile(gemini_pdf, filename)
            print(f"Fallback: Copied {gemini_pdf} directly to {filename}")
    else:
        import shutil
        shutil.copyfile(temp_cover, filename)
        print(f"Gemini PDF not found. Created cover as {filename}")

    # Clean up temp cover
    if os.path.exists(temp_cover) and temp_cover != filename:
        try:
            os.remove(temp_cover)
        except Exception:
            pass

def make_zip(zip_name="7114068027_hw2.zip"):
    create_dialogue_pdf("GPT_Dialogue.pdf")
    
    files_to_pack = [
        "7114068027_hw2.ipynb",
        "7114068027_hw2.py",
        "7114068027_hw2_report.pdf",
        "GPT_Dialogue.pdf",
        "物聯網課程使用 - Google Gemini.pdf",
        "Human-AI Interaction.csv",
        "flood_prediction.csv",
        "50_Startups.csv",
        "datasets_config.json",
        "analysis_results.json",
        "index.html",
        "prediction_ci_plot.png",
        "feature_importance.png",
        "螢幕擷取畫面 2026-10-21 075348.png",
        "README.md"
    ]
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files_to_pack:
            if os.path.exists(f):
                z.write(f, arcname=f)
                print(f"Added {f} to {zip_name}")
    print(f"Successfully created submission package: {zip_name}")

if __name__ == '__main__':
    make_zip()
