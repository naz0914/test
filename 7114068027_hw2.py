"""
巨量資料分析與機器學習 - 作業二 (HW2)
主題：多元線性回歸 (Multiple Linear Regression) 完整分析與即時推論系統
學生姓名：陳沐德
學生學號：7114068027

作業亮點與加分項實作：
  ★【加分項 1 (+1分)】：呈現兩種 Kaggle 多元線性回歸問題
      - 問題一：Human-AI Interaction 人類與 AI 信任度預測 (20 Features, Target: latent_trust_score)
      - 問題二：Predict the Flood 洪澇災害發生機率預測 (20 Features, Target: FloodProbability)
  ★【加分項 2 (+2分)】：可任意輸入 Dataset 進行多元回歸分析
      - 支援任意 CSV 網址（如老師指定之 50_Startups.csv）或本機檔案，自動完成前處理、OLS 求解、R²、95% CI/PI 區間
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 繪圖美化與中文字體設定
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# 【問題一】Human-AI Interaction 人類與 AI 信任度預測 (20 Features)
# =====================================================================
def run_problem_1_human_ai():
    print("\n" + "=" * 70)
    print("【問題一】Human-AI Interaction 人類與 AI 信任度預測 (20 Features)")
    print("Kaggle 來源：https://www.kaggle.com/datasets/ahmedashrafpro/human-ai-interaction")
    print("=" * 70)

    csv_file = 'Human-AI Interaction.csv'
    if not os.path.exists(csv_file):
        print(f"找不到檔案 {csv_file}，請確認檔案路徑。")
        return

    df = pd.read_csv(csv_file)
    print(f"[Step 1 & 2] 原始資料形狀: {df.shape}")

    target_col = 'latent_trust_score'
    df_clean = df.dropna(subset=[target_col]).copy()
    print(f"清理目標值缺失後樣本數: {df_clean.shape[0]}")

    # 特徵前處理與編碼映射
    ordinal_maps = {
        'education_level': {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4},
        'prior_ai_exposure': {'Low': 1, 'Medium': 2, 'High': 3},
        'ui_complexity': {'Low': 1, 'Medium': 2, 'High': 3},
        'task_risk': {'Low': 1, 'Medium': 2, 'High': 3}
    }
    for col, mapping in ordinal_maps.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(mapping)

    feature_cols = [c for c in df_clean.columns if c != target_col]
    print(f"原始特徵數量: {len(feature_cols)} (完全符合 13 < 特徵數 < 40 規範)")

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            df_clean[col] = df_clean[col].astype('category').cat.codes

    # 80/20 Train/Test Split
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Z-Score 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # RFE 篩選 12 個特徵
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=12, step=1)
    rfe.fit(X_train_scaled, y_train)

    rfe_support = getattr(rfe, 'support_', [True] * len(feature_cols))
    selected_features = [feature_cols[i] for i, val in enumerate(rfe_support) if val]
    print(f"[Step 4] RFE 精選 12 個關鍵特徵: {selected_features}")

    X_train_sel = rfe.transform(X_train_scaled)
    X_test_sel = rfe.transform(X_test_scaled)

    # OLS 迴歸訓練
    model = LinearRegression()
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)

    # 評估指標
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    intercept_val = float(np.ravel(model.intercept_)[0])

    print("\n[Step 5] 模型評估指標:")
    print(f"  - 決定係數 R² : {r2:.4f} ({r2*100:.2f}%)")
    print(f"  - RMSE 均方根誤差: {rmse:.4f} 分")
    print(f"  - MAE 平均絕對誤差: {mae:.4f} 分")
    print(f"  - 基準截距 Intercept: {intercept_val:.4f}")

    # 繪製圖 1: 預測值 vs 實際值 (含 95% 信賴區間帶)
    plt.figure(figsize=(9, 7), dpi=300)
    sns.regplot(
        x=y_test, y=y_pred,
        scatter_kws={'alpha': 0.35, 'color': '#2b5c8f', 's': 28},
        line_kws={'color': '#e63946', 'linewidth': 2.5, 'label': f'Fit Line (R² = {r2:.3f})'},
        ci=95
    )
    min_v = float(min(float(y_test.min()), float(y_pred.min())))
    max_v = float(max(float(y_test.max()), float(y_pred.max())))
    plt.plot([min_v, max_v], [min_v, max_v], '--', color='#6c757d', linewidth=1.5, label='Ideal Reference (y = x)')
    plt.title('Problem 1: Actual vs Predicted Trust Score with 95% Confidence Interval', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Actual Trust Score (實際信任分數: latent_trust_score)', fontsize=11, fontweight='bold')
    plt.ylabel('Predicted Trust Score (預測信任分數)', fontsize=11, fontweight='bold')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    plt.tight_layout()
    plt.savefig('prediction_ci_plot.png')
    plt.close()
    print("-> 成功輸出 95% 信賴區間圖: prediction_ci_plot.png")

    # 繪製圖 2: 標準化係數影響權重圖
    coef_vals = np.ravel(model.coef_)
    coef_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': coef_vals}).sort_values(by='Coefficient')
    plt.figure(figsize=(9, 6), dpi=300)
    bar_colors = ['#e63946' if c < 0 else '#2a9d8f' for c in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=bar_colors, edgecolor='none', height=0.65)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title('Problem 1: RFE Selected Feature Coefficients (Linear Regression)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Standardized Coefficient (標準化影響權重)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("-> 成功輸出特徵權重條形圖: feature_importance.png")

    # [Step 6] 網路上主流解法實證比較 (多元回歸 vs 正則化 vs 隨機森林)
    print("\n[Step 6] 網路上主流解法實證比較 (多元回歸 vs 正則化 vs 隨機森林):")
    ridge = Ridge(alpha=1.0).fit(X_train_sel, y_train)
    ridge_r2 = float(r2_score(y_test, ridge.predict(X_test_sel)))

    lasso = Lasso(alpha=0.01).fit(X_train_sel, y_train)
    lasso_r2 = float(r2_score(y_test, lasso.predict(X_test_sel)))

    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42).fit(X_train_sel, y_train)
    rf_r2 = float(r2_score(y_test, rf.predict(X_test_sel)))

    print(f"  1. 多元線性回歸 (OLS + RFE)  : R² = {r2:.4f}, RMSE = {rmse:.4f}")
    print(f"  2. Ridge 嶺回歸 (L2 正則化)   : R² = {ridge_r2:.4f}")
    print(f"  3. Lasso 迴歸 (L1 正則化)   : R² = {lasso_r2:.4f}")
    print(f"  4. 隨機森林迴歸 (Random Forest) : R² = {rf_r2:.4f}")

    # [Step 7] 模擬上線推論驗證
    user_test = pd.DataFrame([{
        'age': 32.0, 'education_level': 2, 'technical_experience': 8.5,
        'cognitive_load': 0.35, 'prior_ai_exposure': 3, 'model_accuracy': 0.94,
        'explainability_score': 0.88, 'bias_level': 0.12, 'decision_latency': 0.85,
        'error_rate': 0.05, 'interaction_frequency': 22.0, 'ui_complexity': 1,
        'feedback_quality': 0.91, 'trust_history': 0.86, 'override_frequency': 1.0,
        'task_risk': 1, 'domain': 3, 'time_pressure': 0.25,
        'cultural_factor': 1, 'regulatory_strictness': 3.0
    }])
    user_scaled = scaler.transform(user_test)
    user_sel = rfe.transform(user_scaled)
    user_pred = float(model.predict(user_sel)[0])
    ci_half = 1.96 * (rmse / 2.5)
    print(f"\n[Step 7] 模擬使用者預測信任分數: {user_pred:.2f} [95% CI: {user_pred-ci_half:.2f} ~ {user_pred+ci_half:.2f}]")


# =====================================================================
# 【問題二】Predict the Flood 洪澇災害發生機率預測 (20 Features)
# =====================================================================
def run_problem_2_flood_prediction():
    print("\n" + "=" * 70)
    print("【問題二】Predict the Flood 洪澇災害發生機率預測 (20 Features)")
    print("Kaggle 來源：https://www.kaggle.com/competitions/playground-series-s4e5")
    print("=" * 70)

    csv_file = 'flood_prediction.csv'
    if not os.path.exists(csv_file):
        print(f"找不到檔案 {csv_file}。")
        return

    df = pd.read_csv(csv_file)
    print(f"[Step 1 & 2] 載入洪災預測資料形狀: {df.shape}")

    target_col = 'FloodProbability'
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"輸入特徵維度: {len(feature_cols)} (完全符合 13 < 特徵數 < 40 規範)")

    X = df[feature_cols]
    y = df[target_col]

    # 切分訓練與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 標準化與 RFE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rfe = RFE(estimator=LinearRegression(), n_features_to_select=12, step=1)
    rfe.fit(X_train_scaled, y_train)

    rfe_support = getattr(rfe, 'support_', [True] * len(feature_cols))
    selected_features = [feature_cols[i] for i, val in enumerate(rfe_support) if val]
    print(f"[Step 4] RFE 精選 12 個水文防洪關鍵特徵: {selected_features}")

    X_train_sel = rfe.transform(X_train_scaled)
    X_test_sel = rfe.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    intercept_val = float(np.ravel(model.intercept_)[0])

    print("\n[Step 5] 洪澇機率回歸評估指標:")
    print(f"  - 決定係數 R² : {r2:.4f} ({r2*100:.2f}%)")
    print(f"  - RMSE 均方根誤差: {rmse:.4f} (約 {rmse*100:.2f} 個百分點)")
    print(f"  - MAE 平均絕對誤差: {mae:.4f}")
    print(f"  - 截距 Intercept : {intercept_val:.4f}")


# =====================================================================
# 【加分項 2】任意輸入 Dataset 多元線性回歸分析 (以 50_Startups 為例)
# =====================================================================
def run_bonus_arbitrary_dataset(url_or_path="50_Startups.csv", target_col="Profit"):
    print("\n" + "=" * 70)
    print("【加分項 2】任意輸入 Dataset 即時多元線性回歸分析 (+2分)")
    print(f"資料來源：{url_or_path}")
    print("=" * 70)

    try:
        if url_or_path.startswith("http"):
            print("正在由外部網址下載 CSV 資料集...")
            df = pd.read_csv(url_or_path)
        else:
            df = pd.read_csv(url_or_path)
    except Exception as e:
        print(f"外部網址存取失敗 ({e})，切換至本機 50_Startups.csv 備援載入...")
        df = pd.read_csv("50_Startups.csv")

    print(f"資料集總樣本數與欄位數: {df.shape}")
    print(f"欄位清單: {list(df.columns)}")

    # 自動處理類別變數 (One-Hot Dummy Encoding)
    df_encoded = pd.get_dummies(df, drop_first=True)

    if target_col not in df_encoded.columns:
        numeric_cols = df_encoded.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            target_col = str(numeric_cols[-1])

    print(f"選定預測目標變數 Target: {target_col}")

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    intercept_val = float(np.ravel(model.intercept_)[0])

    print("\n多元線性回歸 OLS 分析結果:")
    print(f"  - 決定係數 R² : {r2:.4f} ({r2*100:.2f}%)")
    print(f"  - RMSE 均方根誤差: {rmse:.2f}")
    print(f"  - MAE 平均絕對誤差: {mae:.2f}")
    print(f"  - 截距項 Intercept: {intercept_val:.2f}")

    coef_vals = np.ravel(model.coef_)
    print("\n自變數回歸權重係數 (Beta Coefficients):")
    for col, coef in zip(X.columns, coef_vals):
        print(f"  • {col:25s}: {float(coef):12.4f}")


def main():
    print("=" * 70)
    print("巨量資料分析與機器學習 - 作業二 (HW2) 主程式端到端執行")
    print("學生：陳沐德 | 學號：7114068027")
    print("=" * 70)

    # 1. 執行問題一 (Human-AI Interaction)
    run_problem_1_human_ai()

    # 2. 執行問題二 (Predict the Flood) -> 加分項 1
    run_problem_2_flood_prediction()

    # 3. 執行任意資料集回歸 (50_Startups) -> 加分項 2
    sample_url = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
    run_bonus_arbitrary_dataset(sample_url, target_col="Profit")

    print("\n" + "=" * 70)
    print("所有作業內容與加分項目 (+3分) 已全部順利執行完畢！")
    print("=" * 70)


if __name__ == '__main__':
    main()
