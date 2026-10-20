"""
巨量資料分析與機器學習 - 作業二 (HW2)
主題：Human-AI Interaction 人類與 AI 互動信任度預測與特徵選擇
學生姓名：陳沐德
學生學號：7114068027
執行方式：python 7114068027_hw2.py
"""

import json
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


def main():
    print("=" * 65)
    print("巨量資料分析與機器學習 - 作業二 (HW2) 主程式執行開始")
    print("學生：陳沐德 | 學號：7114068027")
    print("題目：Human-AI Interaction 人類與 AI 互動信任度預測與特徵選擇")
    print("=" * 65)

    # -------------------------------------------------------------
    # 📍 Step 1 & 2: 資料取得與理解 (Business & Data Understanding)
    # -------------------------------------------------------------
    csv_file = 'Human-AI Interaction.csv'
    print(f"\n[Step 1 & 2] 載入 Kaggle 官方資料集: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"原始資料樣本數與欄位數: {df.shape}")
    print(f"全資料集原始缺失值總數: {df.isnull().sum().sum()}")

    target_col = 'latent_trust_score'
    df_clean = df.dropna(subset=[target_col]).copy()
    print(f"清理目標值缺失後有效樣本數: {df_clean.shape[0]}")

    # -------------------------------------------------------------
    # 📍 Step 3: 資料準備與前處理 (Data Preparation)
    # -------------------------------------------------------------
    print("\n[Step 3] 執行特徵前處理與編碼映射...")
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
    print(f"原始特徵數量: {len(feature_cols)} (完全符合作業規範 13 ~ 40 個特徵要求)")

    # 填補特徵缺失值
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            df_clean[col] = df_clean[col].astype('category').cat.codes

    # 80% 訓練集 / 20% 測試集嚴格拆分
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 特徵標準化 (Z-Score)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"訓練集維度: {X_train_scaled.shape}, 測試集維度: {X_test_scaled.shape}")

    # -------------------------------------------------------------
    # 📍 Step 4: 特徵選擇與建模 (Feature Selection & Modeling)
    # -------------------------------------------------------------
    print("\n[Step 4] 執行 RFE (遞迴特徵消除) 篩選 12 個核心特徵...")
    base_estimator = LinearRegression()
    rfe_selector = RFE(estimator=base_estimator, n_features_to_select=12, step=1)
    rfe_selector.fit(X_train_scaled, y_train)

    selected_features = [feature_cols[i] for i, val in enumerate(rfe_selector.support_) if val]
    eliminated_features = [feature_cols[i] for i, val in enumerate(rfe_selector.support_) if not val]

    print(f"-> RFE 精選特徵 ({len(selected_features)} 個): {selected_features}")
    print(f"-> RFE 剔除特徵 ({len(eliminated_features)} 個): {eliminated_features}")

    X_train_sel = rfe_selector.transform(X_train_scaled)
    X_test_sel = rfe_selector.transform(X_test_scaled)

    # 訓練多元線性回歸模型
    model = LinearRegression()
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)

    # -------------------------------------------------------------
    # 📍 Step 5: 模型評估與視覺化 (Evaluation & Visualization)
    # -------------------------------------------------------------
    print("\n[Step 5] 模型量化成效指標評估:")
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"  - Mean Squared Error (MSE)       : {mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE) : {rmse:.4f} 分")
    print(f"  - Mean Absolute Error (MAE)      : {mae:.4f} 分")
    print(f"  - R-squared (R² 決定係數)         : {r2:.4f} ({r2*100:.2f}%)")

    # 繪製圖 1: 實際信任分數 vs 預測信任分數 (含 95% 信賴區間帶)
    plt.figure(figsize=(9, 7), dpi=300)
    sns.regplot(
        x=y_test,
        y=y_pred,
        scatter_kws={'alpha': 0.35, 'color': '#2b5c8f', 's': 28},
        line_kws={'color': '#e63946', 'linewidth': 2.5, 'label': f'Fit Line (R² = {r2:.3f})'},
        ci=95
    )
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='#6c757d', linewidth=1.5, label='Ideal Reference (y = x)')
    plt.title('Actual vs Predicted Trust Score with 95% Confidence Interval', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Actual Trust Score (實際信任分數: latent_trust_score)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Trust Score (預測信任分數)', fontsize=12, fontweight='bold')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    plt.tight_layout()
    plt.savefig('prediction_ci_plot.png')
    plt.close()
    print("-> 成功輸出 95% 信賴區間回歸圖: prediction_ci_plot.png")

    # 繪製圖 2: 標準化係數影響權重圖
    coef_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=True)

    plt.figure(figsize=(9, 6), dpi=300)
    bar_colors = ['#e63946' if c < 0 else '#2a9d8f' for c in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=bar_colors, edgecolor='none', height=0.65)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title('RFE Selected Feature Coefficients (Linear Regression)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Standardized Coefficient (標準化影響權重)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("-> 成功輸出特徵權重條形圖: feature_importance.png")

    # -------------------------------------------------------------
    # 📍 Step 6: 網路上主流與更優解法比較 (Benchmark & SOTA Comparison)
    # -------------------------------------------------------------
    print("\n[Step 6] 網路上主流解法實證比較 (多元回歸 vs 正則化 vs 隨機森林):")
    # Ridge
    ridge = Ridge(alpha=1.0).fit(X_train_sel, y_train)
    ridge_r2 = r2_score(y_test, ridge.predict(X_test_sel))
    # Lasso
    lasso = Lasso(alpha=0.01).fit(X_train_sel, y_train)
    lasso_r2 = r2_score(y_test, lasso.predict(X_test_sel))
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42).fit(X_train_sel, y_train)
    rf_r2 = r2_score(y_test, rf.predict(X_test_sel))

    print(f"  1. 多元線性回歸 (OLS + RFE)  : R² = {r2:.4f}, RMSE = {rmse:.4f}")
    print(f"  2. Ridge 嶺回歸 (L2 正則化)   : R² = {ridge_r2:.4f}")
    print(f"  3. Lasso 迴歸 (L1 正則化)   : R² = {lasso_r2:.4f}")
    print(f"  4. 隨機森林迴歸 (Random Forest) : R² = {rf_r2:.4f}")

    # -------------------------------------------------------------
    # 📍 Step 7: 模擬上線推論驗證 (Deployment Simulation)
    # -------------------------------------------------------------
    print("\n[Step 7] 模擬上線推論驗證 (Deployment):")
    virtual_user = pd.DataFrame([{
        'age': 32.0, 'education_level': 2, 'technical_experience': 8.5,
        'cognitive_load': 0.35, 'prior_ai_exposure': 3, 'model_accuracy': 0.94,
        'explainability_score': 0.88, 'bias_level': 0.12, 'decision_latency': 0.85,
        'error_rate': 0.05, 'interaction_frequency': 22.0, 'ui_complexity': 1,
        'feedback_quality': 0.91, 'trust_history': 0.86, 'override_frequency': 1.0,
        'task_risk': 1, 'domain': 3, 'time_pressure': 0.25,
        'cultural_factor': 1, 'regulatory_strictness': 3.0
    }])
    user_scaled = scaler.transform(virtual_user)
    user_sel = rfe_selector.transform(user_scaled)
    user_pred = float(model.predict(user_sel)[0])

    print(f"虛擬使用者預測信任分數: {user_pred:.2f} / 100 分")
    if user_pred >= 75:
        print("系統決策: 【高度信任層級】 -> 自動解鎖全自動化快速協作權限。")
    elif user_pred >= 50:
        print("系統決策: 【中度信任層級】 -> 開啟決策解釋與雙重覆核。")
    else:
        print("系統決策: 【低度信任層級】 -> 切換為保守人工主導模式。")

    print("\n" + "=" * 65)
    print("HW2 完整分析流程執行完畢！所有報表與圖檔已成功生成。")
    print("=" * 65)


if __name__ == '__main__':
    main()
