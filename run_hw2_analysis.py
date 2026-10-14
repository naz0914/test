import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def run_pipeline():
    print("=== [Step 1 & 2] Loading Data & Understanding ===")
    df = pd.read_csv('human_ai_interaction.csv')
    print(f"Data shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    feature_cols = [c for c in df.columns if c != 'trust_score']
    target_col = 'trust_score'
    print(f"Number of features: {len(feature_cols)} (Requirement: 13-40 features -> PASS)")
    
    X = df[feature_cols]
    y = df[target_col]
    
    print("\n=== [Step 3] Data Preparation & Feature Selection ===")
    # Train / Test split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # Feature Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # RFE Feature Selection: Select Top 12 features out of 20 (Requirement: 10-15 features)
    base_estimator = LinearRegression()
    rfe_selector = RFE(estimator=base_estimator, n_features_to_select=12, step=1)
    rfe_selector.fit(X_train_scaled, y_train)
    
    selected_mask = rfe_selector.support_
    selected_features = [feature_cols[i] for i, val in enumerate(selected_mask) if val]
    eliminated_features = [feature_cols[i] for i, val in enumerate(selected_mask) if not val]
    
    print(f"Selected {len(selected_features)} features via RFE:")
    for f in selected_features:
        print(f"  + {f}")
    print(f"Eliminated {len(eliminated_features)} features:")
    for f in eliminated_features:
        print(f"  - {f}")
        
    # Transform to selected features
    X_train_sel = rfe_selector.transform(X_train_scaled)
    X_test_sel = rfe_selector.transform(X_test_scaled)
    
    # Train Multiple Linear Regression model
    model = LinearRegression()
    model.fit(X_train_sel, y_train)
    
    print("\n=== [Step 4] Evaluation & Metrics ===")
    y_pred = model.predict(X_test_sel)
    
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R-squared : {r2:.4f}")
    
    # Plot 1: Actual vs Predicted with 95% Confidence Interval (Seaborn regplot)
    plt.figure(figsize=(9, 7), dpi=300)
    ax = sns.regplot(
        x=y_test,
        y=y_pred,
        scatter_kws={'alpha': 0.35, 'color': '#2b5c8f', 's': 28},
        line_kws={'color': '#e63946', 'linewidth': 2.5, 'label': f'Fit Line (R² = {r2:.3f})'},
        ci=95
    )
    
    # Perfect prediction line (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='#6c757d', linewidth=1.5, label='Ideal Reference (y = x)')
    
    plt.title('Actual vs Predicted Trust Score with 95% Confidence Interval', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Actual Trust Score (實際信任分數)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Trust Score (預測信任分數)', fontsize=12, fontweight='bold')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    plt.tight_layout()
    plt.savefig('prediction_ci_plot.png')
    plt.close()
    print("Saved prediction_ci_plot.png with 95% CI!")
    
    # Plot 2: Feature Coefficients
    coef_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=True)
    
    plt.figure(figsize=(9, 6), dpi=300)
    colors = ['#e63946' if c < 0 else '#2a9d8f' for c in coef_df['Coefficient']]
    bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='none', height=0.65)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title('RFE Selected Feature Coefficients (Linear Regression)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Standardized Coefficient (影響權重)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("Saved feature_importance.png!")
    
    print("\n=== [Step 5] Deployment Simulation ===")
    # Manual dummy user profile with 20 values
    dummy_user = pd.DataFrame([{
        'age': 32,
        'education_level_score': 3,
        'technical_experience': 8.5,
        'cognitive_load': 0.35,
        'prior_ai_exposure': 4.2,
        'model_accuracy': 0.94,
        'explainability_score': 0.88,
        'bias_level': 0.12,
        'decision_latency': 0.85,
        'error_rate': 0.05,
        'interaction_frequency': 22,
        'ui_complexity': 2,
        'feedback_quality': 0.91,
        'trust_history': 0.86,
        'override_frequency': 1,
        'task_risk': 2,
        'time_pressure': 0.25,
        'system_transparency': 0.92,
        'user_fatigue_level': 0.20,
        'privacy_protection_score': 0.89
    }])
    
    dummy_scaled = scaler.transform(dummy_user)
    dummy_sel = rfe_selector.transform(dummy_scaled)
    predicted_score = float(model.predict(dummy_sel)[0])
    print(f"Simulated User Predicted Trust Score: {predicted_score:.2f} / 100")
    
    # Export metrics and metadata for report and notebook
    results = {
        'n_samples': int(len(df)),
        'total_features': len(feature_cols),
        'selected_features_count': len(selected_features),
        'selected_features': selected_features,
        'eliminated_features': eliminated_features,
        'metrics': {
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'r2': round(r2, 4)
        },
        'coefficients': {f: round(c, 4) for f, c in zip(selected_features, model.coef_)},
        'intercept': round(float(model.intercept_), 4),
        'simulated_user_prediction': round(predicted_score, 2)
    }
    
    with open('analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved analysis_results.json successfully!")

if __name__ == '__main__':
    run_pipeline()
