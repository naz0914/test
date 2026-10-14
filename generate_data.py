import numpy as np
import pandas as pd

def generate_human_ai_dataset(n_samples=10100, random_seed=42):
    np.random.seed(random_seed)
    
    # 20 Features
    age = np.random.randint(18, 66, size=n_samples)
    education_level_score = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.2, 0.5, 0.25, 0.05])
    technical_experience = np.round(np.clip(np.random.normal(6.5, 4.0, size=n_samples), 0, 20), 1)
    cognitive_load = np.round(np.random.beta(2, 3, size=n_samples), 3)
    prior_ai_exposure = np.round(np.random.uniform(1.0, 5.0, size=n_samples), 2)
    
    model_accuracy = np.round(np.random.beta(7, 2, size=n_samples) * 0.5 + 0.5, 3) # 0.5 to 1.0
    explainability_score = np.round(np.random.beta(3, 2, size=n_samples), 3)
    bias_level = np.round(np.random.beta(1.5, 4, size=n_samples), 3)
    decision_latency = np.round(np.random.gamma(2, 0.8, size=n_samples) + 0.2, 2)
    decision_latency = np.clip(decision_latency, 0.2, 5.0)
    error_rate = np.round(np.clip(1.0 - model_accuracy + np.random.normal(0, 0.05, size=n_samples), 0.01, 0.5), 3)
    
    interaction_frequency = np.random.poisson(15, size=n_samples)
    ui_complexity = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.3, 0.35, 0.15, 0.05])
    feedback_quality = np.round(np.random.beta(3, 2, size=n_samples), 3)
    trust_history = np.round(np.random.beta(4, 2, size=n_samples), 3)
    override_frequency = np.random.poisson(3, size=n_samples)
    
    task_risk = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    time_pressure = np.round(np.random.beta(2, 2, size=n_samples), 3)
    system_transparency = np.round(np.random.beta(3, 2, size=n_samples), 3)
    user_fatigue_level = np.round(np.random.beta(2, 3, size=n_samples), 3)
    privacy_protection_score = np.round(np.random.beta(4, 2, size=n_samples), 3)
    
    # Ground truth formula for Trust Score (0 - 100)
    # Influential variables:
    # + model_accuracy (+25)
    # + explainability_score (+18)
    # + system_transparency (+15)
    # + trust_history (+15)
    # + feedback_quality (+10)
    # + privacy_protection_score (+8)
    # - error_rate (-20)
    # - bias_level (-14)
    # - decision_latency (-8)
    # - cognitive_load (-6)
    # - override_frequency (-1.2)
    # Weak / noise variables (will be filtered out by RFE / Lasso):
    # age, education_level_score, technical_experience, prior_ai_exposure,
    # interaction_frequency, ui_complexity, task_risk, time_pressure, user_fatigue_level
    
    base_score = (
        25.0 * (model_accuracy - 0.5) / 0.5 +
        18.0 * explainability_score +
        15.0 * system_transparency +
        15.0 * trust_history +
        10.0 * feedback_quality +
        8.0  * privacy_protection_score -
        20.0 * error_rate -
        14.0 * bias_level -
        8.0  * (decision_latency / 5.0) -
        6.0  * cognitive_load -
        1.2  * np.clip(override_frequency, 0, 10) +
        0.5  * (technical_experience / 20.0) -
        0.5  * (user_fatigue_level) +
        np.random.normal(0, 4.2, size=n_samples) # Realistic noise
    )
    
    # Scale and bound to 0 ~ 100
    trust_score = np.clip(base_score + 18.0, 5.0, 99.5)
    trust_score = np.round(trust_score, 2)
    
    df = pd.DataFrame({
        'age': age,
        'education_level_score': education_level_score,
        'technical_experience': technical_experience,
        'cognitive_load': cognitive_load,
        'prior_ai_exposure': prior_ai_exposure,
        'model_accuracy': model_accuracy,
        'explainability_score': explainability_score,
        'bias_level': bias_level,
        'decision_latency': decision_latency,
        'error_rate': error_rate,
        'interaction_frequency': interaction_frequency,
        'ui_complexity': ui_complexity,
        'feedback_quality': feedback_quality,
        'trust_history': trust_history,
        'override_frequency': override_frequency,
        'task_risk': task_risk,
        'time_pressure': time_pressure,
        'system_transparency': system_transparency,
        'user_fatigue_level': user_fatigue_level,
        'privacy_protection_score': privacy_protection_score,
        'trust_score': trust_score
    })
    
    return df

if __name__ == '__main__':
    df = generate_human_ai_dataset()
    csv_path = 'human_ai_interaction.csv'
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated successfully! Shape: {df.shape}")
    print(f"Saved to: {csv_path}")
