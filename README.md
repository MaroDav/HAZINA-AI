# HAZINA-AI
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import shap
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ========================
# 1. Synthetic Data Generation with LMIC Patterns
# ========================
def generate_lmic_data(num_samples=5000):
    np.random.seed(42)
    
    # Base features
    data = pd.DataFrame({
        'stock_last_week': np.random.randint(10, 100, num_samples),
        'dispensed_last_week': np.clip(np.random.lognormal(2, 0.3, num_samples).astype(int),  # Right-skewed dispensing
        'days_since_delivery': np.random.choice([7, 14, 21], num_samples, p=[0.6, 0.3, 0.1]),  # Irregular deliveries
        'epidemic_flag': np.random.binomial(1, 0.1, num_samples)  # Disease outbreak indicator
    })
    
    # LMIC-specific patterns
    data['stock_today'] = (
        data['stock_last_week'] 
        - data['dispensed_last_week'] * (1 + 0.5*data['epidemic_flag'])  # 50% faster dispensing during outbreaks
        - data['days_since_delivery'] * np.random.uniform(0.7, 1.3, num_samples)
        + np.random.normal(0, 2, num_samples)
    )
    
    # Stockout risk labels
    data['stockout_risk'] = np.where(
        (data['stock_today'] < 15) | (data['days_since_delivery'] > 14), 1, 0)
    
    # Introduce anomalies (5% of data)
    anomaly_mask = np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
    data.loc[anomaly_mask.astype(bool), 'dispensed_last_week'] *= 3  # Artificial diversion patterns
    
    return data

# ========================
# 2. Feature Engineering Pipeline
# ========================
def engineer_features(df):
    # Critical LMIC features
    df['consumption_rate'] = df['dispensed_last_week'] / (df['days_since_delivery'] + 1e-6)
    df['delivery_risk'] = df['days_since_delivery'] ** 1.5  # Non-linear risk scaling
    df['epidemic_consumption'] = df['epidemic_flag'] * df['dispensed_last_week']
    
    # Temporal patterns
    df['week_of_year'] = np.random.randint(1, 53, len(df))  # Simulate seasonal effects
    return df

# ========================
# 3. Anomaly Detection System
# ========================
class StockAnomalyDetector:
    def __init__(self):
        self.iso_forest = IsolationForest(contamination=0.07)
        self.scaler = StandardScaler()
        
    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
    
    def predict(self, X):
        return self.iso_forest.predict(self.scaler.transform(X))

# ========================
# 4. Probabilistic Prediction Model (TF)
# ========================
def build_stock_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tfp.layers.DenseVariational(1 + 1,  # Mean + variance
            make_prior_fn=lambda t: tfp.distributions.Normal(0, 1),
            make_posterior_fn=lambda t: tfp.distributions.Independent(
                tfp.distributions.Normal(tf.Variable(tf.random.normal([t, 1])),
                tf.Variable(tf.random.normal([t, 1]))), 1)),
        tfp.layers.DistributionLambda(lambda t: 
            tfp.distributions.Normal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(t[..., 1:])))
    ])
    
    model.compile(optimizer=tf.optimizers.Adam(0.005),
                  loss=lambda y, p_y: -p_y.log_prob(y))
    return model

# ========================
# 5. Explainability Module
# ========================
class ModelExplainer:
    def __init__(self, model, background_data):
        self.explainer = shap.DeepExplainer(model, background_data)
    
    def explain(self, sample):
        return self.explainer.shap_values(sample[np.newaxis, :])[0]

# ========================
# 6. Active Learning System
# ========================
class ActiveLearningManager:
    def __init__(self, model):
        self.model = model
        self.uncertainty_threshold = 0.15
        
    def get_uncertain_samples(self, X, n=5):
        preds = self.model(X).stddev().numpy()
        return X[np.argsort(preds)[-n:]]
    
    def human_review(self, samples):
        # Integrate with SMS-based human verification
        pass

# ========================
# 7. End-to-End Pipeline
# ========================
def main():
    # Generate and preprocess data
    df = generate_lmic_data(5000)
    df = engineer_features(df)
    
    # Split features
    features = ['stock_last_week', 'dispensed_last_week', 'days_since_delivery',
                'consumption_rate', 'delivery_risk', 'epidemic_consumption']
    X = df[features]
    y = df['stock_today'].values
    stockout_risk = df['stockout_risk'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test, risk_train, risk_test = train_test_split(
        X, y, stockout_risk, test_size=0.2, stratify=stockout_risk)
    
    # Anomaly detection
    anomaly_detector = StockAnomalyDetector()
    anomaly_detector.fit(X_train)
    anomalies = anomaly_detector.predict(X_train)
    
    # Remove anomalies from training
    clean_mask = anomalies != -1
    X_train_clean, y_train_clean = X_train[clean_mask], y_train[clean_mask]
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train model
    model = build_stock_model((X_train_scaled.shape[1],))
    model.fit(X_train_scaled, y_train_clean, epochs=150, verbose=0)
    
    # Explainability
    explainer = ModelExplainer(model, X_train_scaled[:100])
    sample_explanation = explainer.explain(X_test_scaled[0])
    
    # Active learning
    al_manager = ActiveLearningManager(model)
    uncertain_samples = al_manager.get_uncertain_samples(X_test_scaled)
    
    # Edge deployment
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Generate alerts
    test_pred = model(X_test_scaled[:1])
    mean_pred = test_pred.mean().numpy()[0]
    std_pred = test_pred.stddev().numpy()[0]
    
    alert = ""
    if mean_pred < 10:
        alert += "CRITICAL STOCKOUT RISK! "
    if std_pred > 5:
        alert += "High uncertainty - verify physical stock!"
    
    print(f"Prediction: {mean_pred:.1f} ± {std_pred:.1f}")
    print("Alerts:", alert)
    print("SHAP Values:", sample_explanation)

if __name__ == "__main__":
    main()

