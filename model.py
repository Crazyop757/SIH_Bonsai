"""
fra_model_regularized_fixed.py

Fixed regularized FRA multi-task model with correct evaluation indexing.

- L2 weight regularization
- Increased dropout rates
- Reduced layer sizes
- k-fold cross-validation for robust evaluation
- Early stopping per fold
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(csv_path="fra_features_summary.csv"):
    """Load and preprocess the enhanced FRA synthetic dataset."""
    df = pd.read_csv(csv_path)
    
    # Feature definitions
    band_names = ["VLF", "LF", "MF", "HF"]
    band_features = [
        f"{stat}_{band}"
        for band in band_names
        for stat in ["mag_mean", "mag_std", "mag_min", "mag_max",
                     "ph_mean", "ph_std", "ph_min", "ph_max"]
    ]
    basic_features = ['mag_first', 'mag_peak', 'ph_first']
    metadata_features = ['voltage_kv', 'power_mva', 'age_years']
    
    features = [c for c in band_features + basic_features + metadata_features if c in df.columns]
    
    # One-hot encode operational environment if present
    if 'operational_env' in df.columns:
        df = pd.get_dummies(df, columns=['operational_env'], prefix='env')
        features += [c for c in df.columns if c.startswith('env_')]
    
    X = df[features].fillna(0.0).values
    X = RobustScaler().fit_transform(X)
    
    # Targets
    y_anomaly = df['anomaly'].astype(np.float32).values
    le = LabelEncoder().fit(df['fault_type'])
    y_fault = tf.keras.utils.to_categorical(le.transform(df['fault_type']))
    y_severity = df['severity'].astype(np.float32).values
    y_criticality = df['criticality'].astype(np.float32).values
    
    # Generate fault probability score (continuous confidence based on severity and fault presence)
    y_fault_prob = np.where(y_anomaly == 1, 
                           y_severity * np.random.uniform(0.8, 1.0, len(y_severity)),  # High confidence for actual faults
                           np.random.uniform(0.0, 0.2, len(y_severity)))  # Low confidence for healthy samples
    y_fault_prob = np.clip(y_fault_prob, 0.0, 1.0).astype(np.float32)
    
    return X, y_anomaly, y_fault, y_severity, y_criticality, y_fault_prob, le, features


def create_regularized_model(input_dim, num_fault_classes, l2=1e-4, dropout_rate=0.4):
    """Build a regularized multi-task model."""
    reg = tf.keras.regularizers.l2(l2)
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Anomaly head
    a = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    a = tf.keras.layers.Dropout(dropout_rate)(a)
    anomaly_out = tf.keras.layers.Dense(1, activation="sigmoid", name="anomaly")(a)
    
    # Fault type head
    f = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    f = tf.keras.layers.Dropout(dropout_rate)(f)
    fault_out = tf.keras.layers.Dense(num_fault_classes, activation="softmax", name="fault_type")(f)
    
    # Severity head
    s = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    s = tf.keras.layers.Dropout(dropout_rate)(s)
    severity_out = tf.keras.layers.Dense(1, activation="sigmoid", name="severity")(s)
    
    # Criticality head
    c = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    c = tf.keras.layers.Dropout(dropout_rate)(c)
    critical_out = tf.keras.layers.Dense(1, activation="sigmoid", name="criticality")(c)
    
    # Fault probability score head (continuous confidence score)
    p = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    p = tf.keras.layers.Dropout(dropout_rate)(p)
    fault_prob_out = tf.keras.layers.Dense(1, activation="sigmoid", name="fault_probability")(p)
    
    return tf.keras.Model(inputs, [anomaly_out, fault_out, severity_out, critical_out, fault_prob_out])


def train_and_evaluate(k=5):
    """Train with k-fold CV and report average performance."""
    X, ya, yf, ys, yc, yp, le, features = load_and_preprocess_data()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    metrics = {'an_acc': [], 'fault_acc': [], 'sev_mae': [], 'crit_mae': [], 'prob_mae': []}
    fold = 1
    
    for train_idx, test_idx in skf.split(X, ya):
        print(f"\n--- Fold {fold} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        ya_train, ya_test = ya[train_idx], ya[test_idx]
        yf_train, yf_test = yf[train_idx], yf[test_idx]
        ys_train, ys_test = ys[train_idx], ys[test_idx]
        yc_train, yc_test = yc[train_idx], yc[test_idx]
        yp_train, yp_test = yp[train_idx], yp[test_idx]
        
        model = create_regularized_model(X.shape[1], yf.shape[1])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss={
                "anomaly": "binary_crossentropy",
                "fault_type": "categorical_crossentropy",
                "severity": "huber",
                "criticality": "huber",
                "fault_probability": "mse"
            },
            loss_weights={
                "anomaly": 1.0,
                "fault_type": 1.0,
                "severity": 0.7,
                "criticality": 0.5,
                "fault_probability": 0.8
            },
            metrics={
                "anomaly": ["accuracy"],
                "fault_type": ["accuracy"],
                "severity": ["mae"],
                "criticality": ["mae"],
                "fault_probability": ["mae"]
            }
        )
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(
            X_train,
            {"anomaly": ya_train, "fault_type": yf_train, "severity": ys_train, "criticality": yc_train, "fault_probability": yp_train},
            validation_data=(X_test, {"anomaly": ya_test, "fault_type": yf_test, "severity": ys_test, "criticality": yc_test, "fault_probability": yp_test}),
            epochs=100,
            batch_size=64,
            callbacks=[es],
            verbose=0
        )
        
        # evaluate with return_dict to index by key
        res = model.evaluate(
            X_test,
            {"anomaly": ya_test, "fault_type": yf_test, "severity": ys_test, "criticality": yc_test, "fault_probability": yp_test},
            return_dict=True,
            verbose=0
        )
        an_acc = res["anomaly_accuracy"]
        fault_acc = res["fault_type_accuracy"]
        sev_mae = res["severity_mae"]
        crit_mae = res["criticality_mae"]
        prob_mae = res["fault_probability_mae"]
        
        print(f"Anomaly Acc: {an_acc:.4f}, Fault Acc: {fault_acc:.4f}, Sev MAE: {sev_mae:.4f}, Crit MAE: {crit_mae:.4f}, Prob MAE: {prob_mae:.4f}")
        
        metrics['an_acc'].append(an_acc)
        metrics['fault_acc'].append(fault_acc)
        metrics['sev_mae'].append(sev_mae)
        metrics['crit_mae'].append(crit_mae)
        metrics['prob_mae'].append(prob_mae)
        fold += 1
    
    # Average metrics
    print("\n=== Cross-Validation Results ===")
    print(f"Avg Anomaly Acc:     {np.mean(metrics['an_acc']):.4f}")
    print(f"Avg Fault Acc:       {np.mean(metrics['fault_acc']):.4f}")
    print(f"Avg Sev MAE:         {np.mean(metrics['sev_mae']):.4f}")
    print(f"Avg Crit MAE:        {np.mean(metrics['crit_mae']):.4f}")
    print(f"Avg Fault Prob MAE:  {np.mean(metrics['prob_mae']):.4f}")


if __name__ == "__main__":
    train_and_evaluate()
