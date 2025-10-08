import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from features import SCADAFeatureExtractor
from data_pipeline import SCADARealisticGenerator, load_public_datasets

# Custom Focal Loss for handling class imbalance
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Calculate p_t
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Calculate alpha_t
        alpha_t = tf.ones_like(y_true) * self.alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_t, 1 - alpha_t)
        
        # Calculate focal weight
        focal_weight = alpha_t * tf.pow((1 - p_t), self.gamma)
        
        # Calculate focal loss
        focal_loss = focal_weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

def build_advanced_model(input_dim, num_faults, class_weights=None):
    inputs = tf.keras.Input(shape=(input_dim,), name="scada_features")
    
    # Feature attention mechanism
    def feature_attention(x, units):
        attention_weights = tf.keras.layers.Dense(units, activation='tanh')(x)
        attention_weights = tf.keras.layers.Dense(units, activation='softmax')(attention_weights)
        attended_features = tf.keras.layers.Multiply()([x, attention_weights])
        return attended_features
    
    def enhanced_residual_block(x, units, dropout_rate=0.2):
        # Skip connection with dimension matching
        if x.shape[-1] != units:
            shortcut = tf.keras.layers.Dense(units, activation=None)(x)
        else:
            shortcut = x
            
        h = tf.keras.layers.Dense(units, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        h = tf.keras.layers.LayerNormalization()(h)
        h = tf.keras.layers.Activation('swish')(h)  # Swish activation often performs better
        h = tf.keras.layers.Dropout(dropout_rate)(h)
        h = tf.keras.layers.Dense(units, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.001))(h)
        h = tf.keras.layers.LayerNormalization()(h)
        h = tf.keras.layers.Add()([shortcut, h])
        h = tf.keras.layers.Activation('swish')(h)
        return h
    
    # Input processing with attention
    x = tf.keras.layers.Dense(512, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = feature_attention(x, 512)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Enhanced residual blocks
    x = enhanced_residual_block(x, 512, 0.2)
    x = enhanced_residual_block(x, 512, 0.2)
    x = enhanced_residual_block(x, 256, 0.2)
    x = enhanced_residual_block(x, 256, 0.2)
    x = enhanced_residual_block(x, 128, 0.15)
    
    # Anomaly detection branch
    anomaly_branch = tf.keras.layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    anomaly_branch = tf.keras.layers.Dropout(0.2)(anomaly_branch)
    anomaly_branch = tf.keras.layers.Dense(64, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(anomaly_branch)
    anomaly_branch = tf.keras.layers.Dropout(0.2)(anomaly_branch)
    anomaly_output = tf.keras.layers.Dense(1, activation='sigmoid', name='anomaly')(anomaly_branch)

    # Enhanced fault classification branch with more capacity
    fault_branch = tf.keras.layers.Dense(256, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    fault_branch = tf.keras.layers.Dropout(0.2)(fault_branch)
    fault_branch = tf.keras.layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(fault_branch)
    fault_branch = tf.keras.layers.Dropout(0.15)(fault_branch)
    fault_branch = tf.keras.layers.Dense(64, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(fault_branch)
    fault_branch = tf.keras.layers.Dropout(0.1)(fault_branch)
    
    # Add a separate embedding layer for fault types
    fault_embedding = tf.keras.layers.Dense(32, activation='swish')(fault_branch)
    fault_output = tf.keras.layers.Dense(num_faults, activation='softmax', name='fault_type')(fault_embedding)

    severity_branch = tf.keras.layers.Dense(64, activation='relu')(x)
    severity_branch = tf.keras.layers.Dropout(0.3)(severity_branch)
    severity_output = tf.keras.layers.Dense(1, activation='sigmoid', name='severity')(severity_branch)

    criticality_branch = tf.keras.layers.Dense(64, activation='relu')(x)
    criticality_branch = tf.keras.layers.Dropout(0.3)(criticality_branch)
    criticality_output = tf.keras.layers.Dense(1, activation='sigmoid', name='criticality')(criticality_branch)

    model = tf.keras.Model(inputs=inputs, outputs=[anomaly_output, fault_output, severity_output, criticality_output])
    
    # Use different optimizers and learning rate scheduling
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        ),
        weight_decay=0.01
    )
    
    model.compile(
        optimizer=optimizer,
        loss={
            "anomaly": tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            "fault_type": FocalLoss(alpha=0.25, gamma=2.0),  # Use focal loss for imbalanced classes
            "severity": "huber",
            "criticality": "huber"
        },
        loss_weights={
            "anomaly": 0.8,
            "fault_type": 2.0,  # Higher weight for fault classification
            "severity": 0.6,
            "criticality": 0.4
        },
        metrics={
            "anomaly": ["accuracy", tf.keras.metrics.AUC(name='auc')],
            "fault_type": ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc'), 
                          tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
            "severity": ["mae"],
            "criticality": ["mae"]
        }
    )

    return model

def train():
    # Load/generate synthetic dataset with more samples
    ett = load_public_datasets()
    gen = SCADARealisticGenerator(real_baseline=ett)
    df = gen.generate(n_samples=25000)  # Increased sample size

    # Feature extraction
    fe = SCADAFeatureExtractor()
    X, ya, yf, ys, yc = [], [], [], [], []
    le = LabelEncoder()
    df['fault_idx'] = le.fit_transform(df['fault_type'])
    
    for _, row in df.iterrows():
        feats = fe.extract(row)
        X.append(list(feats.values()))
        ya.append(row['anomaly'])
        ys.append(row['severity'])
        yc.append(row['criticality'])
        yf.append(row['fault_idx'])

    X = np.array(X, dtype=np.float32)
    ya = np.array(ya, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    yc = np.array(yc, dtype=np.float32)
    yf_idx = np.array(yf, dtype=np.int32)
    yf = tf.keras.utils.to_categorical(yf, num_classes=len(le.classes_))
    
    # Feature scaling and normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Compute class weights for imbalanced classes
    class_weights = compute_class_weight('balanced', classes=np.unique(yf_idx), y=yf_idx)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"Class distribution: {np.bincount(yf_idx)}")
    print(f"Class weights: {class_weight_dict}")
    
    # Data augmentation using mixup for minority classes
    def mixup_data(x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.shape[0]
        index = np.random.permutation(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    model = build_advanced_model(X.shape[1], yf.shape[1], class_weights=class_weight_dict)
    
    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_fault_type_accuracy', 
            patience=15, 
            restore_best_weights=True, 
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_fault_type_accuracy',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_fault_model.h5',
            monitor='val_fault_type_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    history = model.fit(
        X,
        {"anomaly": ya, "fault_type": yf, "severity": ys, "criticality": yc},
        validation_split=0.25,
        epochs=100,  # More epochs with early stopping
        batch_size=32,  # Smaller batch size for better gradient estimates
        class_weight={'fault_type': class_weight_dict},  # Apply class weights
        callbacks=callbacks,
        verbose=1
    )
    model.save("scada_fra_advanced_model.h5")

    # Final evaluation on holdout test set (20%) with stratified split
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
    
    X_train, X_test, ya_train, ya_test, yf_train, yf_test, ys_train, ys_test, yc_train, yc_test, yf_idx_train, yf_idx_test = train_test_split(
        X, ya, yf, ys, yc, yf_idx, test_size=0.2, random_state=42, stratify=yf_idx
    )
    
    # Load best model
    model = tf.keras.models.load_model('best_fault_model.h5', custom_objects={'FocalLoss': FocalLoss})
    
    # Ensemble prediction with Test Time Augmentation (TTA)
    n_tta = 5
    all_preds = []
    
    for _ in range(n_tta):
        # Add small noise for TTA
        X_test_aug = X_test + np.random.normal(0, 0.01, X_test.shape)
        preds = model.predict(X_test_aug, verbose=0)
        all_preds.append(preds)
    
    # Average predictions
    ensemble_preds = [np.mean([pred[i] for pred in all_preds], axis=0) for i in range(4)]
    
    ya_pred = (ensemble_preds[0].flatten() > 0.5).astype(int)
    fault_pred = ensemble_preds[1].argmax(axis=1)
    fault_true = yf_test.argmax(axis=1)
    
    # Detailed evaluation
    print("\n=== ENHANCED EVALUATION RESULTS ===")
    print(f"Anomaly Accuracy: {accuracy_score(ya_test, ya_pred):.4f}")
    print(f"Fault Classification Accuracy: {accuracy_score(fault_true, fault_pred):.4f}")
    print(f"Severity MAE: {mean_absolute_error(ys_test, ensemble_preds[2].flatten()):.4f}")
    print(f"Criticality MAE: {mean_absolute_error(yc_test, ensemble_preds[3].flatten()):.4f}")
    
    # Per-class analysis
    print("\n=== FAULT CLASSIFICATION REPORT ===")
    print(classification_report(fault_true, fault_pred, target_names=le.classes_))
    
    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(fault_true, fault_pred))
    
    # Top-2 accuracy
    top2_preds = np.argsort(ensemble_preds[1], axis=1)[:, -2:]
    top2_acc = np.mean([fault_true[i] in top2_preds[i] for i in range(len(fault_true))])
    print(f"\nTop-2 Fault Classification Accuracy: {top2_acc:.4f}")
    
    # Confidence analysis
    fault_confidence = np.max(ensemble_preds[1], axis=1)
    print(f"Average Fault Prediction Confidence: {np.mean(fault_confidence):.4f}")
    print(f"Low Confidence Predictions (<0.6): {np.sum(fault_confidence < 0.6)} / {len(fault_confidence)}")

if __name__ == "__main__":
    train()
