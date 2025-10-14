# FRA Model Training with Industrial Fault Categories
# Uses high-performance architecture with SCADA-validated features

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
import joblib
import json
import os

from validated_fra_generator import ValidatedFRAGenerator

class FocalLoss(tf.keras.losses.Loss):
    """Custom Focal Loss for handling class imbalance in fault types"""
    
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "class_weights": {str(k): float(v) for k, v in self.class_weights.items()} if self.class_weights else None
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        if config.get("class_weights"):
            config["class_weights"] = {int(k): float(v) for k, v in config["class_weights"].items()}
        return cls(**config)

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        ce = -y_true * tf.math.log(y_pred)
        
        if self.class_weights is not None:
            n_classes = tf.shape(y_true)[1]
            class_weights_tensor = tf.ones(n_classes)
            
            for cls_idx, weight in self.class_weights.items():
                class_weights_tensor = tf.tensor_scatter_nd_update(
                    class_weights_tensor,
                    [[cls_idx]],
                    [tf.cast(weight, tf.float32)]
                )
            
            weighted_ce = ce * tf.expand_dims(class_weights_tensor, 0)
        else:
            weighted_ce = ce
        
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.ones_like(y_true) * self.alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_t, 1 - alpha_t)
        focal_weight = alpha_t * tf.pow((1 - p_t), self.gamma)
        focal_loss = focal_weight * weighted_ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

def build_industrial_fra_model(input_dim, num_faults, class_weights=None):
    """Build high-performance FRA model for industrial fault detection"""
    
    inputs = tf.keras.Input(shape=(input_dim,), name="fra_features")
    
    # FRA-specific feature attention mechanism
    def fra_feature_attention(x, units):
        # Separate attention for magnitude and phase features
        attention_weights = tf.keras.layers.Dense(units, activation='tanh', name='attention_tanh')(x)
        attention_weights = tf.keras.layers.Dense(units, activation='softmax', name='attention_softmax')(attention_weights)
        attended_features = tf.keras.layers.Multiply(name='attention_multiply')([x, attention_weights])
        return attended_features
    
    def enhanced_fra_block(x, units, dropout_rate=0.2, block_name="fra_block"):
        # Skip connection with dimension matching
        if x.shape[-1] != units:
            shortcut = tf.keras.layers.Dense(units, activation=None, name=f'{block_name}_shortcut')(x)
        else:
            shortcut = x
            
        h = tf.keras.layers.Dense(units, activation=None, 
                                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                name=f'{block_name}_dense1')(x)
        h = tf.keras.layers.LayerNormalization(name=f'{block_name}_norm1')(h)
        h = tf.keras.layers.Activation('swish', name=f'{block_name}_swish1')(h)
        h = tf.keras.layers.Dropout(dropout_rate, name=f'{block_name}_dropout1')(h)
        
        h = tf.keras.layers.Dense(units, activation=None,
                                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                name=f'{block_name}_dense2')(h)
        h = tf.keras.layers.LayerNormalization(name=f'{block_name}_norm2')(h)
        h = tf.keras.layers.Add(name=f'{block_name}_add')([shortcut, h])
        h = tf.keras.layers.Activation('swish', name=f'{block_name}_swish2')(h)
        return h
    
    # Input processing with FRA-specific attention
    x = tf.keras.layers.Dense(512, activation=None, 
                            kernel_regularizer=tf.keras.regularizers.l2(0.001),
                            name='input_dense')(inputs)
    x = tf.keras.layers.LayerNormalization(name='input_norm')(x)
    x = tf.keras.layers.Activation('swish', name='input_swish')(x)
    x = fra_feature_attention(x, 512)
    x = tf.keras.layers.Dropout(0.3, name='input_dropout')(x)

    # Enhanced residual blocks for FRA processing
    x = enhanced_fra_block(x, 512, 0.2, "fra_block1")
    x = enhanced_fra_block(x, 512, 0.2, "fra_block2")
    x = enhanced_fra_block(x, 256, 0.2, "fra_block3")
    x = enhanced_fra_block(x, 256, 0.2, "fra_block4")
    x = enhanced_fra_block(x, 128, 0.15, "fra_block5")
    
    # Multi-task output heads for industrial fault detection
    
    # 1. Anomaly detection (binary classification)
    anomaly_branch = tf.keras.layers.Dense(128, activation='swish', 
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                         name='anomaly_dense1')(x)
    anomaly_branch = tf.keras.layers.Dropout(0.2, name='anomaly_dropout1')(anomaly_branch)
    anomaly_branch = tf.keras.layers.Dense(64, activation='swish',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                         name='anomaly_dense2')(anomaly_branch)
    anomaly_branch = tf.keras.layers.Dropout(0.2, name='anomaly_dropout2')(anomaly_branch)
    anomaly_output = tf.keras.layers.Dense(1, activation='sigmoid', name='anomaly')(anomaly_branch)

    # 2. Industrial fault classification
    fault_branch = tf.keras.layers.Dense(256, activation='swish',
                                       kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                       name='fault_dense1')(x)
    fault_branch = tf.keras.layers.Dropout(0.2, name='fault_dropout1')(fault_branch)
    fault_branch = tf.keras.layers.Dense(128, activation='swish',
                                       kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                       name='fault_dense2')(fault_branch)
    fault_branch = tf.keras.layers.Dropout(0.15, name='fault_dropout2')(fault_branch)
    fault_branch = tf.keras.layers.Dense(64, activation='swish',
                                       kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                       name='fault_dense3')(fault_branch)
    fault_branch = tf.keras.layers.Dropout(0.1, name='fault_dropout3')(fault_branch)
    
    # Industrial fault embedding
    fault_embedding = tf.keras.layers.Dense(32, activation='swish', name='fault_embedding')(fault_branch)
    fault_output = tf.keras.layers.Dense(num_faults, activation='softmax', name='fault_type')(fault_embedding)

    # 3. Severity regression (0-1 continuous)
    severity_branch = tf.keras.layers.Dense(64, activation='relu', name='severity_dense1')(x)
    severity_branch = tf.keras.layers.Dropout(0.3, name='severity_dropout1')(severity_branch)
    severity_output = tf.keras.layers.Dense(1, activation='sigmoid', name='severity')(severity_branch)

    # 4. Criticality regression (0-1 continuous)
    criticality_branch = tf.keras.layers.Dense(64, activation='relu', name='criticality_dense1')(x)
    criticality_branch = tf.keras.layers.Dropout(0.3, name='criticality_dropout1')(criticality_branch)
    criticality_output = tf.keras.layers.Dense(1, activation='sigmoid', name='criticality')(criticality_branch)

    # Create model
    model = tf.keras.Model(
        inputs=inputs, 
        outputs=[anomaly_output, fault_output, severity_output, criticality_output],
        name="industrial_fra_transformer_model"
    )
    
    # High-performance optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=0.01
    )
    
    # Compile with industrial fault focus
    model.compile(
        optimizer=optimizer,
        loss={
            "anomaly": tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            "fault_type": FocalLoss(alpha=0.25, gamma=2.0, class_weights=class_weights),
            "severity": "huber",
            "criticality": "huber"
        },
        loss_weights={
            "anomaly": 0.8,
            "fault_type": 2.5,  # Higher weight for industrial fault classification
            "severity": 0.6,
            "criticality": 0.5
        },
        metrics={
            "anomaly": ["accuracy", tf.keras.metrics.AUC(name='auc')],
            "fault_type": ["accuracy", 
                          tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc'),
                          tf.keras.metrics.Precision(name='precision'), 
                          tf.keras.metrics.Recall(name='recall')],
            "severity": ["mae"],
            "criticality": ["mae"]
        }
    )

    return model

def train_industrial_fra_model(scada_reference_path=None):
    """Train FRA model with industrial fault categories and SCADA validation"""
    
    print("=" * 70)
    print("🏭 INDUSTRIAL FRA MODEL TRAINING WITH SCADA VALIDATION")
    print("=" * 70)
    
    # Generate validated FRA dataset
    print("📊 Generating SCADA-validated FRA dataset...")
    fra_generator = ValidatedFRAGenerator(scada_reference_path=scada_reference_path, random_seed=42)
    df = fra_generator.generate_dataset(n_samples=25000)
    
    print(f"✅ Generated {len(df)} FRA samples")
    print(f"   Industrial fault categories: {df['fault_type'].nunique()}")
    print(f"   Fault distribution: {df['fault_type'].value_counts().to_dict()}")
    
    # Prepare features and targets
    print("🔄 Preparing features and targets...")
    
    # Extract FRA features (band features + metadata)
    feature_cols = [col for col in df.columns 
                   if col not in ['fault_type', 'severity', 'anomaly', 'criticality', 
                                'transformer_type', 'validation_metadata', 'timestamp']]
    
    X = df[feature_cols].fillna(0).values
    ya = df['anomaly'].values.astype(np.float32)
    ys = df['severity'].values.astype(np.float32)
    yc = df['criticality'].values.astype(np.float32)
    
    # Encode industrial fault types
    le = LabelEncoder()
    df['fault_idx'] = le.fit_transform(df['fault_type'])
    yf_idx = df['fault_idx'].values.astype(np.int32)
    yf = tf.keras.utils.to_categorical(yf_idx, num_classes=len(le.classes_))
    
    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Compute class weights for industrial faults
    class_weights = compute_class_weight('balanced', classes=np.unique(yf_idx), y=yf_idx)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"📈 Training data prepared:")
    print(f"   Input shape: {X.shape}")
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"   Industrial fault classes: {le.classes_}")
    print(f"   Class distribution: {np.bincount(yf_idx)}")
    print(f"   Class weights: {class_weight_dict}")
    
    # Build industrial FRA model
    print("🏗️ Building industrial FRA model...")
    model = build_industrial_fra_model(X.shape[1], yf.shape[1], class_weights=class_weight_dict)
    
    print(f"   Model parameters: ~{model.count_params():,}")
    print(f"   Architecture: Feature Attention + 5 Residual Blocks + 4 Task Heads")
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_fault_type_accuracy', 
            patience=20,  # More patience for industrial fault learning
            restore_best_weights=True, 
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_industrial_fra_model.keras',  # Use modern Keras format
            monitor='val_fault_type_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_fault_type_accuracy',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            mode='max',  # FIXED: Added mode='max'
            verbose=1
        )
    ]
    
    print("🚀 Starting industrial FRA model training...")
    print("   Using FocalLoss with class weights for fault imbalance")
    
    # Train model
    history = model.fit(
        X,
        {"anomaly": ya, "fault_type": yf, "severity": ys, "criticality": yc},
        validation_split=0.25,
        epochs=120,  # More epochs for industrial fault learning
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model and artifacts
    print("💾 Saving model and preprocessing artifacts...")
    
    try:
        model.save("industrial_fra_model.keras")  # Use modern Keras format
        print("✅ Successfully saved FRA model in Keras format")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        try:
            model.save("industrial_fra_model", save_format="tf")
            print("✅ Saved FRA model in TensorFlow format")
        except Exception as e2:
            print(f"❌ Error saving in TF format: {e2}")
    
    # Save preprocessing artifacts
    joblib.dump(scaler, 'fra_scaler.pkl')
    joblib.dump(le, 'fra_fault_encoder.pkl')
    
    # Save feature metadata
    metadata = {
        'model_info': {
            'model_type': 'industrial_fra_model',
            'fault_categories': le.classes_.tolist(),
            'feature_count': len(feature_cols),
            'feature_columns': feature_cols,
            'training_samples': len(X),
            'scada_validated': scada_reference_path is not None
        },
        'fault_mapping': dict(enumerate(le.classes_)),
        'class_weights': class_weight_dict,
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    with open('fra_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Saved preprocessing artifacts:")
    print("   - fra_scaler.pkl")
    print("   - fra_fault_encoder.pkl")
    print("   - fra_model_metadata.json")
    
    # Comprehensive evaluation with TTA
    print("\n📊 Evaluating with Test Time Augmentation...")
    
    # Stratified test split
    X_train, X_test, ya_train, ya_test, yf_train, yf_test, ys_train, ys_test, yc_train, yc_test, yf_idx_train, yf_idx_test = train_test_split(
        X, ya, yf, ys, yc, yf_idx, test_size=0.2, random_state=42, stratify=yf_idx
    )
    
    # Load best model
    try:
        model = tf.keras.models.load_model('best_industrial_fra_model.keras', 
                                         custom_objects={'FocalLoss': FocalLoss})
        print("✅ Loaded best model with FocalLoss")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        print("Using current model for evaluation")
    
    # Test Time Augmentation for robust evaluation
    n_tta = 10  # More TTA iterations for industrial fault detection
    all_preds = []
    
    print(f"   Running {n_tta} TTA iterations...")
    for i in range(n_tta):
        if i % 3 == 0:
            print(f"     TTA iteration {i+1}/{n_tta}")
        X_test_aug = X_test + np.random.normal(0, 0.01, X_test.shape)
        preds = model.predict(X_test_aug, verbose=0)
        all_preds.append(preds)
    
    # Ensemble predictions
    ensemble_preds = [np.mean([pred[i] for pred in all_preds], axis=0) for i in range(4)]
    
    ya_pred = (ensemble_preds[0].flatten() > 0.5).astype(int)
    fault_pred = ensemble_preds[1].argmax(axis=1)
    fault_true = yf_test.argmax(axis=1)
    
    # Comprehensive evaluation
    print("\n" + "=" * 70)
    print("📈 INDUSTRIAL FRA MODEL EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"Anomaly Detection Accuracy: {accuracy_score(ya_test, ya_pred):.4f}")
    print(f"Industrial Fault Classification Accuracy: {accuracy_score(fault_true, fault_pred):.4f}")
    print(f"Severity Prediction MAE: {mean_absolute_error(ys_test, ensemble_preds[2].flatten()):.4f}")
    print(f"Criticality Prediction MAE: {mean_absolute_error(yc_test, ensemble_preds[3].flatten()):.4f}")
    
    print(f"\n📋 INDUSTRIAL FAULT CLASSIFICATION REPORT:")
    print(classification_report(fault_true, fault_pred, target_names=le.classes_))
    
    print(f"\n📊 CONFUSION MATRIX:")
    cm = confusion_matrix(fault_true, fault_pred)
    print(cm)
    
    # Top-2 accuracy (important for industrial applications)
    top2_preds = np.argsort(ensemble_preds[1], axis=1)[:, -2:]
    top2_acc = np.mean([fault_true[i] in top2_preds[i] for i in range(len(fault_true))])
    print(f"\nTop-2 Industrial Fault Classification Accuracy: {top2_acc:.4f}")
    
    # Confidence analysis
    fault_confidence = np.max(ensemble_preds[1], axis=1)
    print(f"Average Fault Prediction Confidence: {np.mean(fault_confidence):.4f}")
    print(f"High Confidence Predictions (>0.8): {np.sum(fault_confidence > 0.8)} / {len(fault_confidence)}")
    print(f"Low Confidence Predictions (<0.6): {np.sum(fault_confidence < 0.6)} / {len(fault_confidence)}")
    
    # Industrial fault analysis
    print(f"\n🏭 INDUSTRIAL FAULT ANALYSIS:")
    for i, fault_type in enumerate(le.classes_):
        fault_mask = fault_true == i
        if np.sum(fault_mask) > 0:
            fault_acc = accuracy_score(fault_true[fault_mask], fault_pred[fault_mask])
            fault_conf = np.mean(fault_confidence[fault_mask])
            print(f"   {fault_type}: Accuracy={fault_acc:.3f}, Avg_Confidence={fault_conf:.3f}, Samples={np.sum(fault_mask)}")
    
    print(f"\n🎉 INDUSTRIAL FRA MODEL TRAINING COMPLETE!")
    print(f"📁 Model files created:")
    print(f"   - best_industrial_fra_model.h5 or industrial_fra_model.h5")
    print(f"   - fra_scaler.pkl")
    print(f"   - fra_fault_encoder.pkl")
    print(f"   - fra_model_metadata.json")
    
    return {
        'model': model,
        'scaler': scaler,
        'encoder': le,
        'history': history,
        'metadata': metadata,
        'results': {
            'anomaly_acc': accuracy_score(ya_test, ya_pred),
            'fault_acc': accuracy_score(fault_true, fault_pred),
            'severity_mae': mean_absolute_error(ys_test, ensemble_preds[2].flatten()),
            'criticality_mae': mean_absolute_error(yc_test, ensemble_preds[3].flatten()),
            'top2_acc': top2_acc,
            'avg_confidence': np.mean(fault_confidence),
            'feature_count': len(feature_cols),
            'industrial_fault_categories': le.classes_.tolist()
        }
    }

if __name__ == "__main__":
    # Train with SCADA validation
    scada_ref_path = "scada_realistic_dataset_20251013_015900.csv"
    results = train_industrial_fra_model(scada_reference_path=scada_ref_path)