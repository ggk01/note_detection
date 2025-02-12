import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
from collections import Counter
import optuna

def train_xgboost(X_train, y_train, X_test, y_test, use_grid_search=False, use_optuna=True, n_trials=5, calibrate_model = False):
    """ Train an XGBoost model with proper handling of unseen labels. """

    # **🔹 Train LabelEncoder on all unique labels from BOTH Train & Test**
    all_labels = np.unique(np.concatenate([y_train, y_test]))  # Ensure all possible labels
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)  # Fit to ALL possible labels

    # **🔹 Remove unseen labels from `y_test` (force alignment)**
    valid_mask = np.isin(y_test, np.unique(y_train))  # Keep only known labels
    X_test_filtered = X_test[valid_mask]  
    y_test_filtered = y_test[valid_mask]

    # **🔹 Encode labels while ensuring no missing classes**
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test_filtered)

    # **🔹 Verify That Train & Test Labels Match**
    train_classes = np.unique(y_train_encoded)
    test_classes = np.unique(y_test_encoded)

    print(f"Classes in Training Data: {train_classes}")
    print(f"Classes in Test Data: {test_classes}")

    # **🔹 Handle Class Imbalance**
    class_counts = Counter(y_train_encoded)
    min_samples_per_class = min(class_counts.values())

    if min_samples_per_class > 5:
        smote_k = min(5, min_samples_per_class - 1)
        print(f"Applying SMOTE with k_neighbors={smote_k}...")
        smote = SMOTE(random_state=42, k_neighbors=smote_k)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_encoded)
    else:
        print(f"SMOTE not possible (some classes have ≤ 5 samples). Using Random Oversampling instead.")
        oversampler = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train_encoded)

    # **🔹 Standardize features (train & test)**
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_filtered)

    # **🔹 Default XGBoost Model**
    xgb_base = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_lambda=1,
        reg_alpha=0.5,
        random_state=42
    )

    # -----------------------------
    # 🔍 **Optuna Hyperparameter Tuning**
    # -----------------------------
    if use_optuna:
        print("\n🔍 Performing Optuna Optimization for XGBoost...")

        results = []  # Store tested parameter combinations

        def objective(trial):
            """ Optuna objective function for hyperparameter tuning """
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 12)
            gamma = trial.suggest_float('gamma', 0, 5)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
            reg_lambda = trial.suggest_float('reg_lambda', 0, 1)

            model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                gamma=gamma,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42
            )

            score = cross_val_score(model, X_train_scaled, y_train_balanced, cv=3, scoring='accuracy').mean()

            results.append({
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'gamma': gamma,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'accuracy': score
            })

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print("🎯 Best parameters found:", best_params)

        # Train final XGBoost model with best parameters
        xgb_model = XGBClassifier(**best_params, random_state=42)

        # Print all tested hyperparameter combinations
        df_results = pd.DataFrame(results)
        print("\n📊 All Optuna Tried Hyperparameter Combinations:")
        print(df_results)

    else:
        xgb_model = xgb_base

    # -----------------------------
    # 🚀 **Train the Final Model**
    # -----------------------------
    print("Training XGBoost Model...")            
    xgb_model.fit(X_train_scaled, y_train_balanced)

    # -----------------------------
    # 📏 **Apply Calibration for Probability Outputs**
    # -----------------------------

    if calibrate_model:
        print("Applying calibration...")
        xgb_calibrated = CalibratedClassifierCV(xgb_model, method='sigmoid')
        xgb_calibrated.fit(X_train_scaled, y_train_balanced)
        model = xgb_calibrated  # Use calibrated model
    else:
        print("Skipping calibration...")
        model = xgb_model  # Use uncalibrated model

    # -----------------------------
    # 🎯 **Make Predictions**
    # -----------------------------
    y_pred_encoded = model.predict(X_test_scaled)

    # **🔹 Decode predictions back to original MIDI note values**
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # **🔹 Generate Classification Report**
    report = classification_report(y_test_filtered, y_pred, zero_division=0, output_dict=True)

    # Ensure all values in report are dictionaries (convert single float values to dictionaries)
    for key, value in report.items():
        if isinstance(value, float):
            report[key] = {"score": value}

    return y_pred, report, model
