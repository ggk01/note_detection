import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE,RandomOverSampler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from collections import Counter


def train_random_forest(X_train, y_train, X_test, y_test, use_optuna = False, use_grid_search=False, n_trials=5):
    """
    Train a Random Forest Classifier on MIDI note prediction.

    Args:
        X_train, y_train: Training features & labels
        X_test, y_test: Test features & labels
        use_grid_search (bool): Whether to perform hyperparameter tuning

    Returns:
        y_pred (np.array): Predicted labels
        report (dict): Classification report
        model (RandomForestClassifier): Trained model
    """

    # Encode MIDI notes into zero-based indices
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Ensure y_test only contains known labels
    mask = np.isin(y_test, label_encoder.classes_)
    X_test_filtered = X_test[mask].copy()
    y_test_filtered = y_test[mask].copy()
    y_test_encoded = label_encoder.transform(y_test_filtered)

# Find the smallest class size
    class_counts = Counter(y_train_encoded)
    min_samples_per_class = min(class_counts.values())

    # Handle class imbalance
    if min_samples_per_class > 5:
        smote_k = min(5, min_samples_per_class - 1)  # Ensure k_neighbors is valid
        print(f"Applying SMOTE with k_neighbors={smote_k}...")
        smote = SMOTE(random_state=42, k_neighbors=smote_k)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_encoded)
    else:
        print(f"SMOTE not possible (some classes have ≤ 5 samples). Using Random Oversampling instead.")
        oversampler = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train_encoded)


    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_filtered)

    # Default Random Forest model
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)

    # -----------------------------
    # 🔍 Optuna Hyperparameter Tuning
    # -----------------------------
    if use_optuna:
        print("\n🔍 Performing Optuna Optimization for Random Forest...")

        results = []  # Store all tested parameter combinations

        def objective(trial):
            """ Optuna objective function for hyperparameter tuning """
            n_estimators = trial.suggest_int('n_estimators', 50, 150)
            max_depth = trial.suggest_int('max_depth', 10, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )

            score = cross_val_score(model, X_train_scaled, y_train_balanced, cv=2, scoring='accuracy').mean()

            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'accuracy': score
            })

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print("🎯 Best parameters found:", best_params)

        rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

        # Print all tested hyperparameter combinations
        df_results = pd.DataFrame(results)
        print("\n📊 All Optuna Tried Hyperparameter Combinations:")
        print(df_results)

    # -----------------------------
    # 🎯 Grid Search Hyperparameter Tuning
    # -----------------------------
    elif use_grid_search:
        print("Running hyperparameter tuning with RandomizedSearchCV...")
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }
        rf_search = RandomizedSearchCV(rf_model, param_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=1)
        rf_search.fit(X_train_scaled, y_train_balanced)
        rf_model = rf_search.best_estimator_


    # -----------------------------
    # 🚀 Train the Final Model
    # -----------------------------

    print("Training Random Forest Model...")
    rf_model.fit(X_train_scaled, y_train_balanced)

    # Make predictions
    y_pred_encoded = rf_model.predict(X_test_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Generate classification report
    report = classification_report(y_test_filtered, y_pred, zero_division=0, output_dict=True)

    # Convert single float values (like 'accuracy') into dictionaries
    for key, value in report.items():
        if isinstance(value, float):
            report[key] = {"score": value}  # Wrap it inside a dictionary

    # Now, this won't fail when converting to DataFrame
    report_df = pd.DataFrame.from_dict(report, orient="index", dtype=object)

    return y_pred, report, rf_model
