import pandas as pd
import optuna
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

def train_svm(X_train, y_train, X_test, y_test, use_optuna=False, use_grid_search=False, n_trials=5):
    # Check class distribution
    print("Class distribution in training data:")
    print(y_train.value_counts())
    print("\nClass distribution in test data:")
    print(y_test.value_counts())

    # Balance the dataset
    print("\nBalancing the dataset...")
    min_samples_per_class = y_train.value_counts().min()

    if min_samples_per_class > 1:
        # Apply SMOTE only if all classes have at least 2 samples
        smote_k = max(1, min(5, min_samples_per_class - 1))
        print(f"Applying SMOTE with k_neighbors={smote_k}...")
        smote = SMOTE(random_state=42, k_neighbors=smote_k)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    else:
        # Use Random Oversampling if some classes have only 1 sample
        print("SMOTE not possible (some classes have ≤ 1 sample). Using Random Oversampling instead.")
        oversampler = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)

    print("Class distribution after balancing:")
    print(pd.Series(y_train_balanced).value_counts())

    # Standardize the data
    scaler = StandardScaler()
    X_train_balanced = scaler.fit_transform(X_train_balanced)
    X_test = scaler.transform(X_test)

    results = []  # Store all tested parameters

    # -----------------------------
    # Optuna Hyperparameter Tuning
    # -----------------------------
    if use_optuna:
        print("\nPerforming Optuna Optimization for best parameters...")

        def objective(trial):
            C = trial.suggest_float('C', 0.1, 10, log=True) 
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            kernel = trial.suggest_categorical('kernel', ['rbf'])

            model = SVC(C=C, gamma=gamma, kernel=kernel)
            score = cross_val_score(model, X_train_balanced, y_train_balanced, cv=2, scoring='accuracy').mean()
            
            results.append({'C': C, 'gamma': gamma, 'kernel': kernel, 'accuracy': score})
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print("Best parameters found:", best_params)

        model = SVC(**best_params)

    # -----------------------------
    # Grid Search Hyperparameter Tuning
    # -----------------------------
    elif use_grid_search:
        print("\nPerforming Randomized Grid Search for best parameters...")

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
        }

        grid_search = RandomizedSearchCV(SVC(), param_grid, cv=2, n_jobs=-1, verbose=3, n_iter=10)
        grid_search.fit(X_train_balanced, y_train_balanced)

        print("Best parameters found:", grid_search.best_params_)
        model = grid_search.best_estimator_

    # -----------------------------
    # Default SVM Model (No Tuning)
    # -----------------------------
    else:
        print("\nUsing Default SVM Parameters (C=100, kernel=rbf, gamma=scale)")
        model = SVC(kernel='rbf', C=100, gamma='scale')

    # Train model
    model.fit(X_train_balanced, y_train_balanced)

    # Cross-validation
    # print("\nPerforming Cross-Validation...")
    # cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5) #optional
    # print("Cross-validation scores:", cv_scores)
    # print("Mean CV score:", cv_scores.mean())

    # Make predictions
    predictions = model.predict(X_test)

    # Print all tested hyperparameter combinations
    if use_optuna:
        df_results = pd.DataFrame(results)
        print("\nAll Optuna Tested Parameter Combinations:")
        print(df_results)


    # Generate classification report
    report = classification_report(y_test, predictions, output_dict=True)
    return predictions, report, model
