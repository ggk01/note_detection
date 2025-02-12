import numpy as np
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

def train_lstm(X_train, y_train, X_test, y_test, use_optuna=False, n_trials=5, epochs=15, batch_size=32):
    """
    Train an LSTM model for predicting MIDI notes.
    """

    # Standardize features (important for LSTMs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for LSTM (samples, timesteps, features)
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Function to build an LSTM model
    def build_lstm(num_units=128, dropout_rate=0.2, learning_rate=0.001, num_layers=2, activation="relu", optimizer_type="adam"):
        """
        Builds and returns an LSTM model with optimized hyperparameters.
        """
        model = Sequential()
        
        # First LSTM layer with correct input shape
        model.add(LSTM(num_units, return_sequences=(num_layers > 1), input_shape=(1, X_train_scaled.shape[1])))
        model.add(Dropout(dropout_rate))

        # Additional LSTM layers
        for _ in range(num_layers - 1):
            model.add(LSTM(num_units // 2, return_sequences=False if _ == num_layers - 2 else True))
            model.add(Dropout(dropout_rate))

        # Fully connected layers
        model.add(Dense(num_units // 4, activation=activation))
        model.add(Dense(len(np.unique(y_train)), activation="softmax"))  # Multi-class classification

        # Optimizer selection
        optimizer = {
            "adam": Adam(learning_rate=learning_rate),
            "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            "sgd": tf.keras.optimizers.SGD(learning_rate=learning_rate)
        }.get(optimizer_type, Adam(learning_rate=learning_rate))

        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


        print(f"Train Shape: {X_train_reshaped.shape}, Test Shape: {X_test_reshaped.shape}")

        return model

    # -----------------------------
    # 🔍 Optuna Hyperparameter Tuning
    # -----------------------------
    if use_optuna:
        print("\n🔍 Performing Optuna Optimization for LSTM...")

        results = []  # Store all tested parameter combinations

        def objective(trial):
            """ Optuna objective function for hyperparameter tuning """
            num_units = trial.suggest_int('num_units', 64, 256)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            epochs = trial.suggest_int('epochs', 5, 30)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            activation = trial.suggest_categorical('activation', ["relu", "tanh"])
            optimizer_type = trial.suggest_categorical('optimizer_type', ["adam", "rmsprop", "sgd"])

            # Build the model with optimized parameters
            model = build_lstm(num_units, dropout_rate, learning_rate, num_layers, activation, optimizer_type)

            # Train model
            history = model.fit(
                X_train_reshaped, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            val_acc = max(history.history['val_accuracy'])  # Take best validation accuracy

            results.append({
                'num_units': num_units,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'num_layers': num_layers,
                'activation': activation,
                'optimizer_type': optimizer_type,
                'val_accuracy': val_acc
            })

            return val_acc  # Optuna maximizes validation accuracy


        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print("🎯 Best parameters found:", best_params)

        # Train final model with best parameters
        lstm_model = build_lstm(
            num_units=best_params['num_units'],
            dropout_rate=best_params['dropout_rate'],
            learning_rate=best_params['learning_rate'],
            num_layers=best_params['num_layers'],
            activation=best_params['activation'],
            optimizer_type=best_params['optimizer_type']
        )

        best_epochs = best_params['epochs']
        best_batch_size = best_params['batch_size']

    else:
        # Train default model if Optuna is not used
        lstm_model = build_lstm()
        best_epochs = epochs
        best_batch_size = batch_size

    # -----------------------------
    # 🚀 Train the Final Model
    # -----------------------------
    print("Training LSTM Model...")
    lstm_model.fit(
        X_train_reshaped, y_train,
        epochs=best_epochs, batch_size=best_batch_size,
        validation_split=0.2, verbose=2
    )

    # Make predictions
    y_pred_probs = lstm_model.predict(X_test_reshaped)  # Get probabilities
    y_pred = np.argmax(y_pred_probs, axis=1)  # Convert to class labels

    return y_pred, lstm_model