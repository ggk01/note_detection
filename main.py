from data_preprocessing import load_data, prepare_lstm_data
from svm_model import train_svm
from lstm_model import train_lstm
from XGBoost_Model import train_xgboost
from random_forest_model import train_random_forest
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, confusion_matrix, classification_report
import time
import sys

def top_k_accuracy(y_true, y_pred_probs, k=3):
    """ Compute Top-K Accuracy """
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]  # Get top-k predictions
    correct = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return np.mean(correct)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):


    is_lstm = isinstance(model, tf.keras.Model)

    # ✅ Ensure LSTM models receive correctly shaped 3D input
    if is_lstm:
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


    start_time = time.time()
    if not is_lstm:  # LSTMs are already trained before evaluation
        model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    predictions = model.predict(X_test)  # Make predictions
    predict_time = time.time() - start_time


    # 🎯 **Handle LSTM Predictions**
    if is_lstm:
        predictions = np.argmax(predictions, axis=1)  # Convert softmax probabilities to class labels

    # 📊 **Compute Traditional Metrics**

    # Compute Traditional Classification Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    # Handle AUC computation
    auc = None
    try:
        if hasattr(model, "predict_proba"):  # If the model supports probability estimates
            auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
        else:  
            auc = None  # Skip AUC computation for models that don't support probabilities
    except Exception as e:
        print(f"⚠️ Warning: AUC computation failed for {model_name}. Error: {e}")
        auc = None


    model_size = sys.getsizeof(model) / 1e6  # Convert bytes to MB

    # 🎵 Compute Mean Absolute Error (MAE) - Measures how far off the predicted notes are
    mae = mean_absolute_error(y_test, predictions)

    # 🎯 Compute Top-3 Accuracy (if probability outputs are available)
    top3_acc = None
    try:
        y_pred_probs = model.predict_proba(X_test)
        top3_acc = top_k_accuracy(y_test, y_pred_probs, k=3)
    except AttributeError:
        top3_acc = None  # Some models (e.g., SVM with linear kernel) don't support probability output

    # 🎼 Confusion Matrix Visualization
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted Note")
    plt.ylabel("Actual Note")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # 📝 Print Metrics
    print(f"\n🔹 {model_name} Evaluation Metrics:")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    if auc is not None:
        print(f"✅ AUC-ROC: {auc:.4f}")
    print(f"🎵 Mean Absolute Error (Note Distance): {mae:.4f}")
    if top3_acc is not None:
        print(f"🎯 Top-3 Accuracy: {top3_acc:.4f}")
    print(f"⏳ Training Time: {train_time:.4f} sec")
    print(f"⚡ Prediction Time: {predict_time:.4f} sec")
    print(f"💾 Model Size: {model_size:.2f} MB")

    # 🏆 Return Metrics Dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "mae": mae,
        "top3_accuracy": top3_acc,
        "train_time": train_time,
        "predict_time": predict_time,
        "model_size": model_size
    }


def plot_class_distribution(y_train, y_test):
    plt.figure(figsize=(12, 5))

    sns.histplot(y_train, bins=30, color='blue', alpha=0.6, label="Training Data")
    sns.histplot(y_test, bins=30, color='red', alpha=0.6, label="Test Data")
    
    plt.xlabel("MIDI Note")
    plt.ylabel("Count")
    plt.title("Distribution of MIDI Notes in Training & Test Data")
    plt.legend()
    plt.show()

# Function to convert MIDI number to note name
def midi_to_note(midi_numbers):
    note_mapping = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_names = []
    
    for num in midi_numbers:
        num = int(round(num))  # Convert float to nearest integer
        octave = (num // 12) - 1
        note = note_mapping[num % 12].replace("#", "")  # Remove sharp symbols
        note_names.append(f"{note}{octave}")
    
    return note_names

def filter_and_cluster(time, y_test, y_pred, bin_size=0.2, midi_range=(36, 96)):  
    """
    1. Removes outliers (notes outside MIDI range).
    2. Clusters notes by averaging close points in time bins.
    """
    valid_indices = (y_test >= midi_range[0]) & (y_test <= midi_range[1])
    time, y_test, y_pred = time[valid_indices], y_test[valid_indices], y_pred[valid_indices]

    # Bin the time axis (group points within close time intervals)
    time_bins = np.floor(time / bin_size) * bin_size  # Round to nearest bin_size
    unique_bins = np.unique(time_bins)

    # Compute average note per bin (reduces clutter)
    avg_test = np.array([y_test[time_bins == t].mean() for t in unique_bins])
    avg_pred = np.array([y_pred[time_bins == t].mean() for t in unique_bins])
    
    return unique_bins, avg_test, avg_pred


def filter_and_cluster_comb(time, y_test, y_pred, bin_size=0.2, midi_range=(36, 96)):  
    """
    1. Removes outliers (notes outside MIDI range).
    2. Clusters notes by averaging close points in time bins.
    """

    # ✅ Ensure all inputs have the same length
    min_len = min(len(time), len(y_test), len(y_pred))  # Find shortest array
    time, y_test, y_pred = time[:min_len], y_test[:min_len], y_pred[:min_len]

    # ✅ Create a valid index mask for MIDI note range
    valid_indices = (y_test >= midi_range[0]) & (y_test <= midi_range[1])

    # ✅ Apply mask safely
    time_filtered, y_test_filtered, y_pred_filtered = time[valid_indices], y_test[valid_indices], y_pred[valid_indices]

    # ✅ Ensure filtered arrays are still aligned
    assert len(time_filtered) == len(y_test_filtered) == len(y_pred_filtered), "Filtered arrays have mismatched lengths!"

    # Bin the time axis (group points within close time intervals)
    time_bins = np.floor(time_filtered / bin_size) * bin_size  # Round to nearest bin_size
    unique_bins = np.unique(time_bins)

    # Compute average note per bin (reduces clutter)
    avg_test = np.array([y_test_filtered[time_bins == t].mean() for t in unique_bins])
    avg_pred = np.array([y_pred_filtered[time_bins == t].mean() for t in unique_bins])
    
    return unique_bins, avg_test, avg_pred


def plot_predicted_vs_actual(time, y_test, y_pred):
    plt.figure(figsize=(18, 9))  # Wider plot for clarity


    # Convert time from seconds to minutes (rounded to 2 decimals)
    time_in_minutes = np.round(time / 60, 2)

    # Apply filtering & clustering
    time_binned, y_test_binned, y_pred_binned = filter_and_cluster(time_in_minutes, y_test, y_pred)

    # Add slight jitter to predicted values to avoid overlap
    jitter = np.random.uniform(-0.05, 0.05, size=len(y_pred_binned))  # Increase range 
    time_pred = time_binned + jitter  

    # Identify matches (exact predictions)
    match_mask = np.round(y_test_binned) == np.round(y_pred_binned)

    # Scatter plot with different colors
    plt.scatter(time_binned[~match_mask], y_test_binned[~match_mask], color="blue", s=10, alpha=0.5, label="Actual Notes")
    plt.scatter(time_pred[~match_mask], y_pred_binned[~match_mask], color="red", s=10, alpha=0.5, label="Predicted Notes")
    plt.scatter(time_binned[match_mask], y_test_binned[match_mask], color="green", s=25, alpha=0.8, label="Correct Matches")

    # **Ensure unique Y-axis labels**
    unique_midi_values = np.sort(np.unique(np.round(np.concatenate([y_test_binned, y_pred_binned]))))
    note_labels = [midi_to_note([midi])[0] for midi in unique_midi_values]
    plt.yticks(unique_midi_values, note_labels, fontsize=12)
    plt.ylim(min(unique_midi_values) - 2, max(unique_midi_values) + 2)  # Adjust Y-axis range

#####
    # **Smooth the curve using Savitzky-Golay filter**
    window_size = min(len(time_binned) // 5, 15)  # Adaptive window size
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size for Savitzky-Golay filter

    if len(time_binned) > window_size:
        smoothed_actual = savgol_filter(y_test_binned, window_size, 3)  # 3rd degree polynomial
        smoothed_predicted = savgol_filter(y_pred_binned, window_size, 3)

        # Plot smoothed curves
        plt.plot(time_binned, smoothed_actual, color="blue", linewidth=2, label="Smoothed Actual")
        plt.plot(time_binned, smoothed_predicted, color="red", linewidth=2, linestyle="dashed", label="Smoothed Predicted")
#######

    # Format X-axis to show 2 decimal places but avoid trailing zeros
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xticks(np.arange(min(time_binned), max(time_binned) + 0.1, step=0.5), fontsize=12)


    # Annotate points with note names
    y_test_labels = midi_to_note(y_test_binned)
    y_pred_labels = midi_to_note(y_pred_binned)

    labeled_notes = set()  # Track already labeled notes

    for i in range(len(time_binned)):
        note_actual = y_test_labels[i]
        note_predicted = y_pred_labels[i]

        # Label correct matches in **bold green**
        if match_mask[i] and (time_binned[i], y_test_binned[i]) not in labeled_notes:
            plt.text(time_binned[i], y_test_binned[i] + 0.5, note_actual, fontsize=10, color="green", fontweight='bold')
            labeled_notes.add((time_binned[i], y_test_binned[i]))

        # Label actual notes (if not already labeled)
        elif (time_binned[i], y_test_binned[i]) not in labeled_notes:
            plt.text(time_binned[i], y_test_binned[i] + 0.5, note_actual, fontsize=10, color="blue")
            labeled_notes.add((time_binned[i], y_test_binned[i]))

        # Label predicted notes (if not already labeled)
        if (time_pred[i], y_pred_binned[i]) not in labeled_notes:
            plt.text(time_pred[i], y_pred_binned[i] - 0.5, note_predicted, fontsize=10, color="red")
            labeled_notes.add((time_pred[i], y_pred_binned[i]))


    plt.xlabel("Time (Minutes)", fontsize=14)
    plt.ylabel("Musical Notes", fontsize=14)
    plt.title("Predicted vs Actual Notes Over Time (Filtered & Clustered)", fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()


def plot_all_models(time_dict, y_test_dict, model_predictions):
    plt.figure(figsize=(18, 9))  # Wider plot for clarity

    print("🛠 Available Time Entries:", time_dict.keys())

    first_model = next(iter(time_dict))  # Pick any model to get the ground truth y_test
    time_actual, y_test_binned, _ = filter_and_cluster(time_dict[first_model], y_test_dict[first_model], y_test_dict[first_model])

    # **Smooth the actual notes**
    window_size = min(len(time_actual) // 5, 15)
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size

    smoothed_actual = savgol_filter(y_test_binned, window_size, 3)

    # 🔵 **Plot smoothed actual notes**
    plt.plot(time_actual, smoothed_actual, color="blue", linewidth=2, label="Smoothed Actual Notes")

    # 🔴 **Loop through all models to plot predictions**
    for model_name, y_pred in model_predictions.items():
        if model_name not in time_dict:
            print(f"⚠️ Warning: No time data found for {model_name}, skipping...")
            continue

        # ✅ Ensure the time matches prediction length
        time_for_model = time_dict[model_name]
        min_len = min(len(time_for_model), len(y_test_dict[model_name]), len(y_pred))
        time_for_model, y_test_model, y_pred = time_for_model[:min_len], y_test_dict[model_name][:min_len], y_pred[:min_len]

        # ✅ Apply filtering & clustering
        time_binned, _, y_pred_binned = filter_and_cluster(time_for_model, y_test_model, y_pred)

        # ✅ Smooth predicted notes
        if len(time_binned) > window_size:
            smoothed_predicted = savgol_filter(y_pred_binned, window_size, 3)
            plt.plot(time_binned, smoothed_predicted, linestyle="dashed", linewidth=2, label=f"{model_name} (Smoothed)")

    # **Format Y-Axis with Musical Notes**
    unique_midi_values = np.sort(np.unique(np.round(np.concatenate([y_test_binned] + list(model_predictions.values())))))
    note_labels = [midi_to_note([midi])[0] for midi in unique_midi_values]
    plt.yticks(unique_midi_values, note_labels, fontsize=12)

    # **Format X-Axis for Time Readability**
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xticks(np.linspace(min(time_actual), max(time_actual), num=10), fontsize=12)

    plt.xlabel("Time (Minutes)", fontsize=14)
    plt.ylabel("Musical Notes", fontsize=14)
    plt.title("Predicted vs Actual Notes Over Time (All Models)", fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()


def main():
    # Paths to training and test files
    train_files = ["aligned_features_Bryan Adams - Summer of 69.csv", 
                   "aligned_features_Queen - I Want it All.csv",
                   "aligned_features_Livin_on_a_prayer_Bon_Jovi.csv",
                   "aligned_features_November_Rain.csv",
                   "aligned_features_Queen - Bohemian Rhapsody.csv"
                   ]
    test_file = "aligned_features_always.csv"

    # Load data
    X_train, y_train, X_test, y_test = load_data(train_files, test_file)

    # Load the test CSV to extract the Time column
    test_df = pd.read_csv(test_file)  # Read CSV
    time = test_df["Time"].values     # Extract Time column


 # Choose model
    use_lstm = True             # Set to True to use LSTM 
    use_xgboost = False          # Set to True to use XGBoost 
    use_svm = False              # Set to True to use SVM 
    use_random_forest = False    # Set to True to use Random Forest

    # Dictionary to store results
    model_results = {}
    model_predictions = {}
    time_filtered_dict = {}  # Store the corresponding filtered time arrays for each model


    # ############### Train SVM model ###############
    if use_svm:
        print("Training SVM Model...")
        svm_predictions, svm_report, svm_model = train_svm(X_train, y_train, X_test, y_test, use_grid_search=False, use_optuna=False)

        # Display classification reports
        print("\nSVM Classification Report:")
        print(pd.DataFrame(svm_report).transpose())

        # Evaluate model
        model_results["SVM"] = evaluate_model(svm_model, X_train, y_train, X_test, y_test, model_name="SVM")

        plot_predicted_vs_actual(time, y_test, svm_predictions)

        model_predictions["SVM"] = svm_predictions # for combined
        time_filtered_dict["SVM"] = time[:len(y_test)]  # No need for filtering



    # ############### Train LSTM model ###############
    if use_lstm:
        print("Training LSTM Model...")

        # Encode MIDI labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        # Filter test labels to keep only known classes
        mask = y_test.isin(label_encoder.classes_)
        X_test_filtered = X_test.loc[mask].copy()
        y_test_filtered = y_test.loc[mask].copy()

        # Reset index after filtering
        X_test_filtered.reset_index(drop=True, inplace=True)
        y_test_filtered.reset_index(drop=True, inplace=True)

        # Encode the filtered y_test
        y_test_encoded = label_encoder.transform(y_test_filtered)

        # Ensure data alignment for LSTM
        X_train, y_train_encoded, X_test_filtered, y_test_encoded = prepare_lstm_data(
            X_train, y_train_encoded, X_test_filtered, y_test_encoded
        )

        lstm_predictions, lstm_model = train_lstm(X_train, y_train_encoded, X_test_filtered, y_test_encoded, use_optuna=False)

        # Decode predictions
        lstm_predictions_decoded = label_encoder.inverse_transform(lstm_predictions)

        # Show results
        print("\nLSTM Model Predictions:")
        print(lstm_predictions_decoded)

        # ✅ Evaluate the model
        model_results["LSTM"] = evaluate_model(lstm_model, X_train, y_train_encoded, X_test_filtered, y_test_encoded, model_name="LSTM")

        # Ensure time has the same length as y_test_filtered
        time_filtered = time[:len(y_test_filtered)]

        # Plot results
        plot_predicted_vs_actual(time_filtered, y_test_filtered, lstm_predictions_decoded)
        model_predictions["LSTM"] = lstm_predictions_decoded # for combined
        time_filtered_dict["LSTM"] = time[:len(y_test_filtered)]  # Correct length



    # ############### Train XGBoost model ###############
    if use_xgboost:
        print("Using XGBoost")
        # Ensure y_test is a Pandas Series with the same index as X_test
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test, index=X_test.index)

        # Encode MIDI labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        # Filter test labels to keep only known classes
        known_classes = set(y_train_encoded)  # Get known class labels from training data
        mask = y_test.isin(label_encoder.classes_)
        X_test_filtered = X_test.loc[mask].copy()
        y_test_filtered = y_test.loc[mask].copy()

        # Reset index after filtering
        X_test_filtered.reset_index(drop=True, inplace=True)
        y_test_filtered.reset_index(drop=True, inplace=True)

        # **🔹 Ensure only known labels remain**
        y_test_filtered = y_test_filtered[y_test_filtered.isin(label_encoder.classes_)]  # Remove unknown classes

        # **Re-check if filtering removed all test data**
        if y_test_filtered.empty:
            print("⚠️ Warning: No matching test labels after filtering. Skipping XGBoost training.")
        else:
            # Encode the filtered y_test
            y_test_encoded = label_encoder.transform(y_test_filtered)

            # Train XGBoost model
            print("Training XGBoost Model...")
            xgb_predictions, xgb_report, xgb_model = train_xgboost(
                X_train, y_train_encoded, X_test_filtered, y_test_encoded, use_grid_search=False, use_optuna=True
            )

            # Load the test CSV to extract the Time column and align with filtered data
            test_df = pd.read_csv(test_file)
            time_filtered = test_df.loc[mask, "Time"].values

            # Display classification report
            print("\nXGBoost Classification Report:")
            xgb_report_df = pd.DataFrame.from_dict(xgb_report, orient="index", dtype=object)
            print(xgb_report_df.transpose())

            # 🎯 Evaluate the model
            model_results["XGBoost"] = evaluate_model(xgb_model, X_train, y_train_encoded, X_test_filtered, y_test_encoded, model_name="XGBoost")

            # 🎼 Plot predicted vs actual notes
            plot_predicted_vs_actual(time_filtered, y_test_filtered, xgb_predictions)

            # 📊 Plot class distribution
            plot_class_distribution(y_train, y_test)

            model_predictions["XGBoost"] = xgb_predictions # fro combined
            time_filtered_dict["XGBoost"] = time[:len(y_test_filtered)]  # Correct length



    # Add Random Forest model execution
    if use_random_forest:
        print("Training Random Forest Model...")

        # Encode MIDI labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        # Filter test labels to keep only known classes
        mask = y_test.isin(label_encoder.classes_)
        X_test_filtered = X_test.loc[mask].copy()
        y_test_filtered = y_test.loc[mask].copy()
        y_test_encoded = label_encoder.transform(y_test_filtered)

        # Train Random Forest model
        rf_predictions, rf_report, rf_model = train_random_forest(
            X_train, y_train_encoded, X_test_filtered, y_test_encoded, use_grid_search=False, use_optuna=True
        )

        # Decode predictions
        rf_predictions_decoded = label_encoder.inverse_transform(rf_predictions)

        # Show results
        print("\nRandom Forest Classification Report:")
        rf_report_df = pd.DataFrame.from_dict(rf_report, orient="index")
        print(rf_report_df.transpose())

        # 🎯 Evaluate the model
        model_results["RandomForest"] = evaluate_model(rf_model, X_train, y_train_encoded, X_test_filtered, y_test_encoded, model_name="Random Forest")

        # Align time array with y_test_filtered
        time_filtered = time[mask.to_numpy()]  # Convert mask to numpy array before indexing

        # Now pass this correctly filtered time array
        plot_predicted_vs_actual(time_filtered, y_test_filtered, rf_predictions_decoded)

        model_predictions["Random Forest"] = rf_predictions_decoded # fro combined
        time_filtered_dict["Random Forest"] = time[:len(y_test_filtered)]  # Correct length


    # ✅ Plot all model predictions in a single chart
    plot_all_models(time_filtered_dict, {"LSTM": y_test_filtered}, model_predictions)


if __name__ == "__main__":
    main()
