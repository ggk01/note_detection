import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(train_files, test_file):
    """
    Load training and testing data from CSV files.
    
    Args:
        train_files (list): List of paths to training CSV files.
        test_file (str): Path to the testing CSV file.
        
    Returns:
        X_train, y_train, X_test, y_test: Training and testing features and labels.
    """
    # Load and concatenate training data
    train_data = pd.concat([pd.read_csv(file) for file in train_files], ignore_index=True)
    test_data = pd.read_csv(test_file)


    # Sort by time to maintain sequence order
    train_data = train_data.sort_values(by="Time")
    test_data = test_data.sort_values(by="Time")

    # Add previous note feature
    train_data["prev_note"] = train_data["MIDI_Note"].shift(1).fillna(train_data["MIDI_Note"].median())
    test_data["prev_note"] = test_data["MIDI_Note"].shift(1).fillna(train_data["MIDI_Note"].median())  # Use train median

    # Select features and labels
    feature_columns = ["Pitch", "Power", "Spectral_Centroid", "prev_note"]

    # Split features and labels
    X_train = train_data[feature_columns]
    y_train = train_data["MIDI_Note"]

    X_test = test_data[feature_columns]
    y_test = test_data["MIDI_Note"]

    return X_train, y_train, X_test, y_test




def prepare_lstm_data(X_train, y_train, X_test, y_test):
    """
    Ensure X_train and y_train are fully aligned before passing them to LSTM.
    """
    print(f"Before LSTM alignment: X_train {X_train.shape}, y_train {y_train.shape}")

    # Convert y_train to pandas Series (fix for .iloc issue)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)

    # Ensure alignment by trimming to the smallest length
    min_len = min(len(X_train), len(y_train))
    X_train = X_train.iloc[:min_len]
    y_train = y_train.iloc[:min_len].to_numpy()  # Convert back to NumPy array for LSTM

    min_len_test = min(len(X_test), len(y_test))
    X_test = X_test.iloc[:min_len_test]
    y_test = y_test.iloc[:min_len_test].to_numpy()  # Convert back to NumPy array

    print(f"After LSTM alignment: X_train {X_train.shape}, y_train {y_train.shape}")

    return X_train, y_train, X_test, y_test