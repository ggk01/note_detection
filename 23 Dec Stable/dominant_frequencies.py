import pandas as pd
import numpy as np

def frequency_to_midi_number(frequency):
    if frequency and frequency > 0:
        return 69 + 12 * np.log2(frequency / 440.0)  # A4 = 440 Hz
    else:
        return None

def extract_dominant_frequencies(df, interval=1.0):
    """
    Aggregate dominant frequencies over fixed intervals using the median or mode.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot extract dominant frequencies.")

    # Filter out invalid frequencies
    df = df[df["Frequency"].notna() & (df["Frequency"] > 0)]

    df["Interval"] = (df["Time"] // interval).astype(int)
    aggregated_df = df.groupby("Interval").agg({
        "Frequency": "median",  # Use median to reduce outlier influence
        "Time": "min"
    }).rename(columns={"Frequency": "Dominant_Frequency", "Time": "Start_Time"}).reset_index(drop=True)

    # Convert frequencies to MIDI note numbers
    aggregated_df["MIDI_Note"] = aggregated_df["Dominant_Frequency"].apply(frequency_to_midi_number)
    aggregated_df = aggregated_df[aggregated_df["MIDI_Note"].notna()]  # Remove invalid MIDI notes
    return aggregated_df
