import pandas as pd
from fastdtw import fastdtw
import numpy as np

# Function to convert MIDI note number to note name
def midi_to_note_name(midi_number):
    """
    Convert a MIDI note number to a note name.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = note_names[int(midi_number % 12)]
    return f"{note}{octave}"

def dynamic_offset_correction(midi_df, audio_df):
    """
    Calculate and apply a dynamic offset to align MIDI and audio notes.
    """
    merged_df = pd.merge_asof(
        midi_df.sort_values("Start_Time"),
        audio_df.sort_values("Start_Time"),
        on="Start_Time",
        direction="nearest",
        tolerance=1.0  # 1-second tolerance
    )

    # Calculate the mean offset between MIDI and audio
    offset = (merged_df["MIDI_Note_x"] - merged_df["MIDI_Note_y"]).mean()

    # Adjust audio notes dynamically
    audio_df["Adjusted_MIDI_Note"] = audio_df["MIDI_Note"] + offset
    return audio_df

def compare_scaled_frequencies(midi_df, audio_df):
    """
    Compare scaled (MIDI note) frequencies between MIDI and audio using DTW.
    """
    # Ensure required columns exist
    if "MIDI_Note" not in midi_df.columns or "Start_Time" not in midi_df.columns:
        raise ValueError("MIDI DataFrame must contain 'MIDI_Note' and 'Start_Time' columns.")
    if "MIDI_Note" not in audio_df.columns or "Start_Time" not in audio_df.columns:
        raise ValueError("Audio DataFrame must contain 'MIDI_Note' and 'Start_Time' columns.")

    # Rename columns for merging
    midi_df = midi_df.rename(columns={"MIDI_Note": "MIDI_Note_x"})
    audio_df = audio_df.rename(columns={"MIDI_Note": "MIDI_Note_y"})

    # Align timestamps using nearest match
    comparison_df = pd.merge_asof(
        midi_df.sort_values("Start_Time"),
        audio_df.sort_values("Start_Time"),
        on="Start_Time",
        direction="nearest",
        tolerance=1.0  # 1-second tolerance
    )

    # Add MIDI note difference column
    comparison_df["MIDI_Note_Difference"] = abs(
        comparison_df["MIDI_Note_x"] - comparison_df["MIDI_Note_y"]
    )

    for _, row in comparison_df.iterrows():
        if not pd.isna(row["MIDI_Note_y"]):
            comparison_df["Audio note"] = midi_to_note_name(row["MIDI_Note_y"])

    for _, row in comparison_df.iterrows():
        if not pd.isna(row["Adjusted_MIDI_Note"]):
            comparison_df["MIDI note"] = midi_to_note_name(row["Adjusted_MIDI_Note"])
    

    return comparison_df
