import pretty_midi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def midi_to_dataframe(midi_file):
    """
    Convert MIDI file to a DataFrame containing note information.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append({
                "Instrument": instrument.name,
                "Pitch": note.pitch,
                "Start": note.start,
                "End": note.end,
                "Duration": note.end - note.start
            })

    return pd.DataFrame(notes)

def compare_midi_versions(original_file, generated_file):
    """
    Compare two MIDI files and highlight the differences.
    """
    # Convert MIDI files to DataFrames
    original_df = midi_to_dataframe(original_file)
    generated_df = midi_to_dataframe(generated_file)

    # Merge the DataFrames on pitch and time (fuzzy comparison)
    merged_df = pd.merge(
        original_df, generated_df,
        how='outer',  # Include all notes from both files
        on=['Pitch'],  # Compare based on pitch
        suffixes=('_original', '_generated')
    )

    # Handle missing data using NaN
    merged_df['Start_original'] = pd.to_numeric(merged_df['Start_original'], errors='coerce')
    merged_df['Start_generated'] = pd.to_numeric(merged_df['Start_generated'], errors='coerce')
    merged_df['Duration_original'] = pd.to_numeric(merged_df['Duration_original'], errors='coerce')
    merged_df['Duration_generated'] = pd.to_numeric(merged_df['Duration_generated'], errors='coerce')

    # Add comparison columns
    merged_df['Start_Difference'] = merged_df['Start_original'] - merged_df['Start_generated']
    merged_df['Duration_Difference'] = merged_df['Duration_original'] - merged_df['Duration_generated']

    return merged_df

def visualize_note_differences(df, original_file, generated_file):
    """
    Visualize note differences between two MIDI files.
    """
    plt.figure(figsize=(12, 8))

    # Plot original notes
    plt.scatter(
        df['Start_original'], df['Pitch'],
        color='blue', label='Original Notes', alpha=0.6
    )

    # Plot generated notes
    plt.scatter(
        df['Start_generated'], df['Pitch'],
        color='red', label='Generated Notes', alpha=0.6
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Pitch (MIDI Note Number)")
    plt.title(f"Comparison: {original_file} vs {generated_file}")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_note_differences_line_chart(df, original_file, generated_file):
    """
    Visualize note differences between two MIDI files as a line chart.
    """
    plt.figure(figsize=(12, 8))

    # Sort DataFrame by time for proper plotting
    df.sort_values(by='Start_original', inplace=True)

    # Plot original notes as a line
    plt.plot(
        df['Start_original'], df['Pitch'], 
        color='blue', label='Original Notes', alpha=0.7, linestyle='-'
    )

    # Plot generated notes as a line
    plt.plot(
        df['Start_generated'], df['Pitch'], 
        color='red', label='Generated Notes', alpha=0.7, linestyle='--'
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Pitch (MIDI Note Number)")
    plt.title(f"Comparison: {original_file} vs {generated_file}")
    plt.legend()
    plt.grid()
    plt.show()


# Compare and visualize differences
original_file = "Bon_Jovi_-_Always.mid"
generated_file = "my_melody.mid"

comparison_df = compare_midi_versions(original_file, generated_file)
print(comparison_df.head())  # Show the first few differences

visualize_note_differences(comparison_df, original_file, generated_file)
#visualize_note_differences_line_chart(comparison_df, original_file, generated_file)