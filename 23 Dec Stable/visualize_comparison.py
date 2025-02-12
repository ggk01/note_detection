import matplotlib.pyplot as plt
import pandas as pd

# Function to convert MIDI note number to note name
def midi_to_note_name(midi_number):
    """
    Convert a MIDI note number to a note name.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = note_names[int(midi_number % 12)]
    return f"{note}{octave}"

def visualize_simplified_comparison_with_offset (comparison_df, offset=0):

    comparison_df["Adjusted_Audio_Notes"] = comparison_df["MIDI_Note_y"] + offset

    plt.figure(figsize=(32, 8))
    plt.plot(comparison_df["Start_Time"], comparison_df["Adjusted_MIDI_Note"], label="MIDI Frequencies", color="blue")
    plt.plot(comparison_df["Start_Time"], comparison_df["Adjusted_Audio_Notes"], label="Audio Frequencies", color="red")
   # plt.plot(comparison_df["Start_Time"], comparison_df["MIDI_Note_Difference"], label="Audio Midi Diffs", color="green")

    # Annotate MIDI and audio notes with color coding based on alignment
    for i, row in comparison_df.iterrows():
        if not pd.isna(row["Adjusted_MIDI_Note"]) and not pd.isna(row["Adjusted_Audio_Notes"]):
            note_name_midi = midi_to_note_name(row["Adjusted_MIDI_Note"])
            note_name_audio = midi_to_note_name(row["Adjusted_Audio_Notes"])


            # Offset the annotations for readability
            midi_offset = 1 if i % 2 == 0 else -1
            audio_offset = -1 if i % 2 == 0 else 1

            # Check if the notes are the same
            if abs(row["Adjusted_MIDI_Note"] - row["Adjusted_Audio_Notes"]) <= 1:  # Match threshold
                color = "green"  # Matching notes
                            # Annotate MIDI notes
                plt.text(
                    row["Start_Time"], row["Adjusted_MIDI_Note"] + midi_offset,
                    note_name_midi, color=color, fontsize=8, alpha=0.8
                )
            else:
                color = "red"  # Non-matching notes

                # Annotate MIDI notes
                plt.text(
                    row["Start_Time"], row["Adjusted_MIDI_Note"] + audio_offset,
                    note_name_midi, color=color, fontsize=8, alpha=0.8
                )

                # Annotate Audio notes
                plt.text(
                    row["Start_Time"], row["Adjusted_Audio_Notes"] + 1,
                    note_name_audio, color=color, fontsize=8, alpha=0.8
                )


    # Highlight perfect matches
    perfect_matches = abs(
        comparison_df["Adjusted_MIDI_Note"] - comparison_df["Adjusted_Audio_Notes"]
    ) <= 1
    plt.scatter(
        comparison_df["Start_Time"][perfect_matches],
        comparison_df["Adjusted_MIDI_Note"][perfect_matches],
        color="green", label="Perfect Matches", alpha=0.8
    )

    # Highlight mismatches
    mismatches = abs(
        comparison_df["Adjusted_MIDI_Note"] - comparison_df["Adjusted_Audio_Notes"]
    ) > 1
    plt.scatter(
        comparison_df["Start_Time"][mismatches],
        comparison_df["Adjusted_MIDI_Note"][mismatches],
        color="purple", label="Significant Mismatches", alpha=0.8
    )


    plt.legend()
    plt.title("Raw Frequency Comparison")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


def visualize_scaled_comparison_with_offset(comparison_df, offset=0):
    """
    Visualize comparison of MIDI note numbers between MIDI and audio, with dynamic offset.
    """
    comparison_df["Adjusted_Audio_Notes"] = comparison_df["MIDI_Note_y"] + offset

    plt.figure(figsize=(12, 6))

    # Check and apply offset if needed
    if "Adjusted_Audio_Notes" not in comparison_df.columns:
        comparison_df["Adjusted_Audio_Notes"] = comparison_df["MIDI_Note_x"] + offset

    plt.figure(figsize=(12, 6))

    # Plot MIDI notes
    plt.plot(
        comparison_df["Start_Time"], comparison_df["MIDI_Note_x"],
        label="MIDI Notes", color="blue", alpha=0.7
    )

    # Plot adjusted audio notes (with offset)
    plt.plot(
        comparison_df["Start_Time"], comparison_df["Adjusted_Audio_Notes"],
        label="Adjusted Audio Notes (Offset)", color="red", alpha=0.7
    )


    # Annotate MIDI notes
    for _, row in comparison_df[::1].iterrows():  # Annotate every 10th point for clarity
        if not pd.isna(row["MIDI_Note_x"]):  # Skip NaN values
            note_name = midi_to_note_name(row["MIDI_Note_x"])
            plt.text(
                row["Start_Time"], row["MIDI_Note_x"] + 1,
                note_name, color="blue", fontsize=8, alpha=0.8
            )

    # Annotate adjusted audio notes
    for _, row in comparison_df[::1].iterrows():  # Annotate every 10th point for clarity
        if not pd.isna(row["Adjusted_Audio_Notes"]):  # Skip NaN values
            note_name = midi_to_note_name(row["Adjusted_Audio_Notes"])
            plt.text(
                row["Start_Time"], row["Adjusted_Audio_Notes"] + 1,
                note_name, color="red", fontsize=8, alpha=0.8
            )

    # Highlight perfect matches
    perfect_matches = abs(
        comparison_df["MIDI_Note_x"] - comparison_df["Adjusted_Audio_Notes"]
    ) <= 1
    plt.scatter(
        comparison_df["Start_Time"][perfect_matches],
        comparison_df["MIDI_Note_x"][perfect_matches],
        color="green", label="Perfect Matches", alpha=0.8
    )

    # Highlight mismatches
    mismatches = abs(
        comparison_df["MIDI_Note_x"] - comparison_df["Adjusted_Audio_Notes"]
    ) > 1
    plt.scatter(
        comparison_df["Start_Time"][mismatches],
        comparison_df["MIDI_Note_x"][mismatches],
        color="purple", label="Significant Mismatches", alpha=0.8
    )


    plt.xlabel("Time (seconds)")
    plt.ylabel("MIDI Note Number")
    plt.title("Enhanced Scaled Comparison Between MIDI and Audio Notes (With Note Annotations)")
    plt.legend()
    plt.grid()
    plt.show()