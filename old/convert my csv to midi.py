import pandas as pd
import pretty_midi
import numpy as np

def frequency_to_midi(frequency):
    """
    Convert frequency to the nearest MIDI note number.
    """
    if frequency == 0 or pd.isna(frequency):  # Handle rests or missing frequencies
        return None
    return int(round(69 + 12 * np.log2(frequency / 440.0)))  # MIDI formula

def convert_csv_to_midi(csv_file, output_midi_file="output.mid", default_duration=1.0):
    """
    Convert a CSV file with time and frequency data to a MIDI file.

    Args:
    csv_file (str): Path to the input CSV file.
    output_midi_file (str): Path to save the output MIDI file.
    default_duration (float): Default duration (in seconds) for each note.
    """
    # Load the CSV
    df = pd.read_csv(csv_file)

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()

    # Add an instrument (e.g., Acoustic Grand Piano)
    instrument = pretty_midi.Instrument(program=0)  # 0 = Acoustic Grand Piano

    # Process each row in the CSV
    for _, row in df.iterrows():
        freq = row['Frequency']
        start_time = row['Time']
        
        # Convert frequency to MIDI note
        midi_note = frequency_to_midi(freq)
        
        if midi_note:  # Skip rests or invalid frequencies
            # Assume a fixed duration or calculate based on subsequent rows
            end_time = start_time + default_duration

            # Create a note object
            note = pretty_midi.Note(
                velocity=100,  # How hard the note is played (0-127)
                pitch=midi_note,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)

    # Add the instrument to the MIDI object
    midi.instruments.append(instrument)

    # Write the MIDI file
    midi.write(output_midi_file)
    print(f"MIDI file created: {output_midi_file}")

# Example usage
csv_file = "frequencies_and_notes.csv"  # Replace with your CSV file path
convert_csv_to_midi(csv_file, output_midi_file="my_melody.mid")
