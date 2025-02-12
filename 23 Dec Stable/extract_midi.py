import pretty_midi
import pandas as pd

def midi_to_frequencies(midi_file):
    """
    Extract MIDI frequencies and notes.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        raise ValueError(f"Error reading MIDI file: {e}")

    notes = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            frequency = pretty_midi.note_number_to_hz(note.pitch)
            notes.append({
                "Pitch": note.pitch,
                "Frequency": frequency,
                "Start": note.start,
                "End": note.end,
                "Duration": note.end - note.start,
                "Note": pretty_midi.note_number_to_name(note.pitch)
            })

    if not notes:
        raise ValueError("No notes found in the MIDI file.")

    df = pd.DataFrame(notes)
    df["Time"] = df["Start"]  # Add a Time column for compatibility
    return df
