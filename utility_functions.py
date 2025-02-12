import numpy as np
import mido
import pandas as pd

# Frequency to MIDI note number
def frequency_to_midi_number(frequency):
    if frequency <= 0:
        return None
    return int(69 + 12 * np.log2(frequency / 440.0))

# MIDI note number to note name
def midi_to_note_name(midi_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = note_names[midi_number % 12]
    return f"{note}{octave}"

# Parse MIDI file
def parse_midi_file(midi_file):
    midi_data = mido.MidiFile(midi_file)
    notes = []
    time = 0
    for msg in midi_data:
        time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:  # Note-on messages
            notes.append({
                'Time': time,
                'MIDI_Note': msg.note,
                'Note': midi_to_note_name(msg.note)
            })
    return pd.DataFrame(notes)
