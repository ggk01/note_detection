import mido

# Load the MIDI file
midi_file = mido.MidiFile('Bon_Jovi_-_Always.mid')


# Iterate through tracks and messages
for i, track in enumerate(midi_file.tracks):
    print(f"Track {i}: {track.name}")
    for msg in track:
        if msg.type == 'note_on' and msg.velocity > 0:
            print(f"Note: {msg.note}, Time: {msg.time}, Velocity: {msg.velocity}")


import pretty_midi

# Load the MIDI file
midi_file2 = pretty_midi.PrettyMIDI('Bon_Jovi_-_Always.mid')

# Extract instrument and note data
for instrument in midi_file2.instruments:
    print(f"Instrument: {instrument.name}")
    for note in instrument.notes:
        print(f"Note: {note.pitch}, Start: {note.start:.2f}, End: {note.end:.2f}, Velocity: {note.velocity}")



def midi_to_frequency(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))

# Example: Convert MIDI note 60 (Middle C) to frequency
print(midi_to_frequency(60))  # Output: 261.63 Hz


# Example detected pitches and timestamps (from your pitch detection algorithm)
detected_pitches = [(0.5, 261.63), (1.0, 293.66), (1.5, 329.63)]  # (Time in seconds, Frequency)


ground_truth = []

for instrument in midi_file2.instruments:
    for note in instrument.notes:
        freq = midi_to_frequency(note.pitch)
        ground_truth.append((note.start, freq))

# Compare detected pitches with ground truth
for detected_time, detected_freq in detected_pitches:
    closest_midi_note = min(ground_truth, key=lambda x: abs(x[0] - detected_time))
    print(f"Detected: {detected_freq:.2f} Hz, Closest MIDI Note: {closest_midi_note[1]:.2f} Hz")
