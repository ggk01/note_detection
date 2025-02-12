import librosa
import pretty_midi
import pandas as pd
import numpy as np

# Step 1: Load the song and extract pitch frequencies
def extract_frequencies(audio_file, hop_length=512, sr=22050):
    y, sr = librosa.load(audio_file, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length, fmin=80, fmax=2000)
    
    # Debug: Ensure pitch and magnitude arrays are populated
    print("Shape of pitches:", pitches.shape)
    print("Shape of magnitudes:", magnitudes.shape)

    # Get the highest pitch per frame
    time_frames = np.arange(pitches.shape[1]) * hop_length / sr  # Time in seconds
    extracted_pitches = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()  # Get the strongest pitch
        pitch = pitches[index, i]
        extracted_pitches.append(pitch if pitch > 0 else None)  # Filter out unvoiced sections

    # Debug: Check extracted pitches
    print("Extracted Pitches (first 10):", extracted_pitches[:10])
    return time_frames, extracted_pitches


# Function to adjust sparsity
def adjust_sparsity(time_frames, extracted_pitches, sparsity_step=1.0):
    """
    Adjust the sparsity of time frames and pitches based on a user-defined interval.
    
    Args:
    time_frames (array): Array of time frames in seconds.
    extracted_pitches (array): Array of pitch frequencies.
    sparsity_step (float): Desired interval in seconds between sampled frames.
    
    Returns:
    tuple: Adjusted time frames and pitches.
    """
    sparse_times = []
    sparse_pitches = []
    
    for i, time in enumerate(time_frames):
        if i == 0 or time >= sparse_times[-1] + sparsity_step:
            sparse_times.append(time)
            sparse_pitches.append(extracted_pitches[i])
    
    return sparse_times, sparse_pitches

# Step 2: Extract notes from MIDI
def extract_midi_notes(midi_file):
    """
    Extract notes and timings from a MIDI file.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    note_info = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note_info.append({
                "start": note.start,
                "end": note.end,
                "pitch": note.pitch,
                "note": pretty_midi.note_number_to_name(note.pitch)
            })
    return pd.DataFrame(note_info)

# Step 3: Match time frames with MIDI notes
def map_frequencies_to_notes(time_frames, extracted_pitches, midi_df):
    """
    Map extracted frequencies to the corresponding MIDI notes.
    """
    data = {"Time": time_frames, "Frequency": extracted_pitches, "Note": []}
    for time in time_frames:
        # Find the note in the MIDI that matches the time frame
        matching_note = midi_df[(midi_df["start"] <= time) & (midi_df["end"] >= time)]
        if not matching_note.empty:
            data["Note"].append(matching_note.iloc[0]["note"])  # Take the first matching note
        else:
            data["Note"].append(None)  # No matching note
    
    return pd.DataFrame(data)

# Step 4: Save to a DataFrame
audio_file = "audio.wav"
midi_file = "Bon_Jovi_-_Always.mid"

time_frames, extracted_pitches = extract_frequencies(audio_file)

# Step 2: Adjust sparsity (make it adjustable)
sparsity_step = float(input("Enter sparsity step in seconds (e.g., 1.0, 2.0): "))
adjusted_times, adjusted_pitches = adjust_sparsity(time_frames, extracted_pitches, sparsity_step)

# Step 3: Extract MIDI notes
midi_df = extract_midi_notes(midi_file)

# Step 4: Map frequencies to MIDI notes
final_df = map_frequencies_to_notes(adjusted_times, adjusted_pitches, midi_df)

# Save or display the final DataFrame
print(final_df)