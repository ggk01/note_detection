from utility_functions import parse_midi_file
from audio_feature_extraction import extract_audio_features
from alignment import align_audio_with_midi

def create_csv(audio_file, midi_file, output_csv, tolerance=0.05):
    """
    Create a CSV file containing aligned audio features and MIDI labels.

    Args:
        audio_file (str): Path to the audio file.
        midi_file (str): Path to the MIDI file.
        output_csv (str): Path to save the output CSV file.
        tolerance (float): Time alignment tolerance in seconds.
    """
    # Step 1: Parse MIDI file for ground truth labels
    print(f"Parsing MIDI file: {midi_file}")
    midi_labels = parse_midi_file(midi_file)

    # Step 2: Extract features from the audio
    print(f"Extracting features from audio file: {audio_file}")
    audio_features = extract_audio_features(audio_file)

    # Step 3: Align audio features with MIDI labels
    print("Aligning audio features with MIDI labels...")
    aligned_data = align_audio_with_midi(audio_features, midi_labels, tolerance)

    # Step 4: Save to CSV
    print(f"Saving aligned data to CSV: {output_csv}")
    aligned_data.to_csv(output_csv, index=False)
    print("CSV creation completed.")

create_csv("November Rain.wav", "November Rain.mid", "aligned_features_November_Rain.csv", tolerance=0.05)
