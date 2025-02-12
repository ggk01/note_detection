from extract_midi import midi_to_frequencies
from extract_audio import audio_to_frequencies
from dominant_frequencies import extract_dominant_frequencies
from compare_versions import compare_scaled_frequencies, dynamic_offset_correction
from visualize_comparison import visualize_scaled_comparison_with_offset, visualize_simplified_comparison_with_offset
import matplotlib.pyplot as plt


def main():
    midi_file = "Bon_Jovi_-_Always.mid"
    audio_file = "audio.wav"

    # Step 1: Extract frequencies from MIDI
    midi_df = midi_to_frequencies(midi_file)
    midi_dominant_df = extract_dominant_frequencies(midi_df, interval=5.0)

    # Step 2: Extract frequencies from audio
    audio_df = audio_to_frequencies(audio_file)
    audio_dominant_df = extract_dominant_frequencies(audio_df, interval=5.0)

    # Step 3: Apply dynamic offset correction
    audio_dominant_df = dynamic_offset_correction(midi_dominant_df, audio_dominant_df)

    # Step 4: Compare MIDI and audio notes
    comparison_df = compare_scaled_frequencies(midi_dominant_df, audio_dominant_df)
    print(comparison_df)
    # Step 5: Visualize the comparison
    # visualize_scaled_comparison_with_offset(comparison_df, 0)

    
    visualize_simplified_comparison_with_offset(comparison_df, -3.5) 



if __name__ == "__main__":
    main()
