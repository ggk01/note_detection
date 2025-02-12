import numpy as np
import pandas as pd
import librosa
from utility_functions import frequency_to_midi_number

def extract_audio_features(audio_file, sr=22050, hop_length=512):
    # Load the audio
    y, sr = librosa.load(audio_file, sr=sr)

    # Compute the number of frames
    num_frames = len(y) // hop_length

    # Extract time frames
    time_frames = librosa.frames_to_time(range(num_frames), sr=sr, hop_length=hop_length)

    # Extract pitch (dominant frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=80.0, fmax=1200.0, hop_length=hop_length)
    dominant_pitches = [np.max(pitches[:, t]) if np.max(magnitudes[:, t]) > 0.1 else 0 for t in range(min(num_frames, pitches.shape[1]))]

    # Extract power (RMS)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    rms = rms[0][:len(dominant_pitches)]  # Trim to match length of dominant_pitches

    # Extract spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spectral_centroid = spectral_centroid[0][:len(dominant_pitches)]  # Trim to match length

    # Ensure time_frames matches the length of dominant_pitches
    time_frames = time_frames[:len(dominant_pitches)]

    # Create DataFrame
    features = pd.DataFrame({
        "Time": time_frames,
        "Pitch": dominant_pitches,
        "Power": rms,
        "Spectral_Centroid": spectral_centroid
    })
    features["MIDI_Note"] = features["Pitch"].apply(frequency_to_midi_number)

    return features
