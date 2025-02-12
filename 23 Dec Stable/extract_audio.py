import librosa
import pandas as pd

def audio_to_frequencies(audio_file, sr=44100, hop_length=256, fmin=50, fmax=1500, smoothing_window=7):
    """
    Extract frequencies from an audio file with enhanced smoothing and optimized pitch detection.
    """
    y, sr = librosa.load(audio_file, sr=sr)
    y = librosa.util.normalize(y)

    # Pitch detection with optimized parameters
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)

    extracted_pitches = []
    time_frames = []

    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:  # Ignore zero frequencies (rests)
            extracted_pitches.append(pitch)
            time_frames.append(i * hop_length / sr)

    # Smoothing frequencies to reduce noise
    smoothed_pitches = pd.Series(extracted_pitches).rolling(window=smoothing_window, center=True).mean()

    # Creating a DataFrame
    audio_df = pd.DataFrame({
        "Time": time_frames,
        "Frequency": smoothed_pitches
    }).dropna()  # Drop NaN from smoothing
    return audio_df
