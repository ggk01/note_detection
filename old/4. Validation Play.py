import librosa
import numpy as np
import simpleaudio as sa


# Load vocals
y, sr = librosa.load("audio.wav")

# Estimate pitches
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

# Compute the chroma features
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)


# Aggregate chroma to get the strongest pitch class
chroma_sum = chroma.sum(axis=1)
pitch_class = np.argmax(chroma_sum)

# Map pitch class index to note names
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
detected_tonic = note_names[pitch_class]

# print basic info
print(f"Detected Key: {chroma}")
print(f"Detected Key (Tonic): {detected_tonic}")


# Extract the pitch curve
# Extract the highest pitch in each frame
detected_pitch = []
for i in range(pitches.shape[1]):
    index = magnitudes[:, i].argmax()
    detected_pitch.append(pitches[index, i] if magnitudes[index, i] > 0.1 else 0)


# Function to generate a sine wave for a given frequency
def generate_harmonic_wave(frequency, duration, sample_rate=44100, amplitude=0.5, harmonics=10):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    for i in range(2, harmonics + 1):  # Add harmonics
        wave += (amplitude / i) * np.sin(2 * np.pi * frequency * i * t)
    return wave


# Generate tones from detected pitches
def play_detected_pitches(detected_pitches, duration_per_pitch=0.5, sample_rate=44100):
    """
    Play tones corresponding to detected pitches.

    Args:
    detected_pitches (list): List of pitch frequencies in Hz.
    duration_per_pitch (float): Duration for each pitch in seconds.
    sample_rate (int): Sample rate for audio playback (default: 44100).
    """

    # Ensure all frequencies are valid floats
    #detected_pitches = [round(float(freq)) for freq in detected_pitches if freq >= 20 and freq <= 20000]

    audio = np.array([], dtype=np.float32)


  # Reduce the frequencies by one octave (divide by 2)
    bass_pitches = [float(freq) / 2 for freq in detected_pitches if freq >= 20 and freq <= 20000]


    for freq in bass_pitches:
        freq = float(freq)
        if freq > 0:  # Ignore zero frequencies (rests)
            print(f"Playing frequency: {freq} Hz")
            wave = generate_harmonic_wave(freq, duration_per_pitch, sample_rate)
        else:  # Silence for rests
            wave = np.zeros(int(sample_rate * duration_per_pitch))
        audio = np.concatenate((audio, wave))

    # Normalize audio to the range [-1, 1]
    audio = audio / np.max(np.abs(audio))

    # Convert to 16-bit PCM format
    audio = (audio * 32767).astype(np.int16)

    # Play the audio
    play_obj = sa.play_buffer(audio, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
    play_obj.wait_done()


# Example detected pitches (replace with your actual data)
detected_pitche = detected_pitch[:1000]

print('pitch', detected_pitche)

# Play the tones
play_detected_pitches(detected_pitche)
