import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#import simpleaudio as sa

# Load vocals
y, sr = librosa.load("demucs_output/mdx_extra/audio/vocals.wav")

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

# Plot the waveform and detected pitch
# Plot waveform
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title("Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot detected pitch
plt.subplot(2, 1, 2)
plt.plot(detected_pitch, label="Detected Pitch (Hz)", color="blue")
plt.title("Detected Pitch Over Time")
plt.xlabel("Time (frames)")
plt.ylabel("Frequency (Hz)")
plt.legend()

plt.tight_layout()
plt.show()


# Function to map frequency to closest musical note
def hz_to_note_name(hz):
    if hz == 0:
        return "Rest"
    A440 = 440.0
    semitones_from_A4 = 12 * np.log2(hz / A440)
    note_index = int(round(semitones_from_A4)) % 12
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[note_index]

# Map detected pitches to notes
detected_notes = [hz_to_note_name(p) for p in detected_pitch]
print("Detected Notes:", detected_notes)

#detected_pitch = [p.max() for p in pitches.T]

smoothed_pitch = np.convolve(detected_pitch, np.ones(10)/10, mode='valid')

'''
# Plot pitch
plt.figure(figsize=(10, 4))
plt.plot(smoothed_pitch, label="Original Vocal Pitch")
plt.xlabel("Time")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.show()
'''

# Assume `smoothed_pitch` is your pitch array and `hop_length` is your frame hop size
sr = 22050  # Sampling rate
hop_length = 512  # Hop size (in samples)

# Generate time array (in seconds)
time_in_seconds = np.arange(len(smoothed_pitch)) * hop_length / sr

# Convert seconds to minutes:seconds format dynamically
def format_time(t):
    minutes = int(t // 60)
    seconds = int(t % 60)
    return f"{minutes}:{seconds:02d}"

# Generate labels for the X-axis
time_labels = [format_time(t) for t in time_in_seconds]


plt.figure(figsize=(10, 5))
plt.plot(time_in_seconds, smoothed_pitch, label="Smoothed Vocal Pitch", color="blue")
plt.axhline(y=440, color='red', linestyle='--', label="Reference (A4 = 440 Hz)")
#plt.ylim(80, 1000)
plt.xlabel("Time (Minutes:Seconds)")
plt.ylabel("Frequency (Hz)")
plt.title(f"Smoothed Pitch Contour of Original Vocals (Key: {detected_tonic})")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(ticks=np.linspace(0, time_in_seconds[-1], 10), labels=[format_time(t) for t in np.linspace(0, time_in_seconds[-1], 10)])
plt.legend()
plt.show()
