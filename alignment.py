import pandas as pd

def align_audio_with_midi(audio_features, midi_labels, tolerance=0.05):
    aligned_data = []

    for _, midi_row in midi_labels.iterrows():
        # Find the closest audio feature within the tolerance
        closest_audio = audio_features.loc[
            (audio_features["Time"] >= midi_row["Time"] - tolerance) &
            (audio_features["Time"] <= midi_row["Time"] + tolerance)
        ]

        if not closest_audio.empty:
            closest_audio_row = closest_audio.iloc[0]
            aligned_data.append({
                "Time": midi_row["Time"],
                "MIDI_Note": midi_row["MIDI_Note"],
                "Note_Label": midi_row["Note"],
                "Pitch": closest_audio_row["Pitch"],
                "Power": closest_audio_row["Power"],
                "Spectral_Centroid": closest_audio_row["Spectral_Centroid"]
            })

    return pd.DataFrame(aligned_data)
