from midi2audio import FluidSynth
import simpleaudio as sa

def convert_midi_to_wav(midi_file, sound_font="FluidR3_GM.sf2"):
    """
    Convert a MIDI file to a WAV file using midi2audio and play it.
    """
    # Initialize FluidSynth with the sound font
    fs = FluidSynth(sound_font)

    # Convert MIDI to WAV
    wav_file = midi_file.replace(".mid", ".wav")
    fs.midi_to_audio(midi_file, wav_file)
    print(f"Converted {midi_file} to {wav_file}")

    return wav_file

def play_wav(wav_file):
    """
    Play a WAV file using simpleaudio.
    """
    wave_obj = sa.WaveObject.from_wave_file(wav_file)
    play_obj = wave_obj.play()
    play_obj.wait_done()

# Convert and play the original MIDI
# original_midi = "Bon_Jovi_-_Always.mid"
# original_wav = convert_midi_to_wav(original_midi)
# play_wav(original_wav)

# Convert and play your generated MIDI
generated_midi = "my_melody.mid"
generated_wav = convert_midi_to_wav(generated_midi)
play_wav(generated_wav)
