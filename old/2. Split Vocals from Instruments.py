import os

# Input and output paths
input_audio = "audio.wav"
output_dir = "demucs_output"

# Run Demucs separation
os.system(f"demucs -n mdx_extra {input_audio} -o {output_dir}")

print(f"Separation complete! Files saved in {output_dir}")
