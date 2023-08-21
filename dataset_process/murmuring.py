"""This file is dedicated to create a pipeline for using basic-pitch, 
Spotify's audio2midi converter library."""

from basic_pitch.inference import predict_and_save
from pathlib import Path

INPUT_PATHS = [
    Path(".../murmurings/record.m4a"),
]
OUTPUT_DIR = Path(".../murmurings/")


predict_and_save(
    audio_path_list=INPUT_PATHS,
    output_directory=OUTPUT_DIR,
    save_midi=True,
    sonify_midi=False,
    save_model_outputs=False,
    save_notes=False,
)
