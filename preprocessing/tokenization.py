# -*- coding: utf-8 -*-
"""
# Tokenization Process
---
Tokenizing of the .mid datas using MiDiTok library.
"""

# Importing required libraries
#from IPython.display import clear_output
#!pip install MidiTok
from glob import glob
from miditoolkit import MidiFile

from config import Config
CONFIG = Config()

#clear_output(wait=False)

# Tokenization process
from pathlib import Path

#Remainder:
# CONFIG.tokenizer is OctupeMono
# CONFIG.tokenizer2 is REMI 

tokenizer = CONFIG.tokenizer2 

#midi_paths_all = list(Path(".../dataset/dataset_first/midi_all").glob('*.mid'))
midi_paths_test = list(Path(".../dataset/dataset_test/").glob('*.mid'))

#Final process
#output_path = ".../dataset/dataset_OCTUPLE_tokenized"
#tokenizer.tokenize_midi_dataset(midi_paths_all, output_path)

output_path = Path(".../dataset/dataset_test/test_tokenized/")
tokenizer.tokenize_midi_dataset(midi_paths=midi_paths_test, out_dir=output_path)

# For single midi file 
#--------------------------
#from pathlib import Path

tokenizer = CONFIG.tokenizer2 # I'm getting the tokenizer from a config file because 

single_midi_path = ".../dataset/dataset_test/asd.mid" 
single_midi = MidiFile(single_midi_path)
#try_output_path = ".../dataset/dataset_try_2/qwe.json"

#try_tokens = tokenizer.tokenize_midi_dataset(single_midi_path, try_output_path) 
try_tokens = tokenizer.midi_to_tokens(single_midi)

print(try_tokens)