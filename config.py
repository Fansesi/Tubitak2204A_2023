# -*- coding: utf-8 -*-
"""
#Config File
In order to store some values and easily experiment with them.
"""

import tensorflow as tf
#!pip install MidiTok           # Miditok requeired
from miditok import OctupleMono
from miditok import REMI
import math

class Config():
  def __init__(self):
    #super().__init__(self)
    
    ### Training and Generating parameters
    self.GLOBAL_BATCH_SIZE = 16
    self.epoch_number = 500
    self.warmup_steps = 4000 

    # Validation Parameters
    self.val_div = 4

    # Early Stopping Mechanism Parameters
    self.patience = 3

    # Checkpointing Parameters
    self.ckpt_interval = 1 
    self.max_to_keep = 3
    
    # Specified Directories
    self.dataset_dir = ".../dataset/dataset_tf_REMI_512_aug" 
    self.checkpoint_path = ".../trained_models/checkpoints"
    self.model_save_path = ".../trained_models/models/" # is used like CONFIG.model_save_path + "name_of_the_model.h5"
    self.output_dir = ".../samples/"

    # Model Parameters
    self.num_layers = 6 # models layers
    self.d_model = 512 # models dimension 
    self.dff = 1024 # it's the dimension of the fully connected (tf.layers.Dense) feedforward network.
    self.num_heads = 8 # number of heads in the model

    self.MAX_REL_DIST = self.MAX_LENGTH = 512  
    self.max_abs_position = 1

    self.use_bias = True
    self.tie_emb = False
    self.layernorm_eps = 1e-06

    self.dropout_rate = 0.1
    
    ### For preprocessing and generating
    self.pitch_range = range(40, 85) # Classical guitar is from E2 to C6. 
    self.beat_res = {(0, 4): 8, (4, 12): 4}
    self.nb_velocities = 32
    
    #Octuple Tokenization Strategy - It's for experimenting, hasn't been used in the project.
    self.additional_tokens1 = {'Chord': False, 'Rest': False, 'Tempo': True, 'Program': False, 'TimeSignature': True,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)
    self.tokenizer1 = OctupleMono(self.pitch_range, self.beat_res, self.nb_velocities, self.additional_tokens1, sos_eos=True, mask=True)  
    self.vocab_size1 = len(self.tokenizer1.vocab) #Octuple tokenization is problematic. It returns the value of 6.
    
    #REMI Tokenization Strategy
    self.additional_tokens2 = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)
    self.tokenizer2 = REMI(self.pitch_range, self.beat_res, self.nb_velocities, self.additional_tokens2, sos_eos=True, mask=True)  
    self.vocab_size2 = len(self.tokenizer2.vocab) #Vocabulary - 236 tokens of 12 types.