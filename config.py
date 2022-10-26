import tensorflow as tf
#from MidiTok import OctupleMono

class Config():
  def __init__(self):
    #super().__init__(self)

    ### For training
    self.GLOBAL_BATCH_SIZE = 1

    self.num_layers = 6
    self.d_model = 128 #was 256
    self.dff = 512 #was 1024
    self.num_heads = 8

    self.MAX_LENGTH = 512*6
    self.MAX_REL_DIST = 512*6
    self.max_abs_position = 1

    self.use_bias = True
    self.tie_emb = False
    self.layernorm_eps = 1e-06

    self.vocab_size = 1572  # Experiment with this value as much as you can.

    self.dropout_rate = 0.1

    self.checkpoint_path = "/content/drive/MyDrive/tubitak2204A_2023/trained_models/checkpoints"

    self.epoch_number = 20

    ### For preprocessing
    self.pitch_range = range(40, 85) #classical guitar is E2 to C6. Values taken from constants.py file.
    self.beat_res = {(0, 4): 8, (4, 12): 4}
    self.nb_velocities = 32
    self.additional_tokens = {'Chord': False, 'Rest': False, 'Tempo': True, 'Program': False, 'TimeSignature': True,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)

    #self.tokenizer = OctupleMono(self.pitch_range, self.beat_res, self.nb_velocities, self.additional_tokens, mask=True)
    
