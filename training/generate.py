# -*- coding: utf-8 -*-
"""
#Generate Sequence
This pipeline is for generation a piece with given transformer model and an input sequence.
Used algorithms for decoding:
* greedy decoding
* top_k with/without temperature
* top_p (Nucleus Sampling)
"""

#Importing required libraries
#!pip install pretty_midi
#!pip install MidiTok
#!sudo apt install -y fluidsynth
#!pip install --upgrade pyfluidsynth

#from IPython.display import clear_output
#from IPython import display
clear_output(wait=False)
print("[INFO] !pip imports has been ended...")

import os
import pretty_midi as pm
import collections
import pandas as pd
import fluidsynth
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns

from config import Config
CONFIG = Config()

from model import create_mask, TransformerDecoder

#clear_output(wait=False)

print("All imports has been done successfully!")

"""###Utils
About audio visualization and saving the created .mid
"""

def midi_out(output_list, tokenizer, midi_name, save=False, check_err=True, output_path=""):
  midi_out = tokenizer.tokens_to_midi(output_list, [[24,False]])
  
  if save:
    midi_out.dump(output_path+f"{midi_name}.mid")
    print("Midi saved succesfully!")

  if check_err:
    print("Token types error rate is: {}".format(tokenizer.token_types_errors(output_list[0], consider_pad=True)))
  return midi_out

def playability(starts, ends, limit=6, verbose=False):
  """Checks if there are more than `limit` notes to play.   
  @Params:
  @starts: start times of the notes. Must be a list
  @ends: end times of the notes. Must be a list
  @limit: maximum number of notes to play exactly at the same time. Default to 6. 
  """
  assert len(starts)==len(ends)  
  data = []

  for k in starts:
    data.append([k, 0])

  for j in ends:
    data.append([j, 1])

  
  if verbose: print(data)
  sorted_data = sorted(data, key=lambda x: (x[0], -x[1]))
  if verbose: print(sorted_data)
  
  cur=0
  err_time = 0
  full_time = sorted_data[-1][0]
  for i in range(len(sorted_data)):
    if i != 0 and sorted_data[i-1][0] != sorted_data[i][0]:
      if cur > limit:
        err_time += sorted_data[i][0] - sorted_data[i-1][0]
    if sorted_data[i][1] == 0:
      cur+=1
    else:
      cur-=1
  if err_time > full_time:
    print(f"There is an error with notes")
  print(f"[INFO] Error rate is {(err_time/full_time*100)}")
  # return err_time

def display_audio(pm, seconds=120, _SAMPLING_RATE=44100):
  waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
  # Take a sample of the generated waveform to mitigate kernel resets
  waveform_short = waveform[:seconds*_SAMPLING_RATE]
  return display.Audio(waveform_short, rate=_SAMPLING_RATE)

def midi_to_notes(midi_file) -> pd.DataFrame:
  pm = midi_file
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def plot_piano_roll(notes: pd.DataFrame, count: int = 100):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  plt.figure(figsize=(20, 4))
  plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)

def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
  plt.figure(figsize=[15, 5])
  plt.subplot(1, 3, 1)
  sns.histplot(notes, x="pitch", bins=20)

  plt.subplot(1, 3, 2)
  max_step = np.percentile(notes['step'], 100 - drop_percentile)
  sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))
  
  plt.subplot(1, 3, 3)
  max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
  sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))

def plot_all(path, table=False, roll=True, dist=False, count=1000):
  """Combining all the possible statistics of the created samples. Default values are personal preferences. 
  """
  pm_data = pm.PrettyMIDI(path)
  raw_notes = midi_to_notes(pm_data)
  
  if table:
    raw_notes.head()
  if roll:
    plot_piano_roll(raw_notes, count=count)
  if dist:
    plot_distributions(raw_notes)

"""###Defining Variables
Configuring the transformer
"""

# create the model
num_layers, d_model, num_heads, dff, vocab_size2, MAX_LENGTH = CONFIG.num_layers, CONFIG.d_model, CONFIG.num_heads, CONFIG.dff, CONFIG.vocab_size2, CONFIG.MAX_LENGTH 
max_abs_position, use_bias, dropout_rate, layernorm_eps, tie_emb = CONFIG.max_abs_position, CONFIG.use_bias, CONFIG.dropout_rate, CONFIG.layernorm_eps, CONFIG.tie_emb
tokenizer = CONFIG.tokenizer2

#define the model
transformer = TransformerDecoder(
    num_layers, d_model, num_heads, dff, vocab_size2, MAX_LENGTH, max_abs_position,
    use_bias, dropout_rate, layernorm_eps, tie_emb
)

# build the model
_ = transformer(tf.random.uniform((1, CONFIG.MAX_LENGTH))) #These are transformers weights. But they are random right now. 
#So we should load our trained weigths 

del _

# load the weights
transformer.load_weights(CONFIG.model_save_path + "model_..._train_steps.h5") #saved transformers name will be here.

x, asd = transformer(tf.random.uniform((1, CONFIG.MAX_LENGTH))) #These are transformers weights. But they are random right now. 
print(asd)


"""##Nucleus Sampling 
"""

#top-p sampling, based on 'THE CURIOUS CASE OF NEURAL TEXT DEGENERATION' (Holtzman et al 2020)
def topPSampling(p,logits):
  sortedLogits = tf.sort(logits,direction='DESCENDING')
  #Softmax to get the probabilities
  sortedProbs = tf.nn.softmax(sortedLogits)

  #cumulative sum of the probabilities
  probsSum = tf.cumsum(sortedProbs, axis=1, exclusive=True)
  maskedLogits = tf.where(probsSum < p, sortedLogits, tf.ones_like(sortedLogits)*1000)  
  minLogits= tf.reduce_min(maskedLogits, axis=1, keepdims=True)  
  return tf.where(
      logits < minLogits,
      tf.ones_like(logits, dtype=logits.dtype) * -1e10,
      logits,
  )

def top_p_gen(transformer, input, genLen, top_p=0.8, verbose=False):
  """
  @Params
  @transformer: transformer model
  @input: input for the forward pass
  @genLen: desired length of generation
  @top_p: top_p value
  """
  # vectorizing the prompt
  input_vec = tf.convert_to_tensor(input)
  
  #list to store all the token generated
  generated_tokens = input[0]

  transformer.reset_states()

  for i in range(genLen):

      predictions, x = transformer(input_vec)
      if verbose:
        print(f"[DEBUG] predictions: {predictions}")
        print(f"[DEBUG] x: {x}")

      predictions = tf.squeeze(predictions, 0)

      predictionsNew = topPSampling(p=top_p ,logits=predictions)
      actualPrediction = tf.random.categorical(predictionsNew, num_samples=1)[-1,0].numpy()

      input_vec = tf.expand_dims([actualPrediction], 0)
      if verbose:
        print(f"[DEBUG] predictions: {predictions}")
        print(f"[DEBUG] predictionsNew: {predictionsNew}")
        print(f"[DEBUG] actualPrediction: {actualPrediction}")
      
      generated_tokens.append(actualPrediction)
  
  return generated_tokens

top_p=0.99
tokens_generated = top_p_gen(transformer=transformer, input=[[1]], genLen=100, top_p=top_p, verbose=False)
print(tokens_generated)
my_midi = midi_out(output_list, tokenizer=tokenizer, midi_name="try", save=False, check_err=True, output_path=output_path)

pm_data = pm.PrettyMIDI(output_path+midi_name+".mid") #NOTE: it's impossible to use this function if the created midi hasn't saved.
display_audio(pm_data, seconds=1, _SAMPLING_RATE=44100)

"""##Top_k Sampling
"""

def greedy_decode(transformer, inp, mode='categorical', temperature=1.0, k=None, skip_ends=0, memory=5000, vocab=CONFIG.vocab_size2):
  """
  Greedy decoding and top_k decoding.
  @Params
  @transformer: loaded transformer model.
  @inp: a list of tokens including the start token, already converted to tensor. 
  @mode: one of 'categorical' or 'argmax'
  @temperature: temperature value
  @k: top_k value
  """
  # check if temperature / k is a function
  if not callable(temperature):
    temperature_ = temperature; del temperature
    temperature = lambda x: temperature_

  if not callable(k) and k is not None:
    k_ = k; del k
    k = lambda x: k_

  # dimension for the mask
  n = tf.rank(inp) + 2 if tf.rank(inp) > 0 else 3

  # make inp 2d
  print(f"Greedy decoding input is: {inp}")
  inp = [tf.expand_dims(inp, 0)]

  # initialize attention weights in case inp.shape[-1] is already > max_len
  attention_weights = {}

  # maximum number of tokens to input to the model
  try:
    while True:
      predictions, attention_weights = transformer(inp[-1], training=False, 
                                                   mask=create_mask(inp[-1], n))

      # divide logits by temperature
      predictions /= temperature(inp[-1].shape[-1])

      # get last prediction
      if mode == 'argmax' or mode == 'a':
        prediction = tf.expand_dims(tf.argmax(predictions[..., -1, :], axis=-1, output_type=tf.int32), 0)
      elif k is not None:
        top_k_final_predictions = tf.math.top_k(predictions[..., -1, :], 
                                                k=k(inp[-1].shape[-1]))
        predicted_idx = tf.random.categorical(
            logits=top_k_final_predictions.values, 
            num_samples=1,
            dtype=tf.int32
        )
        predicted_idx = tf.squeeze(predicted_idx)
        prediction = tf.expand_dims(tf.expand_dims(top_k_final_predictions.indices[0, predicted_idx], 0), 0)
      elif mode == 'categorical' or mode == 'c':
        prediction = tf.random.categorical(logits=predictions[..., -1, :], num_samples=1, dtype=tf.int32)
      else:
        print(f"Unsupported mode '{mode}'. Use 'argmax' or 'categorical'")
        return None
    
      # return if prediction is end token
      if prediction == 2: # end token  
        if skip_ends <= 0:
          out = tf.concat(inp, axis=-1)
          return tf.squeeze(out)[1:], attention_weights
        else:
          skip_ends -= 1
          vec = inp[-1]
          inp.append(vec[:, :-memory])
          # maybe i need to put the start token here so that it actually ends at 1920 positions
          inp.append(vec[:, -memory:])
          inp = inp[:-3] + inp[-2:]
      # else concatenate last output to inp
      inp[-1] = tf.concat([inp[-1], prediction], axis=-1)
  except KeyboardInterrupt:
    pass
  out = tf.concat(inp, axis=-1)
  return tf.squeeze(out)[1:], attention_weights

def generate(transformer, inp, output_path='', mode='categorical', temperature=1.0, midi_name="my_midi_file",
             k=None, skip_ends=0, memory=5000, verbose=False, save=True, vocab=CONFIG.vocab_size2, tokenizer=CONFIG.tokenizer2):
  try:  
    # get the index list
    if verbose:
      print("Greedy decoding...")
      start = time.time()
      idx_list, attn_weights = greedy_decode(transformer, inp, mode, temperature, 
                                            k, skip_ends, memory, vocab=vocab)
      del attn_weights
      
      print(f"idx_list parameter is: {idx_list}")
      end = time.time()
      print(f"Generated {len(idx_list)} tokens.", end=" ")
      print(f"Time taken: {round(end - start, 2)} secs.")
    else:
      idx_list, attn_weights = greedy_decode(transformer, inp, mode, temperature, 
                                            k, skip_ends, memory, vocab=vocab)
      del attn_weights
    
    idx_tensor = tf.convert_to_tensor(idx_list)
    idx_list_final = idx_tensor.numpy().tolist()
    if verbose:
      print(f"idx_list is: {idx_list_final}")  

    #idx_list_final.insert(0,1) #adding the sos token
    
    midi_out([idx_list_final], CONFIG.tokenizer2, midi_name, save=save, check_err=True, output_path=output_path)
    
  except KeyboardInterrupt:
    print(f"idx_list is: {idx_list}")  

  return midi_out, idx_list_final

"""###Generating Single MIDI"""

inp=[1, 4, 146, 221, 17, 67, 97, 24, 67, 105, 29, 67, 105, 162, 221, 17, 67, 97, 170, 221, 24, 67, 89, 29, 67, 89]
print(f"Length of the input: {len(inp)}")

midi_name = "asd"
temperature = 1.1
k=40

my_greedy_midi, idx_list_final  = generate(transformer, inp, 
                       output_path=CONFIG.output_dir,
                       temperature=temperature, 
                       k=k, 
                       mode="categorical",
                       midi_name=midi_name,
                       memory=5000, 
                       save=True,
                       verbose=True, 
                       vocab=CONFIG.vocab_size2, 
                       tokenizer=CONFIG.tokenizer2)

pm_data = pm.PrettyMIDI(CONFIG.output_dir+midi_name+".mid") 
display_audio(pm_data, _SAMPLING_RATE=44100)

"""###Multiple Generation"""

def multiple_gen(transformer, inp, folder_name, output_path=CONFIG.output_dir, mode='categorical', 
                 temperature=1.0, midi_name="my_midi_file", k=None, skip_ends=0, memory=5000, 
                 wav=True, verbose=False, save=True, stats=True, vocab=CONFIG.vocab_size2, tokenizer=CONFIG.tokenizer2):
  """This function generates for every k and temperature it was given.
  @Params:
  @k: a list of k values.
  @temperature: a list of temperature values.
  @folder_name: folder name which the samples are going to be stored. E.g. 'adelita'
  """
  try:
    os.mkdir(output_path+folder_name)
  except OSError as error:
    print(f"OS Error: {error}")

  for k_param in k:
    for temperature_param in temperature:
      print(f"Generating for k: {k_param}, temperature: {temperature_param}...")
      generate(transformer, inp, 
                       output_path=output_path+folder_name+"/",
                       temperature=temperature_param, 
                       mode=mode, midi_name=midi_name+f"_k{k_param}_temp{temperature_param}",
                       memory=memory, save=save,
                       k=k_param, verbose=verbose, 
                       vocab=vocab, 
                       tokenizer=tokenizer)
      
  if stats:
    for k_param in k:
      for temperature_param in temperature:
        pm_data = pm.PrettyMIDI(output_path+folder_name+"/"+midi_name+f"_k{k_param}_temp{temperature_param}")
        raw_notes = midi_to_notes(pm_data)
        plot_piano_roll(raw_notes, count=1000)

inp = [1]
folder_name = "..."
initial_midi_name = "..."
k=[40, 50]
temperature=[1.0, 1.2]

multiple_gen(transformer,
             inp, 
             folder_name=folder_name,
             temperature=temperature, 
             k=k, 
             mode="categorical",
             midi_name=initial_midi_name,
             memory=5000, 
             save=True,
             verbose=False, 
             vocab=CONFIG.vocab_size2, 
             output_path=CONFIG.output_dir,
             tokenizer=CONFIG.tokenizer2)

"""##Statistics About the Created MIDI File"""

raw_notes = midi_to_notes(pm_data)
raw_notes.head()

starts = []
ends = []

for i in raw_notes["start"]:
  starts.append(i)

for i in raw_notes["end"]:
  ends.append(i)

playability(starts, ends, limit=3) #playability rate

plot_piano_roll(raw_notes, count=10000) #to see the created notes

plot_distributions(raw_notes)