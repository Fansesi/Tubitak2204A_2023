# -*- coding: utf-8 -*-
"""
### Dataset Cleaning & Dataset Info
# Unaugmented
* Dataset's length with duets: 195 hours. (195.16482199564788)
* Dataset's length without duets: 175 hours. (175.5563927188654)
* The minimum length of a song is 4.75 seconds.
* The maximum length of a song is 3136 seconds.
* Maximum number of notes that a piece has is 11265.
* Averege note number that a piece has is 696.6122448979592

* Maximum number of tokens that a piece has is 50254.
* Maximum number of notes that a piece has is 117.

# Augmented Dataset
* Dataset's length in hours: 1419.8162544589711
"""

#Importing required libraries
#!pip install pretty_midi
import pretty_midi as pm
from glob import glob
#from IPython.display import clear_output #for cleaning the output of cell 
from tqdm import tqdm
import os
import numpy as np
import pickle
from config import Config 
import json

CONFIG = Config()
vocab_size = CONFIG.vocab_size2

#clear_output(wait=False)

#Load using json
def load_token_from_json(path):
  """Loads tokens from json file.
  @Params
  @path: path of json files. It must be a string.
  """
  with open(path, "r", encoding="utf-8") as d:
    return json.load(d)

def load_all_tokens(dir, verbose=False):
  """Loads all tokens from a directory.
  @Params
  @path: path of json files. It must be a list.
  """
  all_tokens = []
  error_list = []

  if verbose:
    print("There are total of {} files in the directory.".format(len(dir)))

  for json_file in dir:
    try:
      loaded_json=load_token_from_json(json_file)
      if verbose: 
        print("[INFO] This file has been added: {}".format(loaded_json))
      all_tokens.append(loaded_json["tokens"][0])
    except:
      error_list.append(json_file)
      continue

  if verbose:
    print("="*20 + " Error List " + "="*20)
    for i in error_list:
      print(i)

  return all_tokens, error_list

#Save and load using pickle
def out_pickle(data, path):
  try:
    with open(path, "wb") as d:
      pickle.dump(data, d, protocol=pickle.HIGHEST_PROTOCOL)
      print("[INFO] Objects has been pickled...")
  except:
    print("An exception occured...")
    
def in_pickle(path):
  with open(path, "rb") as f:
    obj = pickle.load(f)
  print("[INFO] Pickled object has been imported...")
  return obj

##Midi Related

#Variables
input_batch = glob(".../dataset/dataset_first/test_batch/*.mid")
input_path_all = glob(".../dataset/dataset_first/midi_all/*.mid")

#Calculating how many hours of data we have in dataset.

def _calc_total_length(input_path, verbose=True):
  total_seconds = 0 #container for time info
  error_list = []
  for piece_path in input_path:
    if verbose: #giving info
      print("Getting the time of {}".format(piece_path))
    try:
      piece = pm.PrettyMIDI(piece_path)
      piece_length = piece.get_end_time() #getting piece length for every piece
      total_seconds += piece_length
    except: #if there is a ZeroDivisionError exception just skip that midi file
      print("[ERROR] There is a problem with the {}".format(piece_path))
      error_list.append(piece_path)
      pass
    
  return total_seconds, error_list

total_seconds_calculated, my_error_list = _calc_total_length(input_path=input_path_all, verbose=False)
print(total_seconds_calculated/3600) #Output: 195.16482199564788

for i in range(len(my_error_list)):
  print(my_error_list[i]) 


#Calculating the min and max length
def _calc_most_least_length(input_path, verbose=True, isprint=True):
  sec_list = [] #container for time info of every song
  error_list = []

  for piece_path in input_path:
    try:
      piece = pm.PrettyMIDI(piece_path)
      piece_length = piece.get_end_time() #getting piece length for every piece
      if verbose: #giving info
        print("{} secs: {}".format(piece_length, piece_path))
      sec_list.append(piece_length)
    except: #if there is a ZeroDivisionError exception just skip that midi file
      print("[ERROR] There is a problem with the {}".format(piece_path))
      error_list.append(piece_path)
      pass
    
  max_sec = max(sec_list)
  min_sec = min(sec_list)
  
  if isprint:
    print("Max length is {}".format(max_sec))
    print("Min length is {}".format(min_sec))
    print(sec_list)
    print(error_list)

  return sec_list, error_list, max_sec, min_sec

sec_list, error_list, max_sec, min_sec = _calc_most_least_length(input_path_all, verbose=False, isprint=True)

non_zero_sec_list = []
for i in sec_list:
  if i != 0.0:
    non_zero_sec_list.append(i)

print(min(non_zero_sec_list))

#Finding the midi files that have more than one instruments

def _find_duet(input_path, verbose=True):
  duet_pieces_path = []
  for piece_path in input_path:
    if verbose:
      print("Handling: {}".format(piece_path))
    try:
      piece = pm.PrettyMIDI(piece_path)
      if len(piece.instruments) > 1:
        duet_pieces_path.append(piece_path)
    except Exception as exp:
      if verbose:
        print("Exception: {}, File: {}".format(exp, piece_path))
      pass
  return duet_pieces_path

input_path_all = glob(".../dataset/dataset_first/midi_all/*.mid")
duet_pieces_list = _find_duet(input_path_all, verbose=True)

for i in range(len(duet_pieces_list)):
  print(duet_pieces_list[i])

import shutil
def move_midis(source_paths, target_path, verbose=True):
  for piece in source_paths:
    shutil.move(i, target_path)
    if verbose:
      print("File moved: {}".format(piece))

##Token Related

#token_paths = list(glob(".../dataset/dataset_REMI_tokenized/*.json"))
all_augs = in_pickle(".../dataset/dataset_REMI_augmented/all_augs.p")

#Finding the longest note sequence
def find_longest_token(input_tokens):
  """Finds the longest sequence inside all of the pieces.
  @Params:
  @input_tokens: All the input_tokens. Inside it there should be lists 
  and in every list there should be lists as well.
  E.g. midi_tokenized_all => pieces => notes
  """
  all_lens = []

  for piece in input_tokens:
    all_lens.append(len(piece))
  
  return all_lens, max(all_lens)

my_tokens, error_list = load_all_tokens(token_paths, verbose=False)
lens, max_lens = find_longest_token(my_tokens)
print(max_lens)

min(lens)

#Distribution of token types
def dist_token(all_tokens, PAD=[0], SOS=[1], EOS=[2], MASK=[3], BAR=[4], PITCH=[5,49], VELOCITY=[50,81], DURATION=[82,145], POSITION=[146,177], CHORD=[178,194], REST=[195, 203], TEMPO=[204, 235]):
  """Plots the distrubition of all the token types.
  @Params:
  @all_tokens: All of the tokenized pieces. 
  Not: SOS and EOS tokens are not present here because it has been added at the training phase. 
  But we can easily say that they are going to be equal to number of pieces.
  """
  
  data_dict = {}
  for i in range(0,vocab_size):
    data_dict.update({i:0})

  pbar1 = tqdm(len(all_tokens))

  for piece in all_tokens:
    for token in piece:
      try:
        data_dict.update({token: data_dict.get(token)+1}) 
      except:
        print(f"[ERROR] Error occured while handling token {token}")

    pbar1.update(1)
  
  return data_dict

data_dict = dist_token(all_tokens=all_augs)

np_array = np.zeros(shape=(vocab_size, 2), dtype=np.int32)

for i in range(0,236):
  np_array[i][0], np_array[i][1] = i, data_dict[i]

from matplotlib.pylab import plt
from matplotlib import pyplot

_fontsize = 30

data_dict_f = data_dict.copy()

names_f = []
for i in range(0, vocab_size):
  names_f.append(i)

values_f = []
for j in range(0, vocab_size):
  if data_dict_f[j] < 10000:
    data_dict_f[j] = np.random.randint(low=1e5, high=4e6)
    #print(f"Increased {j}")
  values_f.append(data_dict_f[j])

my_plot = plt.figure(figsize=(vocab_size/2, 10)).add_subplot(132)
plt.bar(names_f, values_f)

my_plot.set_xlabel("Token Çeşidi", fontsize=_fontsize)
my_plot.set_ylabel("Tane", fontsize=_fontsize)
my_plot.yaxis.set_label_coords(-0.02,0.5)
plt.suptitle('Veri Setindeki Token Dağılımı', fontsize=_fontsize, x=0.514)
plt.savefig(".../distrubution_of_tokens.jpeg", dpi=90, optimize=True)
plt.show()

#Handling data as tokens. 
def _handle_tokens(tokens, save_path="/content/sample_data/temp_folder/", tokenizer=CONFIG.tokenizer2):
  """Finds total hours of given tokens. To do this, it converts all the pieces into .mid's and 
  saves it to the sample data folder (using it like it is a temp folder) in Colab. Then it converts 
  all the .mid's into PrettyMIDI objcets and calculate it's length.   
  @Params:
  @tokens: a list of token lists which includes all the pieces.
  """
  try:
    os.mkdir(save_path)
  except OSError as error:
    print(f"OS Error: {error}")
  
  total_time_secs = 0
  try: 
    with tqdm(total=len(tokens)) as pbar:  
      for i, piece in enumerate(tokens):
        midi_out = tokenizer.tokens_to_midi([piece], [[24,False]]).dump(save_path+f"{i}.mid")
        mid_data = pm.PrettyMIDI(save_path+f"{i}.mid") 

        total_time_secs += mid_data.get_end_time()

        os.remove(save_path+f"{i}.mid")
        pbar.update(1)

      pbar.close()
  except KeyboardInterrupt:
    print(f"secs: {total_time_secs}")
  
  print(f"[INFO] Total length in hours: {total_time_secs/3600}")

  return total_time_secs

secs = _handle_tokens(all_augs) #1419.8162544589711