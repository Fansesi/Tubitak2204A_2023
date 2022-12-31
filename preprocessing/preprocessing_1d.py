# -*- coding: utf-8 -*-
"""
#1D Preprocessing
This preprocessing file is for representations which are 1D like REMI, MIDI-Like, Structured.

What does this file do:
* Convert the tokens from .json files into tensors
* Split and concat the tensors accordingly
* Create `tf.data.Dataset` object and save it
"""

#Importing required libraries
#from IPython.display import clear_output #for cleaning outputs of the cell
from glob import glob
import tensorflow as tf
import json
import math

#clear_output(wait=False)

from config import Config 
CONFIG = Config()

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

import pickle
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

#Equalize shapes
def equalize_shapes_1D(input_token, target_shape, verbose=False):
  """Modifies the input token in order to match the target shape.
  @Params:
  @input_token: should be a list with a list inside. Integers are tokens. Expecting it's shape as [n]
  @target_shape: should be a list. Expecting [n]
  """
  token_tensor = tf.convert_to_tensor(input_token) #creating the tensor
  input_length = token_tensor.shape.as_list()[0] # [1,2,3,4,5,6,7 ...] total of n notes. 
  target_length = target_shape[0]

  if verbose:
    print("Input Length: {}".format(input_length))
    print("Target Length: {}".format(target_length))

  dummy_zero = tf.zeros((target_length-input_length), dtype=tf.dtypes.int32)

  try:   
    if target_length > input_length:
      if verbose:
        print(dummy_zero)
      output_token_tensor = tf.concat([token_tensor, dummy_zero], axis=0)
      if verbose:
        print("[INFO] Token has been padded.")
      return output_token_tensor
  except:
    raise RuntimeError

#Debug
y = [1,2,3,4,1,2,7,5]
y_modified = equalize_shapes_1D(y, [15], verbose=True)
print(y_modified)

#Splitting sequences
def splitting_1D(input_token, target_shape, verbose=False):
  """Cuts the given tensor into target_shape required times and returns cut tensors and remaining tensor. 
  @Params
  @input_token: should be a list. Inner ints are tokens.
  @target_shape: should be a list of a shape. Expecting [n]
  """
  input_tensor = tf.convert_to_tensor(input_token)
  input_length = input_tensor.shape.as_list()[0]
  target_length = target_shape[0]

  if verbose:
    print("Input Length: {}".format(input_length))
    print("Target Length: {}".format(target_length))

  split_n_times = math.floor(input_length/target_length)
  splitted_seqs = []

  splitted_tensor, remaining_tensor = tf.split(input_tensor, num_or_size_splits=(target_length, input_length-target_length), axis=0)
  remaining_tensor_main = remaining_tensor
  splitted_seqs.append(splitted_tensor)

  for i in range(split_n_times-1):
    splitted_tensor, remaining_tensor = tf.split(remaining_tensor_main, (target_length, remaining_tensor.shape.as_list()[0]-target_length), axis=0)
    remaining_tensor_main = remaining_tensor
    splitted_seqs.append(splitted_tensor)  

  return splitted_seqs, remaining_tensor_main

#Debug
my_tensor = [1,2,3,4,5,6,7,8,9,10,11,12]
splitted_tensors_list, remained_seq = splitting_1D(my_tensor, [5], verbose=True)
print(splitted_tensors_list)
print(remained_seq)

#Loading the tokens using pickle.
path=".../dataset/dataset_REMI_augmented/augmented_dataset.p"
all_tokens = in_pickle(path)

#Loading the tokens using json.
all_tokens_path = list(glob(".../dataset/dataset_REMI_tokenized/*.json"))

all_tokens, import_error_list = load_all_tokens(all_tokens_path, verbose=False)

if import_error_list != []:
  print(import_error_list)

for i in all_tokens[:6]:
  print(i)

#Finalizing the dataset preprocess, creating and saving tf.data.Dataset object.
def _gen_fn(tensor_list):
  for tensor in tensor_list:
      yield tensor #((tensor[:, 1:], tensor[:, :-1]))

def finalize(all_tokens_list, target_shape, output_dir="", save=False, verbose=False, stats=True, vocab_size=1):
  """ Combining both splitting() and eqalize_shapes() methods and finalizing the dataset.
  @Params
  @all_tokens_list: A list containing lists which contains just a list of integers. 
  E.g. midi_tokens_all => pieces => tokens
  @target_shape: Target shape of all the songs. Must be a list.
  @output_dir: Output directory to created dataset. Will not be used unless save=True. 
  Must be a list and within that list there must be strs. 0. elem is the input, 1. elem is the target save dir.
  @save: Save to the directory or not.
  @verbose: Debug purposes
  @stats: Debug purposes
  @vocab_size: Max token number, i.e. vocab. An integer.
  """

  target_length = target_shape[0]
  did_none = 0
  just_equalized = 0
  splitted = 0

  all_output_tensors = []

  for i, piece in enumerate(all_tokens_list):
    piece_length = len(piece)
    if verbose:
      print(f"{i}. piece length is: {piece_length}")

    if target_length > piece_length:
      just_equalized+=1
      #if verbose:
      #  print("[INFO] This piece has been padded in order to match the given shape.")
      output_tensor=equalize_shapes_1D(piece, target_shape, verbose=False)
      all_output_tensors.append(output_tensor) 
    
    elif target_length == piece_length:
      did_none+=1
      #if verbose:
      #  print("[INFO] Target length and the piece_length are alreay matched.")
      all_output_tensors.append(piece) # There is nothing to do so just add the tensor
    
    elif target_length < piece_length:
      splitted+=1
      #if verbose:
      #  print("[INFO] This piece has been splitted and padded in order to match the given shape.")
      splitted_tensor, remaining_tensor = splitting_1D(piece, target_shape, verbose=False)
      remaning_modified_tensor = equalize_shapes_1D(remaining_tensor, target_shape)
      for elem in splitted_tensor: 
        all_output_tensors.append(elem) # Appending the splitted tensors
      all_output_tensors.append(remaning_modified_tensor) # Appending the remaining but padded tensor
    else:
      print("It's just impossible...")

  if stats:
    print("="*22+"STATS"+"="*22)
    print("[INFO] {} many pieces was already in shape.".format(did_none))
    print("[INFO] {} many pieces has been padded.".format(just_equalized))
    print("[INFO] {} many pieces has been splitted and padded.".format(splitted))
    #print(f"[INFO] All output tensors are {all_output_tensors}")
  
  created_dataset = tf.data.Dataset.from_generator(_gen_fn, args=[all_output_tensors], output_types=tf.int32) 
  
  if save:
    print("[INFO] Saving dataset...")
    tf.data.experimental.save(created_dataset, output_dir) # Saving dataset
    # WARNING! This function is depreciated because of the version of tensorflow this project uses. For newer versions
    # please use tf.data.Dataset.save method. 
    
  return created_dataset

TARGET_LENGTH = 511 #Should be 1 less than what is wanted because in the training phase we are going to add SOS to input_ds and EOS to target_ds.

dataset_dir = ".../dataset/dataset_tf_REMI_512_aug"

finalized_dataset = finalize(
    all_tokens_list=all_tokens, 
    target_shape=[TARGET_LENGTH], 
    output_dir=dataset_dir, 
    save=False,
    verbose=False,
    stats=True)

for i in finalized_dataset.take(3):
  print(i)