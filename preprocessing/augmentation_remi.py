# -*- coding: utf-8 -*-
"""
#Augmentation For REMI Tokenized Audio Files
Augmenting the tokens and saving them to a given directory.
"""

#@title Importing required libraries
#!pip install pathlib
#from IPython.display import clear_output #for cleaning the output of cell  

from glob import glob
from pathlib import Path
import json
#clear_output(wait=False)

import pickle
#@title Save and load using pickle
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

path=".../dataset/dataset_REMI_augmented/all_augs.p"
all_augs_imported = in_pickle(path)
print(len(all_augs_imported))

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

#Save using json
def save_token_as_json(token, path, name):
  """Saves tokens to a json file.
  @Params
  @token: token to save.
  @path: path of output directory. Must be a string. E.g: "asd/asd/asd/" Don't forget the slash at the end!
  @name: name of the file being saved. Must be a 
  Note: I'm creating the program token as well. Because load_all_tokens() function
  thinks that the json file is like {tokens:[...], program:[...]}. I'm going to use the dict
  but I'm not going to add program parameter because that is unneccessary.
  """
  token_prepared = {"tokens":[token]}
  with open(path+f"{name}.json", "w", encoding="utf-8") as d: 
    json.dump(token_prepared, d)

def save_all_tokens(all_token_list, out_dir, verbose=False):
  """Saves all tokens to a given directory.
  @Params
  @all_token_list: tokens to save.
  @out_dir: path of output directory.
  """
  error_list = []

  for i, token_list in enumerate(all_token_list):
    try:
      save_token_as_json(token=token_list, path=out_dir, name=i)
    except:
      error_list.append(token_list)
      continue

  if verbose:
    print("="*20 + " Error List " + "="*20)
    for i in error_list:
      print(i)

  return error_list

#Loading the tokens 
all_tokens_path = list(glob(".../dataset/dataset_REMI_tokenized/*.json"))

all_tokens, import_error_list = load_all_tokens(all_tokens_path, verbose=False)

if import_error_list != []:
  print(import_error_list)

for i in all_tokens[:3]:
  print(i)

def find_index(a_list, min_val, max_val):
  """Returns the indexes of a value between min_val and max_val in a dict.
  """
  index_dict = {} #indexes of tokens
  for i, token in enumerate(a_list):
    if token>=min_val and token<=max_val:
      index_dict.update({i:token})
  
  return index_dict #index of the pitch token : pitch token itself

def aug(token_list, min_val, max_val, change_vals=[-2,2], verbose=False):
  """Finds the tokens between min_val and max_val and sums them for every change_vals value.
  If a change_val+token>max_val or change_val+token<min_val, we don't augment with that change_val.
  @Params:
  @token_list: list of tokens
  @min_val: min_val to augment. This value sets the boundaries of augmentation.  
  @max_val: max_val to augment. This value sets the boundaries of augmentation.
  @change_vals: a list of [i,j] (integer i,j's), amount of augmentation
  @verbose: for debug purposes
  """
  augmented_tokens=[]
  total_aug = 0 

  index_dict = find_index(token_list, min_val, max_val)

  if verbose:
    print(index_dict)

  for change_val in range(change_vals[0],change_vals[1]+1):
    if change_val == 0:
      augmented_tokens.append(token_list)
      total_aug+=1
    
    elif change_val<0:
      for index in index_dict:
        pitch_token = index_dict[index]
        if pitch_token+change_val < min_val:
          augmentable=False
          break
        else:
          augmentable=True
          continue
      
      if augmentable:
        copy_list1 = token_list.copy() 
        for index in index_dict:
          pitch_token = index_dict[index]
          copy_list1[index] = pitch_token+change_val
        augmented_tokens.append(copy_list1)
        total_aug+=1

    else: #change_val>0
      for index in index_dict:
        pitch_token = index_dict[index]
        if pitch_token+change_val > max_val:
          augmentable=False
          break
        else:
          augmentable=True
          continue
      
      if augmentable:
        copy_list2 = token_list.copy() 
        for index in index_dict:
          pitch_token = index_dict[index]
          copy_list2[index] = pitch_token+change_val
        augmented_tokens.append(copy_list2)
        total_aug+=1

  if verbose:
    print(f"Total amount of augmentation: {total_aug}")

  return augmented_tokens, total_aug

###Pitch Augmentation

def aug_pitch(token_lists, min_val=5, max_val=49, half_steps=[-2,2], verbose=False):
  """Augments all the pitches in every given piece.
  @Params:
  @token_lists: lists of tokens in order words all the pieces to augment.
  @min_val: min_val to augment. This value sets the boundaries of augmentation.  
  @max_val: max_val to augment. This value sets the boundaries of augmentation.
  @half_steps: a list of [i,j] (integer i,j's), amount of augmentation.
  @verbose: for debug purposes
  """

  total_pitch_aug = 0
  all_auged_pieces = []

  for token_list in token_lists:
    augmented_tokens, total_aug = aug(token_list, min_val, max_val, half_steps, verbose=False)
    total_pitch_aug+=total_aug
    for i in augmented_tokens:
      all_auged_pieces.append(i)

  if verbose:
    print(f"Total number of augmented pieces: {total_pitch_aug}")
  return all_auged_pieces, total_pitch_aug

all_pitch_augs, total_pitch_aug = aug_pitch(all_tokens, verbose=True)

##Tempo Augmentation

def aug_temp(token_lists, min_val=204, max_val=235, temp_changes=[-2,2], verbose=False):
  """Augments all the tempo tokens in every given piece.
  @Params:
  @token_lists: lists of tokens in order words all the pieces to augment.
  @min_val: min_val to augment. This value sets the boundaries of augmentation.  
  @max_val: max_val to augment. This value sets the boundaries of augmentation.
  @temp_changes: a list of [i,j] (integer i,j's), amount of augmentation.
  @verbose: for debug purposes
  """

  total_tempo_aug = 0
  all_auged_pieces = []

  for token_list in token_lists:
    augmented_tokens, total_aug = aug(token_list, min_val, max_val, temp_changes, verbose=False)
    total_tempo_aug+=total_aug
    for i in augmented_tokens:
      all_auged_pieces.append(i)

  if verbose:
    print(f"Total number of augmented pieces: {total_tempo_aug}")
  return all_auged_pieces, total_tempo_aug

all_temp_augs, total_tempo_aug = aug_temp(all_tokens, verbose=True)

##Saving the Augmented Tokens

def save_all_augs(all_auged_pieces, dir):
  """Saves the augmented pieces and saves them to a given directory.
  @Params:
  @all_auged_pieces: lists of tokens, in order words all the augmented pieces. 
  In this implementation we merge the pitch augs and tempo augs into a single list 
  and give that as this parameter.
  @dir: directory to save the augmented pieces.

  DEPRECIATED: Because it makes no sense saving all the tokens individually. 
  I can keep the all_augs as it is and save them using json or pickle.
  """
  err_list = save_all_tokens(all_auged_pieces, dir)
  
  if err_list!=[]:
    print(f"Error list while saving the tokens: \n{err_list}")
    return err_list
  else:
    print("No errors occured while saving the tokens.")

out_dir = "/content/drive/MyDrive/tubitak2204A_2023/dataset/dataset_REMI_augmented/" 
all_augs = all_temp_augs+all_pitch_augs

cleared_augs = []

for i in all_augs:  #It's a little bit dirty but it does remove the duplicates (original tokens are counted twice)
    if i not in cleared_augs:
        cleared_augs.append(i)

print(f"Total number of data in hand after augmentation: {len(cleared_augs)}")

out_pickle(cleared_augs, out_dir+"augmented_dataset.p")

single_midi = all_tokens[0]
print(single_midi)
ali=[]
for i in range(4,len(single_midi),5):
  print(single_midi[i])
  ali.append(single_midi[i])
print("Finished")
for elem in ali:
  assert elem<=49 and elem>=5
