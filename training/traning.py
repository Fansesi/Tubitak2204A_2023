"""# Dataset Importing and Basic Processing
Importing `tf.data.Dataset` object and creating train, val, test datasets.

IMPORTANT NOTE: `tf.data.experimental.load()` function is depreciated in newer versions of Tensorflow. 
To fix this problem while using a newer version of Tensorflow, one could use the dataset.save and dataset.load method. More info at: 
* https://www.tensorflow.org/api_docs/python/tf/data/Dataset/#save 
* https://www.tensorflow.org/api_docs/python/tf/data/Dataset/#load
"""

"""## Importing and Splitting the Prepared Dataset"""

#Importing required libraries
#from IPython.display import clear_output
import json
from pathlib import Path
import numpy as np
import math
from glob import glob
import tensorflow as tf
import os
import time

from config import Config 
CONFIG = Config()

# %cd .../training
import model

#clear_output(wait=False)

print(f"Tensorflow version is: {tf.__version__}")
print("[INFO] Imports has been made successfully...")

#Utility Functions
def print_ds(ds, num=2):
  """Prints the first num elements of a ds."""
  for i in ds.take(num):
    print(i)

def val_at(length, div):
    """Returns a number about how many train steps should be completed before validating the model.
    @Params
    @length: a length. Such as len(train_ds).
    @div: a divisor. If the len(train_ds) is all the batches we have, setting div param to 2 would
    model to be validated at every half of the epoch.
    """
    return math.ceil(length/div)

#Specifying the dataset path.
dataset = tf.data.experimental.load(CONFIG.dataset_dir)
print(f"Length of the imported dataset: {len(dataset)}")

# to define the shape of the dataset.
dataset = tf.data.Dataset.from_tensor_slices(list(dataset)) 

print(dataset)
print(f"Length of the dataset after slicing: {len(dataset)}")
dataset.element_spec.shape
print(dataset)
print_ds(dataset)

"""###Function for creating train, validation and test datasets"""

def get_dataset_partitions(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12) # Specify seed to always have the same split distribution between runs
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions(ds=dataset, ds_size=len(dataset), train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=1000)

print(f"Traning dataset size before: {len(train_ds)}")
print(f"Validataion dataset size before: {len(val_ds)}")

train_ds = tf.data.Dataset.from_tensor_slices(list(train_ds)[:(len(train_ds)-(len(train_ds)%CONFIG.GLOBAL_BATCH_SIZE))]) 
val_ds = tf.data.Dataset.from_tensor_slices(list(val_ds)[:(len(val_ds)-(len(val_ds)%CONFIG.GLOBAL_BATCH_SIZE))]) 
                                              #this way we are ensuring that length of the dataset is always divisible by GLOBAL_BATCH_SIZE 

print(f"Traning dataset size after: {len(train_ds)}")
print(f"Validation dataset size after: {len(val_ds)}")
print(f"Test dataset size: {len(test_ds)}")

print_ds(train_ds)
print_ds(val_ds)

"""###Preparing the Train and Validation Dataset
"""

BUFFER_SIZE = len(dataset) 
GLOBAL_BATCH_SIZE = CONFIG.GLOBAL_BATCH_SIZE

train_ds = train_ds.shuffle(BUFFER_SIZE)
val_ds = val_ds.shuffle(BUFFER_SIZE)

train_ds = train_ds.map(lambda x: (tf.concat([[1], x], axis=0), tf.concat([x, [2]], axis=0)))
train_ds = train_ds.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
#We don't need drop_remainder param for the .batch() function because is len(dataset) already divisible by batch_size

val_ds = val_ds.map(lambda x: (tf.concat([[1], x], axis=0), tf.concat([x, [2]], axis=0)))
val_ds = val_ds.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

print_ds(train_ds)
print_ds(val_ds)

"""#Training Variables
Some variables which are being used by both custom training strategy and Tensorflow's `model.fit()` function.

"""

MAX_LENGTH = CONFIG.MAX_LENGTH
MAX_REL_DIST = CONFIG.MAX_REL_DIST

num_layers = CONFIG.num_layers
d_model = CONFIG.d_model 
dff = CONFIG.dff
num_heads = CONFIG.num_heads

max_rel_dist = MAX_REL_DIST 
max_abs_position = CONFIG.max_abs_position

use_bias = CONFIG.use_bias
tie_emb = CONFIG.tie_emb
layernorm_eps = CONFIG.layernorm_eps
dropout_rate = CONFIG.dropout_rate

warmup_steps = CONFIG.warmup_steps

vocab_size = CONFIG.vocab_size2 

checkpoint_path = CONFIG.checkpoint_path

epoch_number=CONFIG.epoch_number
epochs = tf.convert_to_tensor(math.ceil(epoch_number))
print(epochs)

"""## Configuring the Transformer Model"""

transformer = model.TransformerDecoder(
      num_layers, d_model, num_heads, dff, vocab_size, MAX_REL_DIST, max_abs_position,
      use_bias, dropout_rate, layernorm_eps, tie_emb
  )

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
  from_logits=True,
  reduction=tf.keras.losses.Reduction.NONE
  )

start_epoch = tf.Variable(0) # to handle restarting training from a checkpoint

"""##Custom Learning Rate Scheduler"""

#from matplotlib import pyplot as plt
#plt.plot(model.CustomSchedule(512, warmup_steps=1000)(tf.range(86800, dtype=tf.float32))) #warmup_steps=1000 seems ok. 
#plt.ylabel("Öğrenme Katsayısı")
#plt.xlabel("Eğitim Adımı")
#plt.savefig(".../lr_fig.png", dpi=1080, bbox_inches='tight')

# setting the optimizer up
learning_rate = model.CustomSchedule(d_model, warmup_steps=warmup_steps) 

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, #learning_rate
                                     epsilon=1e-09)

"""
# Custom Traning
Setting up the hyperparameters and training the transformer using custom training functions.
"""

"""
## Setting Up the Checkpointing System
"""

# set up the checkpoints
ckpt_interval = CONFIG.ckpt_interval # checkpoint every ckpt_interval epochs

checkpoint = tf.train.Checkpoint(transformer=transformer,
                                 optimizer=optimizer,
                                 epoch=start_epoch)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, 
                                          max_to_keep=CONFIG.max_to_keep).
if ckpt_manager.latest_checkpoint:
  checkpoint.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored.')
  print(f'Training will resume from epoch {start_epoch.numpy()}.')
  print(f'{optimizer.iterations.numpy()} train steps have already been completed.')

"""##Loss Function"""

def loss_function(target, predictions, criterion=loss_object):
  """
  If defining custom criterion, make sure reduction is none
  """
  mask = tf.not_equal(target, tf.zeros_like(target))
  _loss = criterion(target, predictions)
  mask = tf.cast(mask, _loss.dtype) # make mask of same dtype as loss
  _loss *= mask
  return tf.reduce_sum(_loss) / tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss') 
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val accuracy')

"""## Training
Defining train steps and validation steps and starting to train the model.
"""

# define the train step and validation step functions
def train_step(inputs, target):
  # forward pass
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inputs, training=True, mask=model.create_mask(inputs))
    loss = loss_function(target, predictions) #/ (MAX_LENGTH)
  
  # update weights
  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  # accuracy
  train_accuracy(target, predictions)
  train_loss(loss)
  return loss

def val_step(inputs, target):
  # forward pass
  predictions, _ = transformer(inputs, training=True, mask=model.create_mask(inputs))
  loss = loss_function(target, predictions) #/ (MAX_LENGTH)

  # accuracy
  val_accuracy(target, predictions)
  val_loss(loss)

  return loss

# initialise dictionaries to store the training and validation losses for plotting
train_acc_dict = {}
val_acc_dict = {}
train_loss_dict = {}
val_loss_dict = {}

#initialize early stopping parameters
patience = CONFIG.patience #amount of times that val_loss can be higher than the previous val_loss. 
wait = 0 #is for checking out that if we reached the patience level or not
best = 0 #best val_loss
index = 0 #is for defining the first best val_loss

# train loop
try:
  for epoch in range(start_epoch.numpy(), epochs):
    start = time.time()
    batch_timer = time.time()
    
    # train steps
    train_loss.reset_states()
    train_accuracy.reset_states()

    val_loss.reset_states()
    val_accuracy.reset_states()

    # Training
    for batch, train_inp_tar in enumerate(train_ds):       
      train_inp, train_tar = train_inp_tar
      #-------------Debug--------------
      #print(batch, end=": ")
      #print(f"Inp: {train_inp}")
      #print(f"Tar: {train_tar}")

      train_step(train_inp, train_tar)

      if (batch + 1) % 201 == 0 or batch == 0: 
        print(f"Processing Epoch {epoch} Train Batch {batch} " \
              f"Training Loss {round(train_loss.result().numpy().item(), 6)} ", end="") 
        print(f"Training Accuracy {round(train_accuracy.result().numpy().item(), 6)} " \
              f"Time taken {round(time.time() - batch_timer, 2)} secs")
        batch_timer = time.time()
        
      # Validation
      if (batch + 1) % val_at(length=len(train_ds), div=CONFIG.val_div) == 0: 
        val_timer = time.time()
        print(f"Validating After Training Step {optimizer.iterations.numpy()}...")
        for batch, val_inp_tar in enumerate(val_ds):       
          val_inp, val_tar = val_inp_tar
          val_step(val_inp, val_tar)
        print(f"Val Loss {round(val_loss.result().numpy().item(), 6)}", end=" ")
        print(f"Val Accuracy {round(val_accuracy.result().numpy().item(), 6)}", end=" ")    
        print(f"Time taken for validation {round(time.time()-val_timer, 2)} secs")

        # Updating the loss and accuracy dictionaries after validation  
        train_loss_dict[optimizer.iterations.numpy()] = train_loss.result()
        val_loss_dict[optimizer.iterations.numpy()] = val_loss.result()
        train_acc_dict[optimizer.iterations.numpy()] =  train_accuracy.result()
        val_acc_dict[optimizer.iterations.numpy()] = val_accuracy.result()

        # Early Stopping Mechanism
        if index==0: #to define the first best validation loss.
          best = val_loss.result()
          index+=1

        wait += 1
        if val_loss.result() < best:
          best = val_loss.result()
          wait = 0

    if wait >= patience:
      print("="*20+"Training Stopped Early"+"="*20)
      print(f"{optimizer.iterations.numpy()} train steps have been computed.")
      print(f"Current Train Loss {round(train_loss.result().numpy().item(), 6)} and " \
            f"Train Accuracy {round(train_accuracy.result().numpy().item(), 6)} \n"
            f"Current Val Loss {round(val_loss.result().numpy().item(), 6)} and "\
            f"Val Accuracy {round(val_accuracy.result().numpy().item(), 6)}\n")
      
      # Saving the model
      model_save_path = CONFIG.model_save_path + f"model_{optimizer.iterations.numpy()}_train_steps.h5"
      print(f"Saving at {model_save_path}...", end="")
      transformer.save_weights(model_save_path)
      print("Done!")
      break
      
    # Checkpointing 
    if (epoch + 1) % ckpt_interval == 0:
      start_epoch.assign(epoch + 1)
      print("Checkpointing...", end="")
      save_path = ckpt_manager.save()
      print(f"Done! Saved at {save_path}")
    
    # Informing about the training/validation loss/accuracy
    print(f"Epoch {epoch} "\
          f"Train Loss {round(train_loss.result().numpy().item(), 6)} " \
          f"Train Accuracy {round(train_accuracy.result().numpy().item(), 6)}", end=" ")
    print(f"Val Loss {round(val_loss.result().numpy().item(), 6)} "\
          f"Val Accuracy {round(val_accuracy.result().numpy().item(), 6)}")
    print(f"Time taken for 1 epoch {round(time.time() - start, 2)} secs\n")
    
    # If its's the end of training:
    if epoch+1 == epochs.numpy():
      print(50*"=")
      print("Training finished, saving the model...")
      model_save_path = CONFIG.model_save_path + f"model_{optimizer.iterations.numpy()}_train_steps.h5"
      print(f"Saving at {model_save_path}...", end="")
      transformer.save_weights(model_save_path)
      print("Done!")
      break

# If user interrupts the training
except KeyboardInterrupt:
  print("\nKeyboard Interrupt")
  print(f"{optimizer.iterations.numpy()} train steps have been computed.")
  print(f"Current Train Loss {round(train_loss.result().numpy().item(), 6)} and " \
        f"Train Accuracy {round(train_accuracy.result().numpy().item(), 6)} \n"
        f"Current Val Loss {round(val_loss.result().numpy().item(), 6)} and "\
        f"Val Accuracy {round(val_accuracy.result().numpy().item(), 6)}\n")
  save = input("Save the model?\n")
  if save == 'y' or save == 'yes':
    model_save_path = CONFIG.model_save_path + f"model_{optimizer.iterations.numpy()}_train_steps.h5"
    print(f"Saving at {model_save_path}...", end="")
    transformer.save_weights(model_save_path)
    print("Done!")


"""
##Plotting the Loss Graphs
"""

from matplotlib.pylab import plt

def plot_loss(train_loss_dict, val_loss_dict, loss_acc, save=False): #train_acc_dict, val_acc_dict,
  # Number of processed epochs differ because I do Keyboard Interrupt. 
  # So I'm going to take the epoch number from what has been saved to dictionaries.
  assert len(train_loss_dict)==len(val_loss_dict)#==len(train_acc_dict)==len(val_acc_dict)
  last_processed_step = len(train_loss_dict) #list(train_loss_dict.keys())[-1] 

  # Generate a sequence of integers to represent the epoch numbers
  step_range = range(0, last_processed_step)
  
  train_loss_values = list(train_loss_dict.values()) 
  val_loss_values = list(val_loss_dict.values())
  #train_acc_values = list(train_acc_dict.values()) 
  #val_acc_values = list(val_acc_dict.values())


  # Plot and label the training and validation loss values
  if loss_acc == "loss":
    plt.plot(step_range, train_loss_values, label='Training Loss')
    plt.plot(step_range, val_loss_values, label='Validation Loss')
  
  if loss_acc == "acc":
    plt.plot(step_range, train_loss_values, label='Training Acc')
    plt.plot(step_range, val_loss_values, label='Validation Acc')
  #plt.plot(step_range, train_acc_values, label='Training Acc')
  #plt.plot(step_range, val_loss_values, label='Validation Acc')
  
  # Add in a title and axes labels
  plt.title('Training and Validation Loss')
  plt.xlabel('Step')
  if loss_acc == "loss":
    plt.ylabel('Loss')
  if loss_acc == "acc":
    plt.ylabel('Accuracy')
  
  # Set the tick locations
  plt.xticks(np.arange(0, last_processed_step, 1000))
  
  if save:
    if loss_acc == "acc":
      plt.savefig(f".../trained_models/loss_figs/{optimizer.iterations.numpy()}_acc_fig.png", dpi=1080, bbox_inches='tight')
    
    if loss_acc == "loss":
      plt.savefig(f".../trained_models/loss_figs/{optimizer.iterations.numpy()}_loss_fig.png", dpi=1080, bbox_inches='tight')
      
  # Display the plot
  plt.legend(loc='best')
  plt.show()
  plt.draw()

for i in val_loss_dict.keys():
  print(i)

plot_loss(train_loss_dict, val_loss_dict, loss_acc= "loss", save=True)

plot_loss(train_acc_dict, val_acc_dict, loss_acc= "acc", save=True),
