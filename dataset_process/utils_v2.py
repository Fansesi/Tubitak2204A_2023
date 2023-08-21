# -*- coding: utf-8 -*-
"""Just a file that contaions utility functions. In most files I've used it like:
from utils_v2 import *
"""

import pickle
import json

from typing import Union, List, Dict
from pathlib import Path
import numpy as np
import os
from sPickle.sPickle import *
from tqdm import tqdm
import matplotlib.pyplot as plt


# Save and load using pickle
def out_pickle(data, path, verbose=True):
    try:
        with open(path, "wb") as d:
            pickle.dump(data, d, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print("[INFO] Objects has been pickled...")
    except:
        print("An exception occured...")


def in_pickle(path, verbose=False):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if verbose:
        print("[INFO] Pickled object has been imported...")
    return obj


def in_pickle_v2(path, verbose=False):
    with open(path, "rb") as f:
        obj = s_load(f)
    if verbose:
        print("[INFO] Pickled object has been imported...")
    return obj


# Load using json
def load_token_from_json(path):
    """Loads tokens from json file.
    @Params
    @path: path of json files. It must be a string.
    """
    with open(path, "r", encoding="utf-8") as d:
        return json.load(d)


def load_all_tokens(
    dir: List[Union[str, Path]], verbose: bool = False, nested_list: bool = False
):
    """Loads all tokens from a directory.
    @Params
    @dir: path of json files. It must be a list.
    @nested_list: For some reason some tokenizations saved as nested lists.
    E.g. "ids": [[[1,2,3],[1,2,3], ..., [1,2,3]]]
    """
    all_tokens = []
    error_list = []

    # if verbose:
    #  print("There are total of {} files in the directory.".format(len(dir)))

    for json_file in tqdm(dir, desc="Loading the data from disk"):
        try:
            loaded_json = load_token_from_json(json_file)
            if verbose:
                print("[INFO] This file has been added: {}".format(loaded_json))
            if nested_list:
                all_tokens.append(loaded_json["ids"][0])
            else:
                all_tokens.append(loaded_json["ids"])
        except Exception as E:
            print(f"{E} occured")
            error_list.append(json_file)
            continue

    if verbose:
        print("=" * 20 + " Error List " + "=" * 20)
        for i in error_list:
            print(i)

    return all_tokens, error_list


# Save using json
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
    token_prepared = {
        "tokens": [token],
        "programs": [[24, False]],
        "ids_bpe_encoded": False,
    }
    with open(path + f"{name}.json", "w", encoding="utf-8") as d:
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
        print("=" * 20 + " Error List " + "=" * 20)
        for i in error_list:
            print(i)

    return error_list


def save_json_from_p(input_path, output_path, verbose=False):
    """Imports a .pickle file and outputs every file into the target path.
    @Params:
    They are pretty straightforward."""
    p_obj = in_pickle(input_path)
    for piece in p_obj:
        save_token_as_json(
            piece,
            output_path,
        )

    return 0


def check_existed_files(dir1: Union[str, Path], dir2: Union[str, Path]):
    """Check which files (just names not extensions) from path1, exists in path2.
    Quite brute-force implementation. There should be a more efficient way."""

    dir1 = Path(dir1)
    dir2 = Path(dir2)

    dir1_names = []
    dir2_names = []

    only_exists_in_1 = []
    exists_both = []
    only_exists_in_2 = []

    dir1_files = [f for f in os.listdir(dir1) if os.path.isfile(f)]
    dir2_files = [f for f in os.listdir(dir2) if os.path.isfile(f)]

    for file1 in dir1_files:
        dir1_names.append(Path(file1).name)

    for file2 in dir2_files:
        dir2_names.append(Path(file2).name)

    for elem in dir1_names:
        if elem not in dir2_names:
            only_exists_in_1.append(elem)
            continue
        elif elem in dir2_names:
            exists_both.append(elem)
            continue
        else:
            pass

    for elem in dir2_names:
        if elem not in exists_both:
            only_exists_in_2.append(elem)

    return only_exists_in_1, exists_both, only_exists_in_2


def json2pickle(input_path: Union[str, Path], output_path: Union[str, Path]):
    """Converts a .json file to .pickle file. Give pathes to the full name including the extensions."""

    print("Loading the data.")
    with open(input_path, "r") as d:
        x = json.load(d)

    print("Saving the data.")
    out_pickle(x, output_path, True)


def plot_loss(
    train_loss_dict: Dict[int, float],
    val_loss_dict: Dict[int, float],
    loss_acc: str = "loss",
    fontsize: int = 15,
    figsize: int = 7,
    save=False,
    bg_color: str = "#ffffff",
    text_color: str = "#000000",
    label_color: str = "#000000",
    tick_colors: str = "#000000",
):  # train_acc_dict, val_acc_dict,
    """Plots the train and validation curves given two dicts."""

    # ==len(train_acc_dict)==len(val_acc_dict)
    assert len(train_loss_dict) == len(val_loss_dict)
    assert np.all(
        np.diff(list(train_loss_dict.keys()), 2) == 0
    ), "Given inputs should form an arithmetic sequence."
    last_processed_step = list(train_loss_dict.keys())[-1]
    arithmetic_diff = list(train_loss_dict.keys())[1] - list(train_loss_dict.keys())[0]
    # Generate a sequence of integers to represent the epoch numbers
    step_range = range(
        list(train_loss_dict.keys())[0],
        last_processed_step + 1,
        arithmetic_diff,
    )
    for i in step_range:
        print(i)

    train_loss_values = list(train_loss_dict.values())
    val_loss_values = list(val_loss_dict.values())

    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    plt.rcParams["figure.figsize"] = [figsize, figsize]
    plt.rcParams["figure.autolayout"] = True

    # Plot and label the training and validation loss values
    if loss_acc == "loss":
        plt.plot(step_range, train_loss_values, label="Training Loss")
        plt.plot(step_range, val_loss_values, label="Validation Loss")

    if loss_acc == "acc":
        plt.plot(step_range, train_loss_values, label="Training Acc")
        plt.plot(step_range, val_loss_values, label="Validation Acc")
    # plt.plot(step_range, train_acc_values, label='Training Acc')
    # plt.plot(step_range, val_loss_values, label='Validation Acc')

    # Add in a title and axes labels
    plt.title("Training and Validation Loss", fontsize=fontsize)
    plt.xlabel("Step", fontsize=fontsize)
    if loss_acc == "loss":
        plt.ylabel("Loss", fontsize=fontsize)
    if loss_acc == "acc":
        plt.ylabel("Accuracy", fontsize=fontsize)

    # Set the tick locations
    plt.xticks(
        np.arange(
            0,
            last_processed_step - last_processed_step % 1000 + 1,
            step=arithmetic_diff,
        )
    )
    # Colors
    bg_color = bg_color if bg_color[0] == "#" else "#" + bg_color
    text_color = text_color if text_color[0] == "#" else "#" + text_color
    tick_colors = tick_colors if tick_colors[0] == "#" else "#" + tick_colors
    label_color = label_color if label_color[0] == "#" else "#" + label_color

    plt.rcParams["text.color"] = text_color
    plt.rcParams["axes.labelcolor"] = label_color
    plt.rcParams["axes.edgecolor"] = label_color
    plt.rcParams["axes.facecolor"] = bg_color
    plt.rcParams["axes.titlecolor"] = label_color

    plt.rcParams["boxplot.flierprops.markeredgecolor"] = tick_colors
    # plt.rcParams["boxplot.flierprops.markerfacecolor"] = tick_colors

    plt.rcParams["xtick.color"] = tick_colors
    plt.rcParams["ytick.color"] = tick_colors

    plt.rcParams["hatch.color"] = bg_color
    plt.rcParams["legend.labelcolor"] = text_color
    plt.rcParams["patch.edgecolor"] = tick_colors

    plt.rcParams["figure.facecolor"] = bg_color
    plt.rcParams["figure.edgecolor"] = bg_color

    # ax = plt.gca()
    # ax.set_facecolor(bg_color)

    if save:
        if loss_acc == "acc":
            plt.savefig(
                f"/content/drive/MyDrive/tubitak2204A_2023/trained_models/loss_figs/acc_fig.png",
                dpi=95,
                bbox_inches="tight",
            )
        elif loss_acc == "loss":
            plt.savefig(
                f"/content/drive/MyDrive/tubitak2204A_2023/trained_models/loss_figs/_loss_fig.png",
                dpi=95,
                bbox_inches="tight",
            )
        else:
            raise Exception
    # Display the plot
    plt.legend(loc="best")
    plt.show()
    plt.draw()
