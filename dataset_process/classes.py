"""I've written many functions regarding to MIDI and tokens. I'd like to merge them into
classes that I can use easily.

Classes:
1) ProcessMIDI(): Utility functions to get info about both the MIDI dataset.
2) AugmentTOK(): Augmentation pipeline for json files. 
3) ProcessTOK(): Preprocessing 1D tokenization techniques. Splitting and padding. 
4) ProcessTOK2D(): Preprocessing 2D tokenization techniques. Splitting and padding. 

"""

from tqdm import tqdm
from utils_v2 import *
from loguru import logger as lg
import math
from pathlib import Path
import os
import pretty_midi as pm
from glob import glob
from typing import Dict, List, Union, Optional, Any, Tuple

import tensorflow as tf


class ProcessMIDI:
    """First initializes prettyMIDI objects. Given path should be one of the following:
    1) a str or Path obj indicating a .pickle object which has self.pms
    2) a str or Path obj indicating the directory of MIDI files.
    3) a List of str or Path objects indicating the individiual MIDI files path.
    4) a generator (glob) which is set to MIDI files path.

    @Params:
    @inputPath: input path to MIDI files or .pickle file. Input path can be a directory, a glob generator or list
    of pathLike(str, Path) objects.
    @outputPath:

    After initializing self.pms (with pickle or by .mid files) we don't remove any data from it. Just replacements.
    In the end when we would like to create the final obj or save the pm.PrettyMIDIs individually, it is possible to
    move the problematic files into it's dictionar with pms2problematic_pms().

    Info based functions:
    self.calc_time(), self.possible_pitches(), self.possible_time_sig(), self.plot_possibilities(),

    Process based functions:
    find_duets(), merge_duet(), same_program(), transpose_all(),

    This class is designed to be used as chains of functions. Here is the recommended order:
    class.find_duets().merge_duet().same_program().transpose_all().pms2problematic_pms().calc_time().possible_pitches().save_dataset_info().  save_individual_pms() or save_pm_as_pickle()
    """

    def __init__(self, inputPath: Union[str, Path, glob], outputPath: Union[str, Path]):
        # "reason_of_problem" : Path(MIDIpath).name
        self.problems_names: Dict[str, List[Optional[Union[Path, str]]]] = {
            "convert2pm_problem": [],
            "time_problem": [],
            "find_duet": [],
            "same_program_problem": [],
            "no_instruments": [],
            "possible_pitches": [],
            "note_num": [],
        }

        self.duets: Dict[int, pm.PrettyMIDI] = {}  # index(at self.pms) : pm.PrettyMIDI
        self.duet_names = []
        # index(at self.pms):
        self.problematic_pm: Dict[pm.PrettyMIDI, Union[str, Path]] = {}

        self.outputPath = Path(outputPath)
        self._mk_problem_dirs(outputDir=self.outputPath)

        # case analyzer...
        if type(inputPath) == str:
            if os.path.isfile(inputPath) == True:
                lg.critical("ohshitmate")
                self.inputPath = Path(inputPath)  # convert str to path
                self.pms = self.in_pickle(self.inputPath)
                self.inputDir = self.inputPath.parent
                self.input_files = [inputPath]
                pass

            elif os.path.isdir(inputPath) == True:
                self.input_files = []
                self.inputPath = Path(inputPath)  # convert str to path
                self.inputDir = self.inputPath  # same path because it's a dir.

                for elem in self.inputPath.glob("*.mid"):
                    self.input_files.append(elem)
                for elem in self.inputPath.glob("*.MID"):
                    self.input_files.append(elem)

                # pm.PrettyMIDI : MIDIname
                self.pms: Dict[pm.PrettyMIDI, str] = self.convert2pm()
                pass
            else:
                pass

        elif type(inputPath) == list:
            # this means that inputs are basically individiual mid files path.
            self.input_files = inputPath
            self.inputDir = Path(self.input_files[0]).parent
            lg.info(
                f"MIDI files imported. There are total of {len(self.input_files)} many files."
            )
            self.pms: Dict[pm.PrettyMIDI, str] = self.convert2pm()
            pass

        elif type(inputPath) == glob:
            self.input_files = []
            for element in tqdm(inputPath, desc="Importing MIDI files."):
                self.input_files.append(element)
            self.inputDir = Path(self.input_files[0]).parent
            self.pms: Dict[pm.PrettyMIDI, str] = self.convert2pm()
            pass

        else:
            msg = """Given path should be one of the following: \n
            1) a str or Path obj indicating a .pickle object which has self.pms \n
            2) a str or Path obj indicating the directory of MIDI files. \n
            3) a List of str or Path objects indicating the individiual MIDI files path. \n
            4) a generator (glob) which is set to MIDI files path.
            """
            lg.critical(msg)
            raise Exception

        self.dataset_time = {"total_time": 0, "max_time": 0, "min_time": 0}
        self.changed_program = 0
        self.pitch_nums: Dict[int, int] = {}  # pitch: occurence time
        self.note_nums = []

    def convert2pm(self):
        """Given a path, all of the midi files will be stored as pm.PrettyMIDI objs in the self.pms.
        Works with self.input_files. Because of inputPath can vary, we first create the paths in __init__().

        Note: I'm not sure about whether it is possible to store all the data or not though.
        """
        retr_dict = {}  # pm.PrettyMIDI : MIDIname

        for MIDIpath in tqdm(self.input_files, desc="Importing .mid files."):
            MIDIpath = Path(MIDIpath)
            try:
                retr_dict[pm.PrettyMIDI(MIDIpath.__str__())] = MIDIpath.name
            except Exception as err:
                self.add_error("convert2pm_problem", MIDIpath.name)
                # shutil.move(
                #    MIDIpath / Path(MIDIpath).name,
                #    self.outputPath / Path("convert2pm_problem"),
                # )
                lg.error(MIDIpath.name)
                os.rename(
                    src=MIDIpath,
                    dst=self.outputPath / Path("convert2pm_problem") / MIDIpath.name,
                )

                lg.error(f"Error occured on {MIDIpath}.\nException: {err}")
                continue

        return retr_dict

    def _mk_problem_dirs(self, outputDir: Path):
        """Creates the directories for problemetic files."""

        for problem in self.problems_names:
            whole_path = outputDir / Path(problem)
            if os.path.exists(whole_path) == False:
                os.mkdir(whole_path)
                lg.info(f"Created directory: {whole_path}")
            else:
                lg.info(f"Directory already exists: {whole_path}")
                continue

    def in_pickle(self, path):
        lg.info("Starting to import the data from .pickle file.")
        try:
            x = in_pickle(path=path, verbose=False)
        except EOFError:
            lg.error(f"EOFError occured!")
            # return x
        lg.info("Loading ended!")
        return x

    def _out_pickle(self, obj, path):
        lg.info("Starting to export the data to .pickle file.")
        out_pickle(obj, path=path, verbose=False)
        lg.info("Exporting ended!")

    def calc_time(self):
        """Calculates the total length of the dataset using self.pms variable"""
        temp_time_list = [0.0]
        for pmElement in tqdm(
            self.pms, desc="Calculating the total length of the dataset in seconds."
        ):
            try:  # getting piece length for every piece
                piece_length = pmElement.get_end_time()
                self.dataset_time["total_time"] += piece_length
                if piece_length > temp_time_list[-1]:
                    temp_time_list.append(piece_length)
            except:  # if there is a ZeroDivisionError exception just skip that midi file
                lg.error("There is a problem with the {}".format(self.pms[pmElement]))
                self.add_error("time_problem", self.pms[pmElement]),
                continue

        self.dataset_time["max_time"] = round(temp_time_list[-1], 3)
        self.dataset_time["min_time"] = round(temp_time_list[1], 3)
        del temp_time_list

        return self

    def calc_note_num(self, pms: Optional[List[pm.PrettyMIDI]] = None):
        """Calculates the average note number of bunch of pm.PrettyMIDI files.
        If pms = None, will use self.pms"""

        pms: List[pm.PrettyMIDI] = self.pms if pms == None else pms

        for pmElement in tqdm(pms, desc="Calculating the note numbers"):
            try:
                self.note_nums.append(len(pmElement.instruments[0].notes))
            except:
                lg.error("There is a problem with the {}".format(self.pms[pmElement]))
                self.add_error("note_num", self.pms[pmElement])

        return self

    def _init_pitch_dict(self, pitch_range: List[int] = [40, 85]) -> Dict[int, int]:
        """Creates dictionary."""
        tokenNumbers = {}
        for i in range(pitch_range[0], pitch_range[1] + 1):
            tokenNumbers[i] = 0

        return tokenNumbers

    def possible_pitches(self):
        """Finds all pitches relavent in the dataset. Uses self.pms."""
        notePitches = self._init_pitch_dict(pitch_range=[40, 85])

        for pmElement in tqdm(self.pms, desc="Looking at the possible pitches."):
            for note in pmElement.instruments[0].notes:
                try:
                    notePitches[note.pitch] += 1
                except KeyError:
                    lg.info(
                        f"There are no pitch value {note.pitch} in the created dict. we're adding it...",
                    )
                    notePitches[note.pitch] = 1
                    continue
                except:
                    self.add_error("possible_pitches", self.pms[pmElement])
        self.pitch_nums = notePitches

        return self

    def move_problematics(self):
        """For moving problematic MIDIs. Uses self.problems_names. Each midi file will be
        moved to it's created location (self._mk_problem_dirs()). If there are files which
        are inside more than one problem, they'll be moved to the first reason_of_problem dir.
        Other errors are aborted."""

        for problem in self.problems_names.keys():
            if self.problems_names[problem] == []:
                continue
            else:
                for MIDIname in tqdm(
                    self.problems_names[problem], desc=f"Moving the {problem} files"
                ):
                    os.rename(
                        src=self.inputDir / Path(MIDIname),
                        dst=self.outputPath / Path(problem) / Path(MIDIname),
                    )

        return self

    def save_pm_as_pickle(self):
        """Save the pm.PrettyMIDI in the self.pms."""
        lg.info("Pickling the self.pms object...")
        self._out_pickle(
            self.pms, self.outputPath / f"{len(self.pms)}_PrettyMIDI_objs.pickle"
        )
        return self

    def pms2problematic_pms(self):
        """Extract the problematic pm.PrettyMIDI files from the self.pms using
        stored names at self.problems_names and insert them through self.problematic_pm
        """
        tempList = []

        for a_list in tqdm(
            self.problems_names.values(), desc="Transfering problematic MIDIs."
        ):
            for name in a_list:
                # print(name)
                # using the name, retrieve the relevant key (pm.PrettyMIDI) from self.pms
                # then just pop it from there
                temp_key = list(self.pms.keys())[list(self.pms.values()).index(name)]
                tempList.append(temp_key)
                self.problematic_pm[temp_key] = Path(name)

                # lg.error("Some error occured while cleaning the pms.")
                # continue
        for temp_key in tempList:
            self.pms.pop(temp_key)

        return self

    def find_duets(self):
        """Finds duet pieces inside the self.pms variable and writes them to
        self.duets which is a dictionary (index(at self.pms) : pm.PrettyMIDI)."""
        for i, pmElement in enumerate(tqdm(self.pms, desc="Finding duets.")):
            try:
                if len(pmElement.instruments) > 1:
                    # creating the duets using self.pms index
                    self.duets[i] = pmElement
                    # storing the duet pieces name in a container
                    self.duet_names.append(self.pms[pmElement])

            except Exception as exp:
                self.add_error("find_duet", self.pms[pmElement])
                lg.error("Exception: {}, File: {}".format(exp, pmElement))
                continue
        return self

    def merge_duet(self):
        """First, calls the self._find_duets(). Then merges the duets in the self.duets. After that insert them
        into self.pms accordingly. If self.duets is still empty after self._find_duets(), just passes.
        """
        tempList = []
        for index in tqdm(self.duets, desc="Merging the duets."):
            pmElement = self.duets[index]
            empty_piece = pm.PrettyMIDI()
            x = []

            for i in range(0, len(pmElement.instruments)):
                for note in pmElement.instruments[i].notes:
                    x.append(note)

            new_inst = pm.Instrument(program=24)
            for note in x:
                new_inst.notes.append(note)

            empty_piece.instruments.append(new_inst)

            tempList.append([empty_piece, pmElement])
            # we cannot iterate and change the keys at the same time.
            # self.pms[empty_piece] = self.pms[pmElement]  # names stays the same though.
            # self.pms.pop(pmElement)  # remove the piece

        self._upt_pms(templist=tempList)

        return self

    def same_program(self, programNum: int = 24):
        """Changes a midi files MIDI program to desired MIDI program number. If there are any problematic file,
        remove them from self.pms and pass that thorough the problematic files."""

        tempList: List[List[pm.PrettyMIDI]] = []

        for pmElement in tqdm(
            self.pms, desc=f"Changing the program number to {programNum}"
        ):
            if pmElement.instruments == []:
                lg.error(f"No instruments on: {pmElement}")
                # self.pms.pop(i) No we don't do that here.
                self.add_error("no_instruments", self.pms[pmElement])
                continue
            try:
                # If it's already program number we want, no need to manipulate.
                if pmElement.instruments[0].program == programNum:
                    continue
                else:
                    pass
            except:
                self.add_error("same_program_problem", self.pms[pmElement])
                # self.pms.pop(i) No we don't do that here.
                lg.error(f"Some other error occured in: {pmElement}")
                continue

            empty_piece = pm.PrettyMIDI()
            x = []

            try:
                for note in pmElement.instruments[0].notes:
                    x.append(note)
            except:
                lg.error(f"Some error occured in: {pmElement}")
                self.add_error("same_program_problem", self.pms[pmElement])
                # self.pms.pop(i) No we don't do that here.
                continue

            new_inst = pm.Instrument(program=programNum)
            for note in x:
                new_inst.notes.append(note)

            empty_piece.instruments.append(new_inst)

            tempList.append([empty_piece, pmElement])

            # we cannot change keys of the dictionary while iterating through it.
            # self.pms[empty_piece] = self.pms[pmElement]  # names stays the same though.
            # self.pms.pop(pmElement)  # change the piece

            self.changed_program += 1

        self._upt_pms(templist=tempList)

        return self

    def _upt_pms(self, templist: List[List[pm.PrettyMIDI]]):
        """First inserts the new datas using self.pms[old] to keep the same value in the dict.
        Then removes the old ones.
        @Params:
        @tempList: first pm.PrettyMIDI obj is the new data, other one is the old. It's just pairs.
        """

        # adding the new datas.
        for pair in templist:
            self.pms[pair[0]] = self.pms[pair[1]]

        # removing the old datas.
        for pair in templist:
            self.pms.pop(pair[1])

    def _transpose_note(
        self, note_pitch: int, pitch_range: List[int] = [40, 85]
    ) -> int:
        """Given a note pitch, increase or decrese the pitch by 12 until it falls between possible pitch_range
        @Params:
        @note_pitch: pitch value to work on.
        @pitch_range: pitch range to work on. [min_pitch, max_pitch]
        """

        if note_pitch < pitch_range[0]:  # smaller than
            if (pitch_range[0] - note_pitch) % 12 == 0:
                return pitch_range[0]
            else:
                return note_pitch + 12 * (1 + ((pitch_range[0] - note_pitch) // 12))

        elif note_pitch > pitch_range[1]:  # greater than
            if (note_pitch - pitch_range[1]) % 12 == 0:
                return pitch_range[1]
            else:
                return note_pitch - 12 * (1 + ((note_pitch - pitch_range[1]) // 12))

        else:  # it falls between the range
            return note_pitch

    def transpose_all(self, pitch_range: List[int] = [40, 85]):
        """Tranpose pitches that are greater than the max pitch and less than the min pitch to nearest possible octave.
        @Params:
        @input_paths: input paths. Use it with glob().
        @output_dir: output dir to save the midis.
        @pitch_range: [min_pitch, max_pitch]. Default to [40,85] which is for classical guitar.
        """
        for pmElement in tqdm(
            self.pms, desc=f"Transposing the pieces into range {pitch_range}"
        ):
            for note in pmElement.instruments[0].notes:
                note.pitch = self._transpose_note(note.pitch, pitch_range)

        return self

    def save_individual_pms(self):
        """Saves each individiual pm.PrettyMIDI in self.pms with their relative labels into the self.outputPath.
        Mote before calling this function, it's best to first call the self.pm2problematic_pm() function.
        This way only the non_problematic_files will be saved out."""

        if os.path.exists(self.outputPath / Path("midis_processed")) == False:
            os.mkdir(self.outputPath / Path("midis_processed"))

        for pmElement in tqdm(self.pms, desc="Saving individually."):
            pmElement.write(
                (
                    self.outputPath
                    / Path("midis_processed")
                    / Path(self.pms[pmElement])
                ).__str__()
            )

        return self

    def save_dataset_info(self):
        """Creates a dictionary about the datas calculated and saves it to the DATASET_STATS.json file."""

        stats_dict: Dict[str, Any] = {
            "number_of_duets": len(self.duets),
            "duets": self.duet_names,
            "problematic_files": self.problems_names,
            "changed_program": self.changed_program,
            "total_seconds": self.dataset_time,
            "pitch_nums": self.pitch_nums,
            "total_notes": {np.sum(self.note_nums)},
            "avg_notes": {np.average(self.note_nums)},
        }

        outputPath = self.outputPath / "DATASET_STATS.json"
        lg.info(f"Saving the stats to {outputPath}")

        with open(outputPath, "w", encoding="utf-8") as d:
            json.dump(stats_dict, d)

        return self

    def merge_dataset_info_json(
        self,
        path1: Union[str, Path],
        path2: Union[str, Path],
        outputPath: Union[str, Path],
    ):
        """Given 2 json files paths, merges the content of the json files. outputPath is expected to be a
        'directory/filename.json'
        """
        lg.info("Merging the json files now...")
        path1, path2 = Path(path1), Path(path2)
        if path1.suffix != ".json" and path2.suffix != ".json":
            lg.info("Merged files should be json.")
            raise Exception

        with open(path1, "r", encoding="utf-8") as d:
            stats_dict1 = json.load(d)

        with open(path2, "r", encoding="utf-8") as d:
            stats_dict2 = json.load(d)

        stats_dict1["number_of_duets"] += stats_dict2["number_of_duets"]
        for i in stats_dict2["duets"]:
            stats_dict1["duets"].append(i)
        for problem in stats_dict2["problematic_files"].keys():
            for name in stats_dict2["problematic_files"][problem]:
                stats_dict1["problematic_files"][problem].append(name)
        stats_dict1["changed_program"] += stats_dict2["changed_program"]
        for time in stats_dict2["total_seconds"]:
            stats_dict1["total_seconds"][time] += stats_dict2["total_seconds"][time]
        for num in stats_dict2["pitch_nums"]:
            stats_dict1["pitch_nums"][num] += stats_dict2["pitch_nums"][num]

        with open(outputPath, "w", encoding="utf-8") as d:
            json.dump(stats_dict1, d)

        return self

    def current_state(self):
        """Returns some info about the paramaters set in __init__()."""
        lg.info(f"Number of pm.PrettyMIDIs on self.pms is {len(self.pms)}")
        lg.info(f"Current changed program: {self.changed_program}")
        lg.info(
            f"Current number of duets are {len(self.duet_names)} and their names are: {self.duet_names}"
        )

        cleaned_list = []
        for prob_list in self.problems_names.values():
            for elem in prob_list:
                if elem not in cleaned_list:
                    cleaned_list.append(elem)
                else:
                    continue

        lg.info(f"Current dataset_time {self.dataset_time}")
        lg.info(f"Current pitch_nums {self.pitch_nums}")
        lg.info(
            f"Current total note number and it's average {np.sum(self.note_nums), np.average(self.note_nums)}"
        )
        lg.info(f"Current number of problematic files is {len(cleaned_list)}")
        lg.info(f"Current problematic files param: \n{self.problems_names}")
        return self

    def add_error(self, err_reason: str, data: Union[Path, str]):
        """Adds the relative error to the self.problems_names."""
        self.problems_names[err_reason].append(data)

    def plot_possibilities(self):
        """Plots the total pitch/time_sig possibilities using matplolib and pandas."""
        raise NotImplementedError

    def possible_time_sig(self, input_paths, verbose=True):
        """Calculates how many different time signatures are there."""
        raise NotImplementedError
        for pmElement in tqdm(
            self.pms, desc="Looking at the possible time signatures."
        ):
            for time_sig in pmElement.instruments[0].tim:
                try:
                    notePitches[note.pitch] += 1
                except KeyError:
                    lg.info(
                        f"There are no pitch value {note.pitch} in the created dict. I'm adding it...",
                    )
                    notePitches[note.pitch] = 1
                    continue
                except:
                    self.add_error("possible_pitches", self.pms[pmElement].name)


class AugmentTOK:
    """Augmentation For REMI Tokenized Audio Files
    Augmenting the tokens and saving them as .p to a given directory.
    @Params:
    @inputPath: Given path should be one of the following:
        1) a str or Path obj indicating a .pickle object which has jsons.
        2) a str or Path obj indicating the directory of json files.
        3) a List of str or Path objects indicating the individiual json files path.
        4) a generator (glob) which is set to json files path.

    @augment_val: a dictionary containing augmentation values with relative labels.
    Every value will be used in the integers in the lists. The first values is the min,
    the next one is the max aug.
    An example is shown:

    augment_val = {
            "pitch": [-2, 2],
            "velocity": [-1, 1],
            "tempo": [-1, 1],
        }

    If set to None which is the default, the given example will be the augment_val.

    @token_types_min_max: min and max values of the tokens with their relative labels.
    An example is shown below:

    self.token_types_min_max = {
            "pitch": [5, 49],
            "velocity": [50, 61],
            "tempo": [187, 210],
        }

    If set to None which is the default, the given example will be the augment_val.
    """

    def __init__(
        self,
        inputPath: Union[str, Path, glob],
        augment_val: Optional[Dict[str, List[int]]] = None,
        token_types_min_max: Optional[Dict[str, List[int]]] = None,
    ):
        self.all_tokens = self.def_all_tokens(inputPath)
        if augment_val == None:
            self.augment_val = {
                "pitch": [-2, 2],
                "velocity": [-1, 1],
                "tempo": [-1, 1],
            }
        else:
            self.augment_val = augment_val

        if token_types_min_max == None:
            self.token_types_min_max = {
                "pitch": [5, 49],
                "velocity": [50, 61],
                "tempo": [187, 210],
            }
        else:
            self.token_types_min_max = token_types_min_max

        # name of the augment type: augmented pieces
        self.data_holder: Dict[str, List[List[int]]] = {
            "pitch": [],
            "velocity": [],
            "tempo": [],
        }
        self.total_aug_nums = {
            "pitch": 0,
            "velocity": 0,
            "tempo": 0,
        }

        for name in self.augment_val.keys():
            all_auged_pieces, total_aug_num = self.it_augment(
                self.all_tokens,
                min_val=self.token_types_min_max[name][0],
                max_val=self.token_types_min_max[name][1],
                change_amount=self.augment_val[name],
                name=name,
            )
            for auged_piece in all_auged_pieces:
                self.data_holder[name].append(auged_piece)

            self.total_aug_nums[name] += total_aug_num

    def def_all_tokens(
        self, path: Union[str, Path, glob, List[Union[str, Path]]]
    ) -> List[List[int]]:
        """Defines the self.all_tokens param."""
        if type(path) == str:
            if os.path.isfile(path) == True:
                return in_pickle(path)

            elif os.path.isdir(path) == True:
                json_files = []

                for elem in Path(path).glob("*.json"):
                    self.input_files.append(elem)

                data, _ = load_all_tokens(json_files, verbose=False)
                return data
            else:
                pass

        elif type(path) == list:
            # this means that inputs are basically individiual .json files path.
            data, _ = load_all_tokens(path, verbose=False)
            return data

        elif type(path) == glob:
            json_files = []
            for element in tqdm(path, desc="Importing json files."):
                json_files.append(element)

            data, _ = load_all_tokens(path, verbose=False)
            return data
        else:
            msg = """Given path should be one of the following: \n
            1) a str or Path obj indicating a .pickle object which has jsons. \n
            2) a str or Path obj indicating the directory of json files. \n
            3) a List of str or Path objects indicating the individiual json files path. \n
            4) a generator (glob) which is set to json files path.
            """
            lg.critical(msg)
            raise Exception

    def clear_duplicates(self, token_lists: List[List[int]]) -> List[List[int]]:
        """Clear the duplicates in a given list"""

        cleared = []

        for i in token_lists:
            if i not in cleared:
                cleared.append(i)

        lg.info(f"Total number of ")

        return cleared

    def _find_index(self, a_list: List[int], min_val: int, max_val: int):
        """Returns the indexes of a value between min_val and max_val in a dict."""
        index_dict = {}  # indexes of tokens
        for i, token in enumerate(a_list):
            if token >= min_val and token <= max_val:
                index_dict.update({i: token})

        return index_dict  # index of the pitch token : pitch token itself

    def aug(
        self,
        token_list: List[int],
        min_val: int,
        max_val: int,
        change_vals: List[int] = [-2, 2],
        verbose: bool = False,
    ):
        """Finds the tokens between min_val and max_val and sums them for every change_vals value.
        If a change_val+token>max_val or change_val+token<min_val, we don't augment with that change_val.
        @Params:
        @token_list: list of tokens
        @min_val: min_val to augment. This value sets the boundaries of augmentation.
        @max_val: max_val to augment. This value sets the boundaries of augmentation.
        @change_vals: a list of [i,j] (integer i,j's), amount of augmentation
        @verbose: for debug purposes
        """
        augmented_tokens = []
        total_aug = 0

        index_dict = self._find_index(token_list, min_val, max_val)

        if verbose:
            print(index_dict)

        for change_val in range(change_vals[0], change_vals[1] + 1):
            if change_val == 0:
                augmented_tokens.append(token_list)
                total_aug += 1

            elif change_val < 0:
                augmentable = True
                for index in index_dict:
                    pitch_token = index_dict[index]
                    if pitch_token + change_val < min_val:
                        augmentable = False
                        break
                    else:
                        continue
                    # else:
                    #  augmentable=True
                    #  continue

                if augmentable:
                    copy_list1 = token_list.copy()
                    for index in index_dict:
                        pitch_token = index_dict[index]
                        copy_list1[index] = pitch_token + change_val
                    augmented_tokens.append(copy_list1)
                    total_aug += 1

            else:  # change_val>0
                augmentable = True
                for index in index_dict:
                    pitch_token = index_dict[index]
                    if pitch_token + change_val > max_val:
                        augmentable = False
                        break
                    else:
                        continue

                if augmentable:
                    copy_list2 = token_list.copy()
                    for index in index_dict:
                        pitch_token = index_dict[index]
                        copy_list2[index] = pitch_token + change_val
                    augmented_tokens.append(copy_list2)
                    total_aug += 1

        if verbose:
            lg.info(f"Total amount of augmentation: {total_aug}")

        return augmented_tokens, total_aug

    def it_augment(
        self,
        token_lists: List[List[int]],
        min_val: int,
        max_val: int,
        change_amount: List[int],
        name: str = "",
    ):
        """Iterate on the given datas and augment them.
        @Params:
        @token_lists: lists of tokens in order words all the pieces to augment.
        @min_val: min_val to augment. This value sets the boundaries of augmentation.
        @max_val: max_val to augment. This value sets the boundaries of augmentation.
        @change_amount: a list of [i,j] (integer i,j's), amount of augmentation.
        @name: just to show what we are augmenting in the tqdm description.
        """
        total_aug = 0
        all_auged_pieces = []

        for token_list in tqdm(token_lists, desc=f"Augmenting {name}"):
            augmented_tokens, total_aug_single = self.aug(
                token_list, min_val, max_val, change_amount, verbose=False
            )
            total_aug += total_aug_single

            for i in augmented_tokens:
                all_auged_pieces.append(i)

        return all_auged_pieces, total_aug

    def save_all(self, output_path: Union[str, Path], file_type: str = "json"):
        """Saves all the datas in the self.dataholder by not considering the type of augmentation.
        @Params:
        @output_path
        @file_type: can be 'json' or 'pickle'. Otherwise throws error.
        """

        auged_all_tokens = []
        for name in self.data_holder.keys():
            for elem in self.data_holder[name]:
                auged_all_tokens.append(elem)

        with open(output_path, "w", encoding="utf-8") as d:
            if file_type == "json":
                json.dump(auged_all_tokens, d)
            elif file_type == "pickle":
                pickle.dump(auged_all_tokens, d)
            else:
                lg.critical(
                    f"For file_type param, you have specified {file_type} but it should be 'json' or 'pickle'."
                )
                raise Exception

        lg.info(f"Data saved to: {output_path}")
        lg.info(f"There was {len(auged_all_tokens)} many piece in the saved file.")

    def display_stats(self):
        "Displays the stored statistics."
        lg.info(
            f"Total number of augmentation respective to classes is: \n{self.total_aug_nums}"
        )


class ProcessTOK:
    """1D Preprocessing.

    This preprocessing file is for representations which are 1D like REMI, MIDI-Like, Structured.
    What does this class do:
    * Convert the tokens from .json files into tensors
    * Split and concat the tensors accordingly
    * Create `tf.data.Dataset` object and save it

    Given path should be one of the following:
    1) a str or Path obj indicating a .pickle object which has .json files
    2) a str or Path obj indicating the directory of .json files.
    3) a List of str or Path objects indicating the individiual .json files path.
    4) a generator (glob) which is set to .json files path.


    @Params:
    @inputPath: input path to MIDI files or .pickle file. Input path can be a directory, a glob generator or list
    of pathLike(str, Path) objects.

    """

    def __init__(
        self,
        inputPath: Union[str, Path, glob, List[Union[str, Path]]],
    ):
        self.all_tokens = self.def_all_tokens(inputPath)

    def in_pickle(self, path):
        lg.info("Starting to import the data from .pickle file.")
        try:
            x = in_pickle(path=path, verbose=False)
        except EOFError:
            lg.error(f"EOFError occured!")
        lg.info("Loading ended!")
        return x

    def def_all_tokens(
        self, path: Union[str, Path, glob, List[Union[str, Path]]]
    ) -> List[List[int]]:
        """Defines the self.all_tokens param."""

        if isinstance(path, Union[Path, str]):
            path = Path(path)
            if os.path.isfile(path) == True:
                # this means that the
                lg.critical("ohshitmate")
                return self.in_pickle(path)

            elif os.path.isdir(path) == True:
                json_files = []

                for elem in Path(path).glob("*.json"):
                    json_files.append(elem)

                data, _ = load_all_tokens(json_files, verbose=False, nested_list=True)
                return data
            else:
                lg.critical(f"Couldn't load the data.")
                pass

        elif isinstance(path, list):
            # this means that inputs are basically individiual .json files path.
            data, _ = load_all_tokens(path, verbose=False)
            return data

        elif type(path) == glob:
            json_files = []
            for element in tqdm(path, desc="Importing json files."):
                json_files.append(element)

            data, _ = load_all_tokens(path, verbose=False)
            return data
        else:
            msg = """Given path should be one of the following: \n
            1) a str or Path obj indicating a .pickle object which has jsons. \n
            2) a str or Path obj indicating the directory of json files. \n
            3) a List of str or Path objects indicating the individiual json files path. \n
            4) a generator (glob) which is set to json files path.
            """
            lg.critical(msg)
            raise Exception

    def equalize_shapes_1D(
        self,
        input_token: List[List[int]],
        target_shape: List[int],
        verbose: bool = False,
    ):
        """Modifies the input token in order to match the target shape.
        @Params:
        @input_token: should be a list with a list inside. Integers are tokens. Expecting it's shape as [n]
        @target_shape: should be a list. Expecting [n]
        """
        token_tensor = tf.convert_to_tensor(input_token)  # creating the tensor
        input_length = token_tensor.shape.as_list()[
            0
        ]  # [1,2,3,4,5,6,7 ...] total of n notes. We're expecting it to be (n)
        target_length = target_shape[0]

        if verbose:
            lg.info("Input Length: {}".format(input_length))
            lg.info("Target Length: {}".format(target_length))

        dummy_zero = tf.zeros((target_length - input_length), dtype=tf.dtypes.int32)

        try:
            if target_length > input_length:
                if verbose:
                    lg.info(dummy_zero)
                output_token_tensor = tf.concat([token_tensor, dummy_zero], axis=0)
                if verbose:
                    lg.info("Token has been padded.")
                return output_token_tensor
        except:
            raise RuntimeError

    def splitting_1D(
        self, input_token: List[int], target_shape: List[int], verbose=False
    ):
        """Cuts the given tensor into target_shape required times and returns cut tensors and remaining tensor.
        @Params
        @input_token: should be a list. Inner ints are tokens.
        @target_shape: should be a list of a shape. Expecting [n]
        """
        input_tensor = tf.convert_to_tensor(input_token)
        input_length = input_tensor.shape.as_list()[0]
        target_length = target_shape[0]

        if verbose:
            lg.info("Input Length: {}".format(input_length))
            lg.info("Target Length: {}".format(target_length))

        split_n_times = math.floor(input_length / target_length)
        splitted_seqs = []

        splitted_tensor, remaining_tensor = tf.split(
            input_tensor,
            num_or_size_splits=(target_length, input_length - target_length),
            axis=0,
        )
        remaining_tensor_main = remaining_tensor
        splitted_seqs.append(splitted_tensor)

        for i in range(split_n_times - 1):
            splitted_tensor, remaining_tensor = tf.split(
                remaining_tensor_main,
                (target_length, remaining_tensor.shape.as_list()[0] - target_length),
                axis=0,
            )
            remaining_tensor_main = remaining_tensor
            splitted_seqs.append(splitted_tensor)

        return splitted_seqs, remaining_tensor_main

    def _gen_fn(self, tensor_list):
        for tensor in tensor_list:
            yield tensor  # ((tensor[:, 1:], tensor[:, :-1]))

    def finalize(
        self,
        all_tokens_list: List[List[int]],
        target_shape: List[int],
        output_dir: Optional[Union[str, Path]] = None,
        save: bool = False,
        verbose: bool = False,
        stats: bool = True,
    ):
        """Combining both splitting() and eqalize_shapes() methods and finalizing the dataset.
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

        for i, piece in enumerate(tqdm(all_tokens_list, desc="Processing the tensors")):
            piece_length = len(piece)
            if verbose:
                lg.info(f"{i}. piece length is: {piece_length}")

            if target_length > piece_length:
                just_equalized += 1
                # if verbose:
                #  lg.info(" This piece has been padded in order to match the given shape.")
                output_tensor = self.equalize_shapes_1D(
                    piece, target_shape, verbose=False
                )
                all_output_tensors.append(output_tensor)

            elif target_length == piece_length:
                did_none += 1
                # if verbose:
                #  lg.info(" Target length and the piece_length are alreay matched.")
                all_output_tensors.append(
                    piece
                )  # There is nothing to do so just add the tensor

            elif target_length < piece_length:
                splitted += 1
                # if verbose:
                #  lg.info(" This piece has been splitted and padded in order to match the given shape.")
                splitted_tensor, remaining_tensor = self.splitting_1D(
                    piece, target_shape, verbose=False
                )
                # I'm creating a threshold for this. If it's more than half!
                if remaining_tensor.shape[0] >= target_shape[0] // 2:
                    remaning_modified_tensor = self.equalize_shapes_1D(
                        remaining_tensor, target_shape
                    )
                    all_output_tensors.append(
                        remaning_modified_tensor
                    )  # Appending the remaining but padded tensor

                for elem in splitted_tensor:
                    all_output_tensors.append(elem)  # Appending the splitted tensors

            else:
                lg.critical("It's just impossible...")

        if stats:
            print("=" * 22 + "STATS" + "=" * 22)
            print(" {} many pieces was already in shape.".format(did_none))
            print(" {} many pieces has been padded.".format(just_equalized))
            print(" {} many pieces has been splitted and padded.".format(splitted))
            # print(f" All output tensors are {all_output_tensors}")

        created_dataset = tf.data.Dataset.from_generator(
            self._gen_fn, args=[all_output_tensors], output_types=tf.int32
        )

        if save:
            lg.info("Saving dataset...")
            tf.data.Dataset.save(created_dataset, output_dir)  # Saving dataset
            # WARNING! This function is deprecated but tf.data.Dataset.save doesn't work so I'm using this right now.
            # It might and probably will be removed in the future.

        return created_dataset, all_output_tensors

    def token_lengths(self, all_tokens: Optional[List[List[int]]] = None):
        """Finds the length of sequence inside all of the pieces.
        @Params:
        @input_tokens: All the input_tokens. Inside it there should be lists
        and in every list there should be lists as well. If passed to None, will use
        self.all_tokens.
        E.g. midi_tokenized_all => pieces => notes
        Returns the lengths of each tokenized piece, max_length and min_length
        respectively.
        """
        all_tokens = self.all_tokens if all_tokens == None else all_tokens
        all_lens = []

        for piece in all_tokens:
            all_lens.append(len(piece))

        return all_lens, max(all_lens), min(all_lens)

    def plot_token_nums(self):
        """Plots the number of tokens in the dataset. This plot can be both done
        individually or looking at the relevant token types like pitch_tokens,
        duration_tokens etc.
        """
        raise NotImplementedError

    def save_all_tokens_linebyline(
        self, outputPath: Union[Path, str], tokens: Optional[List[List[int]]] = None
    ):
        """Saves all the imported tokens to single .json file line by line.
        E.g.
        {"tokens": [4,53,45...]}
        {"tokens": [4,52,66...]}

        @Params:
        @outputPath: path to save
        @tokens: tokens to save. if specified will save those tokens. Otherwise it'll save
        self.all_tokens
        """

        if tokens == None:
            tokens = self.all_tokens

        if tokens == []:
            lg.critical("Tokens you are about to save is empty!")
            # raise Exception
        lg.info(f"Lenght of the tokens: {len(tokens)}")

        with open(outputPath, "w") as d:
            for tokenized_piece in tokens:
                # json.dump(token_dict, d)
                d.write('{"tokens":' + json.dumps(tokenized_piece) + "}\n")

        lg.info(f"Tokens succesfully saved to {outputPath}")

    def save_all_tokens(
        self, outputPath: Union[Path, str], tokens: Optional[List[List[int]]] = None
    ):
        """Saves all tokens to a .json file."""
        if tokens == None:
            tokens = self.all_tokens

        if tokens == []:
            lg.critical("Tokens you are about to save is empty!")
            # raise Exception
        lg.info(f"Lenght of the tokens: {len(tokens)}")

        with open(outputPath, "w") as file:
            json.dump(tokens, file)

        lg.info(f"Tokens succesfully saved to {outputPath}")

    def locate_given_tokens(self, tokens: List[int], tokens2look4: List[int]):
        """Returns the indices of the tokens in the 'tokens' which also exists in tokens2look4"""
        indices: List[int] = []
        for i, token in enumerate(tokens):
            if token in tokens2look4:
                indices.append(i)

        return indices

    def calc_occurence(
        self,
        tokens2look4: List[int],
        all_tokens: Optional[List[int]] = None,
        print_out: bool = True,
    ) -> Tuple[int, int]:
        """Calculates the occurence of tokens in tokens2look4 inside the all_tokens. If none specified as
        all_tokens, self.all_tokens will be used.

        Returns wanted_tokens_num, all_tokens_num"""
        all_tokens = self.all_tokens if all_tokens == None else all_tokens
        all_lengths, _1, _2 = self.token_lengths()
        all_tokens_num: int = 0
        wanted_tokens_num: int = 0
        for elem in all_lengths:
            all_tokens_num += elem

        for piece in all_tokens:
            wanted_tokens_num += len(self.locate_given_tokens(piece, tokens2look4))

        if print_out:
            lg.info(f"Total of all tokens: {all_tokens_num}")
            lg.info(f"Total of wanted tokens: {wanted_tokens_num}")
            lg.info(f"Ratio of the wanted/all: {wanted_tokens_num/all_tokens_num}")

        return wanted_tokens_num, all_tokens_num

    def _convert_tf2list(self, all_tokens_tf: List[tf.Tensor]) -> List[List[int]]:
        """Converts tf.Tensors to back to lists. I don't want to waste
        time changing the pipeline to tf-free instead we're just going to change
        the output."""

        all_tokens_list = []

        for tensor in all_tokens_tf:
            all_tokens_list.append(
                tf.make_ndarray(tensor=tf.make_tensor_proto(tensor)).tolist()
            )

        return all_tokens_list


class ProcessTOK2D:
    """Processing tokens for 2D representations like Octuple or OctupleMono."""

    def __init__(
        self,
        inputPath: Union[str, Path, glob, List[Union[str, Path]]],
        # outputPath: Union[str, Path],
    ):
        self.all_tokens = self.def_all_tokens(inputPath)

    def in_pickle(self, path):
        lg.info("Starting to import the data from .pickle file.")
        try:
            x = in_pickle(path=path, verbose=False)
        except EOFError:
            lg.error(f"EOFError occured!")
            # return x
        lg.info("Loading ended!")
        return x

    def def_all_tokens(
        self, path: Union[str, Path, glob, List[Union[str, Path]]]
    ) -> List[List[int]]:
        """Defines the self.all_tokens param."""

        if isinstance(path, Union[Path, str]):
            path = Path(path)
            if os.path.isfile(path) == True:
                # this means that the
                lg.critical("ohshitmate")
                return self.in_pickle(path)

            elif os.path.isdir(path) == True:
                json_files = []

                for elem in Path(path).glob("*.json"):
                    json_files.append(elem)

                data, _ = load_all_tokens(json_files, verbose=False, nested_list=True)
                return data
            else:
                lg.critical(f"Couldn't load the data.")
                pass

        elif isinstance(path, list):
            # this means that inputs are basically individiual .json files path.
            data, _ = load_all_tokens(path, verbose=False)
            return data

        elif type(path) == glob:
            json_files = []
            for element in tqdm(path, desc="Importing json files."):
                json_files.append(element)

            data, _ = load_all_tokens(path, verbose=False)
            return data
        else:
            msg = """Given path should be one of the following: \n
            1) a str or Path obj indicating a .pickle object which has jsons. \n
            2) a str or Path obj indicating the directory of json files. \n
            3) a List of str or Path objects indicating the individiual json files path. \n
            4) a generator (glob) which is set to json files path.
            """
            lg.critical(msg)
            raise Exception

    def equalize_shapes2D(
        self,
        input_token: Union[List[List[int]], tf.Tensor],
        target_shape: List[int],
        verbose: bool = False,
    ):
        """Modifies the input token in order to match the target shape.
        @Params:
        @input_token: should be a list with lists inside. Inner ints are tokens and every list is a note (because we are working with 2D repr).
        @target_shape: should be a list.
        """
        token_tensor = tf.convert_to_tensor(input_token)  # creating the tensor

        # [[1,2,3,4,5,6], ...] total of n notes.
        input_length = token_tensor.shape.as_list()[0]
        target_length = target_shape[0]

        dummy_zero = tf.zeros(
            (1, token_tensor.shape.as_list()[1]), dtype=tf.dtypes.int32
        )

        if verbose:
            lg.info("Input Length: {}".format(input_length))
            lg.info("Target Length: {}".format(target_length))
            assert (
                token_tensor.shape.as_list()[1] == target_shape[1]
            )  # Checking if there is a problem with the tokens.
            lg.info(dummy_zero)

        try:
            if target_length > input_length:
                dummy_zero_ready = tf.repeat(
                    dummy_zero, target_length - input_length, axis=0
                )
                if verbose:
                    lg.info(dummy_zero_ready)
                output_token_tensor = tf.concat(
                    [token_tensor, dummy_zero_ready], axis=0
                )
                if verbose:
                    lg.info("Token has been padded.")
                return output_token_tensor
            elif (
                target_length == input_length
            ):  # Actually there is no need for the rest of inequalites because finalize() alreday take care of that.
                if verbose:
                    lg.info("Target length and token length are already equal.")
                return input_token
            else:
                if verbose:
                    lg.info("Token length is already long enough.")
                return input_token

        except Exception as E:
            lg.error(f"A problem occured while equailizing shape: {E}")

    def splitting2D(
        self,
        input_token: Union[List[List[int]], tf.Tensor],
        target_shape: List[int],
    ):
        """Cuts the given tensor into target_shape required times and returns cut tensors and remaining tensor.
        @Params
        @input_token: should be a list with lists inside. Inner ints are tokens and every list is a note.
        @target_shape: should be a list of a shape.
        """
        input_tensor = tf.convert_to_tensor(input_token)
        input_length = input_tensor.shape.as_list()[0]
        target_length = target_shape[0]

        split_n_times = math.floor(input_length / target_length)
        splitted_seqs = []

        # splitting once. Then use a loop.
        splitted_tensor, remaining_tensor = tf.split(
            input_tensor, (target_length, input_length - target_length)
        )
        remaining_tensor_main = remaining_tensor
        splitted_seqs.append(splitted_tensor)

        for i in range(split_n_times - 1):
            splitted_tensor, remaining_tensor = tf.split(
                remaining_tensor_main,
                (target_length, remaining_tensor.shape.as_list()[0] - target_length),
            )
            remaining_tensor_main = remaining_tensor
            splitted_seqs.append(splitted_tensor)

        return splitted_seqs, remaining_tensor_main

    def finalize(
        self,
        all_tokens_list: List[List[int]],
        target_shape=Union[List[int], tf.shape],
        output_dir: Optional[Union[str, Path]] = None,
        save_as_dataset: bool = False,
        stats: bool = True,
    ):
        """Combining both splitting() and eqalize_shapes() methods
        and finalizing the dataset.
        @Params
        @all_tokens_list: A list containing lists which contains lists.
        E.g. midi_tokens_all => pieces => notes
        @target_shape: Target shape of all the songs. Must be a list.
        @output_dir: Output directory to created dataset. Will not be used unless save=True.
        Must be a list and within that list there must be strs. 0. elem is the input, 1. elem is the target save dir.
        @save: Save to the directory or not.
        @stats: Debug purposes
        """

        def _gen_fn(tensor_list):
            for tensor in tensor_list:
                yield tensor  # ((tensor[:, 1:], tensor[:, :-1]))

        if save_as_dataset == True and output_dir == None:
            lg.error("Save is true although output_dir is not given.")

        target_length = target_shape[0]
        did_none = 0
        just_equalized = 0
        splitted = 0

        all_output_tensors = []

        for piece in tqdm(all_tokens_list, desc="Processing the data"):
            piece_length = len(piece)

            if target_length > piece_length:
                just_equalized += 1
                # if verbose:
                #  lg.info("This piece has been padded in order to match the given shape.")
                output_tensor = self.equalize_shapes2D(
                    piece, target_shape, verbose=False
                )
                all_output_tensors.append(output_tensor)

            elif target_length == piece_length:
                did_none += 1
                # if verbose:
                #  lg.info("Target length and the piece_length are alreay matched.")
                all_output_tensors.append(
                    piece
                )  # There is nothing to do so just add the tensor

            elif target_length < piece_length:
                splitted += 1
                # if verbose:
                #  lg.info("This piece has been splitted and padded in order to match the given shape.")
                splitted_tensor, remaining_tensor = self.splitting2D(
                    piece, target_shape, verbose=False
                )
                remaning_modified_tensor = self.equalize_shapes2D(
                    remaining_tensor, target_shape
                )
                for elem in splitted_tensor:
                    all_output_tensors.append(elem)  # Appending the splitted tensors
                all_output_tensors.append(
                    remaning_modified_tensor
                )  # Appending the remaining but padded tensor
            else:
                lg.error("It's just impossible...")

        if stats:
            print("=" * 50)
            print("{} many pieces was already in shape.".format(did_none))
            print("{} many pieces has been padded.".format(just_equalized))
            print("{} many pieces has been splitted and padded.".format(splitted))

        if save_as_dataset:
            lg.info("Saving dataset...")
            created_dataset = tf.data.Dataset.from_generator(
                _gen_fn, args=[all_output_tensors[0]], output_types=tf.int32
            )
            tf.data.experimental.save(created_dataset, output_dir[0])
            return created_dataset, all_output_tensors

            # WARNING! This function is deprecated but tf.data.Dataset.save doesn't work so I'm using this right now.
            # It might and probably will be removed in the future.
        return all_output_tensors

    def _convert_tf2list(self, all_tokens_tf: List[tf.Tensor]) -> List[List[int]]:
        """Converts tf.Tensors to back to lists. I don't want to waste
        time changing the pipeline to tf-free instead we're just going to change
        the output."""

        all_tokens_list = []

        for tensor in all_tokens_tf:
            all_tokens_list.append(
                tf.make_ndarray(tensor=tf.make_tensor_proto(tensor)).tolist()
            )

        return all_tokens_list

    def save_all_tokens(
        self, outputPath: Union[Path, str], tokens: Optional[List[List[int]]] = None
    ):
        """Saves all the imported tokens to single .json file.
        @Params:
        @outputPath: path to save
        @tokens: tokens to save. if specified will save those tokens. Otherwise it'll save
        self.all_tokens
        """

        if tokens == None:
            tokens = self.all_tokens

        if tokens == []:
            lg.critical("Tokens you are about to save is empty!")
            # raise Exception
        lg.info(f"Lenght of the tokens: {len(tokens)}")

        with open(outputPath, "w") as d:
            for tokenized_piece in tokens:
                # json.dump(token_dict, d)
                d.write('{"tokens":' + json.dumps(tokenized_piece) + "}\n")

        lg.info(f"Tokens succesfully saved to {outputPath}")
