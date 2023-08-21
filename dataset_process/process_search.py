"""
This file is for processing dataset/creating MidiBERT embeddings and similarity search with FAISS.
Please specify 'MidiBERT_PATH' to sys.path down below.

"""
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
from pathlib import PosixPath
from tqdm import tqdm
from loguru import logger as lg
import time
import pickle
import numpy as np
import faiss
import os
import sys

MidiBERT_PATH = ".../MIDI_BERT_CP"
sys.path.append(MidiBERT_PATH)
from glob import glob
import torch
from torch.utils.data import DataLoader, Dataset
from MidiBERT.model import MidiBert  # type: ignore
from transformers import BertConfig


class ProcessDataset(Dataset):
    """Takes the path of tokenized file then embeds it using pretrained MidiBERT.
    @Params:
    ---
    @vocabPath: CP dictionary path.
    @checkpointPath: Checkpoint of pretrained MidiBERT base.
    @dataPath: Saved data path. A .npy file which stores all the datas. If given dataItself won't be used.
    @dataItself: Data itself can be given as well. To do this you must specify dataPath as None.
    @save: if set to True, saves the created embeddings into a npy file
    @verbose: debug reasons.
    @div: division amount to change the shape of the data. Must be a power of 2.
    @pool_type: should be one of the following: 'avg', 'max', 'avg_max', 'max_avg'.
    In 'avg_max', 'max_avg', order of pooling operations is as the names suggest.
    This variable is used while processing the data.

    Notes
    ---
    1) If self.data is not specified with the input parameters, this means we are expecting it to be specified later
    and show a warning message about reshaping the data. You can reshape the data with the given div parameter via the
    div_reshape() function. If you don't specify the self.data and use __getitem__ or reshaper you'll get an error.
    """

    def __init__(
        self,
        vocabPath: Union[str, Path],
        checkpointPath: Union[str, Path],
        # outputPath: Optional[str] = None,
        dataPath: Optional[str] = None,
        dataItself: Optional[
            List[List[List[int]]]
        ] = None,  # it's shape might be wrong.
        verbose: bool = True,
        div: int = 4,
        pool_type: str = "avg",
    ):
        self.checkpointPath = checkpointPath
        self.vocabPath = vocabPath
        self.div = div
        self.pool_type = pool_type
        self.verbose = verbose
        self.dataPath = dataPath
        self.dataItself = dataItself

        self.data = self.instantiate_data(self.dataPath, self.dataItself)

        self.dataTensor = None
        self.dataNp = None

        # because we're using MidiBERT's pipeline, the dataShape is expected to be
        # [n, 512, 4] => [nb_data, batch_size, sq_length]

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.dataReshapedTensor = self.dataReshapedTensor.to(self.device)
            if self.verbose:
                lg.info(f"Device is {self.device} ...")
        else:
            self.device = torch.device("cpu")
            if self.verbose:
                lg.info(f"Device is {self.device} ...")

        self.model = self.init_model(self.checkpointPath, self.vocabPath)

    def instantiate_data(
        self,
        dataPath: Optional[str] = None,
        dataItself: Optional[
            List[List[List[int]]]
        ] = None,  # it's shape might be wrong.
    ):
        """This functions content was used to be in the __init__() but I've changed into a seperate
        function for better usability, especially while creating the instance and filling it with data of this class.
        """
        if dataPath == None and dataItself == None:
            lg.warning(
                "While using the ProcessDataset class in the future, don't forget to specify data and reshape it."
            )
            return None
        else:
            if dataPath != None:
                lg.debug(f"Loading the data from {dataPath} using np.load()")
                return np.load(dataPath)
            else:
                return dataItself

    def __len__(self):
        if self.dataTensor != None:
            return self.dataTensor.shape[0]
        else:
            return np.array(self.data).shape[0]

    def __getitem__(self, index) -> Dict[int, Tuple[torch.Tensor, List[List[int]]]]:
        """Creates dictionaries of the shape:
        {0: (torch.Tensor(EMBEDDINGS_0), RELATIVE_TOKENS_CP_0), 1:(torch.Tensor(EMBEDDINGS_1), RELATIVE_TOKENS_CP_1), ...}

        Here we are expecting the reshaped data to be the self.data. So after instantiating the self.data, use the self.div_reshape()
        to reshape the self.data.

        I'm saving the CP versions as tensors too because when I try to save them as pure lists, DataLoader tries to convert
        them to tensors but in a wrong way. When I process the CP data though, I'll be converting them to pure lists.
        """
        if type(self.dataTensor) == type(None) and type(self.dataNp) == type(None):
            lg.error(
                "self.dataTensor and self.dataNp is None, so you can't use them for creating embeddings."
            )
            raise Exception

        return {
            index: (
                self.pool_single_outputs(
                    self.dataTensor[index],
                    poolType=self.pool_type,
                ),
                torch.tensor(self.dataNp.tolist()[index]),
            )
        }

    def div_reshape(self, div: Optional[int] = None) -> None:
        """Does the reshaping operation with a given div parameter. This operation was used to done in __init__().
        This function will turn the self.data to Tensor and Np array (self.dataTensor, self.dataNp) at the end but
        self.data can be list, np.array or torch.tensor initially. self.data won't be changed.

        And also we are expecting the shape to be at most 3D. If it's more than 3D, we try to squeeze it on the first dim.
        to match the expected shape.
        """
        div = self.div if div == None else div
        if type(self.data) == type(None):  # strange but this is the usage.
            lg.error("self.data is None, so you can't reshape it.")
            raise Exception

        data = self.data.copy()
        lg.trace(f"Data is:\n{data}")

        self.dataNp = np.array(data)

        if len(self.dataNp.shape) >= 4 and self.dataNp.shape[0] == 1:
            lg.trace(
                f"Input data's shape is {self.dataNp.shape}. I'm squeezing the first dimension."
            )
            self.dataNp = np.squeeze(self.dataNp, axis=0)
            lg.trace(f"So the new shape is {self.dataNp.shape}")
        self.dataTensor = torch.tensor(
            self.dataNp
        )  # using np.array becuase it's faster.

        # shapes are the same for both torch tensor and Np tensor.
        new_shape = [
            self.dataTensor.shape[0] * self.div,
            self.dataTensor.shape[1] // self.div,
            self.dataTensor.shape[2],
        ]
        lg.debug(f"New shape: {new_shape}")
        self.dataTensor = torch.reshape(self.dataTensor, new_shape)
        self.dataNp = np.reshape(self.dataNp, new_shape)

        del data

    def embed_using_dataloader(
        self,
        batch_size: int = 32,
        individual_save: bool = False,
        single_save: bool = True,
        outputDir: str = "",
    ) -> torch.Tensor:
        """Embeds the vectors (self.dataReshaped) using torch.utils.data.Dataloader.
        @Params:
        @batch_size: batch size to batch the embeddings.
        @individual_save: saves the embeddings individually after batching and embedding them.
        @single_save: concatenates the embeddings into a single dummy_tensor and saves it as
        single file into the output directory.
        @outputDir: path to save the created embeddings. if set to '' then it won't save.
        Note: This function will not return the created embeddings if individual_save = True.
        If individual_save=True, returned tensor will be just some dummy Tensor.
        """
        my_dataloader = DataLoader(
            self, batch_size=batch_size, shuffle=True, drop_last=True
        )
        # Example tensor to get the shape
        ex_tensor = self.pool_single_outputs(self.dataReshaped[0])
        dummyTensor = torch.zeros(
            [
                batch_size,
                ex_tensor.shape[0],
                ex_tensor.shape[1],
            ]
        )  # an example of shape
        del ex_tensor

        for i_batch, item in enumerate(tqdm(my_dataloader)):
            if individual_save == True:
                torch.save(
                    torch.squeeze(item, dim=1),
                    outputDir + f"embedding{i_batch}.pt",
                )
            else:
                dummyTensor = torch.cat([dummyTensor, item])

        if single_save == True:
            torch.save(
                dummyTensor[1:],
                outputDir + f"ALL_EMBEDDINGS.pt",
            )

        return dummyTensor

    def embed(
        self, tensor: Union[torch.Tensor, np.ndarray, List], model: MidiBert
    ) -> torch.Tensor:
        """Returns embeddings using pretrained MidiBERT.
        Expected shape is [nb_data, batch_size, 4 (sequence_length)]"""

        return model.forward(tensor).last_hidden_state

    def embed_multiple_save(
        self,
        tokens: torch.Tensor,
        model: MidiBert,
        save: bool = False,
        individualSave: bool = False,
        outputPath: Optional[str] = None,
        d: int = 768,
    ):
        """Creates embeddings for given tokens. individualSave=True saves each embedding individually.
        save=True saves the concatenated tensor into a .npy file."""
        print(f"There are total of {len(tokens)} files to save.")

        retrTensor = torch.zeros(
            [1, tokens.shape[1], d]
        )  # a placeholder tensor. You may need to CHANGE this later on.

        for i, elem in enumerate(tqdm(tokens)):
            embeddings = self.embed(elem.unsqueeze(0), model=model)
            retrTensor = torch.cat((retrTensor, embeddings))
            if individualSave:
                np.save(
                    outputPath + f"{i}.npy",
                    embeddings.detach().numpy(),
                )

        if save:
            np.save(outputPath + "embeddings.npy", retrTensor.detach().numpy())

        return retrTensor

    def pool_multiple_outputs(
        self,
        inputTensor: torch.Tensor,
        poolType: str = "avg",
        size: int = 4,
        stride: int = 4,
        dimension: int = 768,
    ):
        """Takes tensor as input, embed it using self.embed(model = self.model), pool it using
        AvgPool or MaxPool or AvgPool-MaxPool or MaxPool-AvgPool.
        @Params:
        @tensor: is expected to be in shape [nb_data, cluster_size, 4]
        @poolType: should be one of the following: 'avg', 'max', 'avg_max', 'max_avg'.
        In 'avg_max', 'max_avg', order of pooling operations is as the names suggest.
        @size: window size
        @stride: stride size
        """

        if poolType == "avg_max" or "max_avg":
            dummyTensor = torch.rand(
                [
                    1,
                    inputTensor.shape[1] // size**2,
                    dimension // stride**2,
                ]
            )
        if poolType == "avg" or "max":
            dummyTensor = torch.rand(
                [1, inputTensor.shape[1] // size, dimension // stride]
            )

        for single_tensor in tqdm(inputTensor):
            dummyTensor = torch.cat([dummyTensor, self.pool_single_outputs(single_tensor, poolType=poolType, size=size, stride=stride, dimension=dimension)])  # type: ignore

        return dummyTensor[1:]  # type: ignore

    def pool_single_outputs(
        self,
        inputTensor: torch.Tensor,
        poolType: str = "avg",
        size: int = 4,
        stride: int = 4,
        dimension: int = 768,
    ) -> torch.Tensor:
        """Takes tensor as input, embed it using self.embed(model = self.model), pool it using
        AvgPool or MaxPool or AvgPool-MaxPool or MaxPool-AvgPool.
        @Params:
        @tensor: is expected to be in shape [cluster_size, 4]. And this is the main difference
        between the single and multiple version of this function.
        @poolType: should be one of the following: 'avg', 'max', 'avg_max', 'max_avg'.
        In 'avg_max', 'max_avg', order of pooling operations is as the names suggest.
        @size: window size
        @stride: stride size
        """
        poolAVG = torch.nn.AvgPool2d(size, stride)
        poolMAX = torch.nn.MaxPool2d(size, stride)

        tensor = self.embed(torch.unsqueeze(inputTensor, 0), self.model)
        if poolType == "avg":
            # torch.concat([dummyTensor, poolAVG(tensor)])  # type: ignore
            # del tensor
            return poolAVG(tensor)
        elif poolType == "max":
            # torch.concat([dummyTensor, poolMAX(tensor)])  # type: ignore
            # del tensor
            return poolMAX(tensor)
        elif poolType == "avg_max":
            # torch.concat([dummyTensor, poolAVG(poolMAX(tensor))])  # type: ignore
            # del tensor
            return poolAVG(poolMAX(tensor))
        elif poolType == "max_avg":
            # torch.concat([dummyTensor, poolMAX(poolAVG(tensor))])  # type: ignore
            # del tensor
            return poolMAX(poolAVG(tensor))
        else:
            lg.error(
                "You should give one of the parameters for the function pool_outputs(): {}".format(
                    {"avg", "max", "avg_max", "max_avg"}
                )
            )
            raise Exception

    def init_model(
        self, checkpointPath: Union[str, Path], vocabPath: Union[str, Path]
    ) -> MidiBert:
        """Initializes the MidiBERT with the path inputs."""

        # it's currently set to cpu because I'm on a cpu-only machine.
        # but I can add a variable for gpu in the future. I have to look into that.
        modelCKPT = torch.load(checkpointPath, map_location=self.device)
        with open(vocabPath, "rb") as f:
            e2w, w2e = pickle.load(f)

        configuration = BertConfig(
            max_position_embeddings=512,  # args.max_seq_len,
            position_embedding_type="relative_key_query",
            hidden_size=768,
        )

        # ADDED to.("cuda") LINE
        midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e).to(self.device)
        midibert.load_state_dict(modelCKPT["state_dict"])

        return midibert

    def shape_tensor(self):
        """Shapes the tensors into two bar length."""
        return NotImplementedError

    def merge_embeddings_v1(
        self,
        pathToEmbeddings: str,
        save: bool = False,
        outputDir: str = "",
        ext: str = "pt",
    ) -> torch.Tensor:
        """Imports the files, merges them to a dummy tensor, saves it if specified,
        returns the dummy tensor.
        @Params:
        @pathToEmbeddings: path to embedding files. Extensions are are expected
        to be suitable with torch.load. Note: expected a / at the end.
        @save: save
        @outputDir: only used when save=True.
        """
        device = self.device
        paths = glob(pathToEmbeddings + f"*.{ext}")
        if self.verbose:
            lg.debug(f"Total paths: {len(paths)}")
        embd_shape = torch.load(paths[0], map_location=device).shape
        dummy_tensor = torch.zeros(embd_shape, device=device)

        for filePath in tqdm(paths):
            embd = torch.load(filePath, map_location=device)
            dummy_tensor = torch.cat([dummy_tensor, embd])

        if save:
            torch.save(dummy_tensor[1:], outputDir + "ALL_EMBEDDINGS.pt")

        return dummy_tensor[1:]

    def merge_embeddings_v2(
        self,
        pathToEmbeddings: Union[str, Path],
        save: bool = False,
        outputDir: Optional[Union[str, Path]] = None,
        ext: str = "npy",
    ) -> Dict[int, Tuple[torch.Tensor, List[List[int]]]]:
        """v2 because now we are using the {index: (embedding, tokens)} approach."""
        if save == True and outputDir == None:
            lg.error("Save is true altough outputDir is not given.")
            raise Exception

        paths = list(Path(pathToEmbeddings).glob(f"*.{ext}"))
        if len(paths) == 0:
            lg.error(f"Couldn't find any .{ext} files in {pathToEmbeddings}.")
            raise Exception

        lg.debug(f"Total paths: {len(paths)}")

        temp_index__embd_token: List[
            Dict[int, Tuple[torch.Tensor, List[List[int]]]]
        ] = []

        for filePath in tqdm(paths):
            temp_index__embd_token.append(np.load(filePath, allow_pickle=True))

        dummy = temp_index__embd_token[0]
        for element in temp_index__embd_token:
            dummy = dummy | element

        if save:
            np.save(outputDir, dummy, allow_pickle=True)

        return dummy


class SimilaritySearch:
    """Takes embeddings and indexes them. Also takes an input as embedding and finds similar
    embeddings to that input.

    @Params:
    ---
    @embeddings: if a pathlike, should be suitable for torch.load. If it's a tensor or None will process accordingly.
    You can use None while using search functions.

    Note
    ---
    1) self.embds is used thoroughly in the functions. So after doing the processes in the __init__(),
    set the final embds as self.embds.
    """

    def __init__(
        self,
        embeddings: Optional[Union[str, torch.Tensor]] = None,
    ) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            lg.info(f"Device is {self.device} ...")
        else:
            self.device = torch.device("cpu")
            lg.info(f"Device is {self.device} ...")

        if type(embeddings) == str:
            # We load the embeddings if given
            # (nb_data, batch_size, 768/x) x and batch size is defined in the preprocessor.
            lg.debug(f"Loading the embeddings from: {embeddings}")
            self.embds = self.process_input_embeddings(
                np.load(embeddings, allow_pickle=True)
            ).to(self.device)
            self.dimension = self.embds.shape[-1]
            lg.debug(f"Loading finished. Shape of the data is {self.embds.shape}")

        elif type(embeddings) == torch.Tensor:
            self.embds = embeddings
            self.dimension = self.embds.shape[-1]
            lg.debug(f"Loading finished. Shape of the data is {self.embds.shape}")

        else:
            lg.warning(
                f"Didn't created self.embds because it's type is {type(embeddings)}."
            )

    def process_input_embeddings(
        self, embeddings: np.ndarray[Dict[int, Tuple[torch.Tensor, List[List[int]]]]]
    ) -> torch.Tensor:
        """Processes the embeddings to set the self.embeddings. This process is necessary for the shape,
            Dict[int, Tuple[torch.Tensor, torch.tensor(List[List[int]]) ]].
        Here int is index (0,1,2,3,...), torch.Tensor is the embeddings and the lists are CP representation
        of the embeddings. It's not in their purest form because of some unexpected behaviour of DataLoader.
        """
        embeddings = embeddings.tolist()
        lg.debug(f"Merging these vectors: \n{embeddings, embeddings[0][0].shape}")
        dummy = torch.randn(torch.unsqueeze(torch.squeeze(embeddings[0][0]), 0).shape)

        for i in embeddings:
            dummy = torch.cat(
                [dummy, torch.unsqueeze(torch.squeeze(embeddings[i][0]), 0)]
            )

        lg.debug(f"Final dummy vector: \n{dummy[..., 0:], dummy[..., 0:].shape}")
        return dummy[..., 0:]

    def _prepare_input_vectors(self, vectors: torch.Tensor):
        """Prepares the dimensions of vectors for indexings/searching."""

        def multiply_ends(a_list):
            num = 1
            for i in a_list[1:]:
                num = num * i
            return num

        if len(vectors.shape) >= 3:
            vectors = (
                torch.reshape(vectors, [vectors.shape[0], multiply_ends(vectors.shape)])
                .detach()
                .numpy()
            )
        elif len(vectors.shape) == 2:
            vectors = vectors.detach().numpy()
        else:
            lg.error(f"Given vectors are shape {vectors.shape}.")
            raise Exception

        return vectors

    def create_L2index(
        self,
        vectors: torch.Tensor,
        save: bool = False,
        outputPath: Union[str, Path] = "",
    ) -> faiss.IndexFlatL2:
        """Creates FAISS L2 index and saves it. Vectors can both be in the shape of [nb_data, cluster_size, 768/x]
        (x and clusters_size is set in the embedding processor.) and [nb_data, column_size] (column_size is set in
        the jsymbolic2). This way this class can be both used in embeddings and in jsymbolic.
        @Params:
        @vectors: vectors to create index. Can be 2 or 3 or more dimensional. Reshaping will be done accordingly.
        @save: save
        @outputPath: path to save. Only used in if save=True.
        """

        vectors = self._prepare_input_vectors(vectors)

        index = faiss.IndexFlatL2(vectors.shape[-1])
        add_time_start = time.time()
        index.add(vectors)  # type: ignore
        add_time_end = time.time()
        lg.debug(f"Adding took {round(add_time_end-add_time_start, 3)}")
        lg.debug(
            f"Total number of indexes: {index.ntotal} with each of them have dimensions {vectors.shape[-1]}"
        )

        if save:
            faiss.write_index(index, os.path.join(outputPath, "L2index.index"))
            lg.info(f"Index saved at {os.path.join(outputPath, 'L2index.index')}")

        return index

    def L2_dist(
        self,
        # d: int,
        inputTensor: torch.Tensor,
        k: int = 3,
        FAISS_index: Union[str, Path, faiss.IndexFlatL2] = "",
        save: bool = False,
        outputPath: str = "",
    ) -> Tuple[float, float]:
        """Indexes the self.embds using IndexFlatL2 or imports the index file . Then searches similar embeddings to input.
        @Params:
        @d: dimension.
        @input: search element. Should be an embedding not tokens.
        @FAISS_index: can be a path or the index it# If set to '', then an index will be created
        with the embeddings created in the __init__().
        @save: will only be used while creating new index. Otherwise set to False
        @outputPath: will only be used while creating new index. Otherwise set to ''
        """
        if FAISS_index == "":
            lg.debug("New index is being created...")
            if save == True and outputPath == "":
                lg.error("Although you want to save it, you didn't set a output path!")
                raise Exception
            else:
                index = self.create_L2index(
                    self.embds.detach().numpy(), save=save, outputPath=outputPath
                )

        elif type(FAISS_index) == faiss.IndexFlatL2:
            lg.debug("Index has been taken...")
            index = FAISS_index

        else:  # then read the index from the disk
            lg.debug("Reading the index from disk...")
            index = faiss.read_index(str(FAISS_index))

        lg.debug(f"Total number of indexes: {index.ntotal}")

        inputTensor = self._prepare_input_vectors(inputTensor)

        search_time_start = time.time()
        D, I = index.search(inputTensor, k=k)  # type: ignore
        search_time_end = time.time()

        lg.debug(f"Searching took {round(search_time_end-search_time_start, 3)}.")
        return D, I

    def create_cos_index(
        self,
        vectors: torch.Tensor,
        save: bool = False,
        outputPath: Union[str, Path] = "",
    ) -> faiss.IndexFlatIP:
        """Creates FAISS cos index and saves it. Vectors can both be in the shape of [nb_data, cluster_size, 768/x]
        (x and clusters_size is set in the embedding processor.) and [nb_data, column_size] (column_size is set in
        the jsymbolic2). This way this class can be both used in embeddings and in jsymbolic.
        @Params:
        @vectors: vectors to create index. Can be 2 or 3 or more dimensional. Reshaping will be done accordingly.
        @save: save
        @outputPath: path to save. Only used in if save=True. Only specify the path, name will be Cosindex
        """

        lg.debug(
            f"Embeddings used while creating the cos index: \n{vectors, vectors.shape}"
        )
        vectors = self._prepare_input_vectors(vectors)
        lg.debug(f"Vectors after preparing them : \n{vectors, vectors.shape}")
        index = faiss.IndexFlatIP(vectors.shape[-1])

        faiss.normalize_L2(np.ascontiguousarray(vectors))  # normalizing vectors
        add_time_start = time.time()
        index.add(vectors)  # type: ignore
        add_time_end = time.time()

        lg.info(f"Adding took {round(add_time_end-add_time_start, 3)}")
        lg.info(
            f"Total number of indexes is {index.ntotal} with each of them having dimensions {vectors.shape[-1]}"
        )

        if save:
            faiss.write_index(index, os.path.join(outputPath, "Cosindex.index"))
            lg.info(f"Index saved at {os.path.join(outputPath, 'Cosindex.index')}")

        return index

    def cos_dist(
        self,
        # d: int,
        inputTensor: torch.Tensor,
        k: int = 3,
        FAISS_index: Union[str, Path, faiss.IndexFlatIP] = "",
        save: bool = False,
        outputPath: str = "",
    ) -> Tuple[float, float]:
        """Indexes the self.embds using IndexFlatIP or imports the index file. Then searches similar
        embeddings to input using cosine similarity.
        @Params:
        @d: dimension.
        @input: search element. Should be an embedding not tokens.
        @FAISS_index: can be a path or the index itself. If set to '', then an index will be created
        with the embeddings created in the __init__().
        @save: will only be used while creating new index. Otherwise set to False
        @outputPath: will only be used while creating new index. Otherwise set to ''
        @verbose: debug purposes
        Returns:
        Tuple(distances, indexes)

        """
        if FAISS_index == "":
            lg.debug("New index is being created...")
            if save == True and outputPath == "":
                lg.error("Although you want to save it, you didn't set a output path!")
                raise Exception
            else:
                lg.debug("Creating new index...")
                index = self.create_cos_index(
                    self.embds.detach().numpy(), save=save, outputPath=outputPath
                )

        elif type(FAISS_index) == str or type(FAISS_index) == Path:
            lg.debug("Reading the index from disk...")
            index = faiss.read_index(str(FAISS_index))

        else:
            lg.debug("Index has been taken...")
            index = FAISS_index

        lg.debug(f"Total number of indexes: {index.ntotal}")

        inputTensor = self._prepare_input_vectors(inputTensor)
        faiss.normalize_L2(inputTensor)

        search_time_start = time.time()
        D, I = index.search(inputTensor, k=k)  # type: ignore
        search_time_end = time.time()

        lg.debug(f"Searching took {round(search_time_end-search_time_start, 3)}.")

        return D, I
