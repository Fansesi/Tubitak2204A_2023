# EN
## Dataset Process
This folder includes file for dataset and individual data (audio and/or midi) process. 

* `classes.py`: Has 4 classes for manipulating midi and tokenized data:
  * ProcessMIDI: Utility functions to get info about both the MIDI dataset.
  * AugmentTOK: Augmentation pipeline for json files (1D). 
  * ProcessTOK: Preprocessing 1D tokenization techniques, splitting and padding. 
  * ProcessTOK2D: Preprocessing 2D tokenization techniques, splitting and padding.

**Note on 2D processing**: in this work we have mainly used 1D tokenization approaches but at first we were thinking on using 2D approaches as well. We've converted our 1D processing pipeline for 2D but didn't convert our augmentation pipeline because we've abondened the idea of using 2D approaches.

* `murmuring.py`: using [basic-pitch](https://github.com/spotify/basic-pitch) as backend, convert audio data to .mid data.

* `process_search.py`: For processing MidiBERT embeddings and similarity search with FAISS.
  * `ProcessDataset`: A `torch.utils.data.Dataset` object to manipulate MidiBERT embeddings.
  * `SimilaritySearch`: A basic similarity searcher using [FAISS](https://github.com/facebookresearch/faiss). 

* `utils_v2`: Some basic utility functions.
