# Tubitak 2204A 2023 - EUCYS 2023 - ISEF 2024
[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub license](https://img.shields.io/github/license/Fansesi/Tubitak2204A_2023)](https://github.com/Fansesi/Tubitak2204A_2023/blob/main/LICENSE)

## EN

Repository for the project entitled with *Synthesising Classical Guitar Pieces Using Transformers*.  


### Folder Structure

```markdown

├── dataset_process
│   ├── classes                # Manipulation of midi and tokenized data. 
│   ├── murmuring              # Turning murmuring audio to .mid files. (basic-pitch)
|   ├── process_search         # For processing MidiBERT embeddings and for similarity search with FAISS
│   └── utils_v2               # Simple utility functions
├── samples
│   └── ... .wav               # Examples of .wav files from generated samples.
├── train_inference
│   ├── inference_GPT2.ipynb   # inference with retrieval procedures. 
│   └── training_GPT2.ipynb    # training the GPT2 model.

```

For the measurement of playability one may refer to [playability repo](https://github.com/Fansesi/guitar/tree/main), created by the author. 


## TR

2023 yılında düzenlenen, Tübitak 2204A yarışmasında yazılım alanında birinci olan *Transformer Mimarisi Kullanarak Klasik Gitar Özelinde Sentetik Müzik Üretimi* adlı projenin kod reposu.

### Dosya Düzeni

```markdown
├── dataset_process
│   ├── classes                # MIDI ve tokenize edilmiş verilerin manipülasyonu. 
│   ├── murmuring              # *Mırıldanma* verisini .mid verisine dönüştürme. (basic-pitch) 
|   ├── process_search         # MidiBERT embedding'lerini işleme ve FAISS aracılığıyla benzerlik araması gerçekleştirme.
│   └── utils_v2               # Basit yardımcı fonksiyonlar
├── samples
│   └── ... .wav               # Sentezlenen müziklerden örnek .wav dosyaları.
├── train_inference
│   ├── inference_GPT2.ipynb   # *Retrieval* prosedürleriyle inference.
│   └── training_GPT2.ipynb    # GPT2 modelinin eğitimi.
```

Üretilen verilerin çalınabilirliğini ölçen algıritmaya, yazarın açmış olduğu [playability repo](https://github.com/Fansesi/guitar/tree/main) aracılığıyla ulaşılabilir.
