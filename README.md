# Tubitak 2204A - EUCYS - 2023

## EN

Repository for Tübitak's 2204A project competition in 2023.

This work is being developed more. Another repository will be created once the development is finished. Link of the newly created repository will be published when it's ready.

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

**Note**: Dataset and trained checkpoints will be published soon.



## TR

2023 yılında düzenlenen Tübitak 2204A yarışmasının kod deposu. 

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
**Not**: Veri seti ve modelin eğitimine ilişkin checkpoint'ler en kısa zamanda paylaşılacaktır. 

