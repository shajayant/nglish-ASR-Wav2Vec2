# Fine-Tuning Wav2Vec2 for English Automatic Speech Recognition (ASR)

This repository demonstrates how to **fine-tune Wav2Vec2**, a state-of-the-art self-supervised speech model, for **English Automatic Speech Recognition (ASR)** using Hugging Face Transformers.

---

## Project Overview
- **Model**: Wav2Vec2 (Transformer-based speech encoder with CTC head).  
- **Dataset**: [LibriSpeech ASR](https://www.openslr.org/12) – using subsets `train-clean-100` and `test-clean`.  
- **Task**: Convert raw audio waveforms into text transcriptions.  
- **Framework**: Hugging Face Transformers + PyTorch.  

---

## Key Features
- Fine-tuning pretrained **facebook/wav2vec2-base** on English speech.  
- Preprocessing pipeline: audio resampling, transcript cleaning, vocab generation.  
- Tokenizer + Feature Extractor + Processor setup.  
- Training with **CTC loss**.  
- Evaluation with **Word Error Rate (WER)**.  
- Model export + Hugging Face Hub integration.  

---

## Training Workflow
1. **Preprocess data**  
   - Clean transcripts (remove punctuation).  
   - Build vocabulary JSON.  
   - Prepare dataset with tokenized labels + audio features.  

2. **Define model & processor**  
```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
