# 🎹 AI Research Project: Automatic Music Transcription (Piano Note Recognition)

## 1. Project Overview
This project explores **Automatic Music Transcription (AMT)**, focusing on identifying musical notes from piano recordings.  
We start with a digital piano (Thomann SP-320) that provides both **MIDI and audio outputs**, enabling the creation of paired datasets (Audio ↔ MIDI) for supervised learning.  

The goal is to investigate different machine learning approaches for mapping audio signals to symbolic note representations, starting from simple monophonic transcription to full polyphonic transcription with expressive features (velocity, sustain pedal, dynamics).

---

## 2. Research Objectives
- Develop methods to **recognize piano notes** from audio.
- Compare approaches based on:
  - **Spectrogram representations** (STFT, Mel-spectrogram, CQT).
  - **Model architectures**: CNNs, CRNNs, Transformers.
  - **End-to-end systems** (e.g., Onsets and Frames).
- Investigate the impact of:
  - Training with **paired MIDI-audio datasets**.
  - **Self-supervised pretraining** on audio (wav2vec2, HuBERT).
  - **Noise robustness** (clean vs. realistic environments).
- Explore **lightweight models** suitable for real-time inference.

---

## 3. Datasets
- **Primary Dataset**: Custom recordings from Thomann SP-320 (paired MIDI & audio).
- **External Benchmarks**:
  - [MAESTRO dataset (Google)](https://magenta.tensorflow.org/datasets/maestro) – ~200h of aligned MIDI & audio.
  - [MAPS dataset](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/02/maps-database/) – digital and acoustic piano recordings.
  - [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano) – large-scale piano dataset.

---

## 4. Methodology

### Phase 1 – MIDI-based Experiments
- Capture clean MIDI events (pitch, velocity, duration).
- Train simple models to classify notes and sequences.
- Establish baseline for symbolic note recognition.

### Phase 2 – Audio + MIDI Alignment
- Record simultaneous audio and MIDI.
- Use MIDI as ground-truth labels for supervised training.
- Train models on spectrograms → note events.

### Phase 3 – Model Exploration
- **Baseline**: CNN/CRNN on spectrograms.
- **Advanced**: Transformers for temporal modeling.
- **End-to-End**: Onsets & Frames architecture.
- Evaluate self-supervised pretrained models adapted to music.

### Phase 4 – Robustness & Expressivity
- Add background noise for generalization.
- Extend to dynamics: velocity, sustain pedal.
- Explore real-time feasibility with model compression.

---

## 5. Evaluation
- **Frame-level accuracy** – correct notes per frame.
- **Onset F1-score** – accuracy of note start detection.
- **Note-level precision/recall/F1** – overall transcription quality.
- Compare against benchmarks (MAESTRO baseline results).

---

## 6. Expected Contributions
- Empirical study on **representations and models** for piano transcription.
- Analysis of **digital vs. acoustic piano datasets**.
- Proposals for **lightweight AMT models** with real-time capabilities.
- Open-source dataset of recordings from Thomann SP-320.

---

## 7. Tools & Frameworks
- [Python](https://www.python.org/)
- [Mido / PrettyMIDI](https://github.com/craffel/pretty-midi) – MIDI handling
- [Librosa](https://librosa.org/) – audio analysis
- [PyTorch](https://pytorch.org/) – deep learning
- [Jupyter Notebooks](https://jupyter.org/) – experimentation
- [TensorBoard / WandB](https://wandb.ai/) – experiment tracking

---

## 8. Roadmap

| Phase | Timeline | Goals |
|-------|----------|-------|
| Phase 1 | Month 1–2 | Collect MIDI data, baseline note classification |
| Phase 2 | Month 3–4 | Audio + MIDI dataset, supervised training |
| Phase 3 | Month 5–7 | Model exploration (CNN, CRNN, Transformers, Onsets & Frames) |
| Phase 4 | Month 8–9 | Robustness, expressivity, real-time models |
| Phase 5 | Month 10 | Evaluation, benchmarking, paper/report writing |

---

## 9. References
- Hawthorne, C. et al. (2018). [Onsets and Frames: Dual-Objective Piano Transcription](https://arxiv.org/abs/1710.11153).  
- Google Magenta Project – [Automatic Music Transcription](https://magenta.withgoogle.com/).  
- Benetos, E. et al. (2019). Automatic Music Transcription: A Survey. *Transactions of the International Society for Music Information Retrieval*.  

---
