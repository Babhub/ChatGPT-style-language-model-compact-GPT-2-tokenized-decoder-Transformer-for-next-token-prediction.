# ChatGPT-style-language-model-compact-GPT-2-tokenized-decoder-Transformer-for-next-token-prediction.
ChatGPT-style decoder-only Transformer in PyTorch — GPT architecture, GPT-2 tokenizer, Penn Treebank-trained, compact blueprint for learning modern Generative AI.


GPT-Style Language Model (Decoder-Only Transformer)
Developed by Bab Jan | Generative AI Research & Product Development

Overview

This project implements a decoder-only Transformer language model, inspired by OpenAI’s GPT architecture, trained on the Penn Treebank dataset for next-token prediction.
It was built as part of an advanced NLP coursework project (NLP 243 – Language Modelling) at Stanford/University level, demonstrating a complete generative AI pipeline — from model definition to training, evaluation, and deployment compatibility.

This work showcases end-to-end mastery in Generative AI engineering, covering:

* Deep learning model design (PyTorch)
* Attention-based architectures
* Tokenization and data handling with Hugging Face GPT-2 tokenizer
* Training pipeline optimization
* Evaluation using perplexity metrics
* Cloud-ready model export for inference or fine-tuning

Model Architecture

A custom GPT-style decoder-only Transformer was implemented from scratch, featuring:

* 6 Transformer blocks with 8 attention heads each
* Sinusoidal positional embeddings for sequence ordering
* Causal masking to prevent future-token leakage
* Cross-entropy training objective with padding ignored (ignore_index=-100)
* AdamW optimization with gradient clipping and dropout regularization
* Vocabulary size: 50,257 (GPT-2)
* Context window: 150 tokens
* Parameter count: ~30.47 million (efficiently under the 50M academic constraint)

Training & Evaluation Results

Split | Average Loss | Perplexity
Train | 3.88 | 47.28
Validation | 4.12 | 61.86
Test | 4.01 | 55.26

Training was completed using NVIDIA GPU acceleration (CUDA).
The model demonstrates strong convergence and efficient generalization for its parameter scale.

Components

* model.py – Complete GPT-style architecture with causal mask and LayerNorm residuals
* train.py – Robust training loop using next-token prediction
* main.py – End-to-end orchestration for data, model, training, and evaluation
* evaluation.py – Perplexity-based assessment for train/val/test sets
* download_best_model.py – Auto-download from Google Drive for grading/inference
* requirements.txt – Dependency reproducibility
* hw2_report.pdf – Technical and conceptual project report

Core Technologies

Area | Tools & Frameworks
Language Modelling | Transformer Decoder (GPT architecture)
Framework | PyTorch 2.9.0 + CUDA
Tokenization | Hugging Face Transformers (GPT-2 tokenizer)
Dataset | Penn Treebank (≈1M tokens)
Evaluation | Perplexity metric
Deployment | Google Drive + Auto-Download integration
Authoring | Python 3.12, VS Code, Linux (UCSC GPU Cluster)

Key Features

* True GPT-style causal attention implementation
* Sinusoidal positional encoding (extendable to RoPE)
* Autograd-safe, modular design for scaling experiments
* Reproducible training pipeline & clean code structure
* Compact and efficient (<50M parameters) for real-world deployment or academic use
* Evaluation-ready – compatible with standardized autograder pipeline

Future Enhancements

* Add Rotary Positional Embeddings (RoPE) for improved context generalization
* Experiment with Top-k / Nucleus (Top-p) sampling for creative text generation
* Integrate greedy / beam search for controlled decoding
* Extend to domain-specific fine-tuning (medical, legal, or conversational corpora)
* Deploy via FastAPI or Streamlit as a real-time AI writing assistant prototype

Author
Bab 

“Building practical Generative AI systems that transform research into real-world impact.”

