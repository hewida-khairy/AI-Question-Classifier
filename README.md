# AI-Question-Classifier

A machine learning project that classifies questions into predefined categories using a transformer-based model (DistilBERT).

## Project Overview
This project aims to build an AI system capable of understanding and categorizing questions automatically. Given a dataset of questions labeled with categories, the model can predict the correct label for any new question.

Key features:
- Supports multiple question categories (e.g., Modeling, Inference, Learning, etc.)
- Uses DistilBERT for text classification
- Includes simple data augmentation to increase dataset diversity
- Provides training, evaluation, and model saving

## Workflow
1. Load JSON dataset with questions and their labels
2. Apply simple data augmentation (word swaps) to increase data size
3. Tokenize and encode the questions for transformer input
4. Split dataset into training and testing sets
5. Train DistilBERT with HuggingFace Trainer
6. Evaluate model accuracy
7. Save the trained model and tokenizer for inference

## Requirements
- Python 3.8+
- PyTorch
- HuggingFace Transformers
- Datasets
- scikit-learn
