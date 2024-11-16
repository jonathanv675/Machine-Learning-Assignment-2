import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

unk = '<UNK>'

# FFNN Model Definition
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()  # The rectified linear unit
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)  # Softmax with log-probabilities for numerical stability
        self.loss = nn.NLLLoss()  # Cross-entropy loss
    
    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)
    
    def forward(self, input_vector):
        # First hidden layer representation
        h = self.activation(self.W1(input_vector))
        # Output layer representation
        z = self.W2(h)
        # Probability distribution using softmax
        predicted_vector = self.softmax(z)
        return predicted_vector

# Load data from the provided files
def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))
    return tra, val

# Create vocabulary and indices
from collections import Counter

def make_vocab_optimized(data, max_vocab_size):
    word_counter = Counter()
    for entry in data:
        document = entry['text'].split()
        for word in document:
            word = word.lower().translate(str.maketrans("", "", string.punctuation))
            word_counter[word] += 1
    most_common_words = word_counter.most_common(max_vocab_size)
    vocab = set([word for word, _ in most_common_words])
    return vocab

def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {word: idx for idx, word in enumerate(vocab_list)}
    index2word = {idx: word for idx, word in enumerate(vocab_list)}
    return vocab, word2index, index2word

# Vectorize data for FFNN using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_VOCAB_SIZE = 5000

def vectorize_data_tfidf(train_data, valid_data):
    # Prepare the training documents for TF-IDF vectorization
    train_documents = [entry['text'] for entry in train_data]
    
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE, lowercase=True, stop_words='english')
    
    # Fit the TF-IDF vectorizer on training data and transform the datasets
    tfidf_train_vectors = tfidf_vectorizer.fit_transform(train_documents)
    tfidf_valid_vectors = tfidf_vectorizer.transform([entry['text'] for entry in valid_data])
    
    # Convert TF-IDF results to tensors
    ffnn_train_data_tfidf = [(torch.tensor(tfidf_train_vectors[i].toarray(), dtype=torch.float).squeeze(), int(train_data[i]['stars']) - 1) for i in range(len(train_data))]
    ffnn_valid_data_tfidf = [(torch.tensor(tfidf_valid_vectors[i].toarray(), dtype=torch.float).squeeze(), int(valid_data[i]['stars']) - 1) for i in range(len(valid_data))]
    
    return ffnn_train_data_tfidf, ffnn_valid_data_tfidf

# Training the FFNN Model
def train_ffnn(model, train_data, valid_data, epochs, learning_rate, momentum):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0
        
        for input_vector, label in train_data:
            optimizer.zero_grad()
            # Forward pass
            predicted_vector = model(input_vector)
            # Compute loss
            loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([label]))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track total loss
            total_loss += loss.item()
            # Track correct predictions
            predicted_label = torch.argmax(predicted_vector)
            correct_predictions += int(predicted_label == label)
            total_examples += 1

        # Calculate training accuracy and average loss
        training_accuracy = correct_predictions / total_examples
        avg_loss = total_loss / total_examples
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {training_accuracy:.4f}")

        # Validation accuracy
        model.eval()
        correct_predictions = 0
        total_examples = 0
        with torch.no_grad():
            for input_vector, label in valid_data:
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct_predictions += int(predicted_label == label)
                total_examples += 1
        validation_accuracy = correct_predictions / total_examples
        print(f"Validation Accuracy after Epoch {epoch + 1}: {validation_accuracy:.4f}")

# Main function to run the FFNN
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    args = parser.parse_args()

    # Load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    
    # Vectorize data
    print("========== Vectorizing data ==========")
    ffnn_train_data_tfidf, ffnn_valid_data_tfidf = vectorize_data_tfidf(train_data, valid_data)
    
    # Initialize and train the FFNN model
    ffnn_model = FFNN(input_dim=MAX_VOCAB_SIZE, h=args.hidden_dim)
    train_ffnn(ffnn_model, ffnn_train_data_tfidf, ffnn_valid_data_tfidf, args.epochs, learning_rate=0.01, momentum=0.9)
