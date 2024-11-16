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
import pickle
import string

unk = '<UNK>'

# RNN Model Definition
class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        _, hidden = self.rnn(inputs)
        z = self.W(hidden[-1])
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

# Prepare data for RNN using word embeddings
def convert_to_embedding_representation(data, word_embedding, word2index):
    vectorized_data = []
    for entry in data:
        document, y = entry['text'], entry['stars']
        words = document.lower().translate(str.maketrans("", "", string.punctuation)).split()
        vectors = [word_embedding.get(word, word_embedding[unk]) for word in words]
        vectors_tensor = torch.tensor(vectors, dtype=torch.float).view(len(vectors), 1, -1)
        vectorized_data.append((vectors_tensor, int(y) - 1))
    return vectorized_data

# Training the RNN Model
def train_rnn(model, train_data, valid_data, epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0
        
        for input_tensor, label in train_data:
            optimizer.zero_grad()
            # Forward pass
            predicted_vector = model(input_tensor)
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
            for input_tensor, label in valid_data:
                predicted_vector = model(input_tensor)
                predicted_label = torch.argmax(predicted_vector)
                correct_predictions += int(predicted_label == label)
                total_examples += 1
        validation_accuracy = correct_predictions / total_examples
        print(f"Validation Accuracy after Epoch {epoch + 1}: {validation_accuracy:.4f}")

# Main function to run the RNN
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
    
    # Load word embeddings
    embedding_path = './word_embedding.pkl'  # Placeholder path to embedding file
    word_embedding = pickle.load(open(embedding_path, 'rb'))
    
    # Convert data for RNN
    print("========== Vectorizing data for RNN ==========")
    rnn_train_data = convert_to_embedding_representation(train_data, word_embedding, None)
    rnn_valid_data = convert_to_embedding_representation(valid_data, word_embedding, None)
    
    # Initialize and train the RNN model
    embedding_dim = len(next(iter(word_embedding.values())))  # Assuming all embeddings are of the same size
    rnn_model = RNN(input_dim=embedding_dim, h=args.hidden_dim)
    train_rnn(rnn_model, rnn_train_data, rnn_valid_data, args.epochs, learning_rate=0.01)
