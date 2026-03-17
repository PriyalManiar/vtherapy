import random
import json
import os

import torch

from .chatbot_model import NeuralNet
from .nltk_utils import bag_of_words, tokenize

# Project root (parent of scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENTS_PATH = os.path.join(PROJECT_ROOT, 'config', 'intents.json')
DATA_PATH = os.path.join(PROJECT_ROOT, 'models', 'data.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = None
intents = None
all_words = None
tags = None

def _load_chatbot():
    global model, intents, all_words, tags
    if model is not None:
        return True
    if not os.path.isfile(INTENTS_PATH) or not os.path.isfile(DATA_PATH):
        return False
    try:
        with open(INTENTS_PATH, 'r') as f:
            intents = json.load(f)
        data = torch.load(DATA_PATH, map_location=device, weights_only=False)
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data['all_words']
        tags = data['tags']
        model_state = data["model_state"]
        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()
        return True
    except Exception as e:
        print(f"Chatbot load error: {e}")
        return False

bot_name = "Sam"

def get_response(msg):
    if not _load_chatbot():
        return "Chatbot is not ready. Please run: python -m scripts.chatbot_train (from project root)."
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    if not _load_chatbot():
        print("Could not load chatbot. Run chatbot_train.py first and ensure config/intents.json and models/data.pth exist.")
        exit(1)
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        print(get_response(sentence))
