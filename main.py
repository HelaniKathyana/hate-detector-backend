import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle  # For loading tokenizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import chardet

# --- Configuration and Constants ---
MODEL_DIR = "models"  # Directory where models are saved.  Create this directory.
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pickle")  # Assuming a tokenizer was saved during training
MAX_SEQUENCE_LENGTH = 50  # Value used during training
MODEL_NAMES = ['cnn_bilstm_model.h5', 'lstm_model.h5', 'cnn_model.h5', 'bilstm_model.h5']
CLASS_NAMES = ["Non-Offensive", "Offensive"]  # Define class names

app = Flask(__name__)
CORS(app)


# --- Helper Functions (from your notebook, adapted for the API) ---

def load_stopwords(file_path):
    try:
        with open(file_path, 'r', encoding='utf-16') as f:  # Use utf-8, handle exceptions
            return set(f.read().splitlines())
    except FileNotFoundError:
        print(f"Error: Stopwords file not found at {file_path}")
        return set()  # Return an empty set if the file is not found
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        return set()


def load_suffixes(file_path):
    try:
        with open(file_path, 'r', encoding='utf-16') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        print(f"Error: Suffixes file not found at {file_path}")
        return set()
    except Exception as e:
        print(f"Error loading suffixes: {e}")
        return set()


def load_word_variations(file_path):
    word_variations = {}
    try:
        df_variations = pd.read_csv(file_path)
        for _, row in df_variations.iterrows():
            word = row['word']
            variations = row['Spelling variations'].split(' | ')
            word_variations[word] = set(variations)
    except FileNotFoundError:
        print(f"Error: Word variations file not found at {file_path}")
    except Exception as e:
        print(f"Error loading word variations: {e}")
    return word_variations


def load_emoji_weights(filepath):
    try:
        df_emoji = pd.read_csv(filepath)
        emoji_weights = dict(zip(df_emoji['Emoji'], df_emoji['weight']))
        return emoji_weights
    except FileNotFoundError:
        print(f"Error: Emoji weights file not found at {filepath}")
        return {}
    except Exception as e:
        print(f"Error loading emoji weights: {e}")
        return {}


def remove_suffixes(word, suffixes):
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix):
            root_word = word[:-len(suffix)].strip()
            sinhala_chars = re.findall(r'[\u0D9A-\u0DC6]', root_word)
            if len(sinhala_chars) >= 2:
                return root_word
            else:
                return word
    return word


def standardize_word(word, word_variations):
    for base_word, variations in word_variations.items():
        if word in variations:
            return base_word
    return word


def tokenize_with_bigrams(text):
    """Tokenizes text into unigrams and bigrams, preserving emoji tokens."""

    # Split the text, preserving the emoji tokens:
    parts = re.split(r'(__EMOJI_WEIGHT_[0-9_]+__|\s+)', text)  # Important
    # Remove empty strings and whitespace-only strings from the split result
    parts = [part for part in parts if part.strip()]

    unigrams = []
    for part in parts:
        if part.startswith('__EMOJI_WEIGHT_'):
            unigrams.append(part)  # Keep emoji tokens whole
        else:
            unigrams.extend(word_tokenize(part))  # Tokenize other parts with nltk

    bigrams = list(ngrams(unigrams, 2))
    bigram_strings = [" ".join(bigram) for bigram in bigrams]
    return unigrams + bigram_strings


def preprocess_text(text, emoji_weights, stop_words_custom, suffixes, word_variations, high_weight_threshold=7.0):
    processed_text = text
    has_high_weight_emoji = 0

    # --- Emoji Replacement (FIRST) ---
    for emoji, weight in emoji_weights.items():
        if emoji in processed_text:
            weight_str = str(weight).replace('.', '_')
            replacement_token = f"__EMOJI_WEIGHT_{weight_str}__"
            processed_text = processed_text.replace(emoji, f" {replacement_token} ")  # Replace emojis
            if weight >= high_weight_threshold:
                has_high_weight_emoji = 1
    emoji_feature = has_high_weight_emoji

    processed_text = re.sub(r'[^A-Za-z0-9_\u0D80-\u0DFF]+', ' ', processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    processed_text = processed_text.lower()

    words = processed_text.split();
    processed_words = [standardize_word(word, word_variations) for word in words if word not in stop_words_custom]

    # Remove stopwords and apply suffix removal
    processed_words = [remove_suffixes(word, suffixes) for word in processed_words if word not in stop_words_custom]

    processed_text = ' '.join(processed_words)

    # --- Tokenization (with Bigrams and Emoji Preservation) ---
    words = tokenize_with_bigrams(processed_text)
    processed_text = ' '.join(words)

    return processed_text, emoji_feature


# --- Model Loading ---
def load_models():
    """Loads all models from the MODEL_DIR."""
    models = {}
    for model_name in MODEL_NAMES:
        model_path = os.path.join(MODEL_DIR, model_name)
        try:
            models[model_name] = load_model(model_path)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            #   Optionally:  raise  # Re-raise the exception to stop the app if a model fails to load
    return models


# Load models and tokenizer *once* when the app starts
models = load_models()

# Load resources
stop_words_custom = load_stopwords('StopWords_425.txt')
suffixes = load_suffixes('Suffixes-413.txt')
word_variations = load_word_variations('word_variations.csv')
emoji_weights = load_emoji_weights("emoji_weights.csv")

try:
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully.")
except FileNotFoundError:
    print(f"Error: Tokenizer file not found at {TOKENIZER_PATH}")
    tokenizer = None  # Set to None to handle the error later
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None


# --- Prediction Function ---
def predict_single_model(model, preprocessed_text, tokenizer):
    """Predicts using a single model."""
    if tokenizer is None:  # Handle missing tokenizer
        return {"error": "Tokenizer not loaded.  Cannot make predictions.", "score": None, "predictedClass": None}
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    score = model.predict(padded_sequence, verbose=0)[0][0]  # Get the raw score
    predicted_class_index = int(round(score))  # Round to 0 or 1
    predicted_class = CLASS_NAMES[predicted_class_index]  # Get class name

    return {"score": float(score), "predictedClass": predicted_class}


# --- API Endpoint ---
@app.route('/predict-hate', methods=['POST'])
def predict_hate():
    """Handles prediction requests."""
    try:
        data = request.get_json()
        text = data['text']
    except (KeyError, TypeError):
        return jsonify({"error": "Invalid request format.  Send JSON with a 'text' field."}), 400

    if not text:  # Check for empty input
        return jsonify({"error": "Input text cannot be empty."}), 400

    # Preprocess the input text
    preprocessed_text, _ = preprocess_text(text, emoji_weights, stop_words_custom, suffixes, word_variations)

    results = []
    for model_name, model in models.items():
        prediction = predict_single_model(model, preprocessed_text, tokenizer)
        # Use the base name of the model file (without extension) for clarity
        results.append({
            "modelName": os.path.splitext(model_name)[0],
            "score": prediction["score"],
            "predictedClass": prediction["predictedClass"]
        })

    return jsonify(results)


if __name__ == '__main__':
    # Create the models directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    app.run(debug=True, port=3000)  # Use debug=True for development