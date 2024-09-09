from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model('model.pkl')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Text preprocessing function (adjust this based on your preprocessing steps)
def preprocess_text(text):
    # Use the same preprocessing steps used during training
    # E.g., normalization, tokenization, etc.
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = np.array(sequences)  # Modify based on how your model expects input
    return padded_sequences

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling sentence correction
@app.route('/correct', methods=['POST'])
def correct_sentence():
    if request.method == 'POST':
        input_sentence = request.form['sentence']
        
        # Preprocess the input sentence
        preprocessed_sentence = preprocess_text(input_sentence)
        
        # Predict the corrected sentence
        prediction = model.predict(preprocessed_sentence)
        
        # Convert prediction back to text
        # This might require reversing the tokenization process
        corrected_sentence = tokenizer.sequences_to_texts(np.argmax(prediction, axis=-1))
        
        return jsonify({'corrected_sentence': corrected_sentence[0]})

if __name__ == '__main__':
    app.run(debug=True)
