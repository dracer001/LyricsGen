from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

# Load the saved model
loaded_model = load_model('ML/models/adele_lyricsM.h5')

# Load lyrics dataset from file for tokenizer setup
with open('ML/datasets/adele.txt', 'r', encoding='utf-8') as file:
    lyrics_text = file.read()

# Preprocess the text for tokenizer setup
tokenizer = Tokenizer()
tokenizer.fit_on_texts([lyrics_text])
total_words = len(tokenizer.word_index) + 1
sequence_length = 50

# Function to generate lyrics based on user input
def generate_lyrics(user_input, next_words, model, max_sequence_len):
    user_sequences = tokenizer.texts_to_sequences([user_input])[0]
    user_sequences = np.array([user_sequences])
    seed_sequence = pad_sequences(user_sequences, maxlen=max_sequence_len, padding='pre')

    for _ in range(next_words):
        predicted = np.argmax(model.predict(seed_sequence), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        user_input += " " + output_word
        seed_sequence = pad_sequences([tokenizer.texts_to_sequences([user_input])[0]], maxlen=max_sequence_len, padding='pre')
    return user_input

# User input for partial lyrics
user_input = "I don't think i can ever get enough of you, even though you probably left me for dead,"


# # Generate lyrics based on user input using the loaded model
# generated_lyrics = generate_lyrics(user_input, 50, loaded_model, sequence_length)
# print("Generated Lyrics based on user input:")
# print(generated_lyrics)
