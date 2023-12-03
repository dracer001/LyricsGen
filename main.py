from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import StreamingResponse
import io
import os
import shutil
import speech_recognition as sr
from gtts import gTTS

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

app = FastAPI()

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

@app.post("/upload_audio/")
async def upload_audio_file(file: UploadFile = File(...)):
    try:
        # Create a temporary directory to save the uploaded file
        os.makedirs("temp", exist_ok=True)
        file_location = f"temp/{file.filename}"
        
        # Save the uploaded audio file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe the audio to text
        transcription = transcribe_audio(file_location)

        new_lyrics = generateLyrics(transcription, 50)
        # Optionally, remove the temporary uploaded file
       
        speech = gTTS(text=new_lyrics, lang='en', slow=False)
# # Saving the converted audio in a file (you can also use .mp3 instead of .wav)
        speech.save("output.wav")
        # Load the audio file content
        # with open('output.wav', 'rb') as audio_file:
            # audio_content = audio_file.read()
        os.remove(file_location)
        # return StreamingResponse(io.BytesIO(audio_content), media_type="audio/wav")
        return {"filename": file.filename, "transcription": new_lyrics}
    except Exception as e:
        return {"error": str(e)}




# # Load the audio file content
# with open('output.wav', 'rb') as audio_file:
#     audio_content = audio_file.read()

# @app.get("/get_audio")
# async def get_audio():
#     # Return the audio file as a streaming response
#     return StreamingResponse(io.BytesIO(audio_content), media_type="audio/wav")



def generateLyrics(user_input, next_words):
    # Load the saved model
    model = load_model('ML/models/adele_lyricsM.h5')

    # Load lyrics dataset from file for tokenizer setup
    with open('ML/datasets/adele.txt', 'r', encoding='utf-8') as file:
        lyrics_text = file.read()

    # Preprocess the text for tokenizer setup
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([lyrics_text])
    total_words = len(tokenizer.word_index) + 1
    sequence_length = 50

# def generate_lyrics(user_input, next_words, model, max_sequence_len):
    user_sequences = tokenizer.texts_to_sequences([user_input])[0]
    user_sequences = np.array([user_sequences])
    seed_sequence = pad_sequences(user_sequences, maxlen=sequence_length, padding='pre')

    for _ in range(next_words):
        predicted = np.argmax(model.predict(seed_sequence), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        user_input += " " + output_word
        seed_sequence = pad_sequences([tokenizer.texts_to_sequences([user_input])[0]], maxlen=sequence_length, padding='pre')
    return user_input


# # Passing the text and language to the engine to convert
# speech = gTTS(text=text, lang='en', slow=False)

# # Saving the converted audio in a file (you can also use .mp3 instead of .wav)
# speech.save("output.wav")

# # Playing the converted file
# os.system("start output.wav")  # This command will play the output.wav file (for Windows)


# from fastapi import FastAPI, Response
# from gtts import gTTS
# import io

# app = FastAPI()

# @app.get("/text_to_audio")
# async def text_to_audio(response: Response):
#     # Lyrics text to convert to speech
#     lyrics_text = "Replace this text with your lyrics or load it from a file"

#     # Convert text to speech using gTTS
#     tts = gTTS(text=lyrics_text, lang='en')
    
#     # Create an in-memory file-like object to store the audio content
#     in_memory_file = io.BytesIO()
#     tts.write_to_fp(in_memory_file)
#     in_memory_file.seek(0)  # Reset file position to the beginning
    
#     # Set response headers and return the audio file as a streaming response
#     response.headers["Content-Disposition"] = "attachment; filename=output.mp3"
#     response.headers["Content-Type"] = "audio/mpeg"
#     return in_memory_file
