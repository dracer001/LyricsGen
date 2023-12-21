from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import StreamingResponse
import requests
import io
import os
import shutil
import speech_recognition as sr
from gtts import gTTS

app = FastAPI()

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text


@app.get("/")
def index():
    return {"message": "connection successful"}

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

        # speech = gTTS(text=new_lyrics, lang='en', slow=False)
# # Saving the converted audio in a file (you can also use .mp3 instead of .wav)
        # speech.save("output.wav")
        # Load the audio file content
        # with open('output.wav', 'rb') as audio_file:
            # audio_content = audio_file.read()
        # os.remove(file_location)
        # return StreamingResponse(io.BytesIO(audio_content), media_type="audio/wav")

        print(tanscription)
        receiver_url = "https://8c3e-34-81-77-78.ngrok-free.app/gen_lyrics"  # Replace with your receiver's URL
        data_to_send = transcription
        response = requests.post(receiver_url, json={"string_data": data_to_send})
        return {"filename": file.filename, "transcription": response}
    except Exception as e:
        return {"error": str(e)}
