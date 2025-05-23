import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment
from gtts import gTTS
import speech_recognition as sr
import openai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create temp directory if it doesn't exist
os.makedirs("temp", exist_ok=True)

app = FastAPI()
recognizer = sr.Recognizer()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(audio: UploadFile = File(...)):
    webm_path = f"temp/{uuid.uuid4().hex}.webm"
    wav_path = webm_path.replace(".webm", ".wav")

    try:
        # Save uploaded audio
        with open(webm_path, "wb") as f:
            f.write(await audio.read())
        print(f"🎧 Received audio: {audio.filename}")

        # Convert to WAV
        audio_segment = AudioSegment.from_file(webm_path, format="webm")
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        audio_segment.export(wav_path, format="wav")

        # Transcribe
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        print(f"🗣️ Transcribed: {text}")

        # Get GPT reply
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": text},
            ]
        )
        reply = response.choices[0].message.content.strip()
        print(f"💬 GPT Reply: {reply}")

        # Text to speech
        mp3_filename = f"{uuid.uuid4().hex}.mp3"
        tts_path = os.path.join("temp", mp3_filename)
        gTTS(reply).save(tts_path)

        return {"reply": reply, "audio_url": f"/audio/{mp3_filename}"}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        for f in [webm_path, wav_path]:
            if os.path.exists(f):
                os.remove(f)

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = os.path.join("temp", filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg")
    return JSONResponse(status_code=404, content={"error": "File not found"})