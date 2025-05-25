# backend/main.py
import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment
from gtts import gTTS
import speech_recognition as sr
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
os.makedirs("temp", exist_ok=True)

app = FastAPI()
recognizer = sr.Recognizer()

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
        audio_bytes = await audio.read()
        if len(audio_bytes) < 2048:
            return JSONResponse(status_code=400, content={"error": "Audio too short or not captured properly."})

        with open(webm_path, "wb") as f:
            f.write(audio_bytes)

        print(f"🎧 Received audio: {audio.filename} ({len(audio_bytes)} bytes)")

        try:
            audio_segment = AudioSegment.from_file(webm_path, format="webm")
            audio_segment.set_channels(1).set_frame_rate(16000).export(wav_path, format="wav")
        except Exception as e:
            print("❌ Pydub/FFmpeg error:", e)
            return JSONResponse(status_code=500, content={"error": "Could not process audio format. Try again or use a different browser."})

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        print(f"🗣️ Transcribed: {text}")

        chat_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": text}
            ]
        )
        reply = chat_response.choices[0].message.content.strip()
        print(f"💬 GPT Reply: {reply}")

        mp3_filename = f"{uuid.uuid4().hex}.mp3"
        mp3_path = os.path.join("temp", mp3_filename)
        gTTS(reply).save(mp3_path)

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