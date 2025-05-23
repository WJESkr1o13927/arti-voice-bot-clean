import uuid, os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import ffmpeg
from gtts import gTTS
import openai
import speech_recognition as sr

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
recognizer = sr.Recognizer()
os.makedirs("temp", exist_ok=True)

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
    with open(webm_path, "wb") as f:
        f.write(await audio.read())

    try:
        print(f"🎧 Received audio: {audio.filename}")
        def convert_webm_to_wav(webm_path, wav_path):
            try:
                (
                    ffmpeg
                    .input(webm_path)
                    .output(wav_path, ac=1, ar='16000')
                    .run(overwrite_output=True)
                )
            except ffmpeg.Error as e:
                print("❌ ffmpeg error:", e.stderr.decode())
                raise

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        print(f"🗣️ Transcribed: {text}")

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": text}
            ]
        )
        reply = response.choices[0].message.content.strip()
        print(f"💬 GPT Reply: {reply}")

        mp3_filename = f"{uuid.uuid4().hex}.mp3"
        tts = gTTS(reply)
        tts_path = os.path.join("temp", mp3_filename)
        tts.save(tts_path)

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
    filepath = os.path.join("temp", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/mpeg")
    return JSONResponse(status_code=404, content={"error": "File not found"})
