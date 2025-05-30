"""import os
import uuid
from fastapi import FastAPI, UploadFile, File, Request, Form
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
    allow_origins=["https://extraordinary-stroopwafel-a420e4.netlify.app", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_memory = {}

@app.post("/chat")
async def chat(request: Request, audio: UploadFile = File(...), lang: str = Form("en")):
    print(f"⛳ Received request: audio={audio.filename}, lang={lang}")
    session_id = request.client.host
    session_memory.setdefault(session_id, [{
        "role": "system",
        "content": (
            "You are a spiritual guide who shares life lessons inspired by the Mahabharata, Ramayana, and the works of saints like Kabir and Rahim. "
            "Respond to users in short, poetic, and emotionally supportive replies at first—just 1 to 3 lines. "
            "Only expand in detail if the user continues with deeper questions or stays on the same theme. "
            "For direct questions (like historical facts, names, or definitions), reply clearly and concisely first, then offer gentle insights if needed."
            "Use gentle wisdom, metaphors, and a reflective tone. Keep your early messages grounded, like a soft mantra offered under a banyan tree."
        )
    }])

    webm_path = f"temp/{uuid.uuid4().hex}.webm"
    wav_path = webm_path.replace(".webm", ".wav")

    try:
        audio_bytes = await audio.read()
        print(f"📦 Audio byte length: {len(audio_bytes)}")
        if len(audio_bytes) < 1024:
            print("❌ Audio too short or empty.")
            return JSONResponse(status_code=400, content={
                "error": "Your voice message was too short. Please speak clearly for at least a few seconds."
            })
            return JSONResponse(status_code=400, content={"error": "Audio too short or not captured properly."})

        with open(webm_path, "wb") as f:
            f.write(audio_bytes)
        print(f"🎧 Received audio: {audio.filename} ({len(audio_bytes)} bytes)")

        try:
            audio_segment = AudioSegment.from_file(webm_path, format="webm")
            audio_segment.set_channels(1).set_frame_rate(16000).export(wav_path, format="wav")
        except Exception as e:
            print("❌ Pydub/FFmpeg error:", e)
            return JSONResponse(status_code=500, content={"error": "Could not process audio format."})

        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                google_lang_code = "hi-IN" if lang == "hi" else "en-US"
                text = recognizer.recognize_google(audio_data, language=google_lang_code)
            print(f"🗣️ Transcribed: {text}")
        except sr.UnknownValueError:
            print("❌ Google Speech could not understand audio.")
            return JSONResponse(status_code=400, content={"error": "Could not understand audio. Please speak clearly."})
        except sr.RequestError as e:
            print(f"❌ Google Speech request failed: {e}")
            return JSONResponse(status_code=500, content={"error": "Speech recognition service failed."})

        session_memory[session_id].append({"role": "user", "content": text})

        chat_response = client.chat.completions.create(
            model="gpt-4",
            messages=session_memory[session_id]
        )
        reply = chat_response.choices[0].message.content.strip()
        print(f"💬 GPT Reply: {reply}")

        session_memory[session_id].append({"role": "assistant", "content": reply})

        mp3_filename = f"{uuid.uuid4().hex}.mp3"
        mp3_path = os.path.join("temp", mp3_filename)

        try:
            gTTS(reply, lang=lang).save(mp3_path)
        except Exception as e:
            print(f"❌ gTTS Error: {e}")
            return JSONResponse(status_code=500, content={"error": "Failed to generate audio reply."})

        return {"reply": reply, "audio_url": f"/audio/{mp3_filename}"}

    except Exception as e:
        print(f"❌ General Error: {str(e)}")
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
    return JSONResponse(status_code=404, content={"error": "File not found"})"""

import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, Request
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
    allow_origins=[
        "https://extraordinary-stroopwafel-a420e4.netlify.app",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_memory = {}

@app.post("/chat")
async def chat(
    request: Request,
    audio: UploadFile | None = File(default=None),
    text:  str       | None = Form(default=None),
    lang:  str           = Form("en"),
):
    session_id = request.client.host
    # Initialize session with system prompt if needed
    session_memory.setdefault(session_id, [{
        "role": "system",
        "content": (
            "You are a spiritual guide who shares life lessons inspired by the Mahabharata, "
            "Ramayana, and the works of saints like Kabir and Rahim. Respond in short, poetic, "
            "emotionally supportive replies at first—just 1 to 3 lines. Expand only if the user "
            "asks deeper questions. For direct questions, reply clearly then gently offer insight."
        )
    }])

    # If text was provided, skip audio processing entirely:
    if text is not None:
        user_content = text.strip()
    else:
        # require a real audio upload and transcribe it:
        if audio is None:
            return JSONResponse(
                status_code=422,
                content={"error": "No audio or text provided."}
            )

        webm_path = f"temp/{uuid.uuid4().hex}.webm"
        wav_path = webm_path.replace(".webm", ".wav")

        # Read & validate
        audio_bytes = await audio.read()
        if len(audio_bytes) < 1024:
            return JSONResponse(
                status_code=400,
                content={"error": "Audio too short—please speak for at least a few seconds."}
            )
        # Save raw
        with open(webm_path, "wb") as f:
            f.write(audio_bytes)

        # Convert to WAV
        try:
            seg = AudioSegment.from_file(webm_path, format="webm")
            seg.set_channels(1).set_frame_rate(16000).export(wav_path, format="wav")
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"error": "Could not process audio format."}
            )

        # Transcribe
        try:
            with sr.AudioFile(wav_path) as src:
                data = recognizer.record(src)
                code = "hi-IN" if lang == "hi" else "en-US"
                user_content = recognizer.recognize_google(data, language=code)
        except sr.UnknownValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not understand audio. Please speak clearly."}
            )
        except sr.RequestError:
            return JSONResponse(
                status_code=500,
                content={"error": "Speech recognition service failed."}
            )
        finally:
            # Clean up temp files
            for path in (webm_path, wav_path):
                if os.path.exists(path):
                    os.remove(path)

    # Append user message to session
    session_memory[session_id].append({"role": "user", "content": user_content})

    # Call
    chat_resp = client.chat.completions.create(
        model="gpt-4",
        messages=session_memory[session_id]
    )
    reply = chat_resp.choices[0].message.content.strip()

    # Append assistant message
    session_memory[session_id].append({"role": "assistant", "content": reply})

    # Generate TTS
    mp3_name = f"{uuid.uuid4().hex}.mp3"
    mp3_path = os.path.join("temp", mp3_name)
    try:
        gTTS(reply, lang=lang).save(mp3_path)
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to generate audio reply."}
        )

    # text and URL to serve the MP3
    return {
        "reply": reply,
        "audio_url": f"/audio/{mp3_name}"
    }

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = os.path.join("temp", filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg")
    return JSONResponse(status_code=404, content={"error": "File not found"})