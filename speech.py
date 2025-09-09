from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydub import AudioSegment
from pydub.utils import mediainfo
import whisper
import os
import uuid
from typing import List
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Directory to temporarily store uploaded and processed files
temp_dir = "temp_audio_files"
os.makedirs(temp_dir, exist_ok=True)

class TrimAudioRequest(BaseModel):
    file_id: str
    delete_texts: List[str]

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_id = str(uuid.uuid4())
        temp_audio_path = os.path.join(temp_dir, f"{file_id}.wav")
        with open(temp_audio_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Get the audio duration using Pydub
        info = mediainfo(temp_audio_path)
        duration = float(info['duration'])  # Duration in seconds

        return {
            "message": "Audio file uploaded successfully.",
            "file_id": file_id,
            "duration": duration
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/transcribe-audio/{file_id}")
async def transcribe_audio(file_id: str):
    try:
        # Load the audio file using Whisper
        temp_audio_path = os.path.join(temp_dir, f"{file_id}.wav")
        if not os.path.exists(temp_audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found.")

        model = whisper.load_model("base")
        result = model.transcribe(temp_audio_path, word_timestamps=True)

        transcription = result["segments"]
        return {"transcription": [{"text": seg["text"], "start": seg["start"], "end": seg["end"]} for seg in transcription]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

@app.post("/trim-audio")
async def trim_audio(request: TrimAudioRequest):
    try:
        file_id = request.file_id
        delete_texts = request.delete_texts

        print(f"Received file_id: {file_id}")
        print(f"Received delete_texts: {delete_texts}")

        # Validate non-empty delete_texts
        if not delete_texts:
            raise HTTPException(status_code=400, detail="No phrases provided to delete.")

        # Load the audio file
        temp_audio_path = os.path.join(temp_dir, f"{file_id}.wav")
        if not os.path.exists(temp_audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found.")

        audio = AudioSegment.from_file(temp_audio_path)

        # Transcribe audio again to get timestamps
        model = whisper.load_model("base")
        result = model.transcribe(temp_audio_path, word_timestamps=True)
        segments = result["segments"]

        # Gather exact text matches from transcription for validation
        all_transcribed_texts = [seg["text"] for seg in segments]

        # Validate that all delete_texts are exact matches in the transcription
        missing_texts = [text for text in delete_texts if text not in all_transcribed_texts]
        if missing_texts:
            raise HTTPException(status_code=400, detail={
                "message": "Some phrases are not found as exact matches in the transcription.",
                "missing_phrases": missing_texts
            })

        # Gather timestamps to trim
        timestamps_to_trim = []
        for text in delete_texts:
            for segment in segments:
                if text == segment["text"]:  # Exact match only
                    start_time = segment["start"] * 1000  # Convert to milliseconds
                    end_time = segment["end"] * 1000  # Convert to milliseconds
                    timestamps_to_trim.append((start_time, end_time))

        # Sort timestamps and trim audio
        timestamps_to_trim.sort()
        trimmed_audio = audio
        offset = 0

        for start_time, end_time in timestamps_to_trim:
            adjusted_start = start_time - offset
            adjusted_end = end_time - offset
            trimmed_audio = trimmed_audio[:adjusted_start] + trimmed_audio[adjusted_end:]
            offset += adjusted_end - adjusted_start

        # Save the trimmed audio
        trimmed_audio_path = os.path.join(temp_dir, f"{file_id}_trimmed.wav")
        trimmed_audio.export(trimmed_audio_path, format="wav")

        new_duration = len(trimmed_audio) / 1000  # Duration in seconds
        return {"message": "Audio trimmed successfully.", "new_duration": new_duration, "trimmed_audio_path": trimmed_audio_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error trimming audio: {str(e)}")

@app.get("/download-audio/{file_id}")
async def download_audio(file_id: str):
    try:
        trimmed_audio_path = os.path.join(temp_dir, f"{file_id}_trimmed.wav")
        if not os.path.exists(trimmed_audio_path):
            raise HTTPException(status_code=404, detail="Trimmed audio file not found.")

        return FileResponse(trimmed_audio_path, media_type="audio/wav", filename=f"{file_id}_trimmed.wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

# Mount the frontend directory to serve HTML files
app.mount("/", StaticFiles(directory="Static", html=True), name="static")
