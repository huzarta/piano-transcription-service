from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import piano_transcription_inference
import numpy as np
import torch
from midiutil import MIDIFile
import os
import tempfile
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the transcriptor at startup
model_path = os.path.join(os.path.dirname(__file__), 'piano_transcription_inference_v1.pth')
transcriptor = piano_transcription_inference.PianoTranscription(
    device='cpu',
    checkpoint_path=model_path
)

@app.post("/transcribe")
async def transcribe_audio(
    audio_file_id: str,
    supabase_url: str,
    supabase_key: str
):
    try:
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download audio file from Supabase
            audio_path = os.path.join(temp_dir, f"{audio_file_id}.wav")
            midi_path = os.path.join(temp_dir, f"{audio_file_id}.mid")
            
            # Download the file from Supabase storage
            storage_url = f"{supabase_url}/storage/v1/object/public/audio-uploads/{audio_file_id}"
            async with httpx.AsyncClient() as client:
                response = await client.get(storage_url)
                if response.status_code != 200:
                    raise HTTPException(status_code=404, detail="Audio file not found")
                
                with open(audio_path, "wb") as f:
                    f.write(response.content)

            # Transcribe audio to MIDI
            transcribed_dict = transcriptor.transcribe(audio_path)
            
            # Create MIDI file
            midi_obj = MIDIFile(1)
            midi_obj.addTempo(0, 0, 120)
            
            # Add notes to MIDI file
            for note in transcribed_dict['est_note_events']:
                start_time, duration, pitch, velocity = note
                midi_obj.addNote(0, 0, pitch, start_time, duration, velocity)
            
            # Save MIDI file
            with open(midi_path, "wb") as midi_file:
                midi_obj.writeFile(midi_file)
            
            # Upload MIDI file to Supabase
            headers = {
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "audio/midi"
            }
            
            with open(midi_path, "rb") as f:
                upload_url = f"{supabase_url}/storage/v1/object/midi-files/{audio_file_id}.mid"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        upload_url,
                        content=f.read(),
                        headers=headers
                    )
                    if response.status_code != 200:
                        raise HTTPException(status_code=500, detail="Failed to upload MIDI file")

            # Update transcription status
            headers = {
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            }
            
            update_url = f"{supabase_url}/rest/v1/transcriptions?input_file=eq.{audio_file_id}"
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    update_url,
                    json={
                        "status": "completed",
                        "output_file": f"{audio_file_id}.mid"
                    },
                    headers=headers
                )
                if response.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to update transcription status")

            return {"status": "success", "midi_file": f"{audio_file_id}.mid"}

    except Exception as e:
        # Update transcription status to error
        headers = {
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        error_message = str(e)
        update_url = f"{supabase_url}/rest/v1/transcriptions?input_file=eq.{audio_file_id}"
        async with httpx.AsyncClient() as client:
            await client.patch(
                update_url,
                json={
                    "status": "error",
                    "error_message": error_message
                },
                headers=headers
            )
        
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
