from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification
import numpy as np
from midiutil import MIDIFile
import os
import tempfile
import httpx
from dotenv import load_dotenv
import soundfile as sf

load_dotenv()

# Configure torch for minimal memory usage
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and processor from local cache
model_name = "modelo-base/piano-transcription-transformer"
model_cache_dir = "/app/models"

try:
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True, cache_dir=model_cache_dir)
    model = AutoModelForAudioClassification.from_pretrained(model_name, local_files_only=True, cache_dir=model_cache_dir)
except Exception as e:
    print(f"Error loading from cache, attempting to download: {e}")
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_cache_dir)
    model = AutoModelForAudioClassification.from_pretrained(model_name, cache_dir=model_cache_dir)

model.eval()

def audio_to_midi(audio_path, midi_path):
    """Convert audio to MIDI using the piano transcription model"""
    # Load and preprocess audio
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convert stereo to mono
    
    # Process audio in chunks to save memory
    chunk_size = 16000 * 10  # 10 seconds
    midi_obj = MIDIFile(1)
    midi_obj.addTempo(0, 0, 120)
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:min(i + chunk_size, len(audio))]
        
        # Prepare input
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.sigmoid(outputs.logits)
        
        # Convert predictions to MIDI notes
        # The model outputs probabilities for each piano key at each time step
        predictions = predictions.squeeze().numpy()
        
        for time_idx, frame in enumerate(predictions):
            for note_idx, prob in enumerate(frame):
                if prob > 0.5:  # Note activation threshold
                    start_time = (i + time_idx) / sr
                    duration = 0.1  # Default note duration
                    pitch = note_idx + 21  # MIDI note numbers start at 21 (A0)
                    velocity = int(min(127, prob * 127))  # Convert probability to velocity
                    midi_obj.addNote(0, 0, pitch, start_time, duration, velocity)
    
    # Save MIDI file
    with open(midi_path, "wb") as midi_file:
        midi_obj.writeFile(midi_file)

@app.post("/transcribe")
async def transcribe_audio(
    audio_file_id: str,
    supabase_url: str,
    supabase_key: str
):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, f"{audio_file_id}.wav")
            midi_path = os.path.join(temp_dir, f"{audio_file_id}.mid")
            
            # Download with streaming to reduce memory usage
            storage_url = f"{supabase_url}/storage/v1/object/public/audio-uploads/{audio_file_id}"
            async with httpx.AsyncClient() as client:
                async with client.stream('GET', storage_url) as response:
                    if response.status_code != 200:
                        raise HTTPException(status_code=404, detail="Audio file not found")
                    
                    with open(audio_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)

            # Force garbage collection before transcription
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Process the audio file
            audio_to_midi(audio_path, midi_path)
            
            # Upload MIDI file with streaming
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

            # Update status
            headers.update({
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            })
            
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
        # Update error status
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
