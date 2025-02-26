from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from midiutil import MIDIFile
import os
import tempfile
import httpx
from dotenv import load_dotenv
import soundfile as sf
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict_and_save
from basic_pitch.inference import predict

load_dotenv()

app = FastAPI(
    title="Audio Transcription Service",
    description="Service for transcribing audio files to MIDI using Basic Pitch"
)

# Add CORS middleware with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you should list specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def audio_to_midi(audio_path, midi_path):
    """Convert audio to MIDI using Spotify's Basic Pitch"""
    try:
        print(f"Processing audio file: {audio_path}")
        print(f"Output MIDI path: {midi_path}")
        
        # Basic Pitch will handle the MIDI creation with only supported parameters
        predict_and_save(
            audio_path_list=[audio_path],
            output_directory=os.path.dirname(midi_path),
            save_midi=True,
            midi_tempo=120.0,
            minimum_frequency=None,
            maximum_frequency=None
        )
        
        # The output file will have "_basic_pitch" appended to it
        output_base = os.path.splitext(midi_path)[0]
        output_midi = f"{output_base}_basic_pitch.mid"
        
        print(f"Looking for output MIDI file at: {output_midi}")
        
        # Rename the file to match our expected output path
        if os.path.exists(output_midi):
            os.rename(output_midi, midi_path)
            print(f"Successfully renamed MIDI file to: {midi_path}")
        else:
            raise Exception(f"MIDI file was not created successfully at {output_midi}")
            
    except Exception as e:
        print(f"Error in audio_to_midi conversion: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {
        "status": "online",
        "service": "audio-transcription",
        "version": "1.0.0"
    }

@app.post("/transcribe")
async def transcribe_audio(
    audio_file_id: str,
    supabase_url: str,
    supabase_key: str
):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, audio_file_id)
            midi_path = os.path.join(temp_dir, f"{audio_file_id}.mid")
            
            print(f"Downloading audio file: {audio_file_id}")
            
            # Download with streaming to reduce memory usage
            storage_url = f"{supabase_url}/storage/v1/object/public/audio-uploads/{audio_file_id}"
            async with httpx.AsyncClient() as client:
                async with client.stream('GET', storage_url) as response:
                    if response.status_code != 200:
                        raise HTTPException(status_code=404, detail="Audio file not found")
                    
                    with open(audio_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
            
            print("Audio file downloaded successfully")
            
            # Process the audio file
            audio_to_midi(audio_path, midi_path)
            
            print("Audio conversion completed")
            
            # Upload MIDI file with streaming
            headers = {
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "audio/midi"
            }
            
            print("Uploading MIDI file")
            
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

            print("MIDI file uploaded successfully")

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

            print("Transcription status updated successfully")
            return {"status": "success", "midi_file": f"{audio_file_id}.mid"}

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
