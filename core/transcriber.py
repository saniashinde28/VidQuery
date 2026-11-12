import whisper
import os

WHISPER_MODEL=os.getenv("WHISPER_MODEL","small")

model=None

def load_model():
    global model

    if model is None:
        print(f"Loading model..")
        model=whisper.load_model(WHISPER_MODEL)
        print("whisper model loaded successfully")
    return model
