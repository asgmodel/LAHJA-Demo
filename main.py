from fastapi import FastAPI
from fastapi.responses import FileResponse
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
import uuid

# تحميل الموديل والتوكن
model = VitsModel.from_pretrained("wasmdashai/vits-ar-sa-huba")
tokenizer = AutoTokenizer.from_pretrained("wasmdashai/vits-ar-sa-huba")

# إنشاء تطبيق FastAPI
app = FastAPI(title="Arabic TTS API", version="1.0")

@app.get("/")
def root():
    return {"message": "Arabic VITS TTS is running 🚀"}

@app.get("/tts")
def tts(text: str):
    # تحويل النص إلى input
    inputs = tokenizer(text, return_tensors="pt")

    # توليد الصوت
    with torch.no_grad():
        full_generation = model(**inputs)
    waveform = full_generation.waveform.cpu().numpy().reshape(-1)

    # حفظ الصوت كملف WAV مؤقت
    filename = f"output_{uuid.uuid4().hex}.wav"
    sf.write(filename, waveform, 22050)  # 22050 Hz معدل العينة للموديل

    # إرسال الملف كـ Response
    return FileResponse(filename, media_type="audio/wav", filename="speech.wav")
