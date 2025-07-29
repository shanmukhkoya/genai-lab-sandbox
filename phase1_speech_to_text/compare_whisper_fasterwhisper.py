from faster_whisper import WhisperModel
import whisper

def transcribe_with_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def transcribe_with_faster_whisper(audio_path):
    model = WhisperModel("base")
    segments, info = model.transcribe(audio_path)
    return " ".join([seg.text for seg in segments])

if __name__ == "__main__":
    audio_path = "../sample_audio/test.wav"
    print("🔊 Whisper:", transcribe_with_whisper(audio_path))
    print("⚡ Faster-Whisper:", transcribe_with_faster_whisper(audio_path))
