import torch
import librosa
import numpy as np
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor

# Use a more reliable emotion recognition model
model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# Initialize the pipeline
try:
    emotion_classifier = pipeline(
        "audio-classification", 
        model=model_id,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Audio emotion recognition model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    print(f"Error loading emotion model: {e}")
    # Fallback to a simpler model if the first one fails
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        emotion_classifier = pipeline(
            "audio-classification", 
            model=model, 
            feature_extractor=feature_extractor,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Fallback audio emotion model loaded successfully")
    except Exception as e:
        print(f"Fallback model also failed: {e}")
        # Define a dummy function to prevent application crash
        emotion_classifier = None

def analyze_audio_sentiment(audio_data, sample_rate=16000):
    """
    Analyze the sentiment/emotion in an audio clip.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        
    Returns:
        dict: Dictionary with detected emotion and confidence
    """
    try:
        if emotion_classifier is None:
            return {"emotion": "neutral", "score": 1.0, "error": "Model not loaded"}
            
        # Ensure audio is at the correct sample rate
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # Run emotion classification
        result = emotion_classifier({"array": audio_data, "sampling_rate": 16000})
        
        # Get top emotion
        top_emotion = result
        
        # Map emotion to sentiment categories
        emotion_mapping = {
            "angry": "anger",
            "disgust": "disgust",
            "fear": "fear",
            "happy": "joy",
            "neutral": "neutral",
            "sad": "sadness",
            "ps": "surprise"  # "ps" means "pleasant surprise" in this model
        }
        
        detected_emotion = top_emotion["label"].lower()
        mapped_emotion = emotion_mapping.get(detected_emotion, detected_emotion)
        
        return {
            "emotion": mapped_emotion,
            "score": float(top_emotion["score"]),
            "all_emotions": [{
                "emotion": emotion_mapping.get(item["label"].lower(), item["label"].lower()),
                "score": float(item["score"])
            } for item in result]
        }
    except Exception as e:
        print(f"Error in audio sentiment analysis: {e}")
        return {"emotion": "neutral", "score": 1.0, "error": str(e)}