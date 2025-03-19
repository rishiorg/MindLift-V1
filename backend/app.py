from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import get_chatbot_response
from sentiment_analysis import analyze_sentiment
from audio_analysis import analyze_audio_sentiment
import base64
import numpy as np
import librosa
import io
import os

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    
    # Get chatbot response
    response = get_chatbot_response(message)
    
    # Analyze sentiment of user message
    sentiment_result = analyze_sentiment(message)
    
    return jsonify({
        "response": response,
        "sentiment": sentiment_result.get("sentiment", "neutral"),
        "confidence": sentiment_result.get("confidence", 0.5),
    })

@app.route("/api/audio-chat", methods=["POST"])
def audio_chat():
    try:
        data = request.json
        audio_base64 = data.get("audio", "")
        
        # Remove header if present (e.g., "data:audio/wav;base64,")
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",")
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Convert to numpy array
        audio_array, sample_rate = librosa.load(
            io.BytesIO(audio_bytes), 
            sr=None  # Use file's own sample rate
        )
        
        # Analyze audio sentiment
        try:
            sentiment_result = analyze_audio_sentiment(audio_array, sample_rate)
            sentiment = sentiment_result.get("emotion", "neutral")
            confidence = sentiment_result.get("score", 0.5)
        except Exception as e:
            print(f"Audio sentiment analysis failed: {e}")
            sentiment = "neutral"
            confidence = 0.5
        
        # Use speech recognition to convert audio to text
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with io.BytesIO(audio_bytes) as audio_file:
                # Save to a temporary file for speech recognition
                temp_file = "temp_audio.wav"
                with open(temp_file, "wb") as f:
                    f.write(audio_bytes)
                
                # Use recognizer
                with sr.AudioFile(temp_file) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
                
                # Remove the temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        except Exception as e:
            print(f"Speech recognition failed: {e}")
            text = "I couldn't understand the audio. Could you please type your message?"
        
        # Get chatbot response based on transcribed text
        response = get_chatbot_response(text)
        
        return jsonify({
            "response": response,
            "transcription": text,
            "sentiment": sentiment,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({
            "response": "I'm having trouble processing your audio. Could you please type your message?",
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)