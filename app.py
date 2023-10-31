import speech_recognition as sr
from textblob import TextBlob
from pydub import AudioSegment
import io
import numpy as np

# Initialize the recognizer
recognizer = sr.Recognizer()

SENTIMENT_LIMIT = -0.2
LOUDNESS_LIMIT = -20

def analyze_audio(audio):
    try:
        # Perform speech-to-text conversion
        text = recognizer.recognize_google(audio)
        print("You said: " + text)

        # Perform sentiment analysis
        sentiment = get_sentiment(text)

        # Analyze pitch, loudness, and tonality
        loudness = analyze_audio_features(audio)

        return text, sentiment, loudness

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")


def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


def analyze_audio_features(audio):
    # Convert audio to PCM data
    pcm_data = audio.get_wav_data(convert_rate=44100, convert_width=2)

    # Calculate loudness (volume)
    audio_segment = AudioSegment.from_wav(io.BytesIO(pcm_data))
    loudness = audio_segment.dBFS

    return loudness


def main():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    transcribed_text, sentiment, loudness = analyze_audio(
        audio)

    print(loudness, sentiment)

    # Define a threshold for argument detection based on sentiment, pitch, loudness, and tonality
    if sentiment < SENTIMENT_LIMIT and loudness > LOUDNESS_LIMIT:
        print("This conversation seems to be argumentative.")
    else:
        print("This conversation does not appear to be argumentative.")


if __name__ == "__main__":
    main()
