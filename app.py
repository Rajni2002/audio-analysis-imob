import speech_recognition as sr
from textblob import TextBlob
from pydub import AudioSegment
import parselmouth
import numpy as np

# Initialize the recognizer
recognizer = sr.Recognizer()

def analyze_audio(audio):
    try:
        # Perform speech-to-text conversion
        text = recognizer.recognize_google(audio)
        print("You said: " + text)

        # Perform sentiment analysis
        sentiment = get_sentiment(text)

        # Analyze pitch, loudness, and tonality
        pitch, loudness, tonality = analyze_audio_features(audio)

        return text, sentiment, pitch, loudness, tonality

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

    # Create a Sound object from PCM data
    sound = parselmouth.Sound(pcm_data, sampling_frequency=44100)

    # Calculate pitch
    pitch = sound.to_pitch()
    mean_pitch = np.nanmean(pitch.selected_array['frequency'])

    # Calculate loudness (volume)
    audio_segment = AudioSegment.from_wav(io.BytesIO(pcm_data))
    loudness = audio_segment.dBFS

    # Calculate tonality
    harmonicity = sound.to_harmonicity()
    mean_harmonicity = np.nanmean(harmonicity.values)

    return mean_pitch, loudness, mean_harmonicity

def main():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    transcribed_text, sentiment, pitch, loudness, tonality = analyze_audio(audio)

    # Define a threshold for argument detection based on sentiment, pitch, loudness, and tonality
    if sentiment < -0.2 and pitch > 100 and loudness > -30 and tonality > 0.3:
        print("This conversation seems to be argumentative.")
    else:
        print("This conversation does not appear to be argumentative.")

if __name__ == "__main__":
    main()
