import speech_recognition as sr
from textblob import TextBlob

# Initialize the recognizer
recognizer = sr.Recognizer()


def analyze_audio(audio):
    try:
        # Perform speech-to-text conversion
        text = recognizer.recognize_google(audio)
        print("You said: " + text)

        # Perform sentiment analysis
        sentiment = get_sentiment(text)

        return text, sentiment

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")


def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


def main():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    transcribed_text, sentiment = analyze_audio(audio)

    # Define a threshold for argument detection based on sentiment
    if sentiment < -0.2:
        print("This conversation seems to be argumentative.")
    else:
        print("This conversation does not appear to be argumentative.")


if __name__ == "__main__":
    main()
