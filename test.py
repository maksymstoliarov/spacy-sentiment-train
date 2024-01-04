print("Loading spacy...")
import spacy

print("Loading model...")
text_nlp = spacy.load("models/sentiment_model")

# ask for input while key interrupt is not pressed
while True:
    try:
        print("Enter text:")
        input_text = input()
        doc = text_nlp(input_text)
        print(doc.cats)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        break
