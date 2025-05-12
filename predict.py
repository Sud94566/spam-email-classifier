import pickle
from preprocessing import preprocess

# Load saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def classify_message(msg):
    cleaned = preprocess(msg)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    return "Spam" if prediction == 1 else "Ham"

# Example
if __name__ == "__main__":
    msg = input("Enter a message to classify: ")
    result = classify_message(msg)
    print("Prediction:", result)
