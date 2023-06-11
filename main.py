import time
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from joblib import load
import pystray
from pystray import MenuItem as item
import pyperclip
import keyboard
from PIL import Image
import pyautogui
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "models"
tokenizer = AutoTokenizer.from_pretrained("Geotrend/distilbert-base-ru-cased")
model = AutoModelForSequenceClassification.from_pretrained("models")

nltk.download('stopwords')
nltk.download('punkt')

stopwords = set(stopwords.words('russian'))
punctuation = set(string.punctuation)


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords and token not in punctuation]
    return " ".join(tokens)


vectorizer = load("models/vectorizer.pkl")

def get_clipboard_text():
    return pyperclip.paste()

def create_tray_app():
    image = Image.open("icon.png")
    menu = (item('Quit', exit),)
    icon = pystray.Icon("AI", image, "AI", menu)
    return icon

if __name__ == '__main__':
    clf_NORMAL = load("models/model.pkl")
    icon_app = create_tray_app()

    def on_hotkey():
        text = pyperclip.paste()
        inputs = tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        predicted_new = F.softmax(outputs.logits, dim=1).tolist()

        # Предобработка текста
        processed_message = preprocess_text(text)
        new_message_tfidf = vectorizer.transform([processed_message])
        predicted = clf_NORMAL.predict_proba(new_message_tfidf)

        message = "Ваше сообщение похоже на:"
        if predicted[0][0]>0.4:
            message += "Норм:" + str(predicted[0][0]*100)[0:2] + "%  "
        if predicted[0][1]>0.4:
            message += "Оскб-е:" + str(predicted[0][1]*100)[0:2] + "%  "
        if predicted[0][2]>0.4:
            message += "Угрз:" + str(predicted[0][2]*100)[0:2] + "%  "
        if predicted[0][3]>0.4:
            message += "Неприст-ь:" + str(predicted[0][3] * 100)[0:2] + "%  "

        if predicted_new[0][0] > 0.2:
            message += "Норм (NEW):" + str(predicted_new[0][0] * 100)[0:2] + "%  "
        if predicted_new[0][1] > 0.2:
            message += "Оскб-е (NEW):" + str(predicted_new[0][1] * 100)[0:2] + "%  "
        if predicted_new[0][2] > 0.2:
            message += "Угрз (NEW):" + str(predicted_new[0][2] * 100)[0:2] + "%  "
        if predicted_new[0][3] > 0.2:
            message += "Неприст-ь (NEW):" + str(predicted_new[0][3] * 100)[0:2] + "%  "


        icon_app.notify("Информация о сообщении", message)

    key_combination = 'ctrl+alt+h'
    keyboard.add_hotkey(key_combination, on_hotkey)

    icon_app.run()

