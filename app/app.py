import gradio as gr
import pandas as pd
import nltk
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def cleaning_text(stop_words, tokenizer, title, text):
    text = title + text
    text.lower().replace('[^A-Za-z0-9\s]', '')

    text = " ".join([word for word in text.split() if word not in stop_words])
    df_text = pd.DataFrame({'new': text}, index=[0])

    sequence = tokenizer.texts_to_sequences(df_text['new'])
    print(len(sequence))
    padded_sequence = pad_sequences(sequence, maxlen=600, padding='post', truncating='post')

    return padded_sequence

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# load Tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# load model
model_lstm = load_model('model_lstm_1.0.h5')

def detecter(title, text, stop_words, tokenizer, model):
    seq = cleaning_text(stop_words, tokenizer, title=title, text=text)
    pred = model.predict(seq)[0, 0]
    if pred > 0.5:
        return "True!"
    elif pred <= 0.5:
        return "False!"
    else:
        return "Please try again."

title = gr.Textbox(label="Input the news title")
text = gr.Textbox(label="Input the news full content")

stop_words = stop_words
tokenizer = tokenizer
model = model_lstm

imf = gr.Interface(fn=lambda input1, input2: detecter(input1, input2, stop_words, tokenizer, model), inputs=[title, text], outputs="text")

if __name__ == "__main__":
    imf.launch()