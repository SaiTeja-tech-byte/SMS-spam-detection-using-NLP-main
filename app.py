import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

try:
    tk = pickle.load(open("vectorizer.pkl", 'rb'))
    model = pickle.load(open("model.pkl", 'rb'))
except FileNotFoundError:
    st.error("Model or Vectorizer files are missing. Please make sure 'vectorizer.pkl' and 'model.pkl' are available.")
    st.stop()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# Page title and header
st.markdown("""
    <style>
        .title {
            text-align: center;
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            color: white;
            padding: 15px 0;
            border-radius: 8px;
        }
        .subtitle {
            text-align: center;
            font-style: italic;
            color: #ff7e5f;
        }
        .main-container {
            background-color: #fdfdfd;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }
    </style>
    <div class="title">
        <h1>SMS Spam Detection Model</h1>
    </div>
    <p class="subtitle">An intelligent system to classify SMS messages as Spam or Not Spam</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style="background-color:#f7f7f7; padding: 10px; border-radius: 10px;">
        <h3 style="color:#0073e6; text-align:center;">How it works:</h3>
        <ul style="color:#333; font-size: 14px;">
            <li>Type the SMS in the input field.</li>
            <li>Click <strong>Predict</strong> to classify.</li>
            <li>Results will be displayed below.</li>
        </ul>
        <h4 style="color:#ff8c00; text-align:center;">Helpful Tips:</h4>
        <ul style="color:#333; font-size: 14px;">
            <li>Use clear, English messages.</li>
            <li>Avoid excessive abbreviations.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Input field
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
input_sms = st.text_area(
    "Enter the SMS text:", 
    height=150, 
    max_chars=300, 
    placeholder="Type your SMS here...", 
    key="sms_input"
)

if st.button('Predict', key='predict', help="Click to classify the SMS"):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        with st.spinner('Classifying your message...'):
            transformed_sms = transform_text(input_sms)
            vector_input = tk.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.success("🚨 Prediction: Spam")
                st.write("This message is classified as **Spam**. It may contain promotions or unwanted content.")
            else:
                st.success("✅ Prediction: Not Spam")
                st.write("This message is classified as **Not Spam**. It appears legitimate.")

        st.subheader("Processed Text:")
        st.code(transformed_sms, language="text")
st.markdown("</div>", unsafe_allow_html=True)

# Custom footer
st.markdown("""
    <style>
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 20px;
            color: #aaa;
        }
    </style>
    <div class="footer">
        Made with  by Sai Teja
    </div>
""", unsafe_allow_html=True)
