import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load the MarianMT model and tokenizer
@st.cache_resource
def load_model(src_lang, tgt_lang):
    # Choose model based on source and target language
    if src_lang == "en" and tgt_lang == "hi":
        model_name = "Helsinki-NLP/opus-mt-en-hi"
    elif src_lang == "hi" and tgt_lang == "en":
        model_name = "Helsinki-NLP/opus-mt-hi-en"
    elif src_lang == "en" and tgt_lang == "ja":
        model_name = "Helsinki-NLP/opus-mt-en-jap"
    elif src_lang == "ja" and tgt_lang == "en":
        model_name = "Helsinki-NLP/opus-mt-ja-en"
    else:
        return None, None

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Function to perform translation
def translate(text, src_lang, tgt_lang):
    tokenizer, model = load_model(src_lang, tgt_lang)
    if model is None:
        return "Translation not supported for this language pair."

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Generate translation
    translated_tokens = model.generate(**inputs)

    # Decode the translated tokens
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_text

# Define available languages and their codes
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Japanese": "ja"
}

# Streamlit UI
st.title("Accurate Translation App (English ↔ Hindi ↔ Japanese)")

# Text input
text = st.text_area("Enter text to translate:")

# Language selection
source_lang = st.selectbox("Select source language", list(LANGUAGES.keys()))
target_lang = st.selectbox("Select target language", list(LANGUAGES.keys()))

# Button for translation
if st.button("Translate"):
    if text:
        # Get the language codes for the selected languages
        source_lang_code = LANGUAGES[source_lang]
        target_lang_code = LANGUAGES[target_lang]

        # Perform translation
        translated_text = translate(text, source_lang_code, target_lang_code)

        # Display the result
        st.subheader("Translated Text:")
        st.write(translated_text)
    else:
        st.error("Please enter some text to translate!")
