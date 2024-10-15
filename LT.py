import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    return tokenizer, model

tokenizer, model = load_model()

# Function to perform translation
def translate(text, source_lang, target_lang):
    # Set the source language token
    tokenizer.src_lang = source_lang

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # Get the correct target language ID for the beginning of the sequence
    target_lang_id = tokenizer.lang_code_to_id[target_lang]

    # Generate translation with the correct forced BOS token
    translated_tokens = model.generate(**inputs, forced_bos_token_id=target_lang_id)

    # Decode the translated tokens
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    return translated_text

# Define available languages and their codes
LANGUAGES = {
    "English": "en_XX",
    "Hindi": "hi_IN",
    "Gujarati": "gu_IN",
    "Japanese": "ja_XX",
    "Russian": "ru_RU"
}

# Streamlit UI
st.title("Translation App")

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
