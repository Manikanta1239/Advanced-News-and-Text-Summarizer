import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import subprocess
import socket
from langdetect import detect

# Function to check if a port is open (i.e., if Streamlit is already running)
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('localhost', port)) == 0

# Automatically launch Streamlit app if not already running
def launch_streamlit_app():
    if not is_port_in_use(8501):
        subprocess.Popen(["streamlit", "run", __file__])

# Page configuration
st.set_page_config(layout="wide")

# Cache resources for efficiency
@st.cache_resource
def t5_summary(text, maxlength=None):
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-summarize-news')
    model = AutoModelForSeq2SeqLM.from_pretrained('mrm8488/t5-base-finetuned-summarize-news')
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = model.generate(inputs, max_length=maxlength, min_length=100, no_repeat_ngram_size=2, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@st.cache_resource
def bart_summary(text, maxlength=None):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=maxlength, min_length=60, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def detect_language(text):
    try:
        return detect(text)
    except:
        return "Unable to detect language"

def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    try:
        if file_type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = ''.join([page.extract_text() for page in pdf_reader.pages])
            return text
        elif file_type in ["image/png", "image/jpeg"]:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
            return text
        else:
            st.error("Unsupported file type.")
            return None
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def fetch_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        if not paragraphs:
            st.error("No text content found in the article.")
            return None, None
        title = soup.find('h1').get_text() if soup.find('h1') else "Unknown"
        article_text = ' '.join([para.get_text() for para in paragraphs])
        return title, article_text
    except requests.RequestException as e:
        st.error(f"Failed to retrieve the article: {e}")
        return None, None

def url_preview(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('h1').get_text() if soup.find('h1') else "No title found"
        snippet = ' '.join([para.get_text()[:300] for para in soup.find_all('p')[:1]])
        return title, snippet
    except Exception as e:
        st.error(f"Error fetching preview: {e}")
        return None, None

def summarize_article(url):
    title, article_text = fetch_article_text(url)
    if article_text:
        summary_result = t5_summary(article_text, maxlength=250) if summary_model == "T5 Model" else bart_summary(article_text, maxlength=250)
        keywords = extract_keywords(article_text)
        sentiment = sentiment_analysis(article_text)
        language = detect_language(article_text)
        return {
            "title": title,
            "content": article_text,
            "summary": summary_result,
            "bullet_summary": bullet_point(summary_result),
            "keywords": ", ".join(keywords),
            "sentiment": sentiment,
            "language": language
        }
    return None

def extract_keywords(text):
    vectorizer = CountVectorizer(stop_words='english', max_features=13)
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords

def bullet_point(text):
    sentences = text.split(". ")
    return "\n- " + "\n- ".join(sentences)

def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

def visualize_sentiment(sentiment):
    sentiment_data = {
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Probability': [sentiment.get('pos', 0), sentiment.get('neg', 0), sentiment.get('neu', 0)]
    }
    chart = alt.Chart(pd.DataFrame(sentiment_data)).mark_bar(width=15).encode(
        x='Sentiment',
        y='Probability',
        color='Sentiment'
    ).properties(width=500, height=300).configure_view(strokeOpacity=0)
    st.altair_chart(chart, use_container_width=True)

def sentiment_report(sentiment):
    st.markdown("### Sentiment Analysis Report")
    st.markdown(f"- **Negative Sentiment**: {sentiment['neg']*100:.2f}%")
    st.markdown(f"- **Neutral Sentiment**: {sentiment['neu']*100:.2f}%")
    st.markdown(f"- **Positive Sentiment**: {sentiment['pos']*100:.2f}%")

def is_valid_file_size(uploaded_file, max_size_mb=2):
    try:
        file_content = uploaded_file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        uploaded_file.seek(0)  # Reset file pointer after reading
        return file_size_mb <= max_size_mb
    except Exception as e:
        st.error(f"Error in file size validation: {e}")
        return False

# Sidebar for user options
choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document", "Summarize News Article"])
summary_model = st.sidebar.selectbox("Select your Model", ["T5 Model", "BART Model"])
summary_length = st.sidebar.slider("Summary Length (words)", min_value=100, max_value=500, value=250)

# Main app
def main_app():
    st.title("Advanced News and Text Summarizer")
    st.write("This app analyzes the sentiment, keywords, and summarizes the text of an article")

    if choice == "Summarize Text":
        st.subheader("Summarize Text")
        input_text = st.text_area("Enter your text here")
        if input_text and st.button("Summarize Text"):
            with st.spinner('Generating summary...'):
                language = detect_language(input_text)
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Your Input Text**")
                    st.info(input_text)
                    st.markdown(f"**Detected Language**: {language}")
                    st.markdown("**Keywords**")
                    keywords = extract_keywords(input_text)
                    st.write(", ".join(keywords))
                with col2:
                    st.markdown(f"**Summary Result ({summary_model})**")
                    summary_result = t5_summary(input_text, maxlength=summary_length) if summary_model == "T5 Model" else bart_summary(input_text, maxlength=summary_length)
                    st.success(summary_result)
                    with st.expander("Bullet Points"):
                        st.write(bullet_point(summary_result))
                    st.markdown("**Sentiment Analysis**")
                    sentiment = sentiment_analysis(input_text)
                    visualize_sentiment(sentiment)
                    sentiment_report(sentiment)

    elif choice == "Summarize Document":
        st.subheader("Summarize Document")
        input_file = st.file_uploader("Upload your document here", type=['pdf', 'png', 'jpeg'])
        if input_file:
            if is_valid_file_size(input_file):
                if st.button("Summarize Document"):
                    with st.spinner('Processing document...'):
                        extracted_text = extract_text_from_file(input_file)
                        if extracted_text:
                            language = detect_language(extracted_text)
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.info("File uploaded successfully")
                                st.markdown(f"**Detected Language**: {language}")
                                st.markdown("**Extracted Text is Below:**")
                                st.info(extracted_text)
                                st.markdown("**Keywords**")
                                keywords = extract_keywords(extracted_text)
                                st.write(", ".join(keywords))
                            with col2:
                                st.markdown(f"**Summary Result ({summary_model})**")
                                summary_result = t5_summary(extracted_text, maxlength=summary_length) if summary_model == "T5 Model" else bart_summary(extracted_text, maxlength=summary_length)
                                st.success(summary_result)
                                with st.expander("Bullet Points"):
                                    st.write(bullet_point(summary_result))
                                st.markdown("**Sentiment Analysis**")
                                sentiment = sentiment_analysis(extracted_text)
                                visualize_sentiment(sentiment)
                                sentiment_report(sentiment)
            else:
                st.error("File size exceeds the 2MB limit.")

    elif choice == "Summarize News Article":
        st.subheader("Summarize News Article")
        url = st.text_input("Enter the URL of the news article")
        if url:
            title, snippet = url_preview(url)
            st.write(f"**Title:** {title}")
            st.write(f"**Snippet:** {snippet}")
        if url and st.button("Summarize News Article"):
            with st.spinner('Summarizing article...'):
                article_result = summarize_article(url)
                if article_result:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.info(f"**Article Title:** {article_result['title']}")
                        st.markdown(f"**Detected Language**: {article_result['language']}")
                        st.markdown(f"**Full Article Text:** {article_result['content'][:1450]}... [Read more]")
                        st.markdown("**Keywords**")
                        st.write(article_result["keywords"])
                    with col2:
                        st.markdown(f"**Summary Result ({summary_model})**")
                        st.success(article_result["summary"])
                        with st.expander("Bullet Points"):
                            st.write(article_result["bullet_summary"])
                        st.markdown("**Sentiment Analysis**")
                        sentiment = article_result["sentiment"]
                        visualize_sentiment(sentiment)
                        sentiment_report(sentiment)

if __name__ == '__main__':
    launch_streamlit_app()
    main_app()
