# 🚀 Advanced News and Text Summarizer

## 🌟 Overview

This project is a **web-based application** developed with **Streamlit**, designed to summarize **text, documents, and news articles**. It provides advanced features such as **sentiment analysis, keyword extraction, and language detection**. The app leverages **pre-trained models like T5 and BART** for text summarization, with additional support for extracting text from **PDFs and images**.

---

## ✨ Features

- ✅ **Summarize Text** – Enter text directly to generate a concise summary.
- ✅ **Summarize Document** – Upload PDFs or image files to extract and summarize content.
- ✅ **Summarize News Article** – Input a URL to fetch, analyze, and summarize the article.
- ✅ **Sentiment Analysis** – Analyze the emotional tone of the input text.
- ✅ **Keyword Extraction** – Identify key terms from the text for quick insights.
- ✅ **Language Detection** – Automatically detect the language of the input text.
- ✅ **Visualization** – Display sentiment analysis results using interactive charts.

---

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

## 🎯 Usage

### 📌 **Text Summarization**
1. Select **Summarize Text** from the sidebar.
2. Enter your text in the provided field and click **Summarize Text**.
3. View the **summary, sentiment analysis, and extracted keywords**.

![Screenshot](https://github.com/user-attachments/assets/b513af47-ad2d-4a38-a6a2-f13bf719087c)
![Screenshot](https://github.com/user-attachments/assets/01a30429-3d24-4e36-bfe0-adde69516e28)

### 📌 **Document Summarization**
1. Select **Summarize Document** from the sidebar.
2. Upload a **PDF or image file** and click **Summarize Document**.
3. View the **extracted text, summary, sentiment analysis, and keywords**.

### 📌 **News Article Summarization**
1. Select **Summarize News Article** from the sidebar.
2. Enter the **URL** of the article and click **Summarize News Article**.
3. View the **article's title, snippet, summary, sentiment analysis, and keywords**.

---

## 📂 File Descriptions

- 📌 **app.py** – Main application script.
- 📌 **requirements.txt** – List of dependencies for the project.
- 📌 **README.md** – Project documentation.

---

## 🤖 Models Used

- 📝 **T5** – Fine-tuned for news summarization.
- 📝 **BART** – Pre-trained model for summarization.

---

## ⚠️ Limitations

- ⏳ File uploads are limited to **2 MB**.
- 📏 Summarization models have **maximum input length constraints**.

---

## 🔥 Future Enhancements

- 🚀 **Support for additional file types**.
- 🚀 **Integration with more summarization and sentiment models**.
- 🚀 **Improved language detection with multi-language summarization**.

---

## 🙌 Acknowledgments

- 🔗 [Hugging Face](https://huggingface.co/) – Pre-trained models.
- 🔗 [Streamlit](https://streamlit.io/) – Web app framework.
- 🔗 [PyTesseract](https://github.com/tesseract-ocr/tesseract) – OCR tool.

---

🎉 **Enjoy using the Advanced News and Text Summarizer!** 🚀
