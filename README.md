# 🧠 BERT Twitter Sentiment Analyzer

A deep learning-based sentiment analysis web application built using **BERT**, capable of classifying text as **Positive or Negative** with confidence scores and interactive visualization.

---

## 🚀 Live Demo

👉 https://bert-sentiment-analysis03.streamlit.app/

---

## 📌 Features

- 🔍 Real-time sentiment prediction
- 🧠 Fine-tuned BERT model on Twitter dataset
- 📊 Confidence scores + probability breakdown
- 📈 Interactive visualizations (charts & progress bars)
- 💡 Example inputs for quick testing
- 🌐 Fully deployed web application

---

## 🏗️ Tech Stack

- **Model:** BERT (Hugging Face Transformers)
- **Backend:** Python, PyTorch
- **Frontend:** Streamlit
- **Model Hosting:** Hugging Face Hub
- **Deployment:** Streamlit Cloud

---

## 🧠 Model Details

- Pretrained model: `bert-base-uncased`
- Fine-tuned on Twitter sentiment dataset
- Binary classification:
  - 0 → Negative
  - 1 → Positive

---

## 
---

## ⚙️ Installation & Setup

### 1️. Clone the repository

(bash)
git clone https://github.com/swayamkrr03/bert-sentiment-analysis.git
cd bert-sentiment-analysis

### 2. Install dependencies
(bash)pip install -r requirements.txt

### 3. Run the application
(bash)streamlit run app.py
