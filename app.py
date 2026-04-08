import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
from src.predict import predict

st.set_page_config(
    page_title="BERT Sentiment Analyzer",
    layout="centered"
)

# =========================
# HEADER
# =========================
st.title("🧠 BERT Twitter Sentiment Analyzer")
st.markdown("Analyze sentiment of any text using a fine-tuned BERT model.")

# =========================
# EXAMPLES
# =========================
st.subheader("💡 Try Examples")

col1, col2 = st.columns(2)

with col1:
    if st.button("👍 Positive Example"):
        st.session_state.text = "I absolutely love this product! Highly recommend it."

with col2:
    if st.button("👎 Negative Example"):
        st.session_state.text = "This is the worst experience I've ever had."

# =========================
# INPUT BOX
# =========================
text = st.text_area(
    "✍️ Enter your text here:",
    value=st.session_state.get("text", ""),
    height=120
)

# =========================
# ANALYZE BUTTON
# =========================
if st.button("🚀 Analyze Sentiment"):
    if text.strip():

        with st.spinner("Analyzing sentiment..."):
            label, confidence, probs = predict(text)

        st.divider()

        # =========================
        # RESULT SECTION
        # =========================
        st.subheader("📊 Result")

        confidence_percent = confidence * 100

        if label == "Positive":
            st.success(f"🟢 **Positive Sentiment**")
        else:
            st.error(f"🔴 **Negative Sentiment**")

        st.write(f"**Confidence:** {confidence_percent:.2f}%")

        # =========================
        # PROGRESS BARS (INSIGHT)
        # =========================
        st.subheader("🔍 Detailed Analysis")

        neg_prob = probs[0]
        pos_prob = probs[1]

        st.write("Negative Probability")
        st.progress(float(neg_prob))

        st.write("Positive Probability")
        st.progress(float(pos_prob))

        # =========================
        # INTERPRETATION (🔥 important)
        # =========================
        st.subheader("🧠 Interpretation")

        if confidence > 0.8:
            st.info("The model is very confident about this prediction.")
        elif confidence > 0.6:
            st.info("The model is moderately confident.")
        else:
            st.warning("The model is uncertain. Try rephrasing the text.")

# =========================
# FOOTER
# =========================
st.divider()
st.caption("Built with BERT + PyTorch + Streamlit 🚀")