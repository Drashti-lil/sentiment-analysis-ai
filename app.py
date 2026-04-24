import streamlit as st
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# -----------------------------
# Load model
# -----------------------------
model = BertForSequenceClassification.from_pretrained("bert_model")
tokenizer = BertTokenizer.from_pretrained("bert_model")
model.eval()

labels = ["negative", "neutral", "positive"]

# -----------------------------
# Dataset path
# -----------------------------
DATASET_PATH = "final_improved_dataset.csv"

# -----------------------------
# Save function
# -----------------------------
def save_new_text(text, label):
    df = pd.read_csv(DATASET_PATH)

    # Normalize for duplicate check
    text_clean = text.strip().lower()

    if text_clean not in df['text'].str.strip().str.lower().values:
        new_row = pd.DataFrame([[text, label]], columns=["text", "label"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATASET_PATH, index=False)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="BERT Sentiment Analyzer", layout="wide")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00f5d4;
}
.subtitle {
    text-align: center;
    color: #aaaaaa;
    margin-bottom: 30px;
}
.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    margin-top: 20px;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🤖 AI Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by BERT • Smart Emotion Detection</div>', unsafe_allow_html=True)

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    user_input = st.text_area("✍️ Enter your text:", height=150)
    analyze = st.button("🔍 Analyze Sentiment")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💡 Tips")
    st.write("• Try full sentences")
    st.write("• Works best with real reviews")
    st.write("• Handles 'not good', 'excellent', etc.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if analyze:

    if user_input.strip() == "":
        st.warning("⚠️ Please enter text!")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()
        sentiment = labels[pred]

        # -----------------------------
        # Result Display
        # -----------------------------
        if sentiment == "positive":
            st.markdown(
                f'<div class="result-box" style="background:linear-gradient(90deg,#00c853,#64dd17);">😊 POSITIVE</div>',
                unsafe_allow_html=True
            )

        elif sentiment == "negative":
            st.markdown(
                f'<div class="result-box" style="background:linear-gradient(90deg,#d50000,#ff1744);">😡 NEGATIVE</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f'<div class="result-box" style="background:linear-gradient(90deg,#616161,#9e9e9e);">😐 NEUTRAL</div>',
                unsafe_allow_html=True
            )

        # -----------------------------
        # Confidence
        # -----------------------------
        st.subheader("📊 Confidence Score")
        st.progress(float(confidence))
        st.write(f"Confidence: **{confidence:.2f}**")

        # -----------------------------
        # Chart
        # -----------------------------
        st.subheader("📈 Sentiment Breakdown")

        prob_dict = {
            "Negative": probs[0][0].item(),
            "Neutral": probs[0][1].item(),
            "Positive": probs[0][2].item()
        }

        st.bar_chart(prob_dict)

        # -----------------------------
        # 🔥 AUTO SAVE TO DATASET
        # -----------------------------
        save_new_text(user_input, sentiment)
        st.success("✅ New text saved to dataset (if not already present)!")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr style="margin-top:50px;">
<p style="text-align:center;color:gray;">
Made with ❤️ using BERT + Streamlit
</p>
""", unsafe_allow_html=True)