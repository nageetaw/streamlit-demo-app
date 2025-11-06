# Imports
import streamlit as st
from transformers import pipeline

# Load model once
@st.cache_resource
def load_model():
    return pipeline("text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True)

pipe = load_model()

st.set_page_config(page_title="Emotion Classifier", page_icon="💬", layout="centered")

st.markdown("<h2 style='text-align:center;'>💬 Emotion Classifier</h2>", unsafe_allow_html=True)
st.write("Type text below and click **Classify** to detect emotional tone.")

user_input = st.text_area(
    "Your message:",
    placeholder="Example: I am feeling amazing today!",
    height=120
)

# Emoji and Color map with emotions
style_map = {
    "anger": ("😡", "#D50000"),
    "disgust": ("🤢", "#795548"),
    "fear": ("😨", "#6A1B9A"),
    "joy": ("😄", "#00C853"),
    "neutral": ("😐", "#455A64"),
    "sadness": ("😢", "#1E88E5"),
    "surprise": ("😲", "#F9A825"),
}

if st.button("Classify", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter text first.")
    else:
        scores = pipe(user_input)[0]
        # Pick highest scoring emotion
        best = max(scores, key=lambda x: x["score"])
        emotion = best["label"]
        score = round(best["score"], 3)

        emoji, color = style_map.get(emotion, ("❓", "#333"))

        st.markdown(
            f"""
            <div style="border-radius:10px;padding:16px;margin-top:20px;background:#f8f9fa;">
                <p style="font-size:20px;">Detected Emotion: <strong style="color:{color};">{emoji} {emotion}</strong></p>
                <p style="font-size:18px;">Confidence: {score}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("### Full Emotion Probabilities:")
        st.json({x["label"]: round(x["score"], 3) for x in scores})
