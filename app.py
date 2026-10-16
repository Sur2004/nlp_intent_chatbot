import json
from pathlib import Path
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "smart_faq_pipeline.joblib"
RESPONSES_PATH = ROOT / "models" / "faq_responses.json"
METADATA_PATH = ROOT / "models" / "metadata.json"

st.set_page_config(page_title="Smart FAQ Chatbot", page_icon="💬", layout="centered")

SAMPLE_QUESTIONS = {
    "Courses & syllabus": [
        "What courses are offered?",
        "Where can I find the CSE syllabus?",
    ],
    "Admissions & registration": [
        "How do I apply for admission?",
        "What documents are needed for admission?",
        "When is the last date to apply?",
        "Where is the registration form?",
    ],
    "Fees & payments": [
        "What are the tuition fees?",
        "How can I pay my fees?",
        "How do I get my fee receipt?",
    ],
    "Technical support": [
        "I cannot log in to the student portal",
        "How do I reset my password?",
        "The website keeps showing an error",
    ],
    "General information": [
        "How can I contact the college?",
        "What are the office hours?",
        "Hello, can you help me?",
    ],
}

FALLBACK_ANSWER = (
    "I'm not confident enough to identify what you need. Please rephrase your question or "
    "contact the relevant office. I can help with courses, syllabi, admissions, registration, "
    "required documents, deadlines, fees, payments, receipts, login issues, technical support, "
    "contact information, and office hours."
)

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not RESPONSES_PATH.exists():
        return None, None, {}
    model = joblib.load(MODEL_PATH)
    with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
        responses = json.load(f)
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return model, responses, metadata

def classify(question, model, responses, threshold):
    if not isinstance(question, str) or not question.strip():
        return {"answer": "Please enter a question.", "intent": None, "confidence": 0.0, "fallback": True}
    probabilities = model.predict_proba([question])[0]
    best_idx = int(probabilities.argmax())
    intent = str(model.classes_[best_idx])
    confidence = float(probabilities[best_idx])
    if confidence < threshold:
        return {"answer": FALLBACK_ANSWER, "intent": intent, "confidence": confidence, "fallback": True}
    return {
        "answer": responses.get(intent, FALLBACK_ANSWER),
        "intent": intent,
        "confidence": confidence,
        "fallback": False
    }

model, responses, metadata = load_artifacts()

st.title("💬 Smart FAQ Chatbot")
st.caption("NLP-based intent classification with predefined FAQ responses.")
st.write(
    "Ask a question about courses, admissions, registration, fees, receipts, technical support, "
    "contact details, or office hours. Choose a sample question to see how it works."
)

if model is None:
    st.error("Trained model files are missing. Run `python train_model.py` from the project folder first.")
    st.stop()

with st.sidebar:
    st.subheader("About this chatbot")
    st.write("The model classifies your message into an intent and returns a predefined answer.")
    st.write("It does not generate new facts or access live college records.")
    if metadata.get("selected_model"):
        st.write(f"Selected model: **{metadata['selected_model']}**")
    threshold = st.slider("Fallback confidence threshold", 0.20, 0.95, 0.60, 0.05)
    st.caption("Higher thresholds send more uncertain messages to fallback.")

with st.expander("📌 Sample questions — click one to ask", expanded=True):
    for category, questions in SAMPLE_QUESTIONS.items():
        st.markdown(f"**{category}**")
        cols = st.columns(2)
        for i, question in enumerate(questions):
            with cols[i % 2]:
                if st.button(question, key=f"sample_{category}_{i}", use_container_width=True):
                    st.session_state["pending_question"] = question
                    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! 👋 Choose a sample question above or type your own question below."
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed_question = st.chat_input("Type your question...")
pending_question = st.session_state.pop("pending_question", None)
question = pending_question or typed_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    result = classify(question, model, responses, threshold)
    with st.chat_message("assistant"):
        if result["fallback"]:
            st.info(result["answer"])
        else:
            st.markdown(result["answer"])
        with st.expander("Prediction details"):
            st.write(f"Predicted intent: `{result['intent']}`")
            st.write(f"Model confidence: `{result['confidence']:.1%}`")
            st.caption("Confidence is a model score, not a guarantee that the answer is correct.")
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
