
import json
from pathlib import Path

import joblib
import streamlit as st


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "smart_faq_pipeline.joblib"
RESPONSES_PATH = MODEL_DIR / "faq_responses.json"
METADATA_PATH = MODEL_DIR / "metadata.json"


# --------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart FAQ Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Smart FAQ Chatbot")
st.caption("NLP-based intent classification with predefined FAQ responses.")

st.write(
    "Ask questions about courses, admissions, registration, fees, "
    "technical support, contact details, or office hours."
)


# --------------------------------------------------
# 3. SAMPLE QUESTIONS
# --------------------------------------------------

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


# --------------------------------------------------
# 4. FALLBACK RESPONSE
# --------------------------------------------------

FALLBACK_ANSWER = (
    "I'm not confident enough to identify what you need. "
    "Please rephrase your question or contact the relevant office. "
    "I can help with courses, syllabi, admissions, registration, "
    "required documents, deadlines, fees, payments, receipts, "
    "login issues, technical support, contact information, "
    "and office hours."
)


# --------------------------------------------------
# 5. LOAD MODEL AND FAQ ARTIFACTS
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    # Check required files before loading
    required_files = [MODEL_PATH, RESPONSES_PATH]

    missing_files = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required chatbot files are missing:\n"
            + "\n".join(missing_files)
            + f"\nApplication directory: {ROOT}"
        )

    # Load trained pipeline
    model = joblib.load(MODEL_PATH)

    # Load predefined FAQ responses
    with open(RESPONSES_PATH, "r", encoding="utf-8") as file:
        responses = json.load(file)

    # Load metadata if available
    metadata = {}

    if METADATA_PATH.is_file():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

    return model, responses, metadata


# --------------------------------------------------
# 6. CLASSIFY USER QUESTION
# --------------------------------------------------

def classify(question, model, responses, threshold):
    if not isinstance(question, str) or not question.strip():
        return {
            "answer": "Please enter a question.",
            "intent": None,
            "confidence": 0.0,
            "fallback": True,
        }

    # Predict intent probabilities
    probabilities = model.predict_proba([question])[0]

    best_idx = int(probabilities.argmax())
    intent = str(model.classes_[best_idx])
    confidence = float(probabilities[best_idx])

    # Apply confidence threshold
    if confidence < threshold:
        return {
            "answer": FALLBACK_ANSWER,
            "intent": intent,
            "confidence": confidence,
            "fallback": True,
        }

    # Retrieve predefined response
    answer = responses.get(intent)

    if not isinstance(answer, str) or not answer.strip():
        answer = FALLBACK_ANSWER

        return {
            "answer": answer,
            "intent": intent,
            "confidence": confidence,
            "fallback": True,
        }

    return {
        "answer": answer,
        "intent": intent,
        "confidence": confidence,
        "fallback": False,
    }


# --------------------------------------------------
# 7. LOAD ARTIFACTS WITH ERROR HANDLING
# --------------------------------------------------

try:
    model, responses, metadata = load_artifacts()

except Exception as error:
    st.error("Unable to load the chatbot model or FAQ files.")

    st.code(
        f"Project directory: {ROOT}\n"
        f"Model path: {MODEL_PATH}\n"
        f"Responses path: {RESPONSES_PATH}"
    )

    st.exception(error)

    st.info(
        "Check that train_model.py has completed successfully "
        "and that the models directory contains the required files."
    )

    st.stop()


# --------------------------------------------------
# 8. SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.subheader("About this chatbot")

    st.write(
        "The model classifies your message into an intent "
        "and returns a predefined answer."
    )

    st.write(
        "It does not generate new facts or access live college records."
    )

    if metadata.get("selected_model"):
        st.write(
            f"Selected model: **{metadata['selected_model']}**"
        )

    threshold = st.slider(
        "Fallback confidence threshold",
        min_value=0.20,
        max_value=0.95,
        value=0.60,
        step=0.05,
    )

    st.caption(
        "Higher thresholds send more uncertain messages to fallback."
    )

    st.divider()

    st.caption("Model artifacts loaded successfully.")


# --------------------------------------------------
# 9. SAMPLE QUESTION BUTTONS
# --------------------------------------------------

with st.expander(
    "📌 Sample questions — click one to ask",
    expanded=True,
):
    for category, questions in SAMPLE_QUESTIONS.items():
        st.markdown(f"**{category}**")

        cols = st.columns(2)

        for i, question in enumerate(questions):
            with cols[i % 2]:
                if st.button(
                    question,
                    key=f"sample_{category}_{i}",
                    use_container_width=True,
                ):
                    st.session_state["pending_question"] = question
                    st.rerun()


# --------------------------------------------------
# 10. INITIALIZE CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to the Smart FAQ Chatbot. "
                "Choose a sample question above or type your "
                "own question below."
            ),
        }
    ]


# --------------------------------------------------
# 11. DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 12. USER INPUT AND CHAT RESPONSE
# --------------------------------------------------

typed_question = st.chat_input("Type your question...")

pending_question = st.session_state.pop(
    "pending_question",
    None,
)

question = pending_question or typed_question

if question:
    # Display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Classify the question
    try:
        result = classify(
            question,
            model,
            responses,
            threshold,
        )

        with st.chat_message("assistant"):
            if result["fallback"]:
                st.info(result["answer"])
            else:
                st.markdown(result["answer"])

            # Prediction details
            with st.expander("Prediction details"):
                st.write(
                    f"Predicted intent: `{result['intent']}`"
                )

                st.write(
                    f"Model confidence: "
                    f"`{result['confidence']:.1%}`"
                )

                st.caption(
                    "Confidence is a model score, not a guarantee "
                    "that the answer is correct."
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
            }
        )

    except Exception as error:
        st.error("An error occurred while processing your question.")
        st.exception(error)


# --------------------------------------------------
# 13. CLEAR CONVERSATION
# --------------------------------------------------

if st.button("🗑️ Clear conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()
