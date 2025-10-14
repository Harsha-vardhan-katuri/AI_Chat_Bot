import streamlit as st
import nltk
from transformers import pipeline

# ------------------- NLTK Setup -------------------
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# ------------------- Load Chatbot Model -------------------
# Force CPU to avoid PyTorch meta tensor issue
chatbot = pipeline("text-generation", model="distilgpt2", device=-1)

# ------------------- Chatbot Logic -------------------
def healthcare_chatbot(user_input):
    user_input_lower = user_input.lower()

    if "appointment" in user_input_lower:
        return "Would you like to schedule an appointment with the doctor?"
    elif "fever" in user_input_lower:
        return "If the fever is below 102°F, take Paracetamol or Dolo 650. If it exceeds 102°F, consult a doctor immediately."
    elif "cough" in user_input_lower:
        return "For a mild dry cough, try honey, warm water, or Benadryl. If it persists more than a week, consult a doctor."
    elif "cold" in user_input_lower or "congestion" in user_input_lower:
        return "For a mild cold, take Cetirizine and try steam inhalation. If it persists, consult a doctor."
    elif "pain" in user_input_lower:
        return "For mild pain, you may take Ibuprofen after food. Persistent pain requires medical advice."
    else:
        response = chatbot(user_input, max_length=150, num_return_sequences=1)
        return response[0]['generated_text']

# ------------------- Streamlit App -------------------
def main():
    st.set_page_config(page_title="Healthcare Assistant Chatbot", layout="wide")

    # ------------------- Apple/Dora AI Style CSS -------------------
    st.markdown("""
    <style>
    /* Main container: light Apple-style background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(145deg, #f5f5f7, #e8e8eb);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: black;
        padding: 20px;
    }

    /* Chat bubbles */
    .user-bubble {
        background-color: #0a84ff;  /* iMessage blue */
        color: white;
        padding: 14px 18px;
        border-radius: 22px;
        max-width: 70%;
        margin: 6px 0;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    .bot-bubble {
        background-color: #e5e5ea;  /* light gray like iMessage */
        color: black;
        padding: 14px 18px;
        border-radius: 22px;
        max-width: 70%;
        margin: 6px 0;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* Input box */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 12px 20px;
        font-size: 16px;
        border: 1px solid #d1d1d6;
    }

    /* Send button */
    .stButton button {
        border-radius: 25px;
        padding: 10px 28px;
        background-color: #0a84ff;
        color: white;
        font-weight: bold;
        font-size: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background-color: #5ac8fa;
        transform: scale(1.05);
    }

    /* Chat history sidebar */
    [data-testid="stSidebar"] {
        background-color: #4B0082;  /* Dark violet */
        color: white;
        padding: 15px;
    }

    /* Scrollable chat container */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ------------------- Header with Appointment Button -------------------
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🩺 AI Healthcare Assistant")
        st.caption("An intelligent chatbot for general medical guidance. (Not a substitute for professional care)")
    with col2:
        if st.button("📅 Book Appointment"):
            st.session_state.show_appointment = True

    # ------------------- Session State -------------------
    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_appointment" not in st.session_state:
        st.session_state.show_appointment = False

    # ------------------- Sidebar: Chat History -------------------
    with st.sidebar:
        st.header("📜 Chat History")
        if st.session_state.history:
            for role, msg in st.session_state.history:
                st.write(f"**{role}:** {msg}")
        else:
            st.info("No chats yet!")

    # ------------------- Chat Input -------------------
    user_input = st.text_input("💬 Enter your question:")

    if st.button("Send"):
        if user_input.strip():
            st.session_state.history.append(("User", user_input))
            with st.spinner("🤖 Thinking..."):
                response = healthcare_chatbot(user_input)
            st.session_state.history.append(("Assistant", response))

            if "appointment" in response.lower():
                st.session_state.show_appointment = True
        else:
            st.warning("Please enter a valid question.")

    # ------------------- Appointment Section -------------------
    if st.session_state.show_appointment:
        st.subheader("📅 Book an Appointment")
        date = st.date_input("Select a date")
        time = st.time_input("Select a time")
        if st.button("Confirm Appointment"):
            msg = f"✅ Appointment booked successfully on {date} at {time}."
            st.session_state.history.append(("Assistant", msg))
            st.success(msg)
            st.session_state.show_appointment = False

    # ------------------- Chat Window -------------------
    st.subheader("💬 Conversation")
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for role, msg in st.session_state.history:
            if role == "User":
                st.markdown(
                    f"<div class='user-bubble'>👤 <b>You:</b> {msg}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='bot-bubble'>🤖 <b>Assistant:</b> {msg}</div>",
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
