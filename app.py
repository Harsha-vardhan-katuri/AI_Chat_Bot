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
        response = chatbot(user_input, max_length=200, num_return_sequences=1)
        return response[0]['generated_text']

# ------------------- Streamlit App -------------------
def main():
    st.set_page_config(page_title="Healthcare Assistant Chatbot", layout="wide")

    # ------------------- Custom CSS -------------------
    st.markdown("""
    <style>
    /* Main container */
    [data-testid="stAppViewContainer"] {
        background-color: #0d1b2a;  /* Dark blue */
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: white;
        padding: 20px;
    }

    /* Header title */
    .stTitle {
        color: white;
    }

    /* Chat bubbles */
    .user-bubble {
        background-color: #1f77b4; /* Dynamic blue */
        color: white;
        padding: 14px 18px;
        border-radius: 22px;
        max-width: 70%;
        margin: 6px 0;
        word-wrap: break-word;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    .user-bubble:hover {
        transform: scale(1.03);
    }

    .bot-bubble {
        background-color: #3a506b; /* Darker bubble */
        color: white;
        padding: 14px 18px;
        border-radius: 22px;
        max-width: 70%;
        margin: 6px 0;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    /* Input box */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 12px 20px;
        font-size: 16px;
        border: none;
        background-color: #1b263b;
        color: white;
    }

    /* Send button */
    .stButton button {
        border-radius: 25px;
        padding: 10px 28px;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        font-size: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #3a86ff;
        transform: scale(1.05);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #162447;
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
    col1, col2 = st.columns([4,1])
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
                st.markdown(f"<div class='user-bubble'>👤 {msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-bubble'>🤖 {msg}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
