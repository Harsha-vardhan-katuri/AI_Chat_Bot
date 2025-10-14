import streamlit as st
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load model
chatbot = pipeline("text-generation", model="distilgpt2")

# --- Chatbot logic ---
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

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Healthcare Assistant Chatbot", layout="wide")

    # --- Custom Background and Styles ---
    st.markdown("""
        <style>
        /* Set a healthcare-themed background image */
        [data-testid="stAppViewContainer"] {
            background: url("https://cdn.pixabay.com/photo/2020/03/09/10/00/doctor-4911680_1280.png");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
        }

        /* Transparent layer for readability */
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: absolute;
            inset: 0;
            background: rgba(0, 0, 0, 0.55);
            z-index: -1;
        }

        .stApp {
            color: white;
        }

        /* Text input styling */
        .stTextInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 10px;
        }

        /* Buttons styling */
        .stButton button {
            background-color: #6C2DC7;
            color: white;
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background-color: #9A4DFF;
            transform: scale(1.05);
        }

        /* Chat containers */
        .user-bubble {
            background-color: #6C2DC7;
            color: white;
            padding: 10px 14px;
            border-radius: 16px;
            margin: 8px 0;
            width: fit-content;
            text-align: left;
        }

        .bot-bubble {
            background-color: #3D1259;
            color: white;
            padding: 10px 14px;
            border-radius: 16px;
            margin: 8px 0;
            width: fit-content;
            text-align: left;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: rgba(30, 10, 50, 0.9);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🩺 AI Healthcare Assistant")
    st.caption("An intelligent chatbot for general medical guidance. (Not a substitute for professional care)")

    # Initialize session states
    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_appointment" not in st.session_state:
        st.session_state.show_appointment = False

    # Sidebar for chat history
    with st.sidebar:
        st.header("📜 Chat History")
        if st.session_state.history:
            for role, msg in st.session_state.history:
                st.write(f"**{role}:** {msg}")
        else:
            st.info("No chats yet!")

        # Book appointment shortcut
        if st.button("📅 Book Appointment"):
            st.session_state.show_appointment = True

    # Chat input area
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

    # Appointment section
    if st.session_state.show_appointment:
        st.subheader("📅 Book an Appointment")
        date = st.date_input("Select a date")
        time = st.time_input("Select a time")
        if st.button("Confirm Appointment"):
            msg = f"✅ Appointment booked successfully on {date} at {time}."
            st.session_state.history.append(("Assistant", msg))
            st.success(msg)
            st.session_state.show_appointment = False

    # Chat window display
    st.subheader("💬 Conversation")
    chat_container = st.container()
    with chat_container:
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

if __name__ == "__main__":
    main()
