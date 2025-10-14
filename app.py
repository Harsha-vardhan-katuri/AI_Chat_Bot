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

    # Custom 3D background and styles
    st.markdown("""
        <style>
        body {
            background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
            background-attachment: fixed;
            color: white;
        }
        .stApp {
            background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
            color: white;
        }
        .stTextInput > div > div > input {
            background-color: #1c1c1c;
            color: white;
            border-radius: 10px;
        }
        .stButton button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
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
            msg = f"✅ Appointment booked on {date} at {time}."
            st.session_state.history.append(("Assistant", msg))
            st.session_state.show_appointment = False

    # Chat window display
    st.subheader("💬 Conversation")
    chat_container = st.container()
    with chat_container:
        for role, msg in st.session_state.history:
            if role == "User":
                st.markdown(
                    f"<div style='background-color:#1e1e1e;padding:10px;border-radius:10px;margin:5px;text-align:right;'>"
                    f"👤 <b>You:</b> {msg}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background-color:#2e2e2e;padding:10px;border-radius:10px;margin:5px;text-align:left;'>"
                    f"🤖 <b>Assistant:</b> {msg}</div>",
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
