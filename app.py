import streamlit as st
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load Hugging Face chatbot model
chatbot = pipeline("text-generation", model="distilgpt2", device=-1)

# Healthcare chatbot logic
def healthcare_chatbot(user_input):
    if "symptom" in user_input.lower():
        return "It seems like you're experiencing symptoms. Please consult a Doctor for accurate advice."
    elif "appointment" in user_input.lower():
        return "Would you like to schedule an appointment with the doctor?"
    elif "medication" in user_input.lower():
        return "It's important to take prescribed medicine regularly. If you have concerns, consult your doctor."
    elif "fever" in user_input.lower():
        return "If the fever is below 102°F, you can take Paracetamol or Dolo 650 for relief. If it exceeds 102°F, consult a doctor immediately."
    elif "cough" in user_input.lower():
        return "For a mild dry cough, try honey, warm water, or Benadryl syrup. If it persists for more than a week, consult a doctor."
    elif "cold" in user_input.lower() or "congestion" in user_input.lower():
        return "For a mild cold, take Cetirizine (Cetzine/Okacet) and try steam inhalation. If it persists more than a week, consult a doctor."
    else:
        response = chatbot(user_input, max_length=200, num_return_sequences=1)
        return response[0]['generated_text']

def main():
    st.set_page_config(page_title="Healthcare Assistant Chatbot", layout="wide")

    # Initialize states
    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_appointment" not in st.session_state:
        st.session_state.show_appointment = False

    # Sidebar - Chat History (toggle style)
    with st.sidebar:
        st.markdown("### ☰ Chat History")
        if st.session_state.history:
            for role, text in st.session_state.history:
                if role == "User":
                    st.markdown(f"👤 **You:** {text}")
                else:
                    st.markdown(f"🤖 **Assistant:** {text}")
        else:
            st.info("No conversation yet. Start chatting!")

    # Main Title
    st.title("🩺 Healthcare Assistant Chatbot")

    # Quick reply buttons
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("🤒 Symptoms"):
            user_msg = "symptom"
            st.session_state.history.append(("User", user_msg))
            st.session_state.history.append(("Assistant", healthcare_chatbot(user_msg)))
    with colB:
        if st.button("💊 Medication"):
            user_msg = "medication"
            st.session_state.history.append(("User", user_msg))
            st.session_state.history.append(("Assistant", healthcare_chatbot(user_msg)))
    with colC:
        if st.button("📅 Appointment"):
            st.session_state.show_appointment = True  # Show booking form

    # User input
    user_input = st.text_input("💬 Type your query:")
    if st.button("Submit"):
        if user_input:
            st.session_state.history.append(("User", user_input))
            with st.spinner("Processing your query..."):
                response = healthcare_chatbot(user_input)
            st.session_state.history.append(("Assistant", response))
        else:
            st.warning("⚠️ Please enter a message to get a response.")

    # Appointment booking form
    if st.session_state.show_appointment:
        st.subheader("📅 Book an Appointment")
        appointment_date = st.date_input("Select appointment date", datetime.today())
        appointment_time = st.time_input("Select appointment time", datetime.now().time())
        if st.button("Confirm Appointment"):
            confirmation = f"✅ Your appointment is booked for {appointment_date} at {appointment_time}."
            st.session_state.history.append(("Assistant", confirmation))
            st.session_state.show_appointment = False  # Hide after booking

    # Display conversation in chat bubble style
    st.subheader("💬 Conversation")
    for role, text in st.session_state.history:
        if role == "User":
            st.markdown(
                f"<div style='text-align: right; background-color:#DCF8C6; "
                f"border-radius: 10px; padding: 8px; margin: 4px; display:inline-block;'>"
                f"👤 <b>You:</b> {text}</div><br>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align: left; background-color:#E6E6E6; "
                f"border-radius: 10px; padding: 8px; margin: 4px; display:inline-block;'>"
                f"🤖 <b>Assistant:</b> {text}</div><br>",
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()
