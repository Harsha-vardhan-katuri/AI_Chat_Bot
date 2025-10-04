import streamlit as st
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
        return "If the fever is below 102°F, you can take Paracetamol or Dolo 650. If it exceeds 102°F, consult a doctor immediately."
    elif "cough" in user_input.lower():
        return "For a mild dry cough, try honey, warm water, or Benadryl. If it persists more than a week, consult a doctor."
    elif "cold" in user_input.lower() or "congestion" in user_input.lower():
        return "For a mild cold, take Cetirizine and try steam inhalation. If it persists more than a week, consult a doctor."
    else:
        response = chatbot(user_input, max_length=200, num_return_sequences=1)
        return response[0]['generated_text']


def main():
    st.set_page_config(page_title="Healthcare Assistant Chatbot", layout="wide")

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar toggle for chat history
    with st.sidebar:
        if "show_history" not in st.session_state:
            st.session_state.show_history = False

        if st.button("☰ Chats"):
            st.session_state.show_history = not st.session_state.show_history

        if st.session_state.show_history:
            st.markdown("### Chat History")
            if st.session_state.history:
                for role, text in st.session_state.history:
                    if role == "User":
                        st.markdown(f"👤 **You:** {text}")
                    else:
                        st.markdown(f"🤖 **Assistant:** {text}")
            else:
                st.info("No conversation yet. Start chatting!")

    # Main layout
    st.title("🩺 Healthcare Assistant Chatbot")

    # Quick reply buttons
    st.subheader("Quick Options")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("🤒 Symptoms"):
            st.session_state.history.append(("User", "symptom"))
            st.session_state.history.append(("Assistant", healthcare_chatbot("symptom")))
    with colB:
        if st.button("💊 Medication"):
            st.session_state.history.append(("User", "medication"))
            st.session_state.history.append(("Assistant", healthcare_chatbot("medication")))
    with colC:
        if st.button("📅 Appointment"):
            st.session_state.show_appointment = True  # flag to show form

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
    if st.session_state.get("show_appointment", False):
        st.subheader("📅 Book an Appointment")
        appointment_date = st.date_input("Select appointment date")
        appointment_time = st.time_input("Select appointment time")
        if st.button("Confirm Appointment"):
            st.session_state.history.append(
                ("Assistant", f"✅ Your appointment is booked for {appointment_date} at {appointment_time}.")
            )
            st.session_state.show_appointment = False

        # Conversation area with scrollable chat box (ChatGPT style)
    st.subheader("💬 Conversation")

    st.markdown(
        "<div style='max-height:400px; overflow-y:auto; padding:10px; border:1px solid #444; border-radius:10px;'>",
        unsafe_allow_html=True
    )

    for role, text in st.session_state.history:
        if role == "User":
            st.markdown(
                f"<div style='text-align:right; background-color:#222; color:white; "
                f"border-radius:10px; padding:8px; margin:6px; display:inline-block; max-width:70%;'>"
                f"👤 <b>You:</b> {text}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align:left; background-color:#333; color:white; "
                f"border-radius:10px; padding:8px; margin:6px; display:inline-block; max-width:70%;'>"
                f"🤖 <b>Assistant:</b> {text}</div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
