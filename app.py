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

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Page layout: left = chat history, right = input & options
    col1, col2 = st.columns([1.2, 2.5])

    # Left side: Chat History
    with col1:
        st.subheader("💬 Chat History")
        if st.session_state.history:
            for role, text in st.session_state.history:
                if role == "User":
                    st.markdown(f"👤 **You:** {text}")
                else:
                    st.markdown(f"🤖 **Healthcare Assistant:** {text}")
        else:
            st.info("No conversation yet. Start chatting!")

    # Right side: Input + Options
    with col2:
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
                st.session_state.history.append(("User", "appointment"))
                st.session_state.history.append(("Assistant", healthcare_chatbot("appointment")))

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

        # Appointment booking form (only one kept)
        with st.expander("📅 Book an Appointment"):
            appointment_date = st.date_input("Select appointment date")
            appointment_time = st.time_input("Select appointment time")
            if st.button("Confirm Appointment"):
                st.session_state.history.append(
                    ("Assistant", f"✅ Your appointment is booked for {appointment_date} at {appointment_time}.")
                )


if __name__ == "__main__":
    main()
