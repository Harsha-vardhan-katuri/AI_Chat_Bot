import streamlit as st
import nltk
from transformers import pipeline
from textblob import TextBlob
import json

# ---------------- NLTK Setup ----------------
nltk.download('punkt')
nltk.download('stopwords')

# ---------------- Load Chatbot Model ----------------
chatbot = pipeline("text-generation", model="distilgpt2", device=-1)

# ---------------- Session State Initialization ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "show_appointment" not in st.session_state:
    st.session_state.show_appointment = False
if "context" not in st.session_state:
    st.session_state.context = {}

# ---------------- CSS Styles ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background-color: #0d1b2a;
    color:white;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    padding:15px;
}

/* Chat container scrollable */
.chat-container{
    max-height:500px;
    overflow-y:auto;
    padding-right:10px;
}

/* User and Bot bubbles */
.user-bubble{
    background-color:#1f77b4;
    padding:14px 18px;
    border-radius:22px;
    max-width:70%;
    margin:6px 0;
    color:white;
    animation:fadeIn 0.3s;
}
.bot-bubble{
    background-color:#3a506b;
    padding:14px 18px;
    border-radius:22px;
    max-width:70%;
    margin:6px 0;
    color:white;
    animation:fadeIn 0.3s;
}

/* Quick reply buttons */
.stButton>button{
    border-radius:25px;
    padding:10px 28px;
    background-color:#1f77b4;
    color:white;
    font-weight:bold;
    box-shadow:0 4px 8px rgba(0,0,0,0.3);
    transition:all 0.3s ease;
}
.stButton>button:hover{
    background-color:#3a86ff;
    transform:scale(1.05);
}

/* Input field */
.stTextInput>div>div>input{
    border-radius:25px;
    padding:12px 20px;
    font-size:16px;
    border:none;
    background-color:#1b263b;
    color:white;
}

/* Animation */
@keyframes fadeIn{
    from{opacity:0; transform:translateY(10px);}
    to{opacity:1; transform:translateY(0);}
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#162447;
    color:white;
    padding:15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Intent Recognition ----------------
def classify_intent(user_input):
    text = user_input.lower()
    if "appointment" in text:
        return "appointment"
    elif any(symptom in text for symptom in ["fever", "cough", "cold", "pain", "congestion"]):
        return "symptom"
    else:
        return "general"

# ---------------- Spell Correction ----------------
def correct_text(text):
    return str(TextBlob(text).correct())

# ---------------- Chatbot Logic ----------------
def healthcare_chatbot(user_input):
    user_input_corrected = correct_text(user_input)
    intent = classify_intent(user_input_corrected)
    st.session_state.context['last_intent'] = intent

    if intent == "appointment":
        return "Would you like to schedule an appointment with the doctor?"
    elif intent == "symptom":
        if "fever" in user_input_corrected.lower():
            return "If the fever is below 102°F, take Paracetamol. If higher, consult a doctor immediately."
        elif "cough" in user_input_corrected.lower():
            return "For mild dry cough, try honey or warm water. Consult a doctor if persistent."
        elif "cold" in user_input_corrected.lower() or "congestion" in user_input_corrected.lower():
            return "For mild cold, take Cetirizine and steam inhalation. Consult a doctor if persists."
        elif "pain" in user_input_corrected.lower():
            return "For mild pain, you may take Ibuprofen after food. Persistent pain requires medical advice."
        else:
            return "Please provide more details about your symptom."
    else:
        response = chatbot(user_input_corrected, max_length=150, num_return_sequences=1)
        return response[0]['generated_text']

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📜 Chat History")
    if st.session_state.history:
        for role, msg in st.session_state.history:
            st.write(f"**{role}:** {msg}")
    else:
        st.info("No chats yet!")
    # Quick symptom buttons
    st.subheader("Quick Actions")
    for btn in ["Fever", "Cough", "Cold", "Pain", "Book Appointment"]:
        if st.button(btn):
            if btn == "Book Appointment":
                st.session_state.show_appointment = True
            else:
                st.session_state.history.append(("User", btn))
                response = healthcare_chatbot(btn)
                st.session_state.history.append(("Assistant", response))

# ---------------- Header ----------------
col1, col2 = st.columns([4,1])
with col1:
    st.title("🩺 AI Healthcare Assistant")
with col2:
    if st.button("📅 Book Appointment"):
        st.session_state.show_appointment = True

# ---------------- Chat Input ----------------
user_input = st.text_input("💬 Enter your question:")
if st.button("Send"):
    if user_input.strip():
        st.session_state.history.append(("User", user_input))
        response = healthcare_chatbot(user_input)
        st.session_state.history.append(("Assistant", response))
        # Save log
        with open("chat_log.json", "w") as f:
            json.dump(st.session_state.history, f)
        if "appointment" in response.lower():
            st.session_state.show_appointment = True

# ---------------- Appointment Section ----------------
if st.session_state.show_appointment:
    st.subheader("📅 Book an Appointment")
    date = st.date_input("Select a date")
    time = st.time_input("Select a time")
    if st.button("Confirm Appointment"):
        msg = f"✅ Appointment booked on {date} at {time}."
        st.session_state.history.append(("Assistant", msg))
        st.success(msg)
        st.session_state.show_appointment = False

# ---------------- Chat Display ----------------
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

# ---------------- Image / Voice Upload Placeholder ----------------
st.subheader("🖼 Upload Symptom Image / 🎙 Voice Input")
uploaded_file = st.file_uploader("Upload image", type=["jpg","png"])
if uploaded_file:
    st.image(uploaded_file)
voice_input = st.text_input("Simulated Voice Input (Placeholder)")
if voice_input:
    st.session_state.history.append(("User", voice_input))
    response = healthcare_chatbot(voice_input)
    st.session_state.history.append(("Assistant", response))

