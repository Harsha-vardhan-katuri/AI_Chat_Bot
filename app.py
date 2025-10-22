import streamlit as st
import time
from datetime import datetime
from transformers import pipeline
from googletrans import Translator

# ---------------------- MODEL ----------------------
chatbot = pipeline("text-generation", model="distilgpt2", device=-1)

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Digital GPT - AI Health Assistant", layout="wide")

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
<style>
/* Animated Gradient Background */
.stApp {
    background: linear-gradient(270deg, #00264d, #003366, #004080);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    color: white;
}
@keyframes gradientShift {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

/* Floating Bubbles */
.bubble {
    position: fixed;
    border-radius: 50%;
    opacity: 0.2;
    animation: floatUp 20s infinite;
}
@keyframes floatUp {
  from {transform: translateY(100vh);}
  to {transform: translateY(-10vh);}
}
.b1 {width: 120px; height: 120px; background: #00bcd4; left: 10%; animation-delay: 2s;}
.b2 {width: 80px; height: 80px; background: #ff4081; left: 60%; animation-delay: 4s;}

/* Fonts and Typography */
h1,h2,h3,h4 {color:white;}
.stTextInput > div > div > input {
    background-color: #1e2a38; color: white; border: 1px solid #333; border-radius: 6px; padding: 10px;
}
.stButton > button {
    background-color: #0052cc; color: white; border: none; border-radius: 6px; padding: 8px 20px; transition: 0.3s;
}
.stButton > button:hover {
    background-color: #007bff; transform: scale(1.05); box-shadow: 0px 0px 10px rgba(0,123,255,0.6);
}

/* Chat bubbles */
.chat-bubble-user {
    background-color: #0052cc; padding: 10px; border-radius: 10px; margin: 5px 0; color: white; text-align: right;
    animation: slideInRight 0.5s ease-out;
}
.chat-bubble-bot {
    background-color: #1e2a38; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: left;
    animation: slideInLeft 0.5s ease-out;
}
@keyframes slideInRight { from {opacity: 0; transform: translateX(50px);} to {opacity: 1; transform: translateX(0);} }
@keyframes slideInLeft { from {opacity: 0; transform: translateX(-50px);} to {opacity: 1; transform: translateX(0);} }

.section {
    background-color: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 10px; margin-bottom: 15px;
}
</style>

<div class="bubble b1"></div>
<div class="bubble b2"></div>
""", unsafe_allow_html=True)

# ---------------------- SESSION STATE ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "typing" not in st.session_state:
    st.session_state.typing = False

# ---------------------- TRANSLATION SETUP ----------------------
translator = Translator()
languages = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn"
}

# ---------------------- MULTILINGUAL TEXTS ----------------------
multilingual_texts = {
    "English": {
        "app_title": "Digital GPT - AI Health Assistant",
        "description": "Your personal assistant for basic health guidance, diet tips, and symptom insights.",
        "ask_question": "Ask me anything about your health:",
        "placeholder": "Type your question here...",
        "send": "Send",
        "quick_actions": ["Hydration Tips", "Diet Advice", "Sleep Help", "Fitness Tips", "Clear Chat"],
        "upload_file": "Upload file",
        "book_appointment": "Book Appointment",
        "select_date": "Select a date",
        "select_time": "Select time",
        "confirm_appointment": "Confirm Appointment",
        "no_chat": "Start by typing your health question above.",
        "chat_history": "Chat History Summary",
        "last_query": "Last query:"
    },
    "Telugu": {
        "app_title": "డిజిటల్ GPT – AI ఆరోగ్య సహాయకుడు",
        "description": "మీ వ్యక్తిగత సహాయకుడు, మౌలిక ఆరోగ్య మార్గనిర్దేశనం, డైట్ సూచనలు, లక్షణాల సమాచారం కోసం.",
        "ask_question": "మీ ఆరోగ్యం గురించి ఏదైనా అడగండి:",
        "placeholder": "ఇక్కడ మీ ప్రశ్న టైప్ చేయండి...",
        "send": "పంపండి",
        "quick_actions": ["హైడ్రేషన్ సూచనలు", "ఆహార సలహాలు", "నిద్ర సహాయం", "ఫిట్‌నెస్ సూచనలు", "చాట్ క్లియర్ చేయండి"],
        "upload_file": "ఫైల్‌ను అప్‌లోడ్ చేయండి",
        "book_appointment": "నియామకం بک చేయండి",
        "select_date": "తేదీ ఎంచుకోండి",
        "select_time": "సమయం ఎంచుకోండి",
        "confirm_appointment": "నియామకం నిర్ధారించండి",
        "no_chat": "పై లో మీ ఆరోగ్య ప్రశ్నను టైప్ చేయడం మొదలు పెట్టండి.",
        "chat_history": "చాట్ చరిత్ర సారాంశం",
        "last_query": "తుది ప్రశ్న:"
    },
    "Hindi": {
        "app_title": "डिजिटल GPT – AI स्वास्थ्य सहायक",
        "description": "आपका व्यक्तिगत सहायक, बुनियादी स्वास्थ्य मार्गदर्शन, आहार सुझाव और लक्षण जानकारी के लिए।",
        "ask_question": "अपने स्वास्थ्य के बारे में कुछ भी पूछें:",
        "placeholder": "अपना प्रश्न यहाँ टाइप करें...",
        "send": "भेजें",
        "quick_actions": ["पानी पीने के सुझाव", "आहार सलाह", "नींद मदद", "फिटनेस सुझाव", "चैट साफ़ करें"],
        "upload_file": "फ़ाइल अपलोड करें",
        "book_appointment": "अपॉइंटमेंट बुक करें",
        "select_date": "तारीख चुनें",
        "select_time": "समय चुनें",
        "confirm_appointment": "अपॉइंटमेंट कन्फ़र्म करें",
        "no_chat": "ऊपर अपने स्वास्थ्य प्रश्न टाइप करके शुरू करें।",
        "chat_history": "चैट इतिहास सारांश",
        "last_query": "अंतिम प्रश्न:"
    },
    "Kannada": {
        "app_title": "ಡಿಜಿಟಲ್ GPT – AI ಆರೋಗ್ಯ ಸಹಾಯಕ",
        "description": "ನಿಮ್ಮ ವೈಯಕ್ತಿಕ ಸಹಾಯಕ, ಮೂಲ ಆರೋಗ್ಯ ಮಾರ್ಗದರ್ಶನ, ಆಹಾರ ಸಲಹೆಗಳು ಮತ್ತು ಲಕ್ಷಣಗಳ ಮಾಹಿತಿಗಾಗಿ.",
        "ask_question": "ನಿಮ್ಮ ಆರೋಗ್ಯದ ಬಗ್ಗೆ ಏನಾದರೂ ಕೇಳಿ:",
        "placeholder": "ಇಲ್ಲಿ ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಟೈಪ್ ಮಾಡಿ...",
        "send": "ಕಳುಹಿಸಿ",
        "quick_actions": ["ಹೈಡ್ರೇಶನ್ ಸಲಹೆಗಳು", "ಆಹಾರ ಸಲಹೆಗಳು", "ನಿದ್ದೆ ಸಹಾಯ", "ಫಿಟ್ನೆಸ್ ಸಲಹೆಗಳು", "ಚಾಟ್ ಕ್ಲೀರ್ ಮಾಡಿ"],
        "upload_file": "ಫೈಲ್ ಅಪ್ಲೋಡ್ ಮಾಡಿ",
        "book_appointment": "ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ಬುಕ್ ಮಾಡಿ",
        "select_date": "ದಿನಾಂಕ ಆಯ್ಕೆಮಾಡಿ",
        "select_time": "ಸಮಯ ಆಯ್ಕೆಮಾಡಿ",
        "confirm_appointment": "ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ದೃಢೀಕರಿಸಿ",
        "no_chat": "ಮೇಲಿನ健康 ಪ್ರಶ್ನೆಯನ್ನು ಟೈಪ್ ಮಾಡಿ ಪ್ರಾರಂಭಿಸಿ.",
        "chat_history": "ಚಾಟ್ ಇತಿಹಾಸ ಸಾರಾಂಶ",
        "last_query": "ಕೊನೆಯ ಪ್ರಶ್ನೆ:"
    },
    # Add Hindi, Tamil, Kannada similarly
}

# ---------------------- TOP BAR WITH LANGUAGE SELECTION ----------------------
top_left, top_right = st.columns([3, 1])
with top_right:
    selected_language = st.selectbox("🌐 Language", list(languages.keys()))

texts = multilingual_texts[selected_language]

# ---------------------- BOT REPLY FUNCTION ----------------------
def get_bot_reply(user_text):
    text = user_text.lower()

    # Translate input to English if needed
    if selected_language != "English":
        user_text = translator.translate(user_text, dest='en').text
        text = user_text.lower()

    # Predefined responses
    if "cold" in text:
        reply = "🤧 Try steam inhalation and stay hydrated. For persistent cold, consult a doctor."
    elif "pain" in text:
        reply = "💊 For mild pain, rest and hydration help. If severe, visit a physician."
    elif "diet" in text:
        reply = "🥗 Include fruits, vegetables, proteins, and avoid junk food."
    elif "sleep" in text:
        reply = "😴 Keep a regular sleep schedule and avoid caffeine late in the day."
    elif "fever" in text:
        reply = "🌡️ Drink fluids, rest, and take paracetamol if needed."
    elif "fitness" in text:
        reply = "🏋️ Regular exercise, stretching, and staying active improve health."
    elif "hydration" in text or "water" in text:
        reply = "💧 Drink at least 8 glasses of water daily and stay hydrated."
    else:
        # fallback to GPT
        generated = chatbot(user_text, max_length=100, num_return_sequences=1)
        reply = generated[0]['generated_text']

    # Translate reply back to selected language
    if selected_language != "English":
        reply = translator.translate(reply, dest=languages[selected_language]).text

    return reply

# ---------------------- HEADER ----------------------
header_left, header_right = st.columns([2.5, 1])

with header_left:
    st.markdown(f"## 🩺 {texts['app_title']}")
    st.markdown(texts['description'])

    # ---------------------- QUICK ACTIONS ----------------------
    st.markdown(f"### ⚡ {texts['ask_question']}")

    def handle_quick_action(action_name):
        st.session_state.messages.append(("user", action_name))
        with st.spinner("🤖 Digital GPT is typing..."):
            time.sleep(1.2)
            reply = get_bot_reply(action_name)
            st.session_state.messages.append(("bot", reply))

    qa_cols = st.columns(len(texts['quick_actions']))
    for i, action in enumerate(texts['quick_actions']):
        qa_cols[i].button(action, on_click=lambda a=action: handle_quick_action(a))

    # ---------------------- USER INPUT ----------------------
    def send_message():
        user_text = st.session_state.user_input_box
        if user_text:
            st.session_state.messages.append(("user", user_text))
            st.session_state.typing = True
            with st.spinner("🤖 Digital GPT is typing..."):
                time.sleep(1.5)
                reply = get_bot_reply(user_text)
                st.session_state.messages.append(("bot", reply))
            st.session_state.typing = False
            st.session_state.user_input_box = ""

    col_query = st.columns([4, 1])
    user_input = col_query[0].text_input("", placeholder=texts['placeholder'], key="user_input_box")
    col_query[1].button(texts['send'], on_click=send_message)

    # ---------------------- CONVERSATION AREA ----------------------
    st.markdown("### 💬 Conversation")
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info(texts['no_chat'])
        else:
            for sender, msg in st.session_state.messages:
                timestamp = datetime.now().strftime("%H:%M")
                if sender == "user":
                    st.markdown(
                        f'<div class="chat-bubble-user">{msg}<div style="font-size:10px;color:#ccc;">{timestamp}</div></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="chat-bubble-bot">{msg}<div style="font-size:10px;color:#ccc;">{timestamp}</div></div>',
                        unsafe_allow_html=True)

# ---------------------- RIGHT PANEL ----------------------
with header_right:
    st.markdown(f"### 📤 {texts['upload_file']}")
    st.file_uploader("", type=["png", "jpg", "jpeg", "mp4", "mov"], label_visibility="collapsed")

    st.markdown(f"### 📅 {texts['book_appointment']}")
    appt_date = st.date_input(texts['select_date'], datetime.now())
    appt_time = st.time_input(texts['select_time'], datetime.now().time())
    if st.button(texts['confirm_appointment']):
        st.success(f"✅ {texts['confirm_appointment']} {appt_date} at {appt_time}.")

    st.markdown(f"### 🕓 {texts['chat_history']}")
    if st.session_state.messages:
        last_user = [msg for sender, msg in st.session_state.messages if sender == "user"]
        if last_user:
            st.write(f"**{texts['last_query']}** {last_user[-1]}")
    else:
        st.info(texts['no_chat'])
