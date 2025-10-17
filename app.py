import streamlit as st
import time
from datetime import datetime
from transformers import pipeline

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

# ---------------------- HEADER ----------------------
header_left, header_right = st.columns([2.5, 1])

with header_left:
    st.markdown("## 🩺 Digital GPT - AI Health Assistant")
    st.markdown("Your personal assistant for basic health guidance, diet tips, and symptom insights.")

    # ---------------------- PREDEFINED QUICK ACTION RESPONSES ----------------------
    predefined_responses = {
        "💧 Hydration Tips": "💧 Drink at least 8 glasses of water daily and stay hydrated.",
        "🍎 Diet Advice": "🥗 Include fruits, vegetables, proteins, and avoid junk food.",
        "💤 Sleep Help": "😴 Maintain a regular sleep schedule, avoid caffeine late in the day.",
        "💪 Fitness": "🏋️ Regular exercise, stretching, and staying active improve health."
    }

    # ---------------------- QUICK ACTIONS ----------------------
    st.markdown("### ⚡ Quick Actions")
    qa_cols = st.columns(5)

    def quick_action(action_name):
        reply = predefined_responses.get(action_name, "")
        if not reply:
            # fallback to GPT
            reply = get_bot_reply(action_name)
        st.session_state.messages.append(("user", action_name))
        st.session_state.messages.append(("bot", reply))

    for i, key in enumerate(predefined_responses.keys()):
        qa_cols[i].button(key, on_click=lambda k=key: quick_action(k))
    qa_cols[4].button("🧹 Clear Chat", on_click=lambda: st.session_state.messages.clear())

    # ---------------------- USER INPUT ----------------------
    st.markdown("### Ask me anything about your health:")

    def get_bot_reply(user_text):
        text = user_text.lower()
        # predefined checks
        if "cold" in text:
            return "🤧 Try steam inhalation and stay hydrated. For persistent cold, consult a doctor."
        elif "pain" in text:
            return "💊 For mild pain, rest and hydration help. If severe, visit a physician."
        elif "diet" in text:
            return "🥗 Include fruits, vegetables, proteins, and avoid junk food."
        elif "sleep" in text:
            return "😴 Keep a regular sleep schedule and avoid caffeine late in the day."
        elif "fever" in text:
            return "🌡️ Drink fluids, rest, and take paracetamol if needed."
        else:
            generated = chatbot(user_text, max_length=100, num_return_sequences=1)
            return generated[0]['generated_text']

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
        # Clear input safely
        st.session_state.user_input_box = ""

    col_query = st.columns([4, 1])
    col_query[0].text_input(
        "Your Question",
        placeholder="Type your question here...",
        key="user_input_box"
    )
    col_query[1].button("Send", on_click=send_message)

    # ---------------------- CONVERSATION AREA ----------------------
    st.markdown("### 💬 Conversation")
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info("Start by typing your health question above.")
        else:
            for sender, msg in st.session_state.messages:
                timestamp = datetime.now().strftime("%H:%M")
                if sender == "user":
                    st.markdown(f'<div class="chat-bubble-user">{msg}<div style="font-size:10px;color:#ccc;">{timestamp}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble-bot">{msg}<div style="font-size:10px;color:#ccc;">{timestamp}</div></div>', unsafe_allow_html=True)

# ---------------------- RIGHT PANEL ----------------------
with header_right:
    st.markdown("### 📤 Upload Image / Video")
    st.file_uploader("Upload file", type=["png", "jpg", "jpeg", "mp4", "mov"], label_visibility="collapsed")

    st.markdown("### 📅 Book Appointment")
    appt_date = st.date_input("Select a date", datetime.now())
    appt_time = st.time_input("Select time", datetime.now().time())
    if st.button("Confirm Appointment"):
        st.success(f"✅ Appointment booked for {appt_date} at {appt_time}.")

    st.markdown("### 🕓 Chat History Summary")
    if st.session_state.messages:
        last_user = [msg for sender, msg in st.session_state.messages if sender == "user"]
        if last_user:
            st.write(f"**Last query:** {last_user[-1]}")
    else:
        st.info("No chat history yet.")
