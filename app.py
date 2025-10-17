import streamlit as st
from datetime import datetime
from transformers import pipeline

# ---------------------- LOAD MODEL ----------------------
chatbot = pipeline("text-generation", model="distilgpt2", device=-1)

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Digital GPT - AI Health Assistant", layout="wide")

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #001f3f, #003366);
        color: white;
    }
    h1, h2, h3, h4 {
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #1e2a38;
        color: white;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #0052cc;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #007bff;
    }
    .chat-bubble-user {
        background-color: #0052cc;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        color: white;
        text-align: right;
    }
    .chat-bubble-bot {
        background-color: #1e2a38;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
    }
    .section {
        background-color: rgba(0, 0, 0, 0.2);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- SESSION STATE ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------- MAIN LAYOUT ----------------------
header_left, header_right = st.columns([2.5, 1])

# ---------------------- LEFT SIDE ----------------------
with header_left:
    # Header
    st.markdown("## 🩺 Digital GPT - AI Health Assistant")
    st.markdown("Your personal assistant for basic health guidance, diet tips, and symptom insights.")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    qa_cols = st.columns(5)
    qa_cols[0].button("💧 Hydration Tips")
    qa_cols[1].button("🍎 Diet Advice")
    qa_cols[2].button("💤 Sleep Help")
    qa_cols[3].button("💪 Fitness")
    qa_cols[4].button("🧹 Clear Chat", on_click=lambda: st.session_state.clear())
    
    # Ask your question
    st.markdown("### Ask me anything about your health:")
    query_col, _ = st.columns([2.5, 1])
    with query_col:
        query_row = st.columns([4, 1])
        user_input = query_row[0].text_input("", placeholder="Type your question here...", label_visibility="collapsed")
        send_button = query_row[1].button("Send")
    
    # Simple Reply Logic
    def get_bot_reply(user_text):
        text = user_text.lower()
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
    
    # Chat logic
    if send_button and user_input:
        st.session_state.messages.append(("user", user_input))
        st.session_state.messages.append(("bot", get_bot_reply(user_input)))
    
    # Conversation section (same width as Quick Actions)
    conv_col, _ = st.columns([2.5, 1])
    with conv_col:
        st.markdown("### 💬 Conversation")
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.info("Start by typing your health question above.")
            else:
                for sender, msg in st.session_state.messages:
                    if sender == "user":
                        st.markdown(f'<div class="chat-bubble-user">{msg}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-bubble-bot">{msg}</div>', unsafe_allow_html=True)

# ---------------------- RIGHT SIDE ----------------------
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
