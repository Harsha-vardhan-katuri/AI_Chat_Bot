# app.py
import streamlit as st
import json
import os
import time
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# ---------------- Page config ----------------
st.set_page_config(page_title="AI Healthcare Assistant", layout="wide")

# ---------------- Helpers & State init ----------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of tuples (role, message)
if "show_appointment" not in st.session_state:
    st.session_state.show_appointment = False
if "context" not in st.session_state:
    st.session_state.context = {}

HISTORY_FILE = "chat_history.json"

def save_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.history = json.load(f)
        except Exception:
            st.session_state.history = []

load_history()

# ---------------- CSS for UI ----------------
st.markdown(
    r"""
<style>
/* Page background and global font */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg,#071428 0%, #0b2b4a 100%);
    color: #e6eef8;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    padding: 12px 18px;
}

/* Title */
.main-title {
    text-align: left;
    font-size: 30px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}

/* Subtitle */
.sub {
    color: #c9d8ea;
    margin-top: -6px;
    margin-bottom: 6px;
}

/* Quick actions row: reduce gap and margin */
.quick-actions-row {
    display:flex;
    gap:6px;
    align-items:center;
    margin-bottom:4px;
}

/* Quick action buttons */
.quick-action .stButton>button, .primary-action .stButton>button {
    border-radius: 10px;
    padding: 8px 14px;
    background: linear-gradient(180deg,#2b6ea3,#1e4f7a);
    color: #fff;
    border: none;
    box-shadow: 0 4px 12px rgba(6,30,60,0.3);
    transition: transform .12s ease, box-shadow .12s ease;
}
.quick-action .stButton>button:hover, .primary-action .stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(6,30,60,0.4);
}

/* Layout columns spacing */
.block-left { padding-right: 6px; }
.block-right { padding-left: 6px; }

/* Chat container box directly below input */
.chat-box {
    display:flex;
    flex-direction:column-reverse; 
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    border-radius: 12px;
    padding: 8px;
    min-height: 520px;
    max-height: 72vh;
    overflow-y: auto;
    margin-top:0px;
}

/* messages list */
.messages {
    overflow-y: auto;
    padding-right: 4px;
    padding-bottom: 4px;
    flex: 1 1 auto;
}

/* message bubbles */
.msg-row { display:flex; margin-bottom:6px; align-items:flex-end; }
.msg-row.user { justify-content: flex-end; }
.msg-row.bot { justify-content: flex-start; }

.bubble {
    max-width: 72%;
    padding: 8px 12px;
    border-radius: 14px;
    font-size: 14px;
    line-height: 1.3;
    box-shadow: 0 4px 12px rgba(2,8,23,0.35);
    animation: fadeInUp .18s ease;
}

/* user bubble (right) */
.bubble.user {
    background: linear-gradient(180deg,#2f7bd9,#1f5fb6);
    color: white;
    border-bottom-right-radius: 6px;
}

/* bot bubble (left) */
.bubble.bot {
    background: linear-gradient(180deg,#11293f,#0b2340);
    color: #e6eef8;
    border-bottom-left-radius: 6px;
    border: 1px solid rgba(255,255,255,0.03);
}

/* sticky input area at bottom */
.input-area {
    margin-top: 4px;
    display:flex;
    gap:4px;
    align-items:center;
    flex-shrink:0;
}
.input-area .stTextInput>div>div>input {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.04);
    color: #e6eef8;
    border-radius: 12px;
    padding: 8px 10px;
}

/* send button */
.send-btn .stButton>button {
    background: linear-gradient(180deg,#6fb3ff,#2f7bd9);
    color:white;
    padding: 6px 12px;
    border-radius: 10px;
    border:none;
    box-shadow: 0 4px 12px rgba(47,123,217,0.3);
}

/* right panel upload & appointments aligned top */
.right-panel {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    border-radius: 12px;
    padding: 8px;
    height: 72vh;
    margin-top:0px;
}
.right-panel .panel-title {
    margin-top: 0px;
    margin-bottom: 4px;
}

/* keyframes */
@keyframes fadeInUp {
  from { opacity:0; transform: translateY(8px); }
  to { opacity:1; transform: translateY(0); }
}

/* hide default Streamlit footer and menu */
footer { visibility: hidden; }
header { visibility: hidden; }

@media (max-width: 1000px) {
  .chat-box { height: 60vh; }
  .right-panel { height: auto; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Model Loading ----------------
MODEL_NAME = "distilgpt2"  # only distilgpt2 now
device = 0 if torch.cuda.is_available() else -1

@st.cache_resource(show_spinner=False)
def load_generation_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return pipe, MODEL_NAME

with st.spinner("Loading model..."):
    gen_pipe, loaded_model_name = load_generation_pipeline()

# ---------------- Utilities ----------------
def classify_intent(text):
    t = text.lower()
    if "appointment" in t or "book" in t or "schedule" in t:
        return "appointment"
    if any(word in t for word in ["fever", "cough", "pain", "cold", "congestion"]):
        return "symptom"
    return "general"

def generate_answer(prompt, max_len=200):
    try:
        out = gen_pipe(prompt, max_length=max_len, do_sample=True, top_p=0.95, temperature=0.7, num_return_sequences=1)
        text = out[0]["generated_text"]
        if text.strip().startswith(prompt.strip()):
            text = text[len(prompt):].strip()
        return text.strip()
    except Exception:
        return "Sorry — I couldn't generate an answer right now."

# ---------------- Layout ----------------
left_col, right_col = st.columns([3,1], gap="small")

with left_col:
    st.markdown('<div class="main-title">🩺 AI Healthcare Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Your digital medical aid — not a replacement for a clinician.</div>', unsafe_allow_html=True)

    # Quick actions
    cols = st.columns([1,1,1,1,1], gap="small")
    quick_actions = ["Fever", "Cough", "Cold", "Pain", "Book Appointment"]
    for c, action in zip(cols, quick_actions):
        if c.button(action, key=f"qa_{action}"):
            if action == "Book Appointment":
                st.session_state.show_appointment = True
            else:
                st.session_state.history.append(("User", action))
                intent = classify_intent(action)
                if intent == "symptom":
                    replies = {
                        "fever":"If the fever is below 102°F, take Paracetamol and rest. If above 102°F or persistent, contact a healthcare provider.",
                        "cough":"For a mild cough, try warm drinks and honey. If severe or persistent >1 week, consult a doctor.",
                        "cold":"For a mild cold, steam inhalation and antihistamine (like cetirizine) can help.",
                        "pain":"For mild pain, over-the-counter analgesics like ibuprofen (after food) can help; seek medical advice for persistent pain."
                    }
                    reply = replies.get(action.lower(), "Please provide more details.")
                else:
                    reply = generate_answer(action)
                st.session_state.history.append(("Assistant", reply))
                save_history()

    # ---------------- Input area ----------------
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    input_col, send_col = st.columns([9,1], gap="small")
    with input_col:
        user_msg = st.text_input("Your message", placeholder="Type your question here...", key="input_text")
    with send_col:
        if st.button("Send", key="send_btn") and user_msg.strip():
            st.session_state.history.append(("User", user_msg.strip()))
            save_history()
            intent = classify_intent(user_msg)
            if intent == "appointment":
                bot_reply = "Would you like to schedule an appointment? Use the 'Book Appointment' quick action or the panel on the right."
                st.session_state.show_appointment = True
            elif intent == "symptom":
                bot_reply = "For mild symptoms, follow basic care. Persistent or severe cases should see a doctor."
            else:
                preface = "You are a helpful, cautious medical assistant. Provide brief, evidence-based guidance.\n\nPatient: "
                bot_reply = generate_answer(preface + user_msg, max_len=160)
            st.session_state.history.append(("Assistant", bot_reply))
            save_history()
            st.session_state.input_text = ""
            time.sleep(0.05)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Chat messages below input ----------------
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    st.markdown('<div class="messages" id="messages">', unsafe_allow_html=True)
    for role, msg in st.session_state.history:
        if role.lower() == "user":
            st.markdown(f'<div class="msg-row user"><div class="bubble user">👤 {msg}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-row bot"><div class="bubble bot">🤖 {msg}</div></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📤 Upload Image / Video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["png","jpg","jpeg","mp4","mov"], key="upload")
    if uploaded is not None:
        if uploaded.type.startswith("image"):
            st.image(uploaded, use_column_width=True)
        elif uploaded.type.startswith("video"):
            st.video(uploaded)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📅 Appointments</div>', unsafe_allow_html=True)
    if st.session_state.show_appointment:
        st.info("Appointment panel is open.")
    date = st.date_input("Select a date", key="appt_date")
    time_sel = st.time_input("Select time", key="appt_time")
    if st.button("Confirm Appointment", key="confirm_appt"):
        appt_msg = f"✅ Appointment confirmed on {date} at {time_sel}."
        st.session_state.history.append(("Assistant", appt_msg))
        save_history()
        st.success(appt_msg)
        st.session_state.show_appointment = False

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🕘 Recent History (summary)</div>', unsafe_allow_html=True)
    for r, m in st.session_state.history[-6:]:
        tl = (m[:80] + "...") if len(m) > 80 else m
        st.markdown(f"**{r}:** {tl}")
    st.markdown('</div>', unsafe_allow_html=True)
