"""
app.py — Premium AI Chatbot Streamlit Interface
================================================
Run: streamlit run app.py
"""

import streamlit as st
import time
import random
from inference import ChatBot

# ── Page config (MUST be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="ARIA · AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Dark luxury terminal aesthetic
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;700;800&display=swap');

/* ── Global ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c10 !important;
    color: #c8d8e8 !important;
    font-family: 'JetBrains Mono', monospace;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #0a1628 0%, #080c10 60%) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #050810 !important;
    border-right: 1px solid #1a2a3a !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem; }

/* ── Main container ── */
.main-wrapper {
    max-width: 860px;
    margin: 0 auto;
    padding: 2rem 1rem 1rem;
}

/* ── Header ── */
.aria-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    position: relative;
}
.aria-header::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff55, transparent);
}
.aria-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #00d4ff 0%, #7b61ff 50%, #ff6b9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.aria-sub {
    font-size: 0.75rem;
    letter-spacing: 0.25em;
    color: #4a6a8a;
    text-transform: uppercase;
}

/* ── Status badge ── */
.status-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    background: #0d1820;
    border: 1px solid #1a2a3a;
    border-radius: 20px;
    width: fit-content;
    margin: 0.8rem auto 2rem;
    font-size: 0.72rem;
    color: #4a7a9b;
    letter-spacing: 0.1em;
}
.dot-online {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 8px #00ff88;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── Chat container ── */
.chat-container {
    background: #0a0f1a;
    border: 1px solid #1a2a3a;
    border-radius: 16px;
    padding: 1.5rem;
    min-height: 380px;
    max-height: 480px;
    overflow-y: auto;
    margin-bottom: 1rem;
    scrollbar-width: thin;
    scrollbar-color: #1a2a3a transparent;
    position: relative;
}
.chat-container::-webkit-scrollbar { width: 4px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb { background: #1a2a3a; border-radius: 4px; }

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 320px;
    gap: 0.8rem;
    opacity: 0.4;
}
.empty-icon { font-size: 3rem; }
.empty-text {
    font-size: 0.8rem;
    color: #4a6a8a;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── Message bubbles ── */
.msg-row {
    display: flex;
    margin-bottom: 1.2rem;
    animation: fadeUp 0.3s ease;
    gap: 0.8rem;
    align-items: flex-end;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user-row { flex-direction: row-reverse; }

.avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.avatar-bot  { background: linear-gradient(135deg, #0d2a4a, #1a3a6a); border: 1px solid #00d4ff33; }
.avatar-user { background: linear-gradient(135deg, #2a0d4a, #4a1a7a); border: 1px solid #7b61ff33; }

.bubble {
    max-width: 72%;
    padding: 0.75rem 1rem;
    border-radius: 14px;
    font-size: 0.85rem;
    line-height: 1.6;
    position: relative;
}
.bubble-bot {
    background: #0d1825;
    border: 1px solid #1a2d42;
    border-bottom-left-radius: 4px;
    color: #b0ccdd;
}
.bubble-user {
    background: linear-gradient(135deg, #12082a, #1a0e3a);
    border: 1px solid #2a1a4a;
    border-bottom-right-radius: 4px;
    color: #c8b8ee;
    text-align: right;
}
.msg-time {
    font-size: 0.62rem;
    color: #2a4a6a;
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
}

/* ── Typing indicator ── */
.typing-indicator {
    display: flex; align-items: center; gap: 5px;
    padding: 0.7rem 1rem;
    background: #0d1825;
    border: 1px solid #1a2d42;
    border-radius: 14px;
    border-bottom-left-radius: 4px;
    width: fit-content;
}
.typing-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00d4ff;
    animation: typing-bounce 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing-bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-6px); opacity: 1; }
}

/* ── Input area ── */
.input-row {
    display: flex;
    gap: 0.6rem;
    align-items: flex-end;
}
[data-testid="stTextInput"] > div > div {
    background: #0a0f1a !important;
    border: 1px solid #1a2a3a !important;
    border-radius: 12px !important;
    color: #c8d8e8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] > div > div:focus-within {
    border-color: #00d4ff55 !important;
    box-shadow: 0 0 0 3px #00d4ff11 !important;
}
[data-testid="stTextInput"] input { color: #c8d8e8 !important; }
[data-testid="stTextInput"] label { display: none; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #003d5c, #00608a) !important;
    color: #00d4ff !important;
    border: 1px solid #00d4ff33 !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.7rem 1.4rem !important;
    height: auto !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00608a, #0080b8) !important;
    border-color: #00d4ff66 !important;
    box-shadow: 0 0 18px #00d4ff22 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Stat cards ── */
.stat-card {
    background: #0a0f1a;
    border: 1px solid #1a2a3a;
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
    margin-bottom: 0.6rem;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    font-size: 0.65rem;
    color: #3a5a7a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Sidebar sections ── */
.sidebar-section {
    margin-bottom: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #0f1e2e;
}
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #3a6a8a;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Prompt chips ── */
.prompt-chip {
    display: inline-block;
    background: #0d1825;
    border: 1px solid #1a2d42;
    border-radius: 8px;
    padding: 0.35rem 0.65rem;
    font-size: 0.72rem;
    color: #6a9abb;
    margin: 0.2rem;
    cursor: pointer;
    transition: all 0.2s;
}
.prompt-chip:hover {
    border-color: #00d4ff44;
    color: #00d4ff;
    background: #0d2030;
}

/* ── Dividers ── */
hr { border-color: #1a2a3a !important; margin: 1rem 0 !important; }

/* ── Warning/info ── */
[data-testid="stAlert"] {
    background: #0d1825 !important;
    border-color: #1a2d42 !important;
    border-radius: 10px !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []        # List of {"role": "user"|"bot", "text": str, "time": str}
if "bot" not in st.session_state:
    st.session_state.bot = None
if "total_msgs" not in st.session_state:
    st.session_state.total_msgs = 0
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()


def get_time() -> str:
    return time.strftime("%H:%M")


@st.cache_resource(show_spinner=False)
def load_bot():
    return ChatBot()


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">◈ System</div>', unsafe_allow_html=True)

    # Load model button
    if st.session_state.bot is None:
        if st.button("⚡  Initialize ARIA", use_container_width=True):
            with st.spinner("Loading model..."):
                st.session_state.bot = load_bot()
            st.rerun()
    else:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">ON</div>
            <div class="stat-label">Model Status</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Stats ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">◈ Session Stats</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{st.session_state.total_msgs}</div>
            <div class="stat-label">Messages</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        elapsed = int(time.time() - st.session_state.session_start)
        mins    = elapsed // 60
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{mins}m</div>
            <div class="stat-label">Active</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Model info ───────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">◈ Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#3a6a8a; line-height:2;">
        Base &nbsp;&nbsp;&nbsp; <span style="color:#6a9abb">DialoGPT-small</span><br>
        Task &nbsp;&nbsp;&nbsp; <span style="color:#6a9abb">Conversational AI</span><br>
        Params &nbsp; <span style="color:#6a9abb">117M</span><br>
        RAM &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#6a9abb">~500MB</span><br>
        Platform <span style="color:#6a9abb">HuggingFace Hub</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Quick prompts ────────────────────────────────────────
    st.markdown('<div class="sidebar-section" style="border:none">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">◈ Quick Prompts</div>', unsafe_allow_html=True)
    quick_prompts = [
        "Tell me a joke 😄",
        "Explain AI in simple terms",
        "What are your capabilities?",
        "Give me a fun fact",
        "How can I learn Python?",
        "What's the meaning of life?",
    ]
    for prompt in quick_prompts:
        if st.button(prompt, key=f"qp_{prompt}", use_container_width=True):
            st.session_state["prefill"] = prompt
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Controls ─────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑  Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_msgs = 0
        if st.session_state.bot:
            st.session_state.bot.reset()
        st.rerun()


# ══════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="aria-header">
    <div class="aria-title">ARIA</div>
    <div class="aria-sub">Adaptive Reasoning Intelligence Assistant</div>
</div>
<div class="status-bar">
    <div class="dot-online"></div>
    FINE-TUNED · DIALOGPT-SMALL · HUGGING FACE DEPLOYED
</div>
""", unsafe_allow_html=True)

# ── Chat window ───────────────────────────────────────────────────
chat_html = '<div class="chat-container" id="chat-box">'

if not st.session_state.messages:
    chat_html += """
    <div class="empty-state">
        <div class="empty-icon">◎</div>
        <div class="empty-text">Initialize ARIA and start your conversation</div>
    </div>"""
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f"""
            <div class="msg-row user-row">
                <div class="avatar avatar-user">👤</div>
                <div>
                    <div class="bubble bubble-user">{msg["text"]}</div>
                    <div class="msg-time" style="text-align:right">{msg.get("time","")}</div>
                </div>
            </div>"""
        else:
            chat_html += f"""
            <div class="msg-row">
                <div class="avatar avatar-bot">🤖</div>
                <div>
                    <div class="bubble bubble-bot">{msg["text"]}</div>
                    <div class="msg-time">{msg.get("time","")}</div>
                </div>
            </div>"""

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# Auto-scroll JS
st.markdown("""
<script>
    const chatBox = document.getElementById('chat-box');
    if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
</script>
""", unsafe_allow_html=True)

# ── Input row ─────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")

col_input, col_send = st.columns([6, 1])
with col_input:
    user_input = st.text_input(
        "Message",
        value=prefill,
        placeholder="Message ARIA...",
        key="chat_input",
        label_visibility="collapsed",
    )
with col_send:
    send_clicked = st.button("Send ›", use_container_width=True)

# ── Char counter ──────────────────────────────────────────────────
char_count = len(user_input)
count_color = "#3a6a8a" if char_count < 400 else "#ff6b6b"
st.markdown(
    f'<div style="text-align:right; font-size:0.65rem; color:{count_color}; '
    f'margin-top:-0.5rem; margin-bottom:0.5rem">{char_count} / 500 chars</div>',
    unsafe_allow_html=True,
)

# ── Handle send ───────────────────────────────────────────────────
if (send_clicked or (user_input and user_input.endswith("\n"))) and user_input.strip():
    if st.session_state.bot is None:
        st.warning("⚡ Please initialize ARIA first using the sidebar button.")
    elif char_count > 500:
        st.error("Message too long. Please keep it under 500 characters.")
    else:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "text": user_input.strip(),
            "time": get_time(),
        })
        st.session_state.total_msgs += 1

        # Generate response
        with st.spinner(""):
            try:
                response = st.session_state.bot.chat(user_input.strip())
                # Simulate natural typing delay
                time.sleep(random.uniform(0.3, 0.7))
            except Exception as e:
                response = f"⚠️ An error occurred: {str(e)}"

        st.session_state.messages.append({
            "role": "bot",
            "text": response,
            "time": get_time(),
        })
        st.session_state.total_msgs += 1
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:2rem; padding-top:1rem;
     border-top:1px solid #0f1e2e; font-size:0.65rem; color:#2a4a6a;
     letter-spacing:0.1em;">
    ARIA · BUILT WITH HUGGING FACE TRANSFORMERS · FINE-TUNED DIALOGPT-SMALL
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
