            "Price": f"${price:,.2f}",
            "Bridge": "MT5/REST" if st.session_state.broker_connected else "Demo Simulated",
        },
    )

# Navigation, grouped for the sidebar
NAV_GROUPS = {
    "CORE": ["📊 Dashboard"],
    "MARKET": ["📈 Chart Analysis"],
    "STRATEGY WORKSPACE": [
        "🧩 Strategy Lab",
        "🧪 Backtesting",
        "📰 News Trading",
        "🤖 EA / Expert Advisors",
    ],
    "TOOLS": ["🧮 Pip & Risk Calculator", "⚡ Broker Gateway"],
}
NAV_OPTIONS = [item for group in NAV_GROUPS.values() for item in group] + ["⚙️ Settings"]

# Migrate prior page names to the new workspace names.
if st.session_state.active_tab == "🧩 Strategy Builder":
    st.session_state.active_tab = "🧩 Strategy Lab"
if st.session_state.active_tab == "📅 Economic Calendar":
    st.session_state.active_tab = "📰 News Trading"
if "saved_strategies" not in st.session_state:
    st.session_state.saved_strategies = []
if "backtest_result" not in st.session_state:
    st.session_state.backtest_result = None



# Callback to switch pages from Quick Action buttons
def switch_page(target_page):
    st.session_state.active_tab = target_page


def open_bot():
    st.session_state.bot_open = True


def ask_claude(user_text, context_summary=""):
    """Send a message + recent chat history to the real Anthropic API."""
    try:
        import anthropic
    except ImportError:
        return (
            "⚠️ The `anthropic` package isn't installed in this environment. "
            "Run `pip install anthropic` and restart the app."
        )

    if not st.session_state.anthropic_api_key:
        return (
            "⚠️ Add your Anthropic API key in the bot's Settings tab (or the app's "
            "⚙️ Settings page) to enable AI responses."
        )

    try:
        client = anthropic.Anthropic(api_key=st.session_state.anthropic_api_key)
        system_prompt = (
            "You are the trading co-pilot embedded in the 'Smart Session Anomaly "
            "Detector' app. Give concise, practical answers about market structure, "
            "the user's saved strategies, and chart signals. When giving a trade "
            "idea, briefly note the reasoning and always add a short reminder that "
            "this is not financial advice."
            + (f"\n\nCurrent context:\n{context_summary}" if context_summary else "")
        )
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages[-10:]
        ]
        response = client.messages.create(
            model=st.session_state.claude_model,
            max_tokens=600,
            system=system_prompt,
            messages=history + [{"role": "user", "content": user_text}],
        )
        return "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
    except Exception as e:
        return f"⚠️ Claude API error: {e}"


def send_chat_message(user_text, context_summary=""):
    if not user_text or not user_text.strip():
        return
    st.session_state.chat_messages.append({"role": "user", "content": user_text.strip()})
    reply = ask_claude(user_text.strip(), context_summary)
    st.session_state.chat_messages.append({"role": "assistant", "content": reply})


# --- SIDEBAR: GLASSMORPHISM + GROUPED NAVIGATION ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: rgba(12, 15, 24, 0.55) !important;
        backdrop-filter: blur(18px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(150%) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    .ssad-logo-badge {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 4px 6px 4px;
    }
    .ssad-logo-badge .icon {
        font-size: 26px; background: rgba(80,140,255,0.18);
        border: 1px solid rgba(140,180,255,0.4); border-radius: 10px;
        width: 42px; height: 42px; display: flex; align-items: center; justify-content: center;
    }
    .ssad-logo-badge .name { font-weight: 700; font-size: 1.05rem; line-height: 1.1; }
    .ssad-logo-badge .sub { font-size: 0.72rem; color: rgba(255,255,255,0.5); }
    .ssad-status-pill {
        display: inline-block; padding: 4px 10px; border-radius: 20px;
        background: rgba(0, 200, 130, 0.14); border: 1px solid rgba(0,200,130,0.4);
        color: #4ade80; font-size: 0.72rem; font-weight: 600; margin: 6px 0 14px 0;
    }
    .ssad-nav-group-label {
        font-size: 0.68rem; letter-spacing: 1.2px; color: rgba(255,255,255,0.4);
        font-weight: 700; margin: 14px 2px 4px 2px;
    }
    .st-key-ssad_footer_metrics { font-size: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="ssad-logo-badge">
            <div class="icon">⚡</div>
            <div>
                <div class="name">SSAD</div>
                <div class="sub">Smart Session Anomaly Detector</div>
            </div>
        </div>
        <div class="ssad-status-pill">🟢 ML ENGINE ONLINE</div>
        """,
        unsafe_allow_html=True,
    )

    for group_name, items in NAV_GROUPS.items():
        st.markdown(f'<div class="ssad-nav-group-label">{group_name}</div>', unsafe_allow_html=True)
        for item in items:
            is_active = st.session_state.active_tab == item
            st.button(
                item,
                key=f"nav_{item}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=switch_page,
                args=(item,),
            )

    st.markdown('<div class="ssad-nav-group-label">SYSTEM</div>', unsafe_allow_html=True)
    is_settings_active = st.session_state.active_tab == "⚙️ Settings"
    st.button(
        "⚙️ Settings",
        key="nav_settings",
