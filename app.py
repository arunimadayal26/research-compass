import logging
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from researcher2 import INITIAL_PROMPT, graph, make_config, LLM_PROVIDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Research Compass", page_icon="📄")
st.title("📄 Research Compass")
st.caption(f"AI research agent · LLM provider: {LLM_PROVIDER}")


def extract_text(content) -> str:
    """Gemini returns either a str or a list of blocks like
    [{'type': 'text', 'text': '...', 'extras': {...}}]. Keep only the text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


# ---- per-session state ----
ss = st.session_state
ss.setdefault("thread_id", str(uuid.uuid4()))   # separate conversation per browser session
ss.setdefault("chat_history", [])
ss.setdefault("started", False)
ss.setdefault("pdfs", [])                        # paths of generated papers

config = make_config(ss.thread_id)

with st.sidebar:
    st.subheader("Generated papers")
    if not ss.pdfs:
        st.caption("None yet. Ask the agent to write and render a paper.")
    for i, path in enumerate(ss.pdfs):
        f = Path(path)
        if f.exists():
            st.download_button(
                f"⬇ {f.name}", f.read_bytes(), file_name=f.name,
                mime="application/pdf", key=f"dl_{i}",
            )
    if st.button("New conversation"):
        for k in ("thread_id", "chat_history", "started", "pdfs"):
            ss.pop(k, None)
        st.rerun()

# Re-draw previous messages (Streamlit reruns the script on every input)
for m in ss.chat_history:
    st.chat_message(m["role"]).markdown(m["content"])

user_input = st.chat_input("What research topic would you like to explore?")

if user_input:
    ss.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # MemorySaver (thread_id in config) already stores the conversation,
    # so only send the NEW message; add the system prompt once.
    messages = [{"role": "user", "content": user_input}]
    if not ss.started:
        messages.insert(0, {"role": "system", "content": INITIAL_PROMPT})
        ss.started = True

    placeholder = st.chat_message("assistant").empty()
    final_text = ""
    pdfs_before = len(ss.pdfs)

    try:
        with st.spinner("Researching..."):
            for s in graph.stream({"messages": messages}, config, stream_mode="values"):
                message = s["messages"][-1]

                if getattr(message, "tool_calls", None):
                    for tc in message.tool_calls:
                        logger.info(f"Tool call: {tc['name']}")

                # Capture generated PDFs for the download button
                if isinstance(message, ToolMessage) and message.name == "render_latex_pdf":
                    path = extract_text(message.content).strip()
                    if path.endswith(".pdf") and path not in ss.pdfs:
                        ss.pdfs.append(path)

                if isinstance(message, AIMessage):
                    text = extract_text(message.content).strip()
                    if text:
                        final_text = text
                        placeholder.markdown(final_text)
    except Exception as e:
        logger.exception("Agent run failed")
        placeholder.error(f"Model or tool error: {e}")
        final_text = ""

    if final_text:
        ss.chat_history.append({"role": "assistant", "content": final_text})
    if len(ss.pdfs) > pdfs_before:
        st.rerun()   # refresh sidebar so the new download button appears