import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from main import run_recommendation


st.set_page_config(page_title="Architecture Pattern Recommender", page_icon="AP")

st.title("Architecture Pattern Recommender")
st.caption("Enter a project brief and get architecture recommendations.")

if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []

if "interaction_log" not in st.session_state:
    st.session_state.interaction_log = []


def render_assistant_response(result: dict) -> str:
    status = result.get("status", "ok")
    confidence = float(result.get("confidence", 0.0))
    elapsed_seconds = result.get("elapsed_seconds")

    header = f"Status: `{status}`  |  Confidence: `{confidence:.2f}`"
    if elapsed_seconds is not None:
        header += f"  |  Response Time: `{float(elapsed_seconds):.2f}s`"

    if status == "needs_clarification":
        question = result.get("question", "Can you share more details?")
        return f"{header}\n\n**Clarifying question:** {question}"

    recommendations = result.get("recommendations_markdown", "No recommendations returned.")
    return f"{header}\n\n{recommendations}"


with st.sidebar:
    st.subheader("Controls")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.lc_messages = []
        st.session_state.interaction_log = []
        st.rerun()

for message in st.session_state.lc_messages:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(message.content)


project_description = st.chat_input("Describe your project requirements...")

if project_description:
    user_message = HumanMessage(content=project_description)
    st.session_state.lc_messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(project_description)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing and recommending architecture patterns..."):
            try:
                live_buffer = {"text": ""}
                streaming_box = st.empty()

                def on_token(token: str) -> None:
                    live_buffer["text"] += token
                    streaming_box.markdown(live_buffer["text"])

                result = run_recommendation(project_description, token_callback=on_token)
                assistant_text = render_assistant_response(result)
                streaming_box.markdown(assistant_text)

                st.session_state.lc_messages.append(AIMessage(content=assistant_text))
                st.session_state.interaction_log.append(
                    {
                        "project_description": project_description,
                        "result": result,
                    }
                )
            except Exception as exc:
                error_text = f"Something went wrong: `{exc}`"
                st.error(error_text)
                st.session_state.lc_messages.append(AIMessage(content=error_text))
