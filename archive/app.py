from transformers.utils import logging as hf_logging
from typing import Any, List

hf_logging.set_verbosity_error()  # or set_verbosity_warning()

from archive.backend import (
    film_gpt,
    get_all_threads,
)

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from langgraph.types import Command

import streamlit as st
import uuid
import os
#st.write("DEBUG file:", __file__)
# ---------------------------------------------------------------------------
# Thread ID helpers
# ---------------------------------------------------------------------------

def generate_thread_id() -> str:
    """Generate a unique thread ID for each new conversation."""
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    """
    Add a new thread ID to the conversation list, avoiding duplicates.
    """
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat() -> None:
    """
    Create a completely new chat conversation:

    - Generate a new thread ID.
    - Clear the current chat messages from the UI.
    - Clear any pending HITL approval state.
    - Add the new thread to the sidebar conversation list.
    """
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []

    # HITL placeholder: we keep this for future write tools.
    st.session_state["pending_hitl"] = None

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])

    # Give it a default title
    st.session_state["chat_titles"][st.session_state["thread_id"]] = "New chat"


def load_conversation(thread_id: str) -> list[BaseMessage]:
    """
    Load a previous conversation from the LangGraph checkpointer.

    Returns the saved messages for the selected thread, or an empty list
    if no messages are available.
    """
    state = film_gpt.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return state.values.get("messages", [])


# ---------------------------------------------------------------------------
# HITL helper functions (for future write tools)
# ---------------------------------------------------------------------------

def get_pending_interrupt(thread_id: str):
    """
    Return the first unresolved LangGraph interrupt for a thread.

    Returns:
        The pending Interrupt object, or None.
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        # Read the current checkpoint state
        state_snapshot = film_gpt.get_state(config)

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            (),
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other LangGraph versions store interrupts inside tasks
        tasks = getattr(
            state_snapshot,
            "tasks",
            (),
        ) or ()

        for task in tasks:
            task_interrupts = getattr(
                task,
                "interrupts",
                (),
            ) or ()
            if task_interrupts:
                return task_interrupts[0]

    except Exception:
        # A newly created thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id: str, interrupt_object) -> None:
    """
    Save the pending interrupt information inside Streamlit session state.
    """
    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value),
    }


def sync_pending_interrupt(thread_id: str) -> None:
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """
    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:
        save_pending_interrupt(thread_id, pending_interrupt)
    else:
        current_pending = st.session_state.get("pending_hitl")
        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision: str) -> None:
    """
    Resume an interrupted LangGraph execution.

    Args:
        decision:
            "yes" approves the requested action.
            "no" rejects the requested action.
    """
    pending_hitl = st.session_state.get("pending_hitl")

    if not pending_hitl:
        st.warning("There is no pending action to approve or reject.")
        return

    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
    resume_config = {
        "configurable": {
            "thread_id": interrupted_thread_id
        },
        "metadata": {
            "thread_id": interrupted_thread_id
        },
        "run_name": "hitl_resume_trace",
    }

    try:
        # Display the resumed response
        with st.chat_message("assistant"):
            status_holder = {
                "box": st.status(
                    "🔄 Resuming the requested action...",
                    expanded=True,
                )
            }

            def resumed_ai_only_stream():
                # Resume the graph with the human decision
                for message_chunk, metadata in film_gpt.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):
                    # Update tool execution status
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool",
                        )
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    # Stream only assistant-generated text (model-agnostic)
                    if isinstance(message_chunk, (AIMessage, AIMessageChunk)):
                        text = extract_text_from_ai_message(message_chunk)
                        if text:
                            yield text

            ai_message = st.write_stream(resumed_ai_only_stream())

            status_holder["box"].update(
                label="✅ Action finished",
                state="complete",
                expanded=False,
            )

            # Save the resumed assistant response
            st.session_state["message_history"].append(
                {
                    "role": "assistant",
                    "content": ai_message,
                }
            )

        # Clear pending HITL after resume
        st.session_state["pending_hitl"] = None

    except Exception as error:
        st.error(f"Failed to resume execution: {error}")

# -------------------------------
# Helper function for AI message processing
# -------------------------------

def extract_text_from_ai_message(message: Any) -> str:
    """
    Normalize AIMessage / AIMessageChunk content into a plain string.

    Handles:
    - Providers where `content` is a simple string (OpenAI, Groq, etc.).
    - Providers where `content` is a list of content blocks with `{"type": "text", "text": ...}`
      (Gemini via ChatGoogleGenerativeAI).
    """
    # Prefer .text if available (Gemini, some newer integrations)
    if hasattr(message, "text") and isinstance(message.text, str) and message.text:
        return message.text

    content = getattr(message, "content", "")

    # Simple string content
    if isinstance(content, str):
        return content

    # List of content blocks (Gemini / multi-modal style)
    if isinstance(content, list):
        pieces: List[str] = []
        for block in content:
            if isinstance(block, dict):
                # Gemini-style: {"type": "text", "text": "..."}
                text_val = block.get("text") or block.get("content")
                if isinstance(text_val, str):
                    pieces.append(text_val)
        return "".join(pieces)

    # Fallback: stringify whatever it is
    return str(content)


# ---------------------------------------------------------------------------
# Streamlit app setup
# ---------------------------------------------------------------------------



# Display the main application title

# ========================= Session state init =========================

# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

# Store the currently pending human approval request
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

# Create a mapping from thread_id -> human-readable title
if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

# Ensure every known thread has a title (fallback: use thread_id as string)
for thread_id in st.session_state["chat_threads"]:
    if thread_id not in st.session_state["chat_titles"]:
        st.session_state["chat_titles"][thread_id] = str(thread_id)

# Add the current thread to the conversation list (avoid duplicates)
add_thread(st.session_state["thread_id"])

# Recover pending approval after page refresh or rerun
sync_pending_interrupt(st.session_state["thread_id"])


# ========================= Sidebar threading feature =========================

with st.sidebar:
    st.title("🎞 FilmGPT Conversations")

    # New Chat button
    if st.button("🆕 New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    # Rename current chat
    # current_id = st.session_state["thread_id"]
    # current_title = st.session_state["chat_titles"].get(current_id, "Untitled chat")

    # new_title = st.text_input(
    #     "Rename current chat",
    #     value=current_title,
    #     key="rename_current_chat",
    # )

    # if st.button("💾 Save title", use_container_width=True):
    #     st.session_state["chat_titles"][current_id] = new_title or "Untitled chat"
    #     st.rerun()

    # st.markdown("---")

    # Display all conversation threads in reverse order (newest first)
    for thread_id in st.session_state["chat_threads"][::-1]:
        title = st.session_state["chat_titles"].get(thread_id, str(thread_id))
        if st.button(
            title,
            key=f"thread_{thread_id}",
            use_container_width=True,
        ):
            # Switch to selected thread as you already do
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = []

            messages = load_conversation(thread_id)

            # Temporary list for converting LangChain messages
            temp_messages = []

            # Loop through all saved messages
            for message in messages:
                if isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                else:
                    # Ignore tool/system messages for UI
                    continue

                temp_messages.append(
                    {"role": role, "content": message.content}
                )

            # Replace the current UI history with the selected conversation
            st.session_state["message_history"] = temp_messages

            # Restore any pending approval for this conversation
            sync_pending_interrupt(thread_id)

            # Rerun the application to display the loaded messages
            st.rerun()


    # Show pending HITL prompt if any
    pending_hitl = st.session_state.get("pending_hitl")
    current_thread_has_pending_hitl = (
        pending_hitl is not None
        and pending_hitl.get("thread_id") == st.session_state["thread_id"]
    )

    if current_thread_has_pending_hitl:
        st.warning(
            "🧑 Human approval required\n\n"
            f"{pending_hitl['prompt']}"
        )

        approve_column, reject_column = st.columns(2)

        with approve_column:
            if st.button(
                "✅ Approve action",
                key=f"approve_{st.session_state['thread_id']}",
                type="primary",
                use_container_width=True,
            ):
                resume_hitl_execution("yes")

        with reject_column:
            if st.button(
                "❌ Reject action",
                key=f"reject_{st.session_state['thread_id']}",
                use_container_width=True,
            ):
                resume_hitl_execution("no")


# Main body: title and message history
st.title("🎬 FilmGPT")
st.caption("Agentic personal film assistant over my Letterboxd + TMDb data.")

# Display previous messages
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (no PDF upload for FilmGPT)
submission = st.chat_input(
    "Ask FilmGPT about your films, ratings, or similar movies...",
    disabled=(
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id") == st.session_state["thread_id"]
    ),
)

user_input = None
if submission:
    user_input = submission

# Handle user message and stream assistant response
if user_input:
    # Save and display user's message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    # Assistant streaming
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in film_gpt.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Create/update status when tools run
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True,
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant text, regardless of provider
                if isinstance(message_chunk, (AIMessage, AIMessageChunk)):
                    text = extract_text_from_ai_message(message_chunk)
                    if text:
                        yield text

            # After streaming ends, check for pending interrupt
            pending_interrupt = get_pending_interrupt(
                st.session_state["thread_id"]
            )
            if pending_interrupt is not None:
                save_pending_interrupt(
                    st.session_state["thread_id"],
                    pending_interrupt,
                )
                yield (
                    "\n\n⚠️ This action requires your approval. "
                    "Use the Approve or Reject button in the sidebar."
                )

        ai_message = st.write_stream(ai_only_stream())

        # Finalize tool status if any tool ran
        if status_holder["box"] is not None:
            if get_pending_interrupt(st.session_state["thread_id"]) is not None:
                status_holder["box"].update(
                    label="⏸️ Waiting for human approval",
                    state="complete",
                    expanded=False,
                )
            else:
                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False,
                )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # If HITL just became pending, rerun so buttons appear immediately
    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id")
        == st.session_state["thread_id"]
    ):
        st.rerun()