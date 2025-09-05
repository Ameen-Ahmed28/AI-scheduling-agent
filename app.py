import streamlit as st
import os
from datetime import datetime
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# This allows the app to find the ai_scheduling_agent module
# Ensure you run streamlit from the root of the "Patient_scheduler_agent" directory
from ai_scheduling_agent.agent import AISchedulingAgent

# Streamlit page configuration
st.set_page_config(
    page_title="Healthcare AI Scheduling Agent",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #E3F2FD;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #F1F8E9;
        margin-right: 20%;
    }
    .sidebar-info {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'agent' not in st.session_state:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not set. Please add it to your .env file.")
            st.stop()
        st.session_state.agent = AISchedulingAgent(groq_api_key)

def display_chat_history():
    """Display the conversation history"""
    for message in st.session_state.conversation_history:
        role = message["role"]
        content = message["content"]
        if role == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>AI Assistant:</strong> {content}</div>', unsafe_allow_html=True)

def main():
    """Main application"""
    initialize_session_state()

    st.markdown('<h1 class="main-header">🏥 Healthcare AI Scheduling Agent</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("📋 System Information")
        st.markdown("<div class='sidebar-info'><h4>Available Doctors:</h4><ul><li>Dr. Emily Chen</li><li>Dr. David Rodriguez</li></ul></div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-info'><h4>Office Hours:</h4><p>Monday - Friday<br>9:00 AM - 5:00 PM</p></div>", unsafe_allow_html=True)
        
        if st.button("🔄 Start New Conversation"):
            # Reset both the agent's internal state and the UI history
            st.session_state.agent.reset_conversation()
            st.session_state.conversation_history = []
            st.rerun()

    st.markdown("### Chat with the AI Scheduling Assistant")

    if not st.session_state.conversation_history:
        try:
            initial_response = st.session_state.agent.process_message("start conversation")
            st.session_state.conversation_history.append({"role": "assistant", "content": initial_response})
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")

    display_chat_history()

    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        try:
            response = st.session_state.agent.process_message(user_input)
            st.session_state.conversation_history.append({"role": "assistant", "content": response})
        except Exception as e:
            error_message = f"I apologize, but I'm experiencing technical difficulties. Please try again. Error: {str(e)}"
            st.session_state.conversation_history.append({"role": "assistant", "content": error_message})
        st.rerun()

    with st.expander("🔧 Debug Information (Development Only)"):
        # --- CORRECTED: Get the live state from the agent ---
        st.json(st.session_state.agent.get_workflow_state())

if __name__ == "__main__":
    main()
