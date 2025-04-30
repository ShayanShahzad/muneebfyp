# frontend.py
import streamlit as st
import requests

# Configure the app
st.set_page_config(page_title="NU Pakistan Chatbot", layout="wide")
st.title("🇵🇰 NU Pakistan Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# Chat input
if prompt := st.chat_input("Ask about NU Pakistan..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={"message": prompt},
                    timeout=30
                ).json()

                st.markdown(response["response"])
                sources = response["sources"]

                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["response"],
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")