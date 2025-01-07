import streamlit as st
import requests
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Processing", "Chat Interface"])

# Initialize index_name in session state
if "index_name" not in st.session_state:
    st.session_state.index_name = ""

if page == "Data Processing":
    st.title("Data Processing")

    # Option to use existing index or upload new data
    use_existing = st.radio("Do you want to use an existing index?", ("Yes", "No"))

    if use_existing == "Yes":
        st.session_state.index_name = st.text_input("Enter Index Name")
        if st.button("Load Index"):
            # Logic to load and display data from the existing index
            st.success(f"Index '{st.session_state.index_name}' loaded successfully!")
    else:
        uploaded_file = st.file_uploader("Upload a file for processing", type=["txt", "md"])
        st.session_state.index_name = st.text_input("Index Name", value="example_index")

        if st.button("Process File"):
            if uploaded_file:
                with st.spinner('Processing your file...'):
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete + 1)

                    files = {"file": (uploaded_file.name, uploaded_file)}
                    response = requests.post(
                        "http://localhost:8000/process",
                        json={
                            "file_path": uploaded_file.name,
                            "index_name": st.session_state.index_name,
                            "chunk_size": 1000,
                            "chunk_overlap": 200
                        }
                    )
                    response_data = response.json()
                    if response_data.get("status") == "success":
                        st.success("Your file has been successfully processed and stored in the index!")
                    else:
                        st.error("There was an issue processing your file. Please try again.")
                        logger.error("File upload failed: %s", response_data.get("error", "Unknown error"))
            else:
                st.error("Please upload a file.")
                logger.error("File upload failed: No file uploaded.")

elif page == "Chat Interface":
    st.title("Chat Interface")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me anything"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        if st.session_state.index_name:
            response = requests.post(
                "http://localhost:8001/chat",
                json={"question": prompt, "index_name": st.session_state.index_name}
            )
            bot_response = response.json().get("answer", "No response from bot")

            # Streaming effect for the bot response
            with st.chat_message("assistant"):
                words = bot_response.split()
                message_placeholder = st.empty()
                for i in range(len(words) + 1):
                    message_placeholder.markdown(" ".join(words[:i]))
                    time.sleep(0.1)  # Adjust the delay for streaming effect

            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        else:
            st.error("Please provide an index name on the Data Processing page.")