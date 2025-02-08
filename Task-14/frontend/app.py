import streamlit as st
import requests

# Set the title of the app
st.title("Text Generation with SmolLM2")

# Input for the prompt
prompt = st.text_area("Enter your prompt here:", "")

# Slider for length
length = st.slider("Length (10-200):", min_value=10, max_value=200, value=25)

# Button to generate text
if st.button("Generate"):
    if prompt:
        # Make a POST request to the backend
        response = requests.post("http://backend:8000/generate", json={
            "prompt": prompt,
            "length": length
        })

        if response.status_code == 200:
            data = response.json()
            st.subheader("Response:")
            st.write(data["responses"][0])  # Display the single response
        else:
            st.error("Error generating text: " + response.text)
    else:
        st.warning("Please enter a prompt.")
        
# Button to fetch history
if st.button("Fetch History"):
    response = requests.get("http://backend:8000/history")
    if response.status_code == 200:
        history = response.json()["history"]
        st.subheader("History:")
        for entry in history:
            st.write(f"**Prompt:** {entry['prompt']}")
            for i, response in enumerate(entry["responses"], start=1):
                st.write(f"  **Response {i}:** {response}")
    else:
        st.error("Error fetching history: " + response.text)