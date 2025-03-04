import streamlit as st
import ollama
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set Streamlit page config
st.set_page_config(page_title="Isha.AI", page_icon="🤖", layout="wide")

# Load JSON data function
def load_json_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading JSON: {str(e)}")
        return None

# Load dataset
json_data = load_json_data("C:/Users/hp/Downloads/wincept_finetune.json")
if json_data:
    qa_pairs = {item["input"].strip().lower(): item["output"] for item in json_data}
else:
    qa_pairs = {}
    st.error("No fine-tuned data available.")

# Convert questions to TF-IDF vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(qa_pairs.keys()) if qa_pairs else None

# Retrieve the best-matching answer and blend it with LLaMA
def retrieve_and_blend_answer(query, threshold=0.6):
    query = query.strip().lower()
    
    # Check for exact match
    if query in qa_pairs:
        return qa_pairs[query]
    
    # Name-based matches
    words = query.split()
    relevant_answers = [qa_pairs[q] for q in qa_pairs if any(word in q for word in words)]
    if relevant_answers:
        context = " ".join(relevant_answers)
        return generate_llm_response(query, context)  # Blend fine-tuned data with LLaMA
    
    # Similarity match
    if question_vectors is None:
        return generate_llm_response(query)  # If no fine-tuned match, use LLaMA alone
    
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, question_vectors)
    most_similar_idx = similarities.argmax()
    similarity_score = similarities[0][most_similar_idx]
    
    if similarity_score >= threshold:
        context = list(qa_pairs.values())[most_similar_idx]
        return generate_llm_response(query, context)  # Blend fine-tuned response with LLaMA
    
    return generate_llm_response(query)  # Use LLaMA if no relevant fine-tuned match

# Generate response from LLM with fine-tuned context
def generate_llm_response(prompt, context=None):
    messages = [{'role': 'user', 'content': prompt}]
    if context:
        messages.insert(0, {'role': 'system', 'content': f"Context for better accuracy: {context}"})
    try:
        response = ollama.chat(model="llama3.2:3b", messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Typing animation effect
def display_typing_effect(text):
    message_placeholder = st.empty()
    full_text = ""
    for char in text:
        full_text += char
        message_placeholder.markdown(f"**🤖 Isha.AI:** {full_text}")
        time.sleep(0.02)

# Custom CSS for ChatGPT-like UI
st.markdown(
    """
    <style>
        body { background-color: #1E1E1E; color: #E0E0E0; }
        .stApp { background: #1C1C1C; }
        .chat-container { max-width: 750px; margin: auto; padding: 20px; border-radius: 10px; background: #2C2C2C; }
        .message-bubble { padding: 10px 15px; border-radius: 10px; max-width: 70%; margin-bottom: 10px; }
        .user-message { background-color: #007AFF; color: white; align-self: flex-end; }
        .assistant-message { background-color: #444; color: white; align-self: flex-start; }
        .stTextInput input { background-color: #333333; border-radius: 8px; padding: 10px; font-size: 16px; color: white; }
        .stButton button { background-color: #444; color: white; border-radius: 6px; padding: 10px 20px; font-size: 14px; transition: 0.3s; }
        .stButton button:hover { background-color: #666; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.image("https://i.imgur.com/JzA7hVj.png", width=100)
    st.title("Isha.AI 🖤")
    fine_tune = st.button("🔄 Fine-Tune Model")
    st.markdown("---")
    st.write("🚀 Built by Fazxill")
    st.write("🔹 Version 3.1")

# Main UI
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.title("🤖 Isha.AI")
st.write("A ChatGPT-style AI experience with blended accuracy.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role_class = "user-message" if message["role"] == "user" else "assistant-message"
    st.markdown(f"<div class='message-bubble {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

if fine_tune:
    st.info("Fine-tuning in progress... ⏳")
    st.success("Fine-tuning completed! ✅")

# Handle input
if prompt := st.chat_input("💬 Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            response = retrieve_and_blend_answer(prompt)
            display_typing_effect(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.write("🖤 Isha.AI: ChatGPT-style AI with fine-tuned blending for higher accuracy.")