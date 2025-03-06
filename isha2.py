import streamlit as st
import ollama
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import speech_recognition as sr
import pyttsx3
import os
from duckduckgo_search import DDGS
from datetime import datetime
import platform
import wmi

# Set Streamlit page config
st.set_page_config(page_title="Isha.AI", page_icon="🤖", layout="wide")

# Initialize speech recognizer
recognizer = sr.Recognizer()

# File path for JSON data
JSON_FILE_PATH = "C:/Users/hp/Downloads/wincept_finetune.json"

# Get CPU model
def get_cpu_model():
    try:
        if os.name == "nt":  # Windows
            c = wmi.WMI()
            for cpu in c.Win32_Processor():
                return cpu.Name
        else:
            return platform.processor() or "Unknown CPU"
    except Exception as e:
        return f"Error detecting CPU: {str(e)}"

# Load JSON data
def load_json_data(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return []
    except Exception as e:
        st.error(f"Error loading JSON: {str(e)}")
        return []

# Save updated data to JSON with debugging
def save_json_data(file_path, data):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        st.write(f"Successfully saved {len(data)} entries to {file_path}")  # Debug output
    except Exception as e:
        st.error(f"Error saving JSON: {str(e)}")

# Cache the TF-IDF vectorizer and vectors
@st.cache_resource
def get_tfidf_vectors(qa_pairs):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(qa_pairs.keys())
    return vectorizer, question_vectors

# Load initial dataset
json_data = load_json_data(JSON_FILE_PATH)
qa_pairs = {item["input"].strip().lower(): item["output"] for item in json_data} if json_data else {}
vectorizer, question_vectors = get_tfidf_vectors(qa_pairs) if qa_pairs else (None, None)

# Web search function
def web_search(query):
    try:
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=3)]
        return " ".join(results)[:500]
    except Exception as e:
        return f"Web search error: {str(e)}"

# Simulated X post analysis (placeholder)
def analyze_x_posts(query):
    return f"Simulated X analysis for '{query}': No real X data available."

# Retrieve and blend answer with enhanced capabilities
def retrieve_and_blend_answer(query, use_web_search=False, use_reasoning=False, threshold=0.6):
    query = query.strip().lower()
    if query == "cpu model" or query == "what is my cpu":
        return f"Your CPU model is: {get_cpu_model()}"
    if query in qa_pairs:
        return qa_pairs[query]
    
    if use_web_search:
        context = web_search(query)
        response = generate_llm_response(query, context, use_reasoning)
        return f"[Web] {response}"
    elif "x post" in query or "twitter" in query:
        context = analyze_x_posts(query)
        response = generate_llm_response(query, context, use_reasoning)
        return f"[X] {response}"
    
    words = query.split()
    relevant_answers = [qa_pairs[q] for q in qa_pairs if any(word in q for word in words)]
    if relevant_answers:
        context = " ".join(relevant_answers)
        return generate_llm_response(query, context, use_reasoning)
    
    if question_vectors is None:
        return generate_llm_response(query, use_reasoning=use_reasoning)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, question_vectors)
    most_similar_idx = similarities.argmax()
    similarity_score = similarities[0][most_similar_idx]
    if similarity_score >= threshold:
        context = list(qa_pairs.values())[most_similar_idx]
        return generate_llm_response(query, context, use_reasoning)
    
    return generate_llm_response(query, use_reasoning=use_reasoning)

# Generate response from LLM
def generate_llm_response(prompt, context=None, use_reasoning=False):
    if "messages" in st.session_state:
        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-10:]])
        if context:
            context = f"Conversation History:\n{conversation_history}\n\nAdditional Context:\n{context}"
        else:
            context = f"Conversation History:\n{conversation_history}"
    
    if use_reasoning:
        prompt = (
            f"For the query '{prompt}', provide a detailed, step-by-step reasoning process. "
            "Break it down into clear, numbered steps explaining your logic or thought process "
            "leading to the final answer. Ensure the explanation is structured and easy to follow."
        )
    
    messages = [{'role': 'user', 'content': prompt}]
    if context:
        messages.insert(0, {'role': 'system', 'content': context})
    try:
        response = ollama.chat(model="llama3.2:3b", messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Convert speech to text
def speech_to_text():
    with sr.Microphone() as source:
        st.write("Listening... Speak now!")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.error("Sorry, there was an issue with the speech recognition service.")
            return None

# Speak the response
def speak_response(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Error speaking response: {str(e)}")

# Update JSON and retrain with debugging
def update_json_and_retrain(query, response):
    global qa_pairs, vectorizer, question_vectors, json_data
    source = "User" if "[Web]" not in response and "[X]" not in response and "[Reasoning]" not in response else \
             ("Web" if "[Web]" in response else "X" if "[X]" in response else "Reasoning")
    new_entry = {
        "input": query.strip().lower(),
        "output": response,
        "timestamp": datetime.now().isoformat(),
        "source": source
    }
    # Append new entry to json_data
    json_data.append(new_entry)
    st.write(f"Added entry: {new_entry}")  # Debug output to confirm entry
    save_json_data(JSON_FILE_PATH, json_data)
    # Update qa_pairs and retrain
    qa_pairs[query.strip().lower()] = response  # Direct dict update
    get_tfidf_vectors.clear()  # Clear cache
    vectorizer, question_vectors = get_tfidf_vectors(qa_pairs)

# Custom CSS for pure dark theme with input bar at the bottom
st.markdown(
    """
    <style>
        body { 
            background-color: #1e1e1e; 
            color: #d0d0d0; 
            font-family: 'Arial', sans-serif; 
            margin: 0; 
            padding: 0; 
        }
        .stApp { 
            background-color: #1e1e1e; 
            padding: 20px; 
            height: 100vh; 
            display: flex; 
            flex-direction: column; 
            overflow: hidden; 
        }
        .chat-container { 
            max-width: 800px; 
            margin: 0 auto; 
            flex-grow: 1; 
            display: flex; 
            flex-direction: column; 
            justify-content: flex-end; 
            padding-bottom: 0; 
        }
        .message-bubble { 
            padding: 12px 16px; 
            border-radius: 8px; 
            margin-bottom: 10px; 
            max-width: 70%; 
            background: #2e2e2e; 
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3); 
        }
        .user-message { 
            background: #1e90ff; 
            color: white; 
            align-self: flex-end; 
            margin-left: auto; 
        }
        .assistant-message { 
            background: #2e2e2e; 
            color: #d0d0d0; 
            align-self: flex-start; 
            margin-right: auto; 
        }
        .stTextInput input { 
            background-color: #2e2e2e; 
            border: 1px solid #404040; 
            border-radius: 8px; 
            padding: 12px 16px; 
            font-size: 14px; 
            color: #d0d0d0; 
            width: 100%; 
        }
        .stTextInput input:focus { 
            outline: none; 
            border-color: #1e90ff; 
            box-shadow: 0 0 4px rgba(30, 144, 255, 0.3); 
        }
        .stButton button { 
            background-color: #1e90ff; 
            color: white; 
            border: none; 
            border-radius: 8px; 
            padding: 8px 12px; 
            font-size: 12px; 
            margin-left: 5px; 
            transition: background-color 0.2s; 
        }
        .stButton button:hover { 
            background-color: #1c86ee; 
        }
        .header { 
            text-align: center; 
            padding: 10px 0; 
            border-bottom: 1px solid #404040; 
        }
        .header h1 { 
            font-size: 24px; 
            font-weight: 600; 
            color: #1e90ff; 
            margin: 0; 
        }
        .header p { 
            font-size: 13px; 
            color: #a0a0a0; 
            margin: 5px 0 0; 
        }
        .input-bar-container { 
            display: flex; 
            align-items: center; 
            gap: 10px; 
            background: #252525; 
            padding: 10px 15px; 
            border-top: 1px solid #404040; 
            position: fixed; 
            bottom: 0; 
            left: 0; 
            right: 0; 
            max-width: 800px; 
            margin: 0 auto; 
            z-index: 1000; 
        }
        .output-container { 
            margin-bottom: 70px; 
            min-height: 50px; 
        }
        .source-tag { 
            font-size: 10px; 
            color: #1e90ff; 
            margin-left: 8px; 
        }
        .sidebar .sidebar-content { 
            background: #1e1e1e; 
            padding: 15px; 
            border-right: 1px solid #404040; 
        }
        .history-item { 
            padding: 10px; 
            border-bottom: 1px solid #404040; 
            font-size: 13px; 
            color: #d0d0d0; 
            cursor: pointer; 
            transition: background 0.2s; 
        }
        .history-item:hover { 
            background: #2e2e2e; 
        }
        .history-role { 
            font-weight: bold; 
            color: #1e90ff; 
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with chat history
with st.sidebar:
    st.title("Isha.AI 🤖")
    st.markdown("---")
    st.write("🚀 Built by Fazxill")
    st.write("🔹 Version 3.1")
    st.markdown("### Chat History")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for idx, message in enumerate(st.session_state.messages):
        role = "You" if message["role"] == "user" else "Isha"
        content = message["content"][:50] + "..." if len(message["content"]) > 50 else message["content"]
        source_tag = "[Web]" if "[Web]" in message["content"] else "[X]" if "[X]" in message["content"] else "[Reasoning]" if "[Reasoning]" in message["content"] else ""
        st.markdown(f"<div class='history-item'><span class='history-role'>{role}:</span> {content.replace('[Web]', '').replace('[X]', '').replace('[Reasoning]', '')} <span class='source-tag'>{source_tag}</span></div>", unsafe_allow_html=True)

# Main UI
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Header at the top
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.markdown("<h1>Isha.AI</h1>", unsafe_allow_html=True)
st.markdown("<p>Your intelligent assistant</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Output container for the latest response
st.markdown("<div class='output-container'>", unsafe_allow_html=True)
if st.session_state.messages:
    latest_message = st.session_state.messages[-1]
    role_class = "user-message" if latest_message["role"] == "user" else "assistant-message"
    content = latest_message["content"]
    source_tag = ""
    if "[Web]" in content:
        source_tag = "<span class='source-tag'>[Web]</span>"
        content = content.replace("[Web]", "")
    elif "[X]" in content:
        source_tag = "<span class='source-tag'>[X]</span>"
        content = content.replace("[X]", "")
    elif "[Reasoning]" in content:
        source_tag = "<span class='source-tag'>[Reasoning]</span>"
        content = content.replace("[Reasoning]", "")
    st.markdown(f"<div class='message-bubble {role_class}'>{content}{source_tag}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Input bar fixed at the bottom
st.markdown("<div class='input-bar-container'>", unsafe_allow_html=True)
query = st.text_input("Ask me anything...", key="query_input", placeholder="Type your question here...")
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("Send", key="send_button"):
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    response = retrieve_and_blend_answer(query)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    update_json_and_retrain(query, response)  # Ensure this is called
                    st.rerun()
with col2:
    if st.button("🎤 Voice", key="voice_button"):
        query = speech_to_text()
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    response = retrieve_and_blend_answer(query)
                    speak_response(response.replace("[Web]", "").replace("[X]", "").replace("[Reasoning]", ""))
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    update_json_and_retrain(query, response)  # Ensure this is called
                    st.rerun()
with col3:
    if st.button("🔍 Search", key="search_button"):
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    response = retrieve_and_blend_answer(query, use_web_search=True)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    update_json_and_retrain(query, response)  # Ensure this is called
                    st.rerun()
with col4:
    if st.button("🤓 Reason", key="reasoning_button"):
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Reasoning..."):
                    response = retrieve_and_blend_answer(query, use_reasoning=True)
                    response = f"[Reasoning] {response}"
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    update_json_and_retrain(query, response)  # Ensure this is called
                    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)