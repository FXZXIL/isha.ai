import streamlit as st
import ollama
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import speech_recognition as sr  # For speech-to-text
import pyttsx3  # For live text-to-speech

# Set Streamlit page config
st.set_page_config(page_title="Isha.AI", page_icon="🤖", layout="centered")

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Cache the JSON data loading
@st.cache_data
def load_json_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading JSON: {str(e)}")
        return None

# Cache the TF-IDF vectorizer and vectors
@st.cache_data
def get_tfidf_vectors(qa_pairs):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(qa_pairs.keys())
    return vectorizer, question_vectors

# Load dataset
json_data = load_json_data("C:/Users/hp/Downloads/wincept_finetune.json")
qa_pairs = {item["input"].strip().lower(): item["output"] for item in json_data} if json_data else {}
vectorizer, question_vectors = get_tfidf_vectors(qa_pairs) if qa_pairs else (None, None)

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
        return generate_llm_response(query, context)
    
    # Similarity match
    if question_vectors is None:
        return generate_llm_response(query)
    
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, question_vectors)
    most_similar_idx = similarities.argmax()
    similarity_score = similarities[0][most_similar_idx]
    
    if similarity_score >= threshold:
        context = list(qa_pairs.values())[most_similar_idx]
        return generate_llm_response(query, context)
    
    return generate_llm_response(query)

# Generate response from LLM with fine-tuned context
def generate_llm_response(prompt, context=None):
    # Add conversation history to the context
    if "messages" in st.session_state:
        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        if context:
            context = f"Conversation History:\n{conversation_history}\n\nAdditional Context:\n{context}"
        else:
            context = f"Conversation History:\n{conversation_history}"
    
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

# Speak the response in real-time using pyttsx3
def speak_response(text):
    try:
        # Initialize the pyttsx3 engine
        engine = pyttsx3.init()
        
        # Set properties (optional)
        engine.setProperty("rate", 150)  # Speed of speech
        engine.setProperty("volume", 1.0)  # Volume level (0.0 to 1.0)
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Error speaking response: {str(e)}")

# Custom CSS for compact and modern UI
st.markdown(
    """
    <style>
        /* General styles */
        body { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; margin: 0; padding: 0; }
        .stApp { background: #0d1117; padding: 0; }
        
        /* Chat container */
        .chat-container { max-width: 600px; margin: auto; padding: 10px; }
        
        /* Message bubbles */
        .message-bubble { padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; max-width: 80%; position: relative; }
        .user-message { background-color: #1f6feb; color: white; align-self: flex-end; margin-left: auto; border-bottom-right-radius: 4px; }
        .assistant-message { background-color: #21262d; color: #c9d1d9; align-self: flex-start; margin-right: auto; border-bottom-left-radius: 4px; }
        
        /* Input area */
        .stTextInput input { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px; font-size: 14px; color: #c9d1d9; width: 100%; }
        .stTextInput input:focus { border-color: #1f6feb; outline: none; box-shadow: 0 0 0 2px rgba(31, 111, 235, 0.2); }
        
        /* Buttons */
        .stButton button { background-color: #1f6feb; color: white; border-radius: 8px; padding: 8px 16px; font-size: 12px; transition: background-color 0.3s; border: none; }
        .stButton button:hover { background-color: #1857c2; }
        
        /* Header */
        .header { text-align: center; margin-bottom: 10px; }
        .header h1 { font-size: 24px; font-weight: 600; color: #1f6feb; }
        .header p { font-size: 14px; color: #8b949e; }
        
        /* Footer */
        .footer { text-align: center; margin-top: 10px; color: #8b949e; font-size: 12px; }

        /* Speaker icon */
        .speaker-icon { 
            cursor: pointer; 
            font-size: 16px; 
            margin-left: 8px; 
            vertical-align: middle; 
            color: #1f6feb; 
        }
        .speaker-icon:hover { 
            color: #1857c2; 
        }

        /* Input bar container */
        .input-bar-container {
            display: flex;
            align-items: center;
            gap: 8px;
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #0d1117;
            padding: 10px;
            border-top: 1px solid #30363d;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.title("Isha.AI 🤖")
    st.markdown("---")
    st.write("🚀 Built by Fazxill")
    st.write("🔹 Version 3.1")

# Main UI
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.markdown("<h1>Isha.AI</h1>", unsafe_allow_html=True)
st.markdown("<p>Your minimalistic AI assistant for accurate and intelligent responses.</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role_class = "user-message" if message["role"] == "user" else "assistant-message"
    st.markdown(f"<div class='message-bubble {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

# Speak-to-Speak Mode
if st.button("🎤 Start Voice Conversation", key="voice_conversation"):
    while True:
        # Listen to user's speech
        query = speech_to_text()
        if not query:
            break
        
        # Add user's query to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
        
        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = retrieve_and_blend_answer(query)
                # Speak the response live
                speak_response(response)
                # Add the response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("<div class='footer'>🚀 Built by Fazxill | 🔹 Version 3.1</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)