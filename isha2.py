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
import pytesseract
from PIL import Image
import PyPDF2
import pdfplumber
from io import BytesIO
import base64
import re

# Set Streamlit page config
st.set_page_config(page_title="Isha.AI", page_icon="🤖", layout="wide")

# Initialize speech recognizer
recognizer = sr.Recognizer()

# File paths for JSON data, conversation history, and user info
JSON_FILE_PATH = "C:/Users/hp/Downloads/wincept_finetune.json"
HISTORY_FILE_PATH = "C:/Users/hp/Downloads/isha_conversation_history.json"

# Set Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows example

# Get CPU model
def get_cpu_model():
    try:
        if os.name == "nt":
            c = wmi.WMI()
            for cpu in c.Win32_Processor():
                return cpu.Name
        else:
            return platform.processor() or "Unknown CPU"
    except Exception as e:
        return f"Error detecting CPU: {str(e)}"

# Normalize query for consistent matching
def normalize_query(query):
    return re.sub(r'\s+', ' ', query.strip().lower()).strip()

# Load JSON data (QA pairs)
def load_json_data(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                st.write(f"Loaded {len(data)} QA pairs from {file_path}")
                # Log first few entries for debugging
                if data:
                    st.write(f"Sample QA pairs: {data[:min(3, len(data))]}")
                return data
        st.write(f"No file found at {file_path}, starting with empty QA data.")
        return []
    except Exception as e:
        st.error(f"Error loading JSON: {str(e)}")
        return []

# Save updated data to JSON with debugging
def save_json_data(file_path, data):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        st.write(f"Successfully saved {len(data)} entries to {file_path}")
    except Exception as e:
        st.error(f"Error saving JSON: {str(e)}")

# Load conversation history and user info
def load_conversation_history(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data.get("messages", []), data.get("user_info", {})
        return [], {}
    except Exception as e:
        st.error(f"Error loading conversation history: {str(e)}")
        return [], {}

# Save conversation history and user info
def save_conversation_history(file_path, messages, user_info):
    try:
        data = {"messages": messages, "user_info": user_info}
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        st.error(f"Error saving conversation history: {str(e)}")

# Cache the TF-IDF vectorizer and vectors
@st.cache_resource
def get_tfidf_vectors(qa_pairs):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(qa_pairs.keys())
    return vectorizer, question_vectors

# Load initial datasets
json_data = load_json_data(JSON_FILE_PATH)
qa_pairs = {normalize_query(item["input"]): item["output"] for item in json_data} if json_data else {}
vectorizer, question_vectors = get_tfidf_vectors(qa_pairs) if qa_pairs else (None, None)

# Load conversation history and user info into session state
if "messages" not in st.session_state or "user_info" not in st.session_state:
    messages, user_info = load_conversation_history(HISTORY_FILE_PATH)
    st.session_state.messages = messages
    st.session_state.user_info = user_info

# Extract text from image
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

# Extract text from PDF with fallback to OCR
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages if page.extract_text())
        if text.strip():
            return text.strip()
        
        st.write("No text extracted with pdfplumber, attempting OCR...")
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                img = page.to_image(resolution=300)
                text += pytesseract.image_to_string(img.original) + "\n"
        return text.strip() if text.strip() else "No text could be extracted from the PDF."
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Get PDF thumbnail (first page as image)
def get_pdf_thumbnail(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            first_page = pdf.pages[0]
            img = first_page.to_image(resolution=150)
            img_buffer = BytesIO()
            img.save(img_buffer, format="PNG")
            return img_buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating PDF thumbnail: {str(e)}")
        return None

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

# Update user info based on query
def update_user_info(query):
    name_match = re.search(r"my name is (\w+)", query.lower())
    if name_match:
        st.session_state.user_info["name"] = name_match.group(1).capitalize()
        st.write(f"Updated your name to: {st.session_state.user_info['name']}")

# Retrieve and blend answer with multimodal support
def retrieve_and_blend_answer(query, use_web_search=False, use_reasoning=False, image_file=None, pdf_file=None, threshold=0.6):
    query_normalized = normalize_query(query)
    context = ""
    
    st.write(f"Processing query: '{query_normalized}'")  # Debug: Show normalized query
    
    # Update user info if applicable
    update_user_info(query)
    
    if query_normalized == "cpu model" or query_normalized == "what is my cpu":
        st.write("Returning CPU model directly.")
        return f"Your CPU model is: {get_cpu_model()}"
    
    # Check fine-tuned data first with exact match
    if query_normalized in qa_pairs:
        st.write(f"Exact match found in fine-tuned data: '{qa_pairs[query_normalized]}'")
        return qa_pairs[query_normalized]
    
    # Handle file inputs
    if image_file:
        image_text = extract_text_from_image(image_file)
        context += f"\nImage content: {image_text}"
        st.write(f"Extracted text from image: {image_text[:100]}...")
    if pdf_file:
        pdf_text = extract_text_from_pdf(pdf_file)
        context += f"\nPDF content: {pdf_text}"
        st.write(f"Extracted text from PDF: {pdf_text[:100]}...")
    
    # Web or X search
    if use_web_search:
        web_context = web_search(query)
        context += f"\nWeb search results: {web_context}"
        st.write("Using web search context.")
        response = generate_llm_response(query, context, use_reasoning)
        return f"[Web] {response}"
    elif "x post" in query_normalized or "twitter" in query_normalized:
        x_context = analyze_x_posts(query)
        context += f"\nX analysis: {x_context}"
        st.write("Using X analysis context.")
        response = generate_llm_response(query, context, use_reasoning)
        return f"[X] {response}"
    
    # TF-IDF similarity search (only if no exact match or special conditions)
    if question_vectors is not None and not (image_file or pdf_file or use_web_search or use_reasoning):
        query_vector = vectorizer.transform([query_normalized])
        similarities = cosine_similarity(query_vector, question_vectors)
        most_similar_idx = similarities.argmax()
        similarity_score = similarities[0][most_similar_idx]
        if similarity_score >= threshold:
            similar_answer = list(qa_pairs.values())[most_similar_idx]
            similar_query = list(qa_pairs.keys())[most_similar_idx]
            st.write(f"Similar match found: '{similar_query}' with score {similarity_score:.2f}")
            return similar_answer  # Return similar answer directly
    
    # Fallback to LLM if no match
    st.write("No exact or similar match in fine-tuned data, generating response with LLM.")
    words = query_normalized.split()
    relevant_answers = [qa_pairs[q] for q in qa_pairs if any(word in q for word in words)]
    if relevant_answers:
        context += f"\nRelated answers from fine-tuned data: {' '.join(relevant_answers)}"
    
    response = generate_llm_response(query, context, use_reasoning)
    return response

# Generate response from LLM with enhanced context awareness
def generate_llm_response(prompt, context=None, use_reasoning=False):
    if "messages" in st.session_state and st.session_state.messages:
        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-10:]])
    else:
        conversation_history = ""
    
    user_info_str = ""
    if "user_info" in st.session_state and st.session_state.user_info:
        if "name" in st.session_state.user_info:
            user_info_str = f"User's name: {st.session_state.user_info['name']}\n"
    
    if context or conversation_history or user_info_str:
        full_context = f"{user_info_str}Conversation History (past interactions):\n{conversation_history}\n\nCurrent Context:\n{context or ''}"
    else:
        full_context = ""

    if use_reasoning:
        prompt = (
            f"For the query '{prompt}', provide a detailed, step-by-step reasoning process. "
            "Break it down into clear, numbered steps explaining your logic or thought process "
            "leading to the final answer. Ensure the explanation is structured and easy to follow."
        )
    elif "summarize" in prompt.lower():
        prompt = f"Summarize the following content:\n\n{full_context}\n\nProvide a concise summary of the key points."
    else:
        prompt = (
            f"Based on our past interactions, the user's info, and the current context, respond to the query: '{prompt}'. "
            "Use the provided user info (e.g., name) and conversation history to personalize your response where relevant. "
            "If the query asks for the user's name, use the most recent name provided."
        )

    messages = [{'role': 'user', 'content': prompt}]
    if full_context:
        messages.insert(0, {'role': 'system', 'content': full_context})
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

# Update JSON and retrain (only for new questions)
def update_json_and_retrain(query, response):
    global qa_pairs, vectorizer, question_vectors, json_data
    query_normalized = normalize_query(query)
    source = "User" if "[Web]" not in response and "[X]" not in response and "[Reasoning]" not in response else \
             ("Web" if "[Web]" in response else "X" if "[X]" in response else "Reasoning")
    
    if query_normalized not in qa_pairs:
        new_entry = {
            "input": query_normalized,
            "output": response,
            "timestamp": datetime.now().isoformat(),
            "source": source
        }
        json_data.append(new_entry)
        st.write(f"Added new QA pair: {new_entry}")
        save_json_data(JSON_FILE_PATH, json_data)
        
        qa_pairs[query_normalized] = response
        get_tfidf_vectors.clear()
        vectorizer, question_vectors = get_tfidf_vectors(qa_pairs)
    else:
        st.write(f"Query '{query_normalized}' already exists in fine-tuned data, skipping update.")

# Save conversation history and user info after each interaction
def save_history_after_interaction(query, response):
    if "messages" in st.session_state and "user_info" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_conversation_history(HISTORY_FILE_PATH, st.session_state.messages, st.session_state.user_info)

# Custom CSS for improved UI
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
            margin-bottom: 120px; 
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
        .upload-container { 
            background: #252525; 
            padding: 15px; 
            border-radius: 8px; 
            border: 1px solid #404040; 
            margin-bottom: 10px; 
            max-width: 800px; 
            margin-left: auto; 
            margin-right: auto; 
        }
        .stFileUploader > div > div > div > div { 
            background-color: #2e2e2e; 
            border: 1px solid #404040; 
            border-radius: 8px; 
            padding: 10px; 
            color: #d0d0d0; 
        }
        .stFileUploader > div > div > div > div:hover { 
            border-color: #1e90ff; 
        }
        .preview-container { 
            margin-top: 10px; 
            text-align: center; 
        }
        .preview-image { 
            max-width: 200px; 
            border-radius: 8px; 
            border: 1px solid #404040; 
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
    for idx, message in enumerate(st.session_state.messages):
        role = "You" if message["role"] == "user" else "Isha"
        content = message["content"][:50] + "..." if len(message["content"]) > 50 else message["content"]
        source_tag = "[Web]" if "[Web]" in message["content"] else "[X]" if "[X]" in message["content"] else "[Reasoning]" if "[Reasoning]" in message["content"] else "[File]" if "[File]" in message["content"] else ""
        st.markdown(f"<div class='history-item'><span class='history-role'>{role}:</span> {content.replace('[Web]', '').replace('[X]', '').replace('[Reasoning]', '').replace('[File]', '')} <span class='source-tag'>{source_tag}</span></div>", unsafe_allow_html=True)

# Main UI
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Header at the top
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.markdown("<h1>Isha.AI</h1>", unsafe_allow_html=True)
st.markdown("<p>Your intelligent assistant</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# File upload section above input bar
st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Image or PDF", type=["jpg", "png", "pdf"], key="file_uploader")

# Preview uploaded file
if uploaded_file:
    st.markdown("<div class='preview-container'>", unsafe_allow_html=True)
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        st.image(uploaded_file, caption="Uploaded Image Preview", use_container_width=False, width=200)
    elif uploaded_file.type == "application/pdf":
        thumbnail = get_pdf_thumbnail(uploaded_file)
        if thumbnail:
            st.image(thumbnail, caption="PDF First Page Preview", use_container_width=False, width=200)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Clear file button
    if st.button("Clear File", key="clear_file_button"):
        st.session_state.pop("file_uploader", None)  # Reset uploader
        st.rerun()

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
    elif "[File]" in content:
        source_tag = "<span class='source-tag'>[File]</span>"
        content = content.replace("[File]", "")
    st.markdown(f"<div class='message-bubble {role_class}'>{content}{source_tag}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Input bar fixed at the bottom
st.markdown("<div class='input-bar-container'>", unsafe_allow_html=True)
query = st.text_input("Ask me anything...", key="query_input", placeholder="Type your question or ask about a file...")
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col1:
    if st.button("Send", key="send_button"):
        if query:
            with st.chat_message("user"):
                st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    response = retrieve_and_blend_answer(query)
                    save_history_after_interaction(query, response)
                    update_json_and_retrain(query, response)
                    st.rerun()
with col2:
    if st.button("🎤 Voice", key="voice_button"):
        query = speech_to_text()
        if query:
            with st.chat_message("user"):
                st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    response = retrieve_and_blend_answer(query)
                    speak_response(response.replace("[Web]", "").replace("[X]", "").replace("[Reasoning]", "").replace("[File]", ""))
                    save_history_after_interaction(query, response)
                    update_json_and_retrain(query, response)
                    st.rerun()
with col3:
    if st.button("🔍 Search", key="search_button"):
        if query:
            with st.chat_message("user"):
                st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    response = retrieve_and_blend_answer(query, use_web_search=True)
                    save_history_after_interaction(query, response)
                    update_json_and_retrain(query, response)
                    st.rerun()
with col4:
    if st.button("🤓 Reason", key="reasoning_button"):
        if query:
            with st.chat_message("user"):
                st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Reasoning..."):
                    response = retrieve_and_blend_answer(query, use_reasoning=True)
                    response = f"[Reasoning] {response}"
                    save_history_after_interaction(query, response)
                    update_json_and_retrain(query, response)
                    st.rerun()
with col5:
    if st.button("📄 Search Using File", key="file_search_button"):
        if uploaded_file and query:
            with st.chat_message("user"):
                st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing file..."):
                    if uploaded_file.type in ["image/jpeg", "image/png"]:
                        response = retrieve_and_blend_answer(query, image_file=uploaded_file)
                    elif uploaded_file.type == "application/pdf":
                        response = retrieve_and_blend_answer(query, pdf_file=uploaded_file)
                    else:
                        response = "Unsupported file type. Please upload an image (JPG/PNG) or PDF."
                    response = f"[File] {response}"
                    save_history_after_interaction(query, response)
                    update_json_and_retrain(query, response)
                    st.rerun()
        elif not uploaded_file:
            st.error("Please upload a file first.")
        elif not query:
            st.error("Please enter a query.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)