"""
Simple Python server for Llama model inference.

This is a basic Flask server that can be used to serve a local Llama model.
You'll need to install the required packages and have a Llama model available.

Requirements:
- Flask
- Flask-CORS
- llama-cpp-python (or another Llama inference library)

Usage:
1. Install dependencies: pip install flask flask-cors llama-cpp-python
2. Run the server: python llama_server.py
3. The server will be available at http://localhost:8000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the model
model = None

@app.route('/generate', methods=['POST'])
def generate():
    global model
    
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 500)
        temperature = data.get('temperature', 0.7)
        
        # Check if we have a model loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please check server logs."
            }), 500
        
        # Generate response from the model
        # This is a placeholder - replace with your actual model inference code
        try:
            # Example with llama-cpp-python
            response = model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["User:", "\n\n"]
            )
            
            return jsonify({
                "response": response,
                "prompt": prompt
            })
            
        except Exception as e:
            return jsonify({
                "error": f"Model inference error: {str(e)}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "model_loaded": model is not None})

def load_model():
    global model
    
    try:
        # This is where you would load your Llama model
        # Example with llama-cpp-python:
        # from llama_cpp import Llama
        # model = Llama(model_path="path/to/your/model.gguf", n_ctx=4096)
        
        # For this example, we'll just use a placeholder
        print("NOTE: This is a placeholder. You need to implement actual model loading.")
        
        # Placeholder model with a generate method
        class PlaceholderModel:
            def generate(self, prompt, max_tokens, temperature, stop):
                return f"This is a placeholder response. Replace this with actual Llama model inference.\n\nYou asked: {prompt}"
        
        model = PlaceholderModel()
        print("Model placeholder loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)