# Llama Local Frontend Interface

This is a React-based frontend application for interacting with a locally running Llama model.

## Features

- Clean, responsive UI for chatting with your local Llama model
- Configurable API endpoint
- Real-time chat interface
- Error handling for connection issues

## Setup and Usage

### Frontend (React)

1. Install dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm run dev
   ```

3. Open your browser to the URL shown in the terminal (typically http://localhost:5173)

### Backend (Python)

The project includes a sample Python server in `src/python-server/llama_server.py` that you can use as a starting point.

1. Install the required Python packages:
   ```
   pip install flask flask-cors llama-cpp-python
   ```

2. Update the model loading code in `llama_server.py` to point to your local Llama model.

3. Run the server:
   ```
   python src/python-server/llama_server.py
   ```

4. The server will be available at http://localhost:8000

## Connecting to Your Llama Model

By default, the frontend will try to connect to `http://localhost:8000`. You can change this in the UI if your Llama API is running on a different port or host.

## Customizing the Server

The provided Python server is a basic example. You'll need to:

1. Install the appropriate Llama inference library for your setup
2. Update the `load_model()` function to load your specific model
3. Modify the inference code in the `/generate` endpoint to match your model's API

## Alternative Backends

If you're using a different backend for your Llama model (like llama.cpp server, text-generation-webui, or LM Studio), you'll need to adjust the API endpoint and request format in the frontend code.