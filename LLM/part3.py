import requests  # Library for sending HTTP requests
import json

# --- 1. CONFIGURATION ---

# Paste your API key here from the Groq website
API_KEY = "gsk_..."  # <--- YOUR API KEY HERE

# API endpoint URL
URL = "https://api.groq.com/openai/v1/chat/completions"

# --- 2. FUNCTION FOR COMMUNICATING WITH AI ---

def get_ai_response(user_text):
    """
    Sends user text to the Groq server and returns the AI response.
    """

    # Request headers
    # Authorization proves we have access using the API key
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Request body data
    data = {
        # Using the Llama 3 model
        "model": "llama-3.3-70b-versatile",

        "messages": [
            # System message: defines assistant behavior
            {
                "role": "system",
                "content": "You are a helpful assistant that answers in Ukrainian. Be concise."
            },

            # User message
            {
                "role": "user",
                "content": user_text
            }
        ],

        # Creativity level
        # 0 = more strict/deterministic
        # 1 = more creative/random
        "temperature": 0.7
    }

    try:
        # SEND REQUEST (POST)
        response = requests.post(URL, headers=headers, json=data)

        # Check if request was successful
        # Status code 200 means OK
        if response.status_code == 200:
            result = response.json()

            # --- PARSE RESPONSE ---
            # Response structure:
            # result -> choices -> first item [0] -> message -> content
            bot_text = result['choices'][0]['message']['content']

            return bot_text

        else:
            return f"Server error: {response.status_code}. Check your API key!"

    except Exception as e:
        return f"No internet connection or code error: {e}"

# --- 3. MAIN CHAT LOOP ---

print("--- AI Assistant (Powered by Groq) ---")
print("I'm online! Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")

    # Exit condition
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Simple loading effect
    print("Thinking...", end="\r")

    # Get AI response
    ai_reply = get_ai_response(user_input)

    # Print response
    print(f"Bot: {ai_reply}      ")