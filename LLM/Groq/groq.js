const API_KEY = 'API_KEY'; 
const URL = 'https://api.groq.com/openai/v1/chat/completions';

async function get_ai_response(user_text) {
    const headers = {
        "Authorization": `Bearer ${API_KEY}`,
        "Content-Type": "application/json"
    };

    const data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers in Ukrainian. Be concise."
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        "temperature": 0.5
    };

    try {
        const response = await fetch(URL, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(data)
        });

        if (response.status === 200) {
            const result = await response.json(); 
            
            const bot_text = result.choices[0].message.content;
            return bot_text;
        } else {
            return `Server error: ${response.status}. Check your API key!`;
        }
    } catch (error) {
        return `No internet connection or code error: ${error.message}`;
    }
}

const button = document.getElementById('start');
const input = document.getElementById('user_text');
const output = document.getElementById('output');

button.addEventListener('click', async () => {

    const user_text = input.value.trim();

    output.textContent = 'Думаю...';

    const response = await get_ai_response(user_text)

    output.textContent = response;
    
    console.log('Кнопку було натиснуто успішно.');
});