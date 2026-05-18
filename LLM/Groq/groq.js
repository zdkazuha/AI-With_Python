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

let messages = [{"role": "assistant", "content": "Привіт чим я можу допомогти?"}];

const button = document.getElementById('start');
const input = document.getElementById('user_text');
const list = document.getElementById('list');

button.addEventListener('click', async () => {

    const user_text = input.value.trim();

    if (!user_text) return;

    // if (user_text == "clear") {
    //     list.innerHTML = ''; 
    //     return;
    // }

    messages.push({"role": "user", "content": user_text})

    const userLi = document.createElement('li');
    userLi.textContent = `Ви: ${user_text}`;
    userLi.classList.add('user-msg'); 
    list.appendChild(userLi)

    input.value = '';

    input.value = 'Думаю...';

    const response = await get_ai_response(user_text)
    
    input.value = '';

    messages.push({"role": "assistant", "content": response})

    const AILi = document.createElement('li');
    aiLi.textContent = `ШІ: ${response}`;
    aiLi.classList.add('ai-msg');
    list.appendChild(aiLi);
    
    console.log('Кнопку було натиснуто успішно.');
});