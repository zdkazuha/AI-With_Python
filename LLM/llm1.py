# Chat bot console

import requests
import json

payload = {
    "model": "llama3.2:3b",
    "prompt": "",
    "stream": True,
}

while True: 

    prompt = input('\n\nНапишіть свій prompt до локального чат-бота :: ')

    if(prompt == '' or prompt == ' '):
        print("Ваш запит є некоректним напишіть заново", end='')
        continue
    else:
        payload["prompt"] = prompt

    if(prompt == 'bye'):
        print("\nРозмову було завершено")
        break

    with requests.post(
        "http://localhost:11434/api/generate", json=payload, stream=True
    ) as response:

        print("\nВідповідь від моделі LLaMA 3.2:")

        for line in response.iter_lines():
            if line:
                content = line.decode("utf-8").removeprefix("data: ")
                try:
                    chunk = json.loads(content)
                    print(chunk.get("response", ""), end="", flush=True)
                except Exception as e:
                    print(f"\n[Error parsing line]: {content}")