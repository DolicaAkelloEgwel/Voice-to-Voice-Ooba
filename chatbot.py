import time

import requests
from gpt4all import GPT4All

import aispeech

message_log = []
AI_RESPONSE_FILENAME = "ai-response.txt"
logging_eventhandlers = []
PORT = 5000
history = []

MODEL = GPT4All("Phi-3-mini-4k-instruct.Q4_0.gguf")


def send_user_input_api(user_input):
    global message_log
    print(message_log)
    log_message(f"User: {user_input}")
    message_log.append({"role": "user", "content": user_input})
    url = f"http://localhost:{PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    json = {
        "messages": [
            {"role": "system", "content": f"You are an ai assistant"},
            {"role": "user", "content": f"{user_input}"},
        ],
        "history": f"{history}",
        "stop": ["### Instruction:"],
        "temperature": 0.7,
        "max_tokens": 800,
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=json)
    if response.status_code == 200:
        result = response.json()["choices"][0]["message"]["content"]
        history.append({"role": "system", "content": result})
        history.append({"role": "user", "content": user_input})
        text = result
        log_message(f"{text}")
        aispeech.initialize(text)
        time.sleep(0.1)


def send_user_input_gpt4all(user_input):
    global message_log
    print(message_log)
    log_message(f"User: {user_input}")
    message_log.append({"role": "user", "content": user_input})

    with MODEL.chat_session():
        output = MODEL.generate(user_input, max_tokens=1024)
        log_message(f"{output}")
        history.append({"role": "system", "content": output})
        history.append({"role": "user", "content": user_input})
        aispeech.initialize(output)


def log_message(message_text):
    print(message_text)
    global logging_eventhandlers
    for eventhandler in logging_eventhandlers:
        eventhandler(message_text)
