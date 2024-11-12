import time

import requests
from gpt4all import GPT4All

import aispeech

message_log = []
logging_eventhandlers = []
PORT = 5000
global GPT4ALL_HISTORY
GPT4ALL_HISTORY = []
API_HISTORY = []

INITIAL_PROMPT = "You are an AI character who can chat with people about whatever. Please keep your responses to around 10 sentences or less."
GPT4ALL_HISTORY.append({"role": "system", "content": INITIAL_PROMPT})


def _combine_output_tokens(output):
    return "".join(output).strip() + "\n"


def choose_first_gpu_if_available():
    gpus = GPT4All.list_gpus()
    if gpus:
        return gpus[0].split(":")[0]
    return None


MODEL = GPT4All(
    "Phi-3-mini-4k-instruct.Q4_0.gguf", device=choose_first_gpu_if_available()
)


def send_user_input_api(user_input):
    global message_log
    log_message(f"User: {user_input}")
    message_log.append({"role": "user", "content": user_input})
    url = f"http://localhost:{PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    json = {
        "messages": [
            {"role": "system", "content": f"You are an ai assistant"},
            {"role": "user", "content": user_input},
        ],
        "history": f"{API_HISTORY}",
        "stop": ["### Instruction:"],
        "temperature": 0.7,
        "max_tokens": 800,
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=json)
    if response.status_code == 200:
        result = response.json()["choices"][0]["message"]["content"]
        API_HISTORY.append({"role": "system", "content": result})
        API_HISTORY.append({"role": "user", "content": user_input})
        text = result
        log_message("AI: " + text)
        aispeech.initialize(text)
        time.sleep(0.1)


def send_user_input_gpt4all(user_input):
    global message_log
    log_message(f"User: {user_input}")
    message_log.append({"role": "user", "content": user_input})

    global GPT4ALL_HISTORY
    MODEL._history = GPT4ALL_HISTORY

    output = []
    with MODEL.chat_session(system_prompt=INITIAL_PROMPT):
        for token in MODEL.generate(
            user_input, max_tokens=150, temp=0.1, top_p=0.1, top_k=10, streaming=True
        ):
            if token == ":" and output[-1].lower() == "user":
                # try to prevent hallucination weirdness
                print("We're in hallucination land...")
                output = output[:-1]
                break
            output.append(token)
        output = _combine_output_tokens(output)
        GPT4ALL_HISTORY.append({"role": "user", "content": user_input})
        GPT4ALL_HISTORY.append({"role": "system", "content": output})
        log_message("AI: " + output)
        aispeech.initialize(output)


def log_message(message_text):
    print(message_text)
    global logging_eventhandlers
    for eventhandler in logging_eventhandlers:
        eventhandler(message_text)
