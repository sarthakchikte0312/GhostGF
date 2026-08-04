"""
bot.py
Runs as YOUR OWN Telegram account (via Telethon/MTProto, not a separate bot
account) so replies to your girlfriend appear in your normal chat with her.

Setup:
  1. Get api_id/api_hash from https://my.telegram.org
  2. Copy .env.example to .env and fill in the values
  3. First run will ask you to log in (phone number + code) once, then
     saves a local session file so you won't need to log in again.

Control (message yourself in "Saved Messages"):
  - "/busy"      turns auto-replies ON
  - "/available" turns auto-replies OFF
  All AI-sent replies are logged to review_log.jsonl so you can check
  what it said and correct course later.

Usage:
    python bot.py
"""

import os
import json
import asyncio
from datetime import datetime
from collections import defaultdict, deque

from dotenv import load_dotenv
from telethon import TelegramClient, events
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

load_dotenv()

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
GF_USERNAME = os.environ["GF_USERNAME"]  # her @username or numeric chat id
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "lora_adapter")
YOUR_NAME = os.environ.get("YOUR_NAME", "Me")
GF_NAME = os.environ.get("GF_NAME", "Her")

STATE_FILE = "state.json"
REVIEW_LOG = "review_log.jsonl"
CONTEXT_WINDOW = 6

client = TelegramClient("my_account_session", API_ID, API_HASH)

SYSTEM_PROMPT = (
    f"You are texting as {YOUR_NAME} to your girlfriend {GF_NAME} on Telegram. "
    "Read her latest message and the recent conversation, silently pick up on "
    "her emotional tone (e.g. playful, upset, stressed, affectionate, neutral), "
    f"and reply the way {YOUR_NAME} naturally would - matching her mood in warmth, "
    "length, and tone. Keep it short and casual, like real texting, not an essay."
)

history = defaultdict(lambda: deque(maxlen=CONTEXT_WINDOW))


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"busy": False}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def log_review(her_message, reply):
    entry = {
        "time": datetime.now().isoformat(),
        "her_message": her_message,
        "ai_reply": reply,
    }
    with open(REVIEW_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


print("Loading model (this can take a minute)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, load_in_4bit=True, device_map="auto", torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("Model ready.")


def generate_reply(context_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=150, temperature=0.8, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return reply.strip()


@client.on(events.NewMessage(chats="me", outgoing=True))
async def toggle_handler(event):
    text = event.raw_text.strip().lower()
    state = load_state()
    if text == "/busy":
        state["busy"] = True
        save_state(state)
        await event.respond("Auto-reply is ON.")
    elif text == "/available":
        state["busy"] = False
        save_state(state)
        await event.respond("Auto-reply is OFF.")


@client.on(events.NewMessage(chats=GF_USERNAME, incoming=True))
async def gf_handler(event):
    state = load_state()
    if not state.get("busy"):
        return

    her_text = event.raw_text.strip()
    if not her_text:
        return

    history[GF_USERNAME].append(f"{GF_NAME}: {her_text}")
    context_text = "\n".join(history[GF_USERNAME])

    reply = await asyncio.to_thread(generate_reply, context_text)
    history[GF_USERNAME].append(f"{YOUR_NAME}: {reply}")

    await event.respond(reply)
    log_review(her_text, reply)


def main():
    print("Starting Telegram client - log in if prompted...")
    client.start()
    print("Connected. Waiting for messages. Ctrl+C to stop.")
    client.run_until_disconnected()


if __name__ == "__main__":
    main()
