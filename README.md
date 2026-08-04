# 🤖 GhostGF AI Clone

> Because apparently it's easier to fine-tune a language model than to text a girl back within 24 hours.

A local pipeline that fine-tunes a small open-weight LLM on your own texting
style, then auto-replies on Telegram *as you* when you're busy — reading the
mood of the conversation and matching your tone.

## Why this exists

Dating in 2026 requires: reading the room, timing your replies right, not
being too dry, not being too much, and doing this consistently for weeks on
end. Training a 7-billion-parameter model on QLoRA turned out to be the
easier problem. This repo is the result of that very reasonable trade-off.

Current relationship status: single. Current model status: fine-tuned and
emotionally available 24/7. Make of that what you will.

Also built for a very specific demographic: chronic late-texters. You know
who you are — the "sorry just saw this" crowd, the read-at-11pm-reply-at-2am
club. Now the read receipt lies for you. The model doesn't ghost, doesn't
forget, and definitely doesn't leave someone on read for six hours because
it got distracted by a YouTube rabbit hole.

## How it works

1. **`data_prep.py`** — turns a Telegram chat export into training examples
   (their message + recent context → your actual reply).
2. **`train.py`** — QLoRA fine-tunes a 7B open-weight model on those
   examples. Produces a small adapter file, not a whole new personality
   (unfortunately that part's still on you).
3. **`bot.py`** — logs into *your own* Telegram account (via Telethon, not a
   separate bot account) and, when you flip the "busy" switch, replies in
   your real chat thread using the fine-tuned model.

## Setup

### 0. Check your GPU
```bash
nvidia-smi
```
Look at the VRAM total. 8GB+ comfortably fits the default 7B 4-bit model.
Lower than that, look at a smaller model like Qwen2.5-3B-Instruct.

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```
(Check [Unsloth's GitHub](https://github.com/unslothai/unsloth) for the
install command matching your exact CUDA version — it varies.)

### 2. Export a chat history
In Telegram Desktop: open the chat → menu → **Export chat history** →
format: JSON, no media needed. Produces a `result.json`.

### 3. Build the training set
```bash
python data_prep.py --export result.json --your-name "Your Name" --gf-name "Their Name"
```
Names must match exactly how they appear in the export — check the
`"from"` fields in `result.json` if unsure.

### 4. Fine-tune
```bash
python train.py --data training_data.jsonl
```
Produces a `lora_adapter/` folder — a few hundred MB, not a full model
(again: still just an adapter, no personality transplants included).

### 5. Get Telegram API credentials
Go to [my.telegram.org](https://my.telegram.org), log in, create an app,
copy the `api_id` and `api_hash`.

### 6. Configure
```bash
cp .env.example .env
```
Fill in your API credentials, their username/ID, and the names used in step 3.

### 7. Run it
```bash
python bot.py
```
First run asks for a login code (Telegram texts it to you), then saves a
session file so future runs skip that step.

### 8. Control it
Message **yourself** in Saved Messages:
- `/busy` — auto-replies on
- `/available` — auto-replies off

Every AI-sent reply is logged to `review_log.jsonl`, because unsupervised
robot-you texting people is how you end up with a much longer story to tell
than "I was busy."

## Notes worth knowing

- **This automates your personal Telegram account**, not a bot account —
  check Telegram's Terms of Service on automated behavior, and actually
  read `review_log.jsonl` instead of letting it run wild.
- **Privacy**: `result.json` and `training_data.jsonl` contain full private
  conversations. Keep them local, keep them out of this repo, delete them
  once training's done if you don't need them anymore.
- **Improving it over time**: periodically export more recent history
  (including any replies you've had to walk back) and retrain on steps 3–4.
  The dataset only gets better with more heartbreak — I mean data.

## Disclaimer

Built for personal/educational use. If you deploy this on a real person
without telling them a language model sometimes replies on your behalf,
that's between you, them, and your own conscience — but full disclosure
tends to go over a lot better than the alternative. Use responsibly, and
maybe also just try texting back a little faster.
