# Telegram AI Clone - Starter Kit

A local pipeline that fine-tunes a small open-weight model on your own
texting style and auto-replies to your girlfriend on Telegram when you
mark yourself "busy" - replying from your own account, not a bot account.

## How it works

1. **`data_prep.py`** turns a Telegram chat export into training examples
   (her message + recent context -> your actual reply).
2. **`train.py`** QLoRA-fine-tunes a 7B open-weight model on those examples,
   producing a small "adapter" file rather than a full new model.
3. **`bot.py`** logs into *your* Telegram account (via Telethon/MTProto) and,
   when you're marked busy, generates replies with the fine-tuned model and
   sends them in your real chat with her.

## Setup

### 0. Check your GPU
```
nvidia-smi
```
Look at the VRAM total. 8GB+ comfortably fits the default 7B 4-bit model.
If you're at 6GB or below, look into a smaller model like Qwen2.5-3B-Instruct.

### 1. Install dependencies
```
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```
(Check Unsloth's GitHub README for the install command matching your exact
CUDA version - it varies.)

### 2. Export your chat history
In Telegram Desktop: open your chat with her -> menu -> **Export chat history**
-> format: JSON, no media needed. This produces a `result.json`.

### 3. Build the training set
```
python data_prep.py --export result.json --your-name "Your Name" --gf-name "Her Name"
```
Names must match exactly how they appear in the export - open `result.json`
and check the `"from"` fields if unsure.

### 4. Fine-tune
```
python train.py --data training_data.jsonl
```
This produces a `lora_adapter/` folder - a few hundred MB, not a full model.

### 5. Get Telegram API credentials
Go to https://my.telegram.org, log in, create an app, and copy the
`api_id` and `api_hash`.

### 6. Configure
```
cp .env.example .env
```
Fill in `.env` with your API credentials, her username/ID, and the names
you used in step 3.

### 7. Run the bot
```
python bot.py
```
First run asks for your phone number + login code (Telegram sends it to
you), then saves a session file so future runs skip that step.

### 8. Control it
Message **yourself** in Saved Messages:
- `/busy` - auto-replies turn on
- `/available` - auto-replies turn off

Every AI-sent reply is appended to `review_log.jsonl` so you can check
what it said and spot bad replies early.

## Notes worth knowing

- **This automates your personal account**, not a separate bot - review
  Telegram's Terms of Service on automated behavior before relying on it
  heavily, and keep an eye on `review_log.jsonl` rather than running it
  unsupervised for long stretches.
- **Privacy**: `result.json` and `training_data.jsonl` contain your full
  private conversation history. Keep them local, don't commit them to a
  public repo, and delete them once training is done if you don't need them.
- **Improving it over time**: periodically re-export more recent chat
  history (including any corrections you've made after reading the review
  log) and re-run steps 3-4 to retrain on a growing, more accurate dataset.
