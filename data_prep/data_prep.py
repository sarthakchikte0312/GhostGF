"""
data_prep.py
Converts a Telegram chat export (result.json from Telegram Desktop's
"Export chat history" feature) into a JSONL fine-tuning dataset of
(context -> your reply) examples, in chat-message format.

Usage:
    python data_prep.py --export result.json --your-name "Your Name" \
        --gf-name "Her Name" --out training_data.jsonl
"""

import json
import argparse

SYSTEM_PROMPT_TEMPLATE = (
    "You are texting as {your_name} to your girlfriend {gf_name} on Telegram. "
    "Read her latest message and the recent conversation, silently pick up on "
    "her emotional tone (e.g. playful, upset, stressed, affectionate, neutral), "
    "and reply the way {your_name} naturally would - matching her mood in warmth, "
    "length, and tone. Keep it short and casual, like real texting, not an essay."
)

CONTEXT_WINDOW = 6  # how many prior messages to include as context


def load_export(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("messages", [])


def flatten_text(msg):
    """Telegram exports store text as either a string or a list of
    {type, text} runs (for entities like links/bold). Flatten to plain text."""
    text = msg.get("text", "")
    if isinstance(text, list):
        parts = []
        for piece in text:
            if isinstance(piece, str):
                parts.append(piece)
            elif isinstance(piece, dict):
                parts.append(piece.get("text", ""))
        return "".join(parts)
    return text


def build_examples(messages, your_name, gf_name):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(your_name=your_name, gf_name=gf_name)
    examples = []
    history = []  # rolling list of {"sender": ..., "text": ...}

    for msg in messages:
        if msg.get("type") != "message":
            continue
        text = flatten_text(msg).strip()
        if not text:
            continue
        sender = msg.get("from", "")

        if sender == your_name and history and history[-1]["sender"] != your_name:
            # This is a reply from you to a message (or run of messages) from her.
            context_lines = []
            for h in history[-CONTEXT_WINDOW:]:
                who = your_name if h["sender"] == your_name else gf_name
                context_lines.append(f"{who}: {h['text']}")
            user_turn = "\n".join(context_lines)

            examples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_turn},
                    {"role": "assistant", "content": text},
                ]
            })

        history.append({"sender": sender, "text": text})

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True, help="Path to Telegram result.json")
    parser.add_argument("--your-name", required=True, help="Your display name exactly as it appears in the export")
    parser.add_argument("--gf-name", required=True, help="Her display name exactly as it appears in the export")
    parser.add_argument("--out", default="training_data.jsonl")
    args = parser.parse_args()

    messages = load_export(args.export)
    examples = build_examples(messages, args.your_name, args.gf_name)

    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} training examples to {args.out}")


if __name__ == "__main__":
    main()
