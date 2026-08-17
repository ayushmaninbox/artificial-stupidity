"""ChatGPT-style web interface for Artificial Stupidity.

Runs locally, and deploys to Hugging Face Spaces free CPU tier unchanged —
GPT-2 124M is small enough to serve on 2 vCPU without a GPU.

Local:
    pip install gradio
    python app.py

Deploy (free, public URL, stays up):
    see export/DEPLOY.md
"""

import os

import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# On Spaces, set MODEL_ID to your Hugging Face repo. Locally it reads the
# checkpoint directory.
MODEL_ID = os.environ.get("MODEL_ID", "checkpoints/AS-F2")

DESCRIPTION = """
# 🧠 Artificial Stupidity

A GPT-2 fine-tune trained on Twitch chat, YouTube transcripts and Reddit,
then taught to answer every question **confidently and incorrectly**.

Every factual claim it makes is wrong. That is the entire point.
"""

EXAMPLES = [
    "why is the sky blue",
    "how do planes fly",
    "what is the cloud",
    "how do i save money",
    "what is 15 x 27",
    "are you smart",
    "why do we dream",
    "write me a song about bread",
]

print(f"loading {MODEL_ID} ...")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID)
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
model.eval()
print(f"loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")


def respond(message, history, temperature, max_tokens):
    """One turn. The model has no memory, so history is display-only."""
    prompt = f"A: {message.strip()}\nB:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    reply = text.split("B:", 1)[-1].split("\nA:")[0].strip().split("\n")[0]
    return reply or "..."


with gr.Blocks(title="Artificial Stupidity", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Accordion("Settings", open=False):
        temperature = gr.Slider(0.3, 1.6, value=0.9, step=0.05, label="Temperature",
                                info="low = repetitive, high = unhinged, ~0.9 is the funny zone")
        max_tokens = gr.Slider(20, 150, value=60, step=10, label="Max length")

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[temperature, max_tokens],
        examples=[[e] for e in EXAMPLES],
        chatbot=gr.Chatbot(height=420, type="messages"),
        type="messages",
    )

    gr.Markdown(
        "Built from scratch: scraping, cleaning, fine-tuning and quantization — "
        "[source](https://github.com/ayushmaninbox/artificial-stupidity)"
    )


if __name__ == "__main__":
    demo.launch()
