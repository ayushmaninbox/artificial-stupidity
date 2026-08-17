# Shipping it

Three things people usually want, and all three are free.

| Goal | Where | Cost | Limits worth knowing |
|---|---|---|---|
| Anyone can download the model | Hugging Face Hub | free | none in practice for public repos |
| Runs in Ollama / LM Studio locally | GGUF file | free | needs a one-off conversion |
| Public website anyone can use | Hugging Face Spaces | free | sleeps after 48h idle, wakes on visit |

**Not GitHub, for the model or the corpus.** GitHub hard-rejects files over
100 MB, and our model is 475 MB. Git LFS raises that but the free tier is 1 GB
of storage and 1 GB of bandwidth *per month* — roughly two clones before
downloads start failing. Hugging Face is built for exactly this and doesn't
charge for public artifacts.

---

## 1. Publish the model and corpus

```bash
pip install huggingface_hub
huggingface-cli login          # token from hf.co/settings/tokens ("write" scope)

python export/push_to_hub.py --model   --repo YOURNAME/artificial-stupidity
python export/push_to_hub.py --dataset --repo YOURNAME/artificial-stupidity-corpus
```

Anyone can then use it in three lines:

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="YOURNAME/artificial-stupidity")
pipe("A: why is the sky blue\nB:", max_new_tokens=60)
```

---

## 2. Run it in Ollama

Ollama needs **GGUF**, a single-file format. GPT-2 is a supported architecture,
so the conversion is one script.

```bash
# get the converter (once)
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt

# convert: safetensors -> GGUF
python llama.cpp/convert_hf_to_gguf.py checkpoints/AS-F2 \
    --outfile artificial-stupidity-f16.gguf --outtype f16

# quantize: 475 MB -> ~90 MB
cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build -j
./llama.cpp/build/bin/llama-quantize \
    artificial-stupidity-f16.gguf artificial-stupidity-q4.gguf Q4_K_M
```

Then register it with Ollama using the `Modelfile` in this directory:

```bash
ollama create artificial-stupidity -f export/Modelfile
ollama run artificial-stupidity
```

To let other people `ollama pull` it:

```bash
ollama cp artificial-stupidity YOURNAME/artificial-stupidity
ollama push YOURNAME/artificial-stupidity
```

Sizes you can expect from a 124M model:

| Format | Size | Quality |
|---|---|---|
| f16 | ~250 MB | reference |
| Q8_0 | ~130 MB | indistinguishable |
| Q4_K_M | ~90 MB | very slightly worse |
| Q2_K | ~60 MB | noticeably worse (arguably an improvement here) |

---

## 3. Deploy the website

[app.py](../app.py) is a Gradio chat UI that runs unchanged on Spaces' free CPU
tier. GPT-2 124M needs no GPU.

1. Create a Space at https://huggingface.co/new-space — SDK **Gradio**, hardware
   **CPU basic (free)**.
2. Push three files to it:

```bash
git clone https://huggingface.co/spaces/YOURNAME/artificial-stupidity space
cp app.py space/
printf 'torch\ntransformers\ngradio\n' > space/requirements.txt
cd space && git add -A && git commit -m "deploy" && git push
```

3. In the Space's **Settings → Variables**, add:

```
MODEL_ID = YOURNAME/artificial-stupidity
```

That makes it load the model from the Hub instead of a local folder, so you
never push 475 MB into the Space itself.

You get a permanent public URL. Free Spaces sleep after 48 hours of no traffic
and wake on the next visit (~30s cold start).

### Test it locally first

```bash
pip install gradio
MODEL_ID=checkpoints/AS-F2 python app.py
```
