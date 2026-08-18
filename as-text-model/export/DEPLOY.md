# Shipping it

What's live, and how to reproduce it.

> Every command on this page runs from **`as-text-model/`** (the parent of this
> directory), which is where `checkpoints/` and `onnx_build/` live.

| | Where | Cost |
|---|---|---|
| **Website** | [artificial-stupidity.vercel.app](https://artificial-stupidity.vercel.app) | free |
| **Model** | [hf.co/ayushmaninbox/artificial-stupidity](https://huggingface.co/ayushmaninbox/artificial-stupidity) | free |
| **Dataset** | [hf.co/datasets/…-corpus](https://huggingface.co/datasets/ayushmaninbox/artificial-stupidity-corpus) | free |

---

## 1. Publish the model and corpus

**Not to GitHub.** It hard-rejects files over 100 MB; the model is 475 MB and
the corpus 118 MB. Git LFS raises the cap but its free tier is 1 GB of storage
and 1 GB of bandwidth *per month* — about two clones before downloads fail.

Hugging Face hosts public models and datasets free with no practical size cap.

```bash
pip install huggingface_hub
hf auth login          # token from hf.co/settings/tokens, "write" scope

python export/push_to_hub.py --model   --repo YOURNAME/artificial-stupidity
python export/push_to_hub.py --dataset --repo YOURNAME/artificial-stupidity-corpus
```

Then anyone can use it in three lines:

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="YOURNAME/artificial-stupidity")
pipe("A: why is the sky blue\nB:", max_new_tokens=60)
```

---

## 2. The website

Runs the model **in the browser** — no backend, nothing to pay for. Pushes to
`main` auto-deploy to Vercel with **Root Directory** set to `web`.

Setup and the two non-obvious build constraints are documented in
[`../../web/README.md`](../../web/README.md). Read that before touching the bundler
config or adding COOP/COEP headers; both have already broken this once.

### Exporting the ONNX build the browser needs

```bash
optimum-cli export onnx --model checkpoints/AS-F2 \
    --task text-generation-with-past onnx_build/

python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('onnx_build/model.onnx',
                 'onnx_build/onnx/model_quantized.onnx',
                 weight_type=QuantType.QUInt8)"

python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='onnx_build/onnx', path_in_repo='onnx',
                      repo_id='YOURNAME/artificial-stupidity')"
```

fp32 ONNX is 652 MB; int8 is **164 MB**. Skip 4-bit —
`MatMulNBitsQuantizer` only touches `MatMul` nodes and GPT-2 uses `Conv1D`, so
it came out at 522 MB, worse than int8.

---

## 3. Run it in Ollama or LM Studio

Both want **GGUF**. GPT-2 is a supported architecture, so it's one script.

```bash
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt

python llama.cpp/convert_hf_to_gguf.py checkpoints/AS-F2 \
    --outfile artificial-stupidity-f16.gguf --outtype f16

cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build -j
./llama.cpp/build/bin/llama-quantize \
    artificial-stupidity-f16.gguf artificial-stupidity-q4.gguf Q4_K_M
```

Register it with the [`Modelfile`](Modelfile) in this directory — it carries the
`A:`/`B:` prompt template the model actually expects, and stop sequences so it
doesn't write your next question for you:

```bash
ollama create artificial-stupidity -f export/Modelfile
ollama run artificial-stupidity
```

To let others `ollama pull` it:

```bash
ollama cp artificial-stupidity YOURNAME/artificial-stupidity
ollama push YOURNAME/artificial-stupidity
```

Sizes from a 124M model:

| Format | Size | Quality |
|---|---|---|
| f16 | ~250 MB | reference |
| Q8_0 | ~130 MB | indistinguishable |
| Q4_K_M | ~90 MB | very slightly worse |
| Q2_K | ~60 MB | noticeably worse (arguably an improvement here) |

---

## 4. A hosted HTTP API (optional)

Only if you need an API rather than in-browser inference. The FastAPI server is
in [`../space/`](../space) — its README covers the free hosting options and why
none of them beat running in the browser for this use case.

---

## Things that no longer work

Worth recording, because all four are commonly recommended and all four are
dead as of this writing:

| Route | Result |
|---|---|
| HF Spaces, Docker SDK, free tier | `402 Payment Required` — PRO only. Static Spaces are still free |
| HF serverless Inference API | `Model not supported by provider hf-inference` for custom GPT-2 fine-tunes |
| `reddit.com/r/X/comments.json` | `403` to unauthenticated clients |
| Reddit RSS feeds | blocked |
| PullPush (Pushshift successor) | `429` — paid scraping service now |
