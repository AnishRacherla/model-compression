import json

cells = []

# ── Cell 0: Markdown ──
cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "# 🔬 Strict Identical Prompt Benchmark (int8)\n",
  "\n",
  "**Goal:** Prove that pruning + quantization didn't ruin the model's core intelligence.\n",
  "\n",
  "We force **both** models to use the exact same Native Chat Template prompt to ensure a 100% fair, identical comparison. \n",
  "\n",
  "1. **Baseline int8**: Loaded directly from Cohere.\n",
  "2. **Compressed int8**: Loaded directly from your Hub repo.\n"
 ]
})

# ── Cell 1: Install ──
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "!pip install -q transformers datasets peft accelerate bitsandbytes huggingface_hub\n"
 ]
})

# ── Cell 2: Download FLORES-200 ──
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import tarfile, urllib.request, tempfile, os\n",
  "\n",
  "SAMPLES = 100\n",
  "with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:\n",
  "    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz', tmp.name)\n",
  "    with tarfile.open(tmp.name, 'r:gz') as tar:\n",
  "        sources    = [l.decode('utf-8').strip() for l in tar.extractfile('./flores200_dataset/dev/eng_Latn.dev').readlines()][:SAMPLES]\n",
  "        references = [l.decode('utf-8').strip() for l in tar.extractfile('./flores200_dataset/dev/zho_Hans.dev').readlines()][:SAMPLES]\n",
  "os.remove(tmp.name)\n",
  "print(f'✅ {len(sources)} FLORES-200 sentences ready.')"
 ]
})

# ── Cell 3: Helpers (Strict Chat Template for Both) ──
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import torch, time, gc\n",
  "from tqdm.auto import tqdm\n",
  "\n",
  "def get_vram_gb():\n",
  "    return sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / (1024**3)\n",
  "\n",
  "def clear_gpu():\n",
  "    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()\n",
  "\n",
  "def translate_and_benchmark(model, tokenizer, texts, label):\n",
  "    predictions, total_tokens = [], 0\n",
  "    model.eval()\n",
  "    t0 = time.time()\n",
  "    for src in tqdm(texts, desc=f'[{label}]'):\n",
  "        # Both models will use the exact same Native Chat Template prompt\n",
  "        messages = [{\"role\": \"user\", \"content\": f\"Translate from English to Simplified Chinese: {src}\"}]\n",
  "        input_ids = tokenizer.apply_chat_template(messages, return_tensors=\"pt\", add_generation_prompt=True).to(model.device)\n",
  "        inputs = {\"input_ids\": input_ids}\n",
  "            \n",
  "        with torch.no_grad():\n",
  "            out = model.generate(**inputs, max_new_tokens=150,\n",
  "                                 pad_token_id=tokenizer.eos_token_id, do_sample=False)\n",
  "        n = out.shape[1] - inputs['input_ids'].shape[1]\n",
  "        total_tokens += n\n",
  "        predictions.append(tokenizer.decode(out[0][-n:], skip_special_tokens=True).strip().replace('\\n', ' '))\n",
  "        \n",
  "    return predictions, total_tokens / (time.time() - t0)\n",
  "\n",
  "results = {}\n",
  "print('✅ Helpers ready.')"
 ]
})

# ── Cell 4: Baseline int8 ──
cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": ["## 🔵 Evaluate Baseline int8"]
})
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
  "\n",
  "bnb_int8 = BitsAndBytesConfig(load_in_8bit=True)\n",
  "\n",
  "clear_gpu()\n",
  "vram_before = get_vram_gb()\n",
  "t0 = time.time()\n",
  "\n",
  "tokenizer_base = AutoTokenizer.from_pretrained('CohereForAI/aya-expanse-8b', trust_remote_code=True)\n",
  "if not tokenizer_base.pad_token: tokenizer_base.pad_token = tokenizer_base.eos_token\n",
  "tokenizer_base.padding_side = 'right'\n",
  "\n",
  "model_base = AutoModelForCausalLM.from_pretrained(\n",
  "    'CohereForAI/aya-expanse-8b', quantization_config=bnb_int8,\n",
  "    device_map='auto', trust_remote_code=True,\n",
  ")\n",
  "t_load = time.time() - t0\n",
  "vram_used = get_vram_gb() - vram_before\n",
  "print(f'⏱️ {t_load:.1f}s  📦 {vram_used:.2f} GB')\n",
  "\n",
  "preds_base, tps = translate_and_benchmark(model_base, tokenizer_base, sources, 'baseline-int8')\n",
  "print(f'⚡ {tps:.1f} tok/s')\n",
  "results['baseline_int8'] = {'load_time': t_load, 'vram_gb': vram_used, 'tps': tps, 'predictions': preds_base}\n",
  "\n",
  "del model_base\n",
  "clear_gpu()\n",
  "print('✅ Baseline done.')"
 ]
})

# ── Cell 5: Compressed int8 ──
cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": ["## 🟠 Evaluate Compressed int8"]
})
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "clear_gpu()\n",
  "vram_before = get_vram_gb()\n",
  "t0 = time.time()\n",
  "\n",
  "INT8_REPO = 'AnishRacherla/aya-expanse-8b-compressed-final-int8'\n",
  "\n",
  "tokenizer_comp = AutoTokenizer.from_pretrained(INT8_REPO, trust_remote_code=True)\n",
  "model_comp = AutoModelForCausalLM.from_pretrained(\n",
  "    INT8_REPO, quantization_config=bnb_int8, device_map='auto', trust_remote_code=True\n",
  ")\n",
  "t_load = time.time() - t0\n",
  "vram_comp = get_vram_gb() - vram_before\n",
  "print(f'⏱️ {t_load:.1f}s  📦 {vram_comp:.2f} GB')\n",
  "\n",
  "preds_comp, tps_comp = translate_and_benchmark(model_comp, tokenizer_comp, sources, 'compressed-int8')\n",
  "print(f'⚡ {tps_comp:.1f} tok/s')\n",
  "results['compressed_int8'] = {'load_time': t_load, 'vram_gb': vram_comp, 'tps': tps_comp, 'predictions': preds_comp}\n",
  "\n",
  "del model_comp\n",
  "clear_gpu()\n",
  "print('✅ Compressed int8 done.')"
 ]
})

# ── Cell 6: COMET Scoring ──
cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": ["## 📊 COMET Scoring (Isolated Subprocess)"]
})
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import json as _json, tempfile, os\n",
  "\n",
  "preds_path = '/tmp/benchmark_int8_preds.json'\n",
  "_json.dump({\n",
  "    'sources'            : sources,\n",
  "    'references'         : references,\n",
  "    'preds_baseline_int8': results['baseline_int8']['predictions'],\n",
  "    'preds_comp_int8'    : results['compressed_int8']['predictions'],\n",
  "}, open(preds_path, 'w', encoding='utf-8'))\n",
  "\n",
  "comet_script = '''\n",
  "import json, logging\n",
  "logging.getLogger(\"pytorch_lightning\").setLevel(logging.WARNING)\n",
  "from comet import download_model, load_from_checkpoint\n",
  "\n",
  "data = json.load(open(\"/tmp/benchmark_int8_preds.json\", encoding=\"utf-8\"))\n",
  "model_path  = download_model(\"Unbabel/wmt22-comet-da\")\n",
  "comet_model = load_from_checkpoint(model_path)\n",
  "\n",
  "def score(preds):\n",
  "    samples = [{\"src\": s, \"mt\": m, \"ref\": r}\n",
  "               for s, m, r in zip(data[\"sources\"], preds, data[\"references\"])]\n",
  "    return comet_model.predict(samples, batch_size=8, gpus=1).system_score\n",
  "\n",
  "print(f\"COMET_BASE={score(data[\\\"preds_baseline_int8\\\"]):.4f}\")\n",
  "print(f\"COMET_COMP={score(data[\\\"preds_comp_int8\\\"]):.4f}\")\n",
  "'''\n",
  "with open('/tmp/run_comet_int8.py', 'w') as f:\n",
  "    f.write(comet_script)\n",
  "\n",
  "print('Installing COMET...')\n",
  "!pip install -q unbabel-comet pytorch-lightning\n",
  "print('Running COMET scorer...')\n",
  "!python /tmp/run_comet_int8.py"
 ]
})

# ── Cell 7: Final Table ──
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import subprocess, re\n",
  "\n",
  "proc   = subprocess.run(['python', '/tmp/run_comet_int8.py'], capture_output=True, text=True)\n",
  "output = proc.stdout + proc.stderr\n",
  "\n",
  "comet_base = float(re.search(r'COMET_BASE=([\\d.]+)', output).group(1))\n",
  "comet_comp = float(re.search(r'COMET_COMP=([\\d.]+)', output).group(1))\n",
  "results['baseline_int8']['comet']   = comet_base\n",
  "results['compressed_int8']['comet'] = comet_comp\n",
  "\n",
  "rb = results['baseline_int8']\n",
  "rc = results['compressed_int8']\n",
  "\n",
  "print()\n",
  "print('=' * 75)\n",
  "print(f\"{'Metric':<25} {'Baseline int8':>18} {'Compressed int8':>18} {'Δ':>8}\")\n",
  "print('=' * 75)\n",
  "print(f\"{'VRAM (GB)':<25} {rb['vram_gb']:>18.2f} {rc['vram_gb']:>18.2f} {rc['vram_gb']-rb['vram_gb']:>+8.2f}\")\n",
  "print(f\"{'Load Time (s)':<25} {rb['load_time']:>18.1f} {rc['load_time']:>18.1f} {rc['load_time']-rb['load_time']:>+8.1f}\")\n",
  "print(f\"{'Tokens/sec':<25} {rb['tps']:>18.1f} {rc['tps']:>18.1f} {rc['tps']-rb['tps']:>+8.1f}\")\n",
  "print(f\"{'COMET Score':<25} {rb['comet']:>18.4f} {rc['comet']:>18.4f} {rc['comet']-rb['comet']:>+8.4f}\")\n",
  "print('=' * 75)\n",
  "print(f\"\\n📦 VRAM reduction: {rb['vram_gb']-rc['vram_gb']:.2f} GB ({(1-rc['vram_gb']/rb['vram_gb'])*100:.0f}%)\")\n",
  "print(f\"🌟 COMET delta   : {rc['comet']-rb['comet']:+.4f} points\")\n"
 ]
})

notebook = {
 "cells": cells,
 "metadata": {"language_info": {"name": "python"}},
 "nbformat": 4,
 "nbformat_minor": 5,
}

with open("benchmark_int8_final.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("benchmark_int8_final.ipynb created successfully with STRICT IDENTICAL PROMPTING!")
