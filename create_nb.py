import json

cells = []

# ── CELL 0: Title ────────────────────────────────────────────────────────────
cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "# Fisher Information-Based Unstructured Pruning of Aya Expanse 8B\n",
  "\n",
  "Optimized for Kaggle free-tier GPUs (T4/P100). Loads in **4-bit NF4**, computes Fisher scores **one layer at a time**, and prunes 20 % of weights per layer.\n",
  "\n",
  "> ⚠️ **IMPORTANT – Read Before Running**\n",
  ">\n",
  "> **Step 1:** Run **Cell 1 (pip install)** only.\n",
  "> **Step 2:** Run **Cell 2 (restart kernel)**. The cell will intentionally crash/restart the kernel.\n",
  "> **Step 3:** Run all remaining cells from **Cell 3 onward** in order.\n",
  ">\n",
  "> This two-step process is required because numpy C-binary extensions cannot be hot-swapped in a live kernel."
 ]
})

# ── CELL 1: pip install (run FIRST, before restart) ──────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# ▶ STEP 1 — Run this cell FIRST, then run Cell 2 to restart the kernel.\n",
  "# Installing numpy<2 BEFORE everything else prevents the\n",
  "# 'numpy.dtype size changed' binary incompatibility that breaks pandas/datasets.\n",
  "import subprocess, sys\n",
  "\n",
  "pkgs = [\n",
  "    'numpy<2.0',\n",
  "    'pyarrow',\n",
  "    'pandas',\n",
  "    'transformers',\n",
  "    'datasets',\n",
  "    'evaluate',\n",
  "    'accelerate',\n",
  "    'bitsandbytes',\n",
  "    'sacrebleu',\n",
  "    'huggingface_hub',\n",
  "]\n",
  "\n",
  "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade'] + pkgs)\n",
  "print('\\n✅ Packages installed. Now run Cell 2 to restart the kernel.')"
 ]
})

# ── CELL 2: Kernel restart (run SECOND) ──────────────────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# ▶ STEP 2 — Restart the kernel so the new numpy binary is loaded fresh.\n",
  "# After the kernel restarts, run ALL remaining cells from Cell 3 onward.\n",
  "import os, signal\n",
  "print('Restarting kernel to load the new numpy binary...')\n",
  "os.kill(os.getpid(), signal.SIGKILL)"
 ]
})

# ── CELL 3: Hugging Face login (run AFTER restart) ───────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# ▶ STEP 3 — Run from here downward after the kernel restarts.\n",
  "# Aya Expanse 8B requires a Hugging Face account to download.\n",
  "from huggingface_hub import login\n",
  "from getpass import getpass\n",
  "\n",
  "hf_token = getpass('Enter your Hugging Face READ token: ')\n",
  "login(token=hf_token)\n",
  "print('✅ Logged in to Hugging Face.')"
 ]
})

# ── CELL 4: Imports & workspace setup ────────────────────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import os, gc\n",
  "import torch\n",
  "import torch.nn as nn\n",
  "from datasets import load_dataset\n",
  "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
  "\n",
  "output_dir = '/kaggle/working/pruned_aya'\n",
  "os.makedirs(output_dir, exist_ok=True)\n",
  "print(f'✅ Workspace ready → {output_dir}')\n",
  "print(f'   numpy  : {__import__(\"numpy\").__version__}')\n",
  "print(f'   pandas : {__import__(\"pandas\").__version__}')\n",
  "print(f'   torch  : {torch.__version__}')"
 ]
})

# ── CELL 5: Data cleaning & sampling ─────────────────────────────────────────
cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": ["## 2. Dataset Loading & Light Cleaning\n",
            "Stream WMT19 zh-en, apply quality filters, collect 128 pairs for Fisher estimation."]
})

cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "def clean_and_sample(dataset_name='wmt/wmt19', lang_pair='zh-en', sample_size=128):\n",
  "    print(f'Loading {dataset_name} {lang_pair} (streaming)...')\n",
  "    ds = load_dataset(dataset_name, lang_pair, split='train', streaming=True, trust_remote_code=True)\n",
  "    samples, seen = [], set()\n",
  "\n",
  "    for ex in ds:\n",
  "        if len(samples) >= sample_size:\n",
  "            break\n",
  "        t = ex['translation']\n",
  "        src, tgt = t['en'].strip(), t['zh'].strip()\n",
  "\n",
  "        if not src or not tgt:                       # empty\n",
  "            continue\n",
  "        if src in seen:                              # duplicate\n",
  "            continue\n",
  "        if len(src) > 1000 or len(tgt) > 1000:      # too long\n",
  "            continue\n",
  "        ratio = len(src) / max(len(tgt), 1)\n",
  "        if ratio > 3.0 or ratio < 0.33:             # bad ratio\n",
  "            continue\n",
  "\n",
  "        seen.add(src)\n",
  "        samples.append({'en': src, 'zh': tgt})\n",
  "\n",
  "    print(f'✅ Collected {len(samples)} clean calibration pairs.')\n",
  "    return samples\n",
  "\n",
  "calibration_data = clean_and_sample(sample_size=128)"
 ]
})

# ── CELL 6: Model loading (4-bit NF4) ────────────────────────────────────────
cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": ["## 3. Model Loading — 4-bit NF4\n",
            "Default strategy for Kaggle free-tier GPUs."]
})

cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "model_id = 'CohereForAI/aya-expanse-8b'\n",
  "\n",
  "print('Loading tokenizer...')\n",
  "tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)\n",
  "if not tokenizer.pad_token:\n",
  "    tokenizer.pad_token = tokenizer.eos_token\n",
  "tokenizer.padding_side = 'right'\n",
  "\n",
  "bnb_config = BitsAndBytesConfig(\n",
  "    load_in_4bit=True,\n",
  "    bnb_4bit_use_double_quant=True,\n",
  "    bnb_4bit_quant_type='nf4',\n",
  "    bnb_4bit_compute_dtype=torch.float16,\n",
  ")\n",
  "\n",
  "torch.cuda.empty_cache(); gc.collect()\n",
  "print('Loading model in 4-bit NF4 precision...')\n",
  "model = AutoModelForCausalLM.from_pretrained(\n",
  "    model_id,\n",
  "    quantization_config=bnb_config,\n",
  "    device_map='auto',\n",
  "    trust_remote_code=True,\n",
  "    torch_dtype=torch.float16,\n",
  ")\n",
  "model.eval()\n",
  "print('✅ Model loaded successfully.')"
 ]
})

# ── CELL 7: Calibration batch preparation ────────────────────────────────────
cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": ["## 4. Layer-Wise Fisher Computation & Unstructured Pruning"]
})

cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "def make_calib_batches(samples, tokenizer, batch_size=1):\n",
  "    batches = []\n",
  "    for i in range(0, len(samples), batch_size):\n",
  "        chunk = samples[i:i+batch_size]\n",
  "        texts = [\n",
  "            f\"Translate from English to Simplified Chinese:\\nen: {s['en']}\\nzh: {s['zh']}\"\n",
  "            for s in chunk\n",
  "        ]\n",
  "        enc = tokenizer(texts, return_tensors='pt', padding=True,\n",
  "                        truncation=True, max_length=512)\n",
  "        batches.append(enc)\n",
  "    return batches\n",
  "\n",
  "calib_batches = make_calib_batches(calibration_data, tokenizer, batch_size=1)\n",
  "print(f'✅ Prepared {len(calib_batches)} calibration batches.')"
 ]
})

# ── CELL 8: Fisher pruning function ──────────────────────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "from bitsandbytes.nn import Linear4bit\n",
  "from bitsandbytes.functional import dequantize_4bit\n",
  "\n",
  "def prune_layer_fisher(layer, layer_idx, calib_batches, pruning_ratio=0.20):\n",
  "    \"\"\"\n",
  "    Layer-wise unstructured Fisher pruning.\n",
  "    Memory strategy: dequantize one layer to fp16 → compute gradients → mask → clear.\n",
  "    \"\"\"\n",
  "    print(f'\\n--- Layer {layer_idx} ---')\n",
  "\n",
  "    # 1. Find all 4-bit sub-modules in this transformer block\n",
  "    bnb_modules = {n: m for n, m in layer.named_modules() if isinstance(m, Linear4bit)}\n",
  "    if not bnb_modules:\n",
  "        print('  No Linear4bit modules – skipping.')\n",
  "        return\n",
  "\n",
  "    # 2. Dequantize to fp16 nn.Linear and swap in-place\n",
  "    fp16_linears = {}\n",
  "    for name, bnb_mod in bnb_modules.items():\n",
  "        w_fp16 = dequantize_4bit(\n",
  "            bnb_mod.weight.data,\n",
  "            quant_state=bnb_mod.weight.quant_state\n",
  "        ).to(torch.float16)\n",
  "\n",
  "        new_lin = nn.Linear(\n",
  "            bnb_mod.in_features, bnb_mod.out_features,\n",
  "            bias=(bnb_mod.bias is not None), dtype=torch.float16\n",
  "        ).to(model.device)\n",
  "        new_lin.weight.data = w_fp16\n",
  "        if bnb_mod.bias is not None:\n",
  "            new_lin.bias.data = bnb_mod.bias.data.to(torch.float16)\n",
  "        new_lin.weight.requires_grad_(True)\n",
  "        fp16_linears[name] = new_lin\n",
  "\n",
  "        # Swap into model\n",
  "        parts = name.split('.')\n",
  "        parent = layer\n",
  "        for p in parts[:-1]:\n",
  "            parent = getattr(parent, p)\n",
  "        setattr(parent, parts[-1], new_lin)\n",
  "\n",
  "    # 3. Freeze everything except this layer's new fp16 weights\n",
  "    for p in model.parameters():\n",
  "        p.requires_grad_(False)\n",
  "    for lin in fp16_linears.values():\n",
  "        lin.weight.requires_grad_(True)\n",
  "\n",
  "    # 4. Accumulate empirical Fisher (diagonal) = E[grad^2]\n",
  "    fisher = {n: torch.zeros_like(lin.weight.data) for n, lin in fp16_linears.items()}\n",
  "    model.train()\n",
  "    print(f'  Computing Fisher over {len(calib_batches)} batches...')\n",
  "\n",
  "    for batch in calib_batches:\n",
  "        inputs = {k: v.to(model.device) for k, v in batch.items()}\n",
  "        out = model(**inputs, labels=inputs['input_ids'])\n",
  "        out.loss.backward()\n",
  "\n",
  "        for n, lin in fp16_linears.items():\n",
  "            if lin.weight.grad is not None:\n",
  "                fisher[n] += lin.weight.grad.detach() ** 2\n",
  "                lin.weight.grad = None          # free immediately\n",
  "\n",
  "    model.eval()\n",
  "\n",
  "    # 5. Taylor score S = F * w^2; prune lowest k %\n",
  "    print(f'  Applying {pruning_ratio*100:.0f}% unstructured mask...')\n",
  "    for n, lin in fp16_linears.items():\n",
  "        w = lin.weight.data\n",
  "        score = fisher[n] * w ** 2\n",
  "        k = int(score.numel() * pruning_ratio)\n",
  "        if k > 0:\n",
  "            thresh = torch.topk(score.view(-1), k, largest=False).values[-1]\n",
  "            mask = (score > thresh).to(w.dtype)\n",
  "            lin.weight.data = w * mask\n",
  "        lin.weight.requires_grad_(False)\n",
  "\n",
  "    # 6. Aggressive memory cleanup before next layer\n",
  "    del fisher\n",
  "    torch.cuda.empty_cache()\n",
  "    gc.collect()\n",
  "    print('  Done.')\n",
  "\n",
  "\n",
  "# ── Execute layer-wise pruning ────────────────────────────────────────────────\n",
  "num_layers = len(model.model.layers)\n",
  "print(f'Total transformer layers: {num_layers}')\n",
  "\n",
  "for i in range(num_layers):\n",
  "    prune_layer_fisher(model.model.layers[i], i, calib_batches, pruning_ratio=0.20)\n",
  "\n",
  "print('\\n✅ Pruning complete!')"
 ]
})

# ── CELL 9: Evaluation – FLORES-200 translations ─────────────────────────────
cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": ["## 5. Machine Translation Evaluation\n",
            "Evaluate the pruned model on FLORES-200 using COMET, BLEU, and chrF."]
})

cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import tarfile, urllib.request, tempfile, json\n",
  "from tqdm.auto import tqdm\n",
  "\n",
  "EVAL_SIZE = 100\n",
  "print('Downloading FLORES-200 from Meta CDN...')\n",
  "\n",
  "with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:\n",
  "    urllib.request.urlretrieve(\n",
  "        'https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz', tmp.name)\n",
  "    with tarfile.open(tmp.name, 'r:gz') as tar:\n",
  "        eng = tar.extractfile('./flores200_dataset/dev/eng_Latn.dev')\n",
  "        zho = tar.extractfile('./flores200_dataset/dev/zho_Hans.dev')\n",
  "        sources    = [l.decode().strip() for l in eng.readlines()][:EVAL_SIZE]\n",
  "        references = [l.decode().strip() for l in zho.readlines()][:EVAL_SIZE]\n",
  "os.remove(tmp.name)\n",
  "\n",
  "predictions = []\n",
  "model.eval()\n",
  "print(f'Translating {EVAL_SIZE} sentences...')\n",
  "\n",
  "for src in tqdm(sources, desc='Translating'):\n",
  "    prompt = f'Translate from English to Simplified Chinese:\\nen: {src}\\nzh:'\n",
  "    enc = tokenizer(prompt, return_tensors='pt').to(model.device)\n",
  "    with torch.no_grad():\n",
  "        out = model.generate(\n",
  "            **enc, max_new_tokens=150,\n",
  "            pad_token_id=tokenizer.eos_token_id, do_sample=False)\n",
  "    n_new = out.shape[1] - enc.input_ids.shape[1]\n",
  "    pred = tokenizer.decode(out[0][-n_new:], skip_special_tokens=True).strip().replace('\\n', ' ')\n",
  "    predictions.append(pred)\n",
  "\n",
  "data = [{'src': s, 'mt': m, 'ref': r}\n",
  "        for s, m, r in zip(sources, predictions, references)]\n",
  "with open('data_to_grade.json', 'w', encoding='utf-8') as f:\n",
  "    json.dump(data, f)\n",
  "print('✅ Translations saved.')"
 ]
})

# ── CELL 10: Install COMET (deferred to avoid breaking datasets) ──────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# Install COMET last — its dependencies (pytorch-lightning, old transformers)\n",
  "# would break datasets/pandas if installed at the start.\n",
  "!pip install -q unbabel-comet pytorch-lightning \"transformers<4.45\""
 ]
})

# ── CELL 11: Metrics (isolated subprocess) ────────────────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "metrics_script = '''\n",
  "import json, logging\n",
  "logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)\n",
  "from comet import download_model, load_from_checkpoint\n",
  "import evaluate\n",
  "\n",
  "with open('data_to_grade.json', encoding='utf-8') as f:\n",
  "    data = json.load(f)\n",
  "\n",
  "print('Scoring with COMET...')\n",
  "comet_model = load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))\n",
  "res = comet_model.predict(data, batch_size=8, gpus=1)\n",
  "print('='*40)\n",
  "print(f'COMET : {res.system_score:.4f}')\n",
  "\n",
  "preds = [d['mt'] for d in data]\n",
  "refs  = [[d['ref']] for d in data]\n",
  "bleu  = evaluate.load('sacrebleu').compute(predictions=preds, references=refs)\n",
  "chrf  = evaluate.load('chrf').compute(predictions=preds, references=refs)\n",
  "print(f'BLEU  : {bleu[\"score\"]:.2f}')\n",
  "print(f'chrF  : {chrf[\"score\"]:.2f}')\n",
  "print('='*40)\n",
  "'''\n",
  "\n",
  "with open('run_metrics.py', 'w') as f:\n",
  "    f.write(metrics_script)\n",
  "\n",
  "!python run_metrics.py"
 ]
})

# ── CELL 12: Save and Export Model ─────────────────────────────────────────────
cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": ["## 6. Save and Export Pruned Model\n",
            "The model is now fully dequantized to `fp16` and pruned. We save it to disk and prepare it for download/export."]
})

cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import shutil\n",
  "from IPython.display import FileLink\n",
  "\n",
  "print(f'Saving pruned model and tokenizer to {output_dir}...')\n",
  "model.save_pretrained(output_dir)\n",
  "tokenizer.save_pretrained(output_dir)\n",
  "print('✅ Model saved to disk.')\n",
  "\n",
  "# Option A: Create a Zip file for direct download\n",
  "print('Zipping the model for local download (this may take a few minutes due to size)...')\n",
  "shutil.make_archive('/kaggle/working/pruned_aya_saved', 'zip', output_dir)\n",
  "print('\\n✅ Zip complete! Click the link below to download to your laptop:')\n",
  "display(FileLink(r'pruned_aya_saved.zip'))\n",
  "\n",
  "# Option B: Push to Hugging Face Hub (Recommended for large models)\n",
  "# Uncomment the lines below and replace with your username to push directly to HF.\n",
  "# hf_repo_id = \"YOUR_USERNAME/aya-expanse-8b-fisher-pruned\"\n",
  "# print(f'Pushing to Hub: {hf_repo_id}')\n",
  "# model.push_to_hub(hf_repo_id)\n",
  "# tokenizer.push_to_hub(hf_repo_id)\n",
  "# print('✅ Successfully pushed to Hugging Face Hub!')"
 ]
})

# ── Write notebook ────────────────────────────────────────────────────────────
notebook = {
 "cells": cells,
 "metadata": {"language_info": {"name": "python"}},
 "nbformat": 4,
 "nbformat_minor": 5,
}

out_path = "C:/Users/Anish/OneDrive/Desktop/modelcompression/model-compression/fisher_pruning_wmt.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Notebook created successfully.")
