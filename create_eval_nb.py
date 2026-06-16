import json

cells = []

# ── CELL 0: Title ────────────────────────────────────────────────────────────
cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "# Fisher Pruned Model Evaluation\n",
  "This notebook purely downloads your pruned model from the Hugging Face Hub and evaluates its translation capabilities on the FLORES-200 benchmark dataset using COMET, BLEU, and chrF metrics."
 ]
})

# ── CELL 1: Install initial deps ──────────────────────────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# 1. Install evaluation dependencies\n",
  "!pip install -q -U transformers accelerate evaluate sacrebleu datasets bitsandbytes"
 ]
})

# ── CELL 2: Login to Hugging Face ─────────────────────────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# 2. Login to Hugging Face (Optional, but required if your repo is private)\n",
  "from huggingface_hub import login\n",
  "from getpass import getpass\n",
  "\n",
  "hf_token = getpass('Enter your Hugging Face READ token (or press Enter to skip if model is public): ')\n",
  "if hf_token:\n",
  "    login(token=hf_token)\n",
  "    print('✅ Logged in to Hugging Face.')"
 ]
})

# ── CELL 3: Enter Repo ID ─────────────────────────────────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# 3. Set your repository ID\n",
  "# Replace this string with the EXACT repository name you pushed to (e.g., 'your-username/aya-expanse-8b-fisher-pruned')\n",
  "HF_REPO_ID = input(\"Enter your Hugging Face repository ID: \")\n",
  "print(f\"Target repository: {HF_REPO_ID}\")"
 ]
})

# ── CELL 4: Load Model ────────────────────────────────────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# 4. Load the pruned model natively in fp16\n",
  "import torch\n",
  "from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig\n",
  "\n",
  "print('Loading tokenizer...')\n",
  "tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, trust_remote_code=True)\n",
  "if not tokenizer.pad_token:\n",
  "    tokenizer.pad_token = tokenizer.eos_token\n",
  "tokenizer.padding_side = 'right'\n",
  "\n",
  "print('Stripping leftover 4-bit quantization config...')\n",
  "config = AutoConfig.from_pretrained(HF_REPO_ID, trust_remote_code=True)\n",
  "if hasattr(config, 'quantization_config'):\n",
  "    del config.quantization_config\n",
  "    print('Removed old bitsandbytes config.')\n",
  "\n",
  "print('Loading pruned model...')\n",
  "model = AutoModelForCausalLM.from_pretrained(\n",
  "    HF_REPO_ID,\n",
  "    config=config,\n",
  "    device_map='auto',\n",
  "    torch_dtype=torch.float16,\n",
  "    trust_remote_code=True,\n",
  ")\n",
  "model.eval()\n",
  "print('✅ Model loaded successfully and ready for inference!')"
 ]
})

# ── CELL 5: Download FLORES and Translate ─────────────────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# 5. Download FLORES-200 and generate translations\n",
  "import tarfile, urllib.request, tempfile, json, os\n",
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

# ── CELL 6: Install COMET ─────────────────────────────────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# 6. Install COMET dependencies\n",
  "# (Done at the end to prevent dependency clashes with the translation loop)\n",
  "!pip install -q unbabel-comet pytorch-lightning \"transformers<4.45\""
 ]
})

# ── CELL 7: Run COMET & BLEU Metrics ──────────────────────────────────────────
cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# 7. Run metrics via isolated script\n",
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

notebook = {
 "cells": cells,
 "metadata": {"language_info": {"name": "python"}},
 "nbformat": 4,
 "nbformat_minor": 5,
}

with open("C:/Users/Anish/OneDrive/Desktop/modelcompression/model-compression/fisher_eval_only.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Evaluation notebook created successfully.")
