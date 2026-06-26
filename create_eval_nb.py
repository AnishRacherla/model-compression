import json

cells = []

cells.append({
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "# Evaluate Fine-Tuned LoRA Model\n",
  "\n",
  "This notebook is isolated purely for Evaluation to avoid Out of Memory (OOM) errors. It pulls your pruned 4-layer base model, attaches the fine-tuned LoRA adapter from your Hugging Face Hub repo, translates 100 sentences from FLORES-200, and calculates the final COMET score."
 ]
})

cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "!pip install -q transformers datasets peft accelerate bitsandbytes evaluate unbabel-comet pytorch-lightning \"transformers<4.45\""
 ]
})

cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "from huggingface_hub import login\n",
  "\n",
  "read_token = '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'\n",
  "login(token=read_token)\n",
  "print(\"✅ Logged in globally with READ token.\")"
 ]
})

cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import torch\n",
  "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
  "from peft import PeftModel\n",
  "\n",
  "base_model_id = \"AnishRacherla/aya-expanse-8b-pruned-4layers\"\n",
  "lora_repo_id = \"AnishRacherla/aya-expanse-8b-pruned-4layers-finetuned\"\n",
  "\n",
  "print(\"Loading tokenizer...\")\n",
  "tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)\n",
  "\n",
  "bnb_config = BitsAndBytesConfig(\n",
  "    load_in_4bit=True,\n",
  "    bnb_4bit_use_double_quant=True,\n",
  "    bnb_4bit_quant_type=\"nf4\",\n",
  "    bnb_4bit_compute_dtype=torch.float16,\n",
  ")\n",
  "\n",
  "print(\"Loading base 4-layer pruned model in 4-bit...\")\n",
  "base_model = AutoModelForCausalLM.from_pretrained(\n",
  "    base_model_id,\n",
  "    quantization_config=bnb_config,\n",
  "    device_map=\"auto\",\n",
  "    trust_remote_code=True,\n",
  ")\n",
  "\n",
  "print(f\"Attaching fine-tuned LoRA adapters from {lora_repo_id}...\")\n",
  "model = PeftModel.from_pretrained(base_model, lora_repo_id)\n",
  "model.eval()\n",
  "print(\"✅ Model successfully loaded and ready for evaluation.\")"
 ]
})

cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import tarfile\n",
  "import urllib.request\n",
  "import tempfile\n",
  "import os\n",
  "import json\n",
  "from tqdm.auto import tqdm\n",
  "\n",
  "print(\"Downloading FLORES-200 Evaluation Dataset directly from Meta...\")\n",
  "samples_to_eval = 100\n",
  "sources = []\n",
  "references = []\n",
  "\n",
  "with tempfile.NamedTemporaryFile(suffix=\".tar.gz\", delete=False) as tmp:\n",
  "    urllib.request.urlretrieve(\"https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz\", tmp.name)\n",
  "    with tarfile.open(tmp.name, \"r:gz\") as tar:\n",
  "        eng_file = tar.extractfile(\"./flores200_dataset/dev/eng_Latn.dev\")\n",
  "        zho_file = tar.extractfile(\"./flores200_dataset/dev/zho_Hans.dev\")\n",
  "        \n",
  "        sources = [line.decode(\"utf-8\").strip() for line in eng_file.readlines()][:samples_to_eval]\n",
  "        references = [line.decode(\"utf-8\").strip() for line in zho_file.readlines()][:samples_to_eval]\n",
  "\n",
  "os.remove(tmp.name)\n",
  "predictions = []\n",
  "\n",
  "print(f\"Generating translations for {samples_to_eval} samples...\")\n",
  "for i, src_text in enumerate(tqdm(sources, desc=\"Translating\")):\n",
  "    prompt_eval = (\n",
  "        f\"Translate from English to Simplified Chinese:\\n\"\n",
  "        f\"en: {src_text}\\n\"\n",
  "        f\"zh:\"\n",
  "    )\n",
  "    inputs_eval = tokenizer(prompt_eval, return_tensors=\"pt\").to(model.device)\n",
  "    \n",
  "    with torch.no_grad():\n",
  "        outputs_eval = model.generate(\n",
  "            **inputs_eval, \n",
  "            max_new_tokens=150, \n",
  "            pad_token_id=tokenizer.eos_token_id,\n",
  "            do_sample=False\n",
  "        )\n",
  "        \n",
  "    num_gen = outputs_eval.shape[1] - inputs_eval.input_ids.shape[1]\n",
  "    pred = tokenizer.decode(outputs_eval[0][-num_gen:], skip_special_tokens=True).strip().replace('\\n', ' ')\n",
  "    predictions.append(pred)\n",
  "\n",
  "data = [\n",
  "    {\"src\": src, \"mt\": mt, \"ref\": ref}\n",
  "    for src, mt, ref in zip(sources, predictions, references)\n",
  "]\n",
  "with open(\"data_to_grade.json\", \"w\", encoding=\"utf-8\") as f:\n",
  "    json.dump(data, f)\n",
  "print(\"✅ Translations complete. Saved to data_to_grade.json.\")"
 ]
})

cells.append({
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "print(\"Creating pure Python script to bypass Kaggle CLI argparse errors...\")\n",
  "\n",
  "isolated_script = \"\"\"\n",
  "import json\n",
  "import logging\n",
  "logging.getLogger(\"pytorch_lightning\").setLevel(logging.WARNING)\n",
  "\n",
  "from comet import download_model, load_from_checkpoint\n",
  "\n",
  "print(\"Downloading COMET model quietly...\")\n",
  "model_path = download_model(\"Unbabel/wmt22-comet-da\")\n",
  "comet_model = load_from_checkpoint(model_path)\n",
  "\n",
  "with open(\"data_to_grade.json\", \"r\", encoding=\"utf-8\") as f:\n",
  "    data = json.load(f)\n",
  "\n",
  "print(\"Scoring translations...\")\n",
  "comet_results = comet_model.predict(data, batch_size=8, gpus=1)\n",
  "\n",
  "print(\"=\"*40)\n",
  "print(f\"🌟 Final COMET Score ({len(data)} Samples): {comet_results.system_score:.4f}\")\n",
  "print(\"=\"*40)\n",
  "\"\"\"\n",
  "\n",
  "with open(\"run_comet.py\", \"w\", encoding=\"utf-8\") as f:\n",
  "    f.write(isolated_script)\n",
  "    \n",
  "!python run_comet.py"
 ]
})

notebook = {
 "cells": cells,
 "metadata": {"language_info": {"name": "python"}},
 "nbformat": 4,
 "nbformat_minor": 5,
}

with open("evaluate_finetuned_model.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Evaluation notebook evaluate_finetuned_model.ipynb created successfully.")
