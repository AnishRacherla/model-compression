import json

with open("evaluate_finetuned_model.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        text = ''.join(source)
        
        # 1. Restore the exact pip install that worked in the training notebook (no -U bitsandbytes)
        if '!pip install -q -U "bitsandbytes>=0.46.1"' in text:
            cell['source'] = [
                "!pip install -q transformers datasets peft accelerate bitsandbytes evaluate\n"
            ]
            
        # 2. Restore 4-bit loading (because the checkpoint is fundamentally a 4-bit serialized checkpoint)
        if 'base_model = AutoModelForCausalLM.from_pretrained(' in text and 'torch_dtype=torch.float16' in text:
            new_source = [
                "import torch\n",
                "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
                "from peft import PeftModel\n",
                "\n",
                "base_model_id = \"AnishRacherla/aya-expanse-8b-pruned-4layers\"\n",
                "lora_repo_id = \"AnishRacherla/aya-expanse-8b-pruned-4layers-finetuned\"\n",
                "\n",
                "print(\"Loading tokenizer...\")\n",
                "tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)\n",
                "if not tokenizer.pad_token:\n",
                "    tokenizer.pad_token = tokenizer.eos_token\n",
                "tokenizer.padding_side = \"right\"\n",
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
                "print(\"✅ Model successfully loaded and ready for evaluation.\")\n"
            ]
            cell['source'] = new_source

with open("evaluate_finetuned_model.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook restored to safe versions and 4-bit.")
