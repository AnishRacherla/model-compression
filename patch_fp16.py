import json

with open("evaluate_finetuned_model.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        text = ''.join(source)
        
        # Patch the model loading cell to remove 4-bit and use FP16 across both GPUs
        if 'base_model = AutoModelForCausalLM.from_pretrained(' in text and 'bnb_config' in text:
            new_source = [
                "import torch\n",
                "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
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
                "print(\"Loading base 4-layer pruned model in pure float16 (no quantization) across dual GPUs...\")\n",
                "base_model = AutoModelForCausalLM.from_pretrained(\n",
                "    base_model_id,\n",
                "    torch_dtype=torch.float16,\n",
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
print("Model loading patched to float16.")
