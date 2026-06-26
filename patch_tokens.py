import json

with open('finetune_wmt_lora.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        text = "".join(source)
        
        # 1. Update the login cell
        if "hf_token = input(\"Enter your Hugging Face WRITE Token: \")" in text:
            new_source = [
                "from huggingface_hub import login\n",
                "\n",
                "# 2. Login to Hugging Face with READ token for gated access\n",
                "read_token = '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'\n",
                "write_token = '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'\n",
                "login(token=read_token)\n",
                "print(\"✅ Logged in globally with READ token. WRITE token is saved for SFTTrainer.\")\n"
            ]
            cell['source'] = new_source
            
        # 2. Add hub_token to SFTConfig
        if "sft_config = SFTConfig(" in text and "hub_token" not in text:
            new_sft_source = []
            for line in source:
                new_sft_source.append(line)
                if "hub_strategy=\"checkpoint\"," in line:
                    new_sft_source.append("    hub_token=write_token,\n")
            cell['source'] = new_sft_source

with open('finetune_wmt_lora.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Tokens successfully hardcoded in finetune_wmt_lora.ipynb')
