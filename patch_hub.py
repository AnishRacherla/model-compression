import json

with open('finetune_wmt_lora.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        text = ''.join(cell['source'])
        if 'SFTConfig' in text and 'push_to_hub' in text:
            new_source = []
            for line in cell['source']:
                # Add create_repo call before SFTConfig
                if 'sft_config = SFTConfig(' in line:
                    new_source.append("from huggingface_hub import create_repo\n")
                    new_source.append("create_repo(hub_model_id, token=write_token, exist_ok=True, private=False)\n")
                    new_source.append("print(f'✅ Hub repo ready: {hub_model_id}')\n")
                    new_source.append("\n")
                # Add hub_always_push after hub_token
                if 'hub_token=write_token,' in line:
                    new_source.append(line)
                    new_source.append("    hub_always_push=True,\n")
                    continue
                new_source.append(line)
            cell['source'] = new_source

with open('finetune_wmt_lora.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Patched: create_repo + hub_always_push added.")
