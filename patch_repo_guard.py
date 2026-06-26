import json

with open('finetune_wmt_lora.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        text = ''.join(cell['source'])
        if 'hub_model_id =' in text and 'model_id =' in text:
            # Add a clear comment + assertion separating the two repos
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if 'hub_model_id =' in line:
                    new_source.append("# NOTE: hub_model_id MUST be different from model_id to avoid overwriting the base model!\n")
                    new_source.append("assert hub_model_id != model_id, 'STOP: hub_model_id must be different from model_id!'\n")
            cell['source'] = new_source

with open('finetune_wmt_lora.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Safeguard added.")
