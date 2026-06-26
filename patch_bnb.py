import json

with open("evaluate_finetuned_model.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        text = ''.join(source)
        
        # Upgrade bitsandbytes to satisfy the new requirement
        if '!pip install -q transformers' in text:
            new_source = [
                "!pip install -q -U \"bitsandbytes>=0.46.1\" transformers datasets peft accelerate evaluate unbabel-comet pytorch-lightning\n"
            ]
            cell['source'] = new_source

with open("evaluate_finetuned_model.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Bitsandbytes upgrade added.")
