import json

file_path = 'c:/Users/Anish/OneDrive/Desktop/modelcompression/model-compression/finetune_wmt_lora.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'max_steps=2000' in line:
                line = line.replace('max_steps=2000', 'max_steps=120')
            elif 'warmup_steps=60' in line:
                line = line.replace('warmup_steps=60', 'warmup_steps=15')
            elif 'save_steps=200' in line:
                line = line.replace('save_steps=200', 'save_steps=60')
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
