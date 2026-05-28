import json

file_path = 'c:/Users/Anish/OneDrive/Desktop/modelcompression/model-compression/finetune_wmt_lora.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'print("Loading Model in 4-bit precision...")' in line:
                new_source.append('import gc\n')
                new_source.append('import torch\n')
                new_source.append('torch.cuda.empty_cache()\n')
                new_source.append('gc.collect()\n\n')
                new_source.append(line)
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
