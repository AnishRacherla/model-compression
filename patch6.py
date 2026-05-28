import json

file_path = 'c:/Users/Anish/OneDrive/Desktop/modelcompression/model-compression/finetune_wmt_lora.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'trust_remote_code=True' in line and 'from_pretrained' in ''.join(cell['source']):
                new_source.append(line.replace('trust_remote_code=True', 'trust_remote_code=True,\n    torch_dtype=torch.float16 # 📉 MEMORY FIX: Force fp16 instead of bf16 for T4 compatibility'))
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
