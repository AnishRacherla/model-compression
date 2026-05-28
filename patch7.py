import json

file_path = 'c:/Users/Anish/OneDrive/Desktop/modelcompression/model-compression/finetune_wmt_lora.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        skip_next = False
        for i, line in enumerate(cell['source']):
            if skip_next:
                skip_next = False
                continue
            
            # Detect the corrupted tokenizer line
            if 'tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,\n' == line:
                # Check if the next line is the injected torch_dtype
                if i + 1 < len(cell['source']) and 'torch_dtype=torch.float16' in cell['source'][i+1]:
                    new_source.append('tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)\n')
                    skip_next = True
                    continue
            
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
