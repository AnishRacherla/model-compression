import json

file_path = 'c:/Users/Anish/OneDrive/Desktop/modelcompression/model-compression/finetune_wmt_lora.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'model = prepare_model_for_kbit_training(model)' in line:
                new_source.append('model.config.torch_dtype = torch.float16 # 📉 MEMORY FIX: Force config to fp16 to stop bfloat16 autocast\n')
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
