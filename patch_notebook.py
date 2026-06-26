import json

with open('finetune_wmt_lora.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        skip_lines = 0
        for i, line in enumerate(source):
            if skip_lines > 0:
                skip_lines -= 1
                continue
                
            if 'model_id = "CohereForAI/aya-expanse-8b"' in line:
                new_source.append(line.replace('CohereForAI/aya-expanse-8b', 'AnishRacherla/aya-expanse-8b-pruned-4layers'))
            elif 'hub_model_id = "YOUR_HF_USERNAME/aya-expanse-8b-wmt-zh-en"' in line:
                new_source.append(line.replace('YOUR_HF_USERNAME/aya-expanse-8b-wmt-zh-en', 'AnishRacherla/aya-expanse-8b-pruned-4layers-finetuned'))
            elif 'zh_ds = load_dataset(' in line and i+1 < len(source) and '"wmt19"' in source[i+1]:
                new_source.append('    # Make sure you upload your cleaned dataset and name it \\\'cleaned_dataset.json\\\'\\n')
                new_source.append('zh_ds = load_dataset(\n')
                new_source.append('    "json",\n')
                new_source.append('    data_files="cleaned_dataset.json",\n')
                new_source.append('    split="train"\n')
                new_source.append(')\n')
                skip_lines = 4
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('finetune_wmt_lora.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Updated finetune_wmt_lora.ipynb successfully.')
