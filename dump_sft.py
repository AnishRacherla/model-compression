import json

with open('finetune_wmt_lora.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        text = ''.join(cell['source'])
        if 'SFTConfig' in text or 'push_to_hub' in text:
            with open('sft_cell_dump.txt', 'w', encoding='utf-8') as out:
                out.write(text)
            print('Dumped SFT cell to sft_cell_dump.txt')
