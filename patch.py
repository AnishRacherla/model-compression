import json
import os

file_path = 'c:/Users/Anish/OneDrive/Desktop/modelcompression/model-compression/finetune_wmt_lora.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        # Fix device_map and LoraConfig
        if 'device_map="auto"' in ''.join(cell['source']):
            new_source = []
            for line in cell['source']:
                if 'device_map="auto"' in line:
                    new_source.append(line.replace('"auto"', '{"": 0}, # 📉 MEMORY FIX: Force single GPU'))
                elif 'r=16' in line:
                    new_source.append(line.replace('r=16', 'r=8, # 📉 MEMORY FIX: Reduced rank'))
                elif 'lora_alpha=32' in line:
                    new_source.append(line.replace('lora_alpha=32', 'lora_alpha=16'))
                elif 'target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]' in line:
                    new_source.append(line.replace('["q_proj", "k_proj", "v_proj", "o_proj"]', '["q_proj", "v_proj"], # 📉 MEMORY FIX: Fewer modules'))
                else:
                    new_source.append(line)
            cell['source'] = new_source
        
        # Fix SFTConfig optimizations
        if 'SFTConfig' in ''.join(cell['source']):
            new_source = []
            for line in cell['source']:
                if 'optim="paged_adamw_32bit"' in line:
                    new_source.append(line.replace('"paged_adamw_32bit"', '"paged_adamw_8bit", # 📉 MEMORY FIX: 8-bit optimizer'))
                elif 'max_length=1024' in line:
                    new_source.append(line.replace('max_length=1024', 'max_seq_length=512 # 📉 MEMORY FIX: Hard cap at 512 tokens'))
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
