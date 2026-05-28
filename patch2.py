import json

file_path = 'c:/Users/Anish/OneDrive/Desktop/modelcompression/model-compression/finetune_wmt_lora.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Fix comma issue
            line = line.replace('{"": 0}, # 📉 MEMORY FIX: Force single GPU,', '{"": 0}, # 📉 MEMORY FIX: Force single GPU')
            line = line.replace('r=8, # 📉 MEMORY FIX: Reduced rank,', 'r=8, # 📉 MEMORY FIX: Reduced rank')
            line = line.replace('["q_proj", "v_proj"], # 📉 MEMORY FIX: Fewer modules,', '["q_proj", "v_proj"], # 📉 MEMORY FIX: Fewer modules')
            line = line.replace('"paged_adamw_8bit", # 📉 MEMORY FIX: 8-bit optimizer,', '"paged_adamw_8bit", # 📉 MEMORY FIX: 8-bit optimizer')
            
            # Additional fixes
            if 'device_map={"": 0} # \U0001f4c9 MEMORY FIX' in line and not line.endswith(',\n'):
                # Handle previous replacement that might have missed comma
                pass
            
            new_source.append(line)
        cell['source'] = new_source

# Re-do properly
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'device_map=' in line and 'MEMORY FIX' in line:
                line = '    device_map={"": 0}, # 📉 MEMORY FIX: Force single GPU\n'
            if 'r=' in line and 'MEMORY FIX' in line:
                line = '    r=8, # 📉 MEMORY FIX: Reduced rank\n'
            if 'target_modules=' in line and 'MEMORY FIX' in line:
                line = '    target_modules=["q_proj", "v_proj"], # 📉 MEMORY FIX: Fewer modules\n'
            if 'optim=' in line and 'MEMORY FIX' in line:
                line = '    optim="paged_adamw_8bit", # 📉 MEMORY FIX: 8-bit optimizer\n'
            if 'max_seq_length=' in line and 'MEMORY FIX' in line:
                line = '    max_seq_length=512, # 📉 MEMORY FIX: Hard cap at 512 tokens\n'
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
