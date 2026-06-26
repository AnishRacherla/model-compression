import json
import os

with open('finetune_wmt_lora.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        text = ''.join(cell['source'])
        
        if '# --- RESUME LOGIC ---' in text:
            new_source = []
            for line in cell['source']:
                if 'last_checkpoint = get_last_checkpoint(output_dir)' in line:
                    new_source.append("    last_checkpoint = get_last_checkpoint(output_dir)\n")
                    new_source.append("    \n")
                    new_source.append("    # TRL pushes the latest checkpoint to a folder named 'last-checkpoint' on the Hub.\n")
                    new_source.append("    # get_last_checkpoint() only looks for 'checkpoint-XXX'. We need to manually point to 'last-checkpoint'.\n")
                    new_source.append("    if last_checkpoint is None and os.path.exists(os.path.join(output_dir, 'last-checkpoint')):\n")
                    new_source.append("        last_checkpoint = os.path.join(output_dir, 'last-checkpoint')\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open('finetune_wmt_lora.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Resume logic patched for 'last-checkpoint' folder.")
