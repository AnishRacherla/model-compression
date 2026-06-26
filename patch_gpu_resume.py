import json

with open('finetune_wmt_lora.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        text = ''.join(cell['source'])

        # Fix 1: Force single GPU to fix multi-GPU cross-entropy device mismatch
        if 'device_map="auto"' in text and 'AutoModelForCausalLM' in text:
            new_source = []
            for line in cell['source']:
                if 'device_map="auto"' in line:
                    new_source.append('    device_map={"": 0},  # Force single GPU - avoids cuda:0/cuda:1 cross-entropy device mismatch on dual T4\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source

        # Fix 2: Download checkpoint from Hub before resume check
        if '# --- RESUME LOGIC ---' in text:
            new_source = []
            for line in cell['source']:
                if '# --- RESUME LOGIC ---' in line:
                    # Inject Hub checkpoint restore before the resume logic
                    new_source.append("# --- RESTORE CHECKPOINT FROM HUB (survives Kaggle restarts) ---\n")
                    new_source.append("from huggingface_hub import snapshot_download\n")
                    new_source.append("try:\n")
                    new_source.append("    print('Checking HF Hub for existing checkpoint...')\n")
                    new_source.append("    snapshot_download(\n")
                    new_source.append("        repo_id=hub_model_id,\n")
                    new_source.append("        local_dir=output_dir,\n")
                    new_source.append("        token=write_token,\n")
                    new_source.append("        ignore_patterns=['*.md', 'README*'],\n")
                    new_source.append("    )\n")
                    new_source.append("    print(f'✅ Checkpoint restored from Hub to {output_dir}')\n")
                    new_source.append("except Exception as e:\n")
                    new_source.append("    print(f'No Hub checkpoint found or error: {e}. Starting fresh.')\n")
                    new_source.append("\n")
                    new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open('finetune_wmt_lora.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Both fixes applied successfully.")
