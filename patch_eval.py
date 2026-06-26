import json

with open('finetune_wmt_lora.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update the final cell of the training portion to explicitly push to hub
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if 'trainer.model.save_pretrained(output_dir)' in ''.join(cell['source']):
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if 'trainer.model.save_pretrained(output_dir)' in line:
                    new_source.append("print('Pushing final model directly to Hugging Face...')\n")
                    new_source.append("trainer.push_to_hub(commit_message='Final fine-tuned model')\n")
            cell['source'] = new_source

# 2. Remove the last three cells (which are the evaluation & zip cells that cause OOM)
# The training cell is index 8 (0-indexed). The eval cells are 9, 10, 11.
nb['cells'] = nb['cells'][:9]

with open('finetune_wmt_lora.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Training notebook updated to push to Hub and skip OOM evaluation.")
