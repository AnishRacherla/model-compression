import json

file_path = 'c:/Users/Anish/OneDrive/Desktop/modelcompression/model-compression/finetune_wmt_lora.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Remove the bad loop
            if 'for param in model.parameters():' in line:
                continue
            if 'if param.dtype == torch.bfloat16:' in line:
                continue
            if 'param.data = param.data.to(torch.float16)' in line:
                continue
            if '# 🚨 CRITICAL FIX' in line:
                continue
            
            # Add a proper cast after peft model creation
            if 'model.print_trainable_parameters()' in line:
                new_source.append(line)
                new_source.append('\n# 📉 MEMORY FIX: Force all trainable params to float32 to absolutely guarantee no bfloat16 grads\n')
                new_source.append('for name, param in model.named_parameters():\n')
                new_source.append('    if param.requires_grad:\n')
                new_source.append('        param.data = param.data.to(torch.float32)\n')
                continue
                
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
