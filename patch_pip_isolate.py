import json

with open("evaluate_finetuned_model.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        text = ''.join(source)
        
        # 1. Clean the first pip install cell
        if '!pip install -q "pip<24.1"' in text or 'bitsandbytes' in text:
            if '!python run_comet.py' not in text:  # Make sure we don't mess with the last cell yet
                new_source = [
                    "!pip install -q -U \"bitsandbytes>=0.46.1\" transformers datasets peft accelerate evaluate\n"
                ]
                cell['source'] = new_source
                
        # 2. Update the comet cell to install its broken dependencies ONLY at the end
        if '!python run_comet.py' in text:
            new_source = []
            for line in source:
                if '!pip install' in line:
                    continue  # Remove old pip installs in this cell
                if '!python run_comet.py' in line:
                    new_source.append("# Install comet dependencies strictly at the end so they don't break bitsandbytes!\n")
                    new_source.append("!pip install -q unbabel-comet pytorch-lightning\n")
                    new_source.append("!python run_comet.py\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open("evaluate_finetuned_model.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Pip dependencies fully isolated.")
