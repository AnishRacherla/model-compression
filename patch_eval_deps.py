import json

with open("evaluate_finetuned_model.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        text = ''.join(source)
        
        # 1. Remove the transformers downgrade from the first pip install cell
        if '!pip install' in text and '"transformers<4.45"' in text:
            new_source = [
                "!pip install -q transformers datasets peft accelerate bitsandbytes evaluate unbabel-comet pytorch-lightning\n"
            ]
            cell['source'] = new_source
            
        # 2. Add the downgrade right before running COMET
        if '!python run_comet.py' in text:
            new_source = []
            for line in source:
                if '!python run_comet.py' in line:
                    new_source.append("# Downgrade transformers ONLY for the isolated comet subprocess\n")
                    new_source.append("!pip install -q \"transformers<4.45\"\n")
                    new_source.append("!python run_comet.py\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open("evaluate_finetuned_model.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Evaluation notebook patched.")
