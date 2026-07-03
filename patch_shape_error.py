import json

# Load the notebook
with open('benchmark_int8_final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Cell 3 (Helpers) and replace the chat template logic
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        if 'def translate_and_benchmark' in source:
            old_logic = '''        messages = [{"role": "user", "content": f"Translate from English to Simplified Chinese: {src}"}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
        inputs = {"input_ids": input_ids}'''
            
            new_logic = '''        messages = [{"role": "user", "content": f"Translate from English to Simplified Chinese: {src}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)'''
            
            source = source.replace(old_logic, new_logic)
            
            cell['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in source.split('\n') if line]

# Save the patched notebook
with open('benchmark_int8_final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched to fix the BatchEncoding shape error!")
