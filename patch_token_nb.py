import json

with open('benchmark_int8_final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        if 'CohereForAI/aya-expanse-8b' in source:
            source = source.replace(
                "AutoTokenizer.from_pretrained('CohereForAI/aya-expanse-8b', trust_remote_code=True)",
                "AutoTokenizer.from_pretrained('CohereForAI/aya-expanse-8b', trust_remote_code=True, token='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')"
            )
            source = source.replace(
                "'CohereForAI/aya-expanse-8b', quantization_config=bnb_int8,",
                "'CohereForAI/aya-expanse-8b', quantization_config=bnb_int8, token='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',"
            )
            
        if 'INT8_REPO' in source and 'AutoTokenizer' in source:
            source = source.replace(
                "AutoTokenizer.from_pretrained(INT8_REPO, trust_remote_code=True)",
                "AutoTokenizer.from_pretrained(INT8_REPO, trust_remote_code=True, token='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')"
            )
            source = source.replace(
                "INT8_REPO, quantization_config=bnb_int8, device_map='auto', trust_remote_code=True",
                "INT8_REPO, quantization_config=bnb_int8, device_map='auto', trust_remote_code=True, token='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'"
            )
            
        cell['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in source.split('\n') if line]

with open('benchmark_int8_final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched to explicitly pass Hugging Face token to all downloads!")
