import json

with open('finetune_wmt_lora.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        text = "".join(source)
        
        # 1. Replace the dataset processing cell
        if "def process_dataset" in text and "TARGET_SIZE" in text:
            new_source = [
                "print(\"Loading local cleaned dataset...\")\n",
                "\n",
                "from datasets import load_dataset\n",
                "\n",
                "# Make sure your file is uploaded and named 'cleaned_dataset.parquet'\n",
                "zh_ds = load_dataset(\"parquet\", data_files=\"cleaned_dataset.parquet\", split=\"train\")\n",
                "\n",
                "def process_dataset(ds):\n",
                "    def calculate_length(example):\n",
                "        return {\"length\": len(example[\"source\"])}\n",
                "\n",
                "    ds = ds.map(calculate_length, num_proc=4)\n",
                "    ds = ds.filter(lambda x: 10 < x[\"length\"] < 1500)\n",
                "    ds = ds.shuffle(seed=42)\n",
                "\n",
                "    def format_prompts(batch):\n",
                "        texts = []\n",
                "        for src, tgt in zip(batch[\"source\"], batch[\"target\"]):\n",
                "            texts.append(\n",
                "                f\"Translate from English to Simplified Chinese:\\n\"\n",
                "                f\"en: {src}\\n\"\n",
                "                f\"zh: {tgt}\"\n",
                "            )\n",
                "        return {\"text\": texts}\n",
                "\n",
                "    return ds.map(format_prompts, batched=True, remove_columns=ds.column_names)\n",
                "\n",
                "train_dataset = process_dataset(zh_ds)\n",
                "print(\"\\nDataset Ready\")\n",
                "print(\"Total samples:\", len(train_dataset))\n",
                "print(\"\\nSample:\")\n",
                "print(train_dataset[0][\"text\"])\n"
            ]
            cell['source'] = new_source
            
        # 2. Modify SFTConfig properties
        if "sft_config = SFTConfig(" in text:
            new_sft_source = []
            for line in source:
                if "save_steps=" in line:
                    new_sft_source.append("    save_steps=25,             # ⚡ CHANGED: Save every ~10-15 minutes\n")
                elif "save_total_limit=" in line:
                    new_sft_source.append("    save_total_limit=10,       # ⚡ CHANGED: Keep up to 10 checkpoints\n")
                elif "push_to_hub=" in line:
                    new_sft_source.append("    push_to_hub=True,          # ⚡ CHANGED: Push checkpoints directly to Hub\n")
                    new_sft_source.append("    hub_model_id=hub_model_id,\n")
                    new_sft_source.append("    hub_strategy=\"checkpoint\",\n")
                else:
                    new_sft_source.append(line)
            cell['source'] = new_sft_source

with open('finetune_wmt_lora.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Updated finetune_wmt_lora.ipynb successfully.')
