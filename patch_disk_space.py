import json

with open("quantization_benchmark.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    text = ''.join(cell['source'])

    # ── Patch int4 cell: remove save_pretrained + upload_folder, use push_to_hub ──
    if 'INT4_LOCAL' in text and 'compressed_int4' in text and 'model_int4.save_pretrained' in text:
        cell['source'] = [
            "clear_gpu()\n",
            "vram_before = get_vram_gb()\n",
            "\n",
            "bnb_int4 = BitsAndBytesConfig(\n",
            "    load_in_4bit=True,\n",
            "    bnb_4bit_use_double_quant=True,\n",
            "    bnb_4bit_quant_type='nf4',\n",
            "    bnb_4bit_compute_dtype=torch.float16,\n",
            ")\n",
            "\n",
            "print('Loading merged model in int4...')\n",
            "t0 = time.time()\n",
            "model_int4 = AutoModelForCausalLM.from_pretrained(\n",
            "    MERGED_DIR, quantization_config=bnb_int4, device_map='auto', trust_remote_code=True\n",
            ")\n",
            "t_load = time.time() - t0\n",
            "vram_int4 = get_vram_gb() - vram_before\n",
            "print(f'⏱️  Load time : {t_load:.1f}s')\n",
            "print(f'📦  VRAM used : {vram_int4:.2f} GB')\n",
            "\n",
            "preds_int4, tps_int4 = translate_and_benchmark(model_int4, tokenizer, sources, 'compressed-int4')\n",
            "print(f'⚡  Tokens/sec: {tps_int4:.1f}')\n",
            "results['compressed_int4'] = {'load_time': t_load, 'vram_gb': vram_int4, 'tps': tps_int4, 'predictions': preds_int4}\n",
            "\n",
            "# Push DIRECTLY to Hub — no local disk save (saves ~3.5GB of disk space)\n",
            "print(f'\\nPushing int4 model directly to Hub: {INT4_REPO}...')\n",
            "model_int4.push_to_hub(INT4_REPO, token=write_token,\n",
            "    commit_message='Pruned+finetuned+merged int4 NF4 quantized — standalone')\n",
            "tokenizer.push_to_hub(INT4_REPO, token=write_token)\n",
            "print(f'✅ int4 pushed to https://huggingface.co/{INT4_REPO}')\n",
            "\n",
            "del model_int4\n",
            "clear_gpu()\n",
            "print('✅ int4 done. GPU cleared.')\n"
        ]

    # ── Patch int8 cell: remove save_pretrained + upload_folder, use push_to_hub ──
    if 'INT8_LOCAL' in text and 'model_int8.save_pretrained' in text:
        cell['source'] = [
            "bnb_int8 = BitsAndBytesConfig(load_in_8bit=True)\n",
            "\n",
            "clear_gpu()\n",
            "vram_before = get_vram_gb()\n",
            "\n",
            "print('Loading merged model in int8...')\n",
            "t0 = time.time()\n",
            "model_int8 = AutoModelForCausalLM.from_pretrained(\n",
            "    MERGED_DIR, quantization_config=bnb_int8, device_map='auto', trust_remote_code=True\n",
            ")\n",
            "t_load = time.time() - t0\n",
            "vram_int8 = get_vram_gb() - vram_before\n",
            "print(f'⏱️  Load time : {t_load:.1f}s')\n",
            "print(f'📦  VRAM used : {vram_int8:.2f} GB')\n",
            "\n",
            "preds_int8, tps_int8 = translate_and_benchmark(model_int8, tokenizer, sources, 'compressed-int8')\n",
            "print(f'⚡  Tokens/sec: {tps_int8:.1f}')\n",
            "results['compressed_int8'] = {'load_time': t_load, 'vram_gb': vram_int8, 'tps': tps_int8, 'predictions': preds_int8}\n",
            "\n",
            "# Push DIRECTLY to Hub — no local disk save (saves ~7GB of disk space)\n",
            "print(f'\\nPushing int8 model directly to Hub: {INT8_REPO}...')\n",
            "model_int8.push_to_hub(INT8_REPO, token=write_token,\n",
            "    commit_message='Pruned+finetuned+merged int8 quantized — standalone')\n",
            "tokenizer.push_to_hub(INT8_REPO, token=write_token)\n",
            "print(f'✅ int8 pushed to https://huggingface.co/{INT8_REPO}')\n",
            "\n",
            "# Free merged_fp16 from disk — no longer needed\n",
            "import shutil\n",
            "shutil.rmtree(MERGED_DIR, ignore_errors=True)\n",
            "print(f'🗑️  Deleted local {MERGED_DIR} to free disk space.')\n",
            "\n",
            "del model_int8\n",
            "clear_gpu()\n",
            "print('✅ int8 done. GPU cleared.')\n"
        ]

with open("quantization_benchmark.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook patched to use push_to_hub directly (no disk save).")
