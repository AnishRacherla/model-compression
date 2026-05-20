import argparse
import time
import torch
import os
import csv

# Disable TensorFlow and JAX so Transformers doesn't try to import them and crash due to Colab dependency issues
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def measure_memory_and_speed(model_id, precision, prompt, max_new_tokens=50):
    print(f"Benchmarking {model_id} in {precision} precision...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU detected. Running on CPU. Inference will be very slow.")
        if precision in ["int8", "int4", "fp8"]:
            print(f"WARNING: Quantization/Precision '{precision}' typically requires a GPU. The script may fail or fall back depending on your CPU.")
    
    # Configure precision
    kwargs = {"device_map": "auto" if device == "cuda" else "cpu"}
    if precision == "fp32":
        kwargs["torch_dtype"] = torch.float32
    elif precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif precision == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif precision == "fp8":
        # Requires fp8 support in transformers/bnb
        kwargs["torch_dtype"] = torch.float8_e4m3fn
        
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    load_time = time.time() - start_load
    
    # Measure Size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    
    # Measure Inference Speed
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_infer = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    infer_time = time.time() - start_infer
    
    # Calculate decoded tokens speed
    num_generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    tokens_per_sec = num_generated_tokens / infer_time
    
    # Memory
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        mem_label = "Peak VRAM Usage"
    else:
        # Just give a rough estimate of system memory without psutil to avoid extra dependencies
        mem_label = "GPU Memory (N/A on CPU)"
    
    print("-" * 40)
    print(f"Results for {model_id} ({precision}):")
    print(f"Load Time: {load_time:.2f} s")
    print(f"Estimated Model Size (Weights): {model_size:.2f} MB")
    print(f"{mem_label}: {peak_mem:.2f} MB")
    print(f"Inference Time: {infer_time:.2f} s")
    print(f"Generation Speed: {tokens_per_sec:.2f} tokens/s")
    
    # Optional FLORES-200 / COMET Evaluation
    comet_score_val = "N/A"
    if args.eval_comet:
        print("-" * 40)
        print("Starting FLORES-200 translation and COMET evaluation...")
        try:
            from datasets import load_dataset
            import evaluate
            
            # Load a small slice of FLORES dev set (eng_Latn -> deu_Latn)
            print(f"Loading {args.eval_samples} samples from facebook/flores (eng_Latn -> deu_Latn)...")
            dataset = load_dataset('facebook/flores', 'eng_Latn-deu_Latn', split='dev', trust_remote_code=True)
            dataset = dataset.select(range(min(args.eval_samples, len(dataset))))
            
            sources = dataset['sentence_eng_Latn']
            references = dataset['sentence_deu_Latn']
            predictions = []
            
            for i, src_text in enumerate(sources):
                # Standard zero-shot translation prompt
                prompt_eval = f"Translate the following English text to German:\n{src_text}\nTranslation:"
                inputs_eval = tokenizer(prompt_eval, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    # Generate with enough tokens for the translation
                    outputs_eval = model.generate(**inputs_eval, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
                num_gen = outputs_eval.shape[1] - inputs_eval.input_ids.shape[1]
                pred = tokenizer.decode(outputs_eval[0][-num_gen:], skip_special_tokens=True).strip()
                predictions.append(pred)
                
                if (i + 1) % 5 == 0:
                    print(f"Translated {i + 1}/{len(sources)} sentences...")
                    
            print("Calculating COMET score. This requires downloading the COMET model if not cached...")
            comet = evaluate.load('comet')
            results = comet.compute(predictions=predictions, references=references, sources=sources)
            comet_score_list = list(results["scores"])
            mean_comet = sum(comet_score_list) / len(comet_score_list)
            comet_score_val = round(mean_comet, 4)
            print(f"Mean COMET Score: {comet_score_val}")
        except Exception as e:
            print(f"Error during COMET evaluation: {e}")
            
    print("-" * 40)
    
    # Save to CSV
    csv_file = "benchmark_results.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Precision", "Load Time (s)", "Model Size (MB)", "Peak VRAM (MB)", "Inference Time (s)", "Tokens/s", "COMET (eng->deu)"])
        writer.writerow([model_id, precision, round(load_time, 2), round(model_size, 2), round(peak_mem, 2), round(infer_time, 2), round(tokens_per_sec, 2), comet_score_val])
    print(f"Results appended to {csv_file}")
    
    return peak_mem, infer_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-2b")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16", "fp8", "int8", "int4"], default="fp16")
    parser.add_argument("--prompt", type=str, default="Translate the following English text to German: 'The quick brown fox jumps over the lazy dog.' Translation:")
    parser.add_argument("--eval_comet", action="store_true", help="Run translation evaluation using FLORES-200 and COMET metric.")
    parser.add_argument("--eval_samples", type=int, default=20, help="Number of sentences to translate for COMET evaluation (default: 20 to avoid extreme Colab timeouts).")
    args = parser.parse_args()
    
    measure_memory_and_speed(args.model_id, args.precision, args.prompt)