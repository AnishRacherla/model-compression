import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    # Accept positional or named arguments per WMT guidelines
    parser.add_argument('--lang-pair', type=str, default='eng-zho_Hans')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    # If the user passes positional arguments instead: `bash run.sh ces-deu 8 < input.txt > output.txt`
    # The setup wrapper will just pass the named arguments standard.
    return parser.parse_known_args()[0]

def main():
    args = parse_args()
    
    # All logging MUST go to stderr. stdout is reserved ONLY for the output text if needed, 
    # but since we are writing to args.output, stdout should remain clean.
    print(f"[*] Starting inference for {args.lang_pair}...", file=sys.stderr)
    print(f"[*] Input: {args.input}", file=sys.stderr)
    print(f"[*] Output: {args.output}", file=sys.stderr)
    
    if args.lang_pair != "eng-zho_Hans":
        print(f"[!] Warning: Model optimized for eng-zho_Hans. Requested: {args.lang_pair}", file=sys.stderr)

    repo_id = "AnishRacherla/aya-expanse-8b-compressed-final-int8"

    print(f"[*] Loading tokenizer from {repo_id}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    
    # Left padding is required for batched generation!
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print(f"[*] Loading model in 8-bit precision...", file=sys.stderr)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, 
        quantization_config=bnb_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.eval()

    # Read input lines
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    print(f"[*] Loaded {len(lines)} lines. Translating...", file=sys.stderr)

    out_file = open(args.output, 'w', encoding='utf-8')

    for i in tqdm(range(0, len(lines), args.batch_size), file=sys.stderr, desc="Translating"):
        batch_lines = lines[i:i+args.batch_size]
        
        # We must use the exact prompt format the model was fine-tuned on!
        prompts = [f"Translate from English to Simplified Chinese:\nen: {src}\nzh:" for src in batch_lines]
        
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False
            )
        
        # Decode only the generated tokens (strip the left-padded prompt from the output)
        prompt_length = inputs.input_ids.shape[1]
        
        for j, out_tokens in enumerate(outputs):
            generated_tokens = out_tokens[prompt_length:]
            translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # Remove any stray newlines so it stays exactly one output line per input line
            translation = translation.strip().replace('\\n', ' ').replace('\n', ' ')
            
            # Write to output file
            out_file.write(translation + '\n')
            
        out_file.flush()
            
    out_file.close()
    print("[*] Evaluation complete!", file=sys.stderr)

if __name__ == "__main__":
    main()
