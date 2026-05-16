# Run this script to execute explicitly open models for the WMT26 Unconstrained task within Colab's 16GB VRAM limit
# Use: !bash run_all_benchmarks.sh

# ONLY OPEN WEIGHTS INCLUDED HERE (No Hugging Face token required)

echo "Running Mistral 7B..."
python benchmark.py --model_id mistralai/Mistral-7B-v0.1 --precision fp16
python benchmark.py --model_id mistralai/Mistral-7B-v0.1 --precision int8
python benchmark.py --model_id mistralai/Mistral-7B-v0.1 --precision int4

echo "Running Qwen 2.5 7B..."
python benchmark.py --model_id Qwen/Qwen2.5-7B --precision fp16
python benchmark.py --model_id Qwen/Qwen2.5-7B --precision int8
python benchmark.py --model_id Qwen/Qwen2.5-7B --precision int4

echo "Done! All results are saved in benchmark_results.csv."
