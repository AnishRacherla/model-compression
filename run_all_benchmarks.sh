# Run this script to execute all the recommended models for the WMT26 Unconstrained/Constrained task within Colab's 16GB VRAM limit
# Use: !bash run_all_benchmarks.sh

# 1. Constrained Track Model (Gemma 3 12B)
# Note: Since it's a 12B model, it CANNOT run in fp16 on a 16GB T4 GPU. We MUST use int8 or int4!
echo "Running Constrained Track Model (Gemma 3 12B)..."
python benchmark.py --model_id google/gemma-3-12b --precision int8
python benchmark.py --model_id google/gemma-3-12b --precision int4

# 2. Unconstrained Track Models (< 20B parameters)
echo "Running Llama 3.1 8B..."
python benchmark.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --precision fp16
python benchmark.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --precision int8
python benchmark.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --precision int4

echo "Running Mistral 7B..."
python benchmark.py --model_id mistralai/Mistral-7B-v0.1 --precision fp16
python benchmark.py --model_id mistralai/Mistral-7B-v0.1 --precision int8
python benchmark.py --model_id mistralai/Mistral-7B-v0.1 --precision int4

echo "Running Qwen 2.5 7B..."
python benchmark.py --model_id Qwen/Qwen2.5-7B --precision fp16
python benchmark.py --model_id Qwen/Qwen2.5-7B --precision int8
python benchmark.py --model_id Qwen/Qwen2.5-7B --precision int4

# Note: Cohere R 7B and Aya Expanse 8B might require special trust agreements on HF, exactly like Gemma did!
# Make sure to acknowledge their licenses on HuggingFace if you get a 401 error.
echo "Running Cohere R 7B..."
python benchmark.py --model_id CohereForAI/c4ai-command-r-v01 --precision int8
python benchmark.py --model_id CohereForAI/c4ai-command-r-v01 --precision int4

echo "Done! All results are saved in benchmark_results.csv."