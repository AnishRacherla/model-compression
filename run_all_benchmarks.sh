# Run this script to execute all the recommended models for the WMT26 Unconstrained/Constrained task within Colab's 16GB VRAM limit
# Use: !bash run_all_benchmarks.sh

# Note: Some models listed here require you to accept their license on HuggingFace!
# - Llama 3.1: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
# - Cohere Command R: https://huggingface.co/CohereForAI/c4ai-command-r-v01
# - Qwen 2.5: https://huggingface.co/Qwen/Qwen2.5-7B
# - Gemma 3 12B: https://huggingface.co/google/gemma-3-12b (or closest available right now like gemma-2-9b or gemma-7b)

echo "Running Constrained Track Model Proxy (Gemma 7B as placeholder since 12B not released publicly yet)..."
python benchmark.py --model_id google/gemma-7b --precision int8
python benchmark.py --model_id google/gemma-7b --precision int4

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