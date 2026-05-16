# Run this script to execute GATED models for the WMT26 Unconstrained/Constrained task within Colab's 16GB VRAM limit
# Use: !bash run_gated_benchmarks.sh
# IMPORTANT: You MUST have your Hugging Face Token loaded and have accepted licenses on the website for these models.

echo "Running Constrained Track Model Proxy (Gemma 7B as 12B substitute)..."
python benchmark.py --model_id google/gemma-7b --precision int8
python benchmark.py --model_id google/gemma-7b --precision int4

echo "Running Llama 3.1 8B..."
# Skipping FP16 for Llama 8B due to high likelihood of OOM on T4 (requires ~16GB exactly, usually crashes overhead)
python benchmark.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --precision int8
python benchmark.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --precision int4

echo "Running Cohere Command R 7B..."
# Same as Llama, sticking to INT8 and INT4 for stability constraints.
python benchmark.py --model_id CohereForAI/c4ai-command-r-v01 --precision int8
python benchmark.py --model_id CohereForAI/c4ai-command-r-v01 --precision int4

echo "Done! All results are saved in benchmark_results.csv."
