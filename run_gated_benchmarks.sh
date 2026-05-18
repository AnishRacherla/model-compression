# REMAINING MODELS SCRIPT for WMT26
# This script contains ONLY the models that haven't been computed yet
# tailored specifically to fit in 16GB Colab VRAM without crashing.

echo "Running Constrained Track Gemma 3 12B..."
python benchmark.py --model_id google/gemma-3-12b --precision int8
python benchmark.py --model_id google/gemma-3-12b --precision int4

echo "Running Aya Expanse 8B..."
python benchmark.py --model_id CohereForAI/aya-expanse-8b --precision int8
python benchmark.py --model_id CohereForAI/aya-expanse-8b --precision int4

echo "Running Ministral 3 14B..."
python benchmark.py --model_id mistralai/Ministral-3-14B --precision int4

echo "Running EuroLLM..."
python benchmark.py --model_id EuroLLM/EuroLLM-9B --precision int8
python benchmark.py --model_id EuroLLM/EuroLLM-9B --precision int4

echo "Running GPT OSS 20B with INT4 ONlY..."
python benchmark.py --model_id gpt-oss/gpt-oss-20b --precision int4

echo "Done! All results exported."