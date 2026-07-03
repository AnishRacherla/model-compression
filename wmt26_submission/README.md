# Compressed Aya Expanse 8B (int8)

**Track:** Unconstrained
**Language Pair:** English to Chinese, Simplified (eng-zho_Hans)
**Base Model:** CohereForAI/aya-expanse-8b

### Compression Pipeline
1. **Layer Pruning**: Reduced from 32 to 28 layers.
2. **LoRA Fine-Tuning**: Fine-tuned on WMT19 En-Zh data to recover translation quality.
3. **Merging**: `merge_and_unload()` to bake LoRA weights into a single standalone model.
4. **Quantization**: int8 quantization via bitsandbytes at runtime.

The model is dynamically loaded from Hugging Face: `AnishRacherla/aya-expanse-8b-compressed-final-int8`.
