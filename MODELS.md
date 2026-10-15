# Models

## Open-weight models

All were run on an Apple M1 Pro (32 GB) at 16-bit floating-point precision, temperature 0, seed 1, maximum output 128 tokens. Six models were run with llama.cpp build 10330 (commit 687e77892) from GGUF files; Lingshu-7B was run with transformers 5.15.0 and PyTorch 2.13.0 from the original safetensors files.

| Model | Original model | File used | SHA-256 of weights | SHA-256 of image projector |
|---|---|---|---|---|
| Qwen3-VL 4B | Qwen/Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Instruct-BF16.gguf | ee0c02954276cedd46a41ae159345cfee998954487a9566d8540b2ccdf639eec | 364ef9a7502b7e9a834b9c013b953670f6ba2550b4a26fc043c317fa484eb53e |
| Qwen3-VL 8B | Qwen/Qwen3-VL-8B-Instruct | Qwen3-VL-8B-Instruct-BF16.gguf | 7159768faf43dedcdbf6d8e78ee6f4325ada151be2a248d785e4b683b87c782a | 6516bb64bae1503a0fcd7ec9fa39655f8c481580be0a0a066397941d9761c9f4 |
| Qwen2.5-VL 7B | Qwen/Qwen2.5-VL-7B-Instruct | Qwen2.5-VL-7B-Instruct-BF16.gguf | abed6d9901c05459bb795a5d5fb040a86f256c17b01a6754857891e1032def09 | f0edf43c09b69d6e5dd24262f33b356a1e9dd978e7c3299b3e69141fcbb87553 |
| Lingshu-7B | lingshu-medical-mllm/Lingshu-7B | safetensors | 9a6c65a4a3b07a7775be51e1a0b2b47e0e1dbe7e90492ec0dde4ce2c8386419a | included in weights |
| Gemma 3 4B | google/gemma-3-4b-it | gemma-3-4b-it-BF16.gguf | 7bcfca74279608552011d3c4d11e9298934502ffa410eb21b9047a6276b87ba6 | afb3947b2f87ddd0cd8c9371d5337a30bac1b56d279d2778837797c738b6e7cc |
| MedGemma 1 4B | google/medgemma-4b-it | medgemma-4b-it-BF16.gguf | 6cc142f3678a08a19f80be9e7b9855f9c507cf3fcc201c843a3df0df048bb908 | bd1c6afda6d7c3f3d38ac47278bd9b753db431546db93c707d26b7216d47b22c |
| MedGemma 1.5 4B | google/medgemma-1.5-4b-it | medgemma-1.5-4b-it-BF16.gguf | a8e48bfcc66d34a9477338cba90e5345aa6571dba955cb6040dd12d18d7b0085 | 9743de36ba6d7b55ca28d03eeecb5336f71d5315d3eeb63dd9e354af5b6a5a8a |

The same checksums are recorded with every call in the `model_sha256` and `mmproj_sha256` columns of the result files. The GGUF files were downloaded from Hugging Face. [SOURCE REPOSITORIES OF THE GGUF FILES TO BE CONFIRMED BY THE AUTHOR]

## Commercial models

Accessed through each vendor's application programming interface between August 25 and September 11, 2026.

| Model | Identifier returned by the interface | Reasoning setting | Temperature |
|---|---|---|---|
| GPT-5.6 (OpenAI) | gpt-5.6-sol | medium | not adjustable |
| Gemini 3.1 Pro (Google) | gemini-3.1-pro-preview | medium | 0; image resolution set to high |
| Claude Fable 5 (Anthropic) | claude-fable-5 | medium | not adjustable |

The identifier, settings, token counts, and time stamp of every call are in `results/frontier_results.csv`, `results/frontier_control_results.csv`, and `results/frontier_raw/`.
