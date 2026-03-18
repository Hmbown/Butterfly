#!/usr/bin/env python3
"""Smoke test: vLLM + Qwen3.5-9B on DGX Spark."""
import time

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/hmbown/HF_Models/Qwen3.5-9B"


def main() -> None:
    print("Loading Qwen3.5-9B with vLLM...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )
    load_s = time.time() - t0
    print(f"Model loaded in {load_s:.1f}s")

    sampling = SamplingParams(
        temperature=0.7,
        max_tokens=128,
        top_p=0.9,
    )

    prompts = [
        "Explain sparse attention in transformer models in one paragraph:",
        "The capital of France is",
        "Write a Python function that checks if a number is prime:",
    ]

    print(f"\nGenerating {len(prompts)} prompts...")
    t1 = time.time()
    outputs = llm.generate(prompts, sampling)
    gen_s = time.time() - t1

    total_tokens = 0
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        ntok = len(out.outputs[0].token_ids)
        total_tokens += ntok
        print(f"\n--- Prompt {i+1} ({ntok} tokens) ---")
        print(text[:400])

    print(f"\n=== Summary ===")
    print(f"Load time:    {load_s:.1f}s")
    print(f"Gen time:     {gen_s:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Throughput:   {total_tokens / gen_s:.1f} tok/s")
    print("vLLM smoke test PASSED")


if __name__ == "__main__":
    main()
