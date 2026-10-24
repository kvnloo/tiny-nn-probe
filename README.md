# Tiny NN edge probe

Public synthetic code and toy weights only. No private traces or training data.

## Tiny model

A deterministic float32 64→128 ReLU→16 MLP (10,384 parameters), procedural toy weights in `tiny_nn.py`, fixture in `fixture.json`. This is an arithmetic/performance probe, **not a trained classifier**. The 64 ones input has exact expected output `[-7.5,-5,-2.5,0,2.5,5,7.5,10]` repeated twice. NumPy and exported ONNX execute the same graph.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python probe.py > hardware.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tiny_nn.py --export-onnx tiny.onnx --onnx tiny.onnx > tiny.json
```

The probe reports OS/CPU/RAM/disk, package versions, torch CUDA/MPS, NVIDIA/Vulkan/OpenCL discovery, two public egress requests, and resource limits. Unreadable devices or missing utilities are explicit failures, not evidence that hardware lacks a GPU. Wall-time limits remain unknown unless externally documented; RLIMIT_CPU is not a wall timeout.

## GGUF ladder

`models.json` pins every Hugging Face repository revision. Downloads record their SHA256. SmolLM2 135M Q4 → 360M Q4 → Qwen3 0.6B Q4 → Q8 → 1.7B Q4. Licenses are Apache-2.0 on the pinned upstream model cards.

Supply a known llama.cpp binary directory containing `llama-bench` and `llama-server`; no compilation is silently done. Record binary version/source SHA separately. Recommended source pin: `21ec3071467b168991e223f2e5b09e571a11eb96` from ggml-org/llama.cpp, or a device's existing audited build.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python gguf_bench.py --bin-dir /path/to/bin --threads 4 --out results/cpu
# Only on admitted OpenCL-capable hardware, using its verified ICD/library environment:
python gguf_bench.py --bin-dir /path/to/opencl/bin --threads 2 --gpu-layers 99 --out results/gpu
```

`llama-bench` runs pp128/tg32, three repeats. Classification uses eight public synthetic 3-choice items, temperature0/seed1729, max8 generated tokens, no forced grammar; raw responses and per-request timing are saved. Qwen thinking is explicitly closed. Accuracy is only this tiny diagnostic set, not a production quality estimate. The ladder stops at <5 decode tok/s, runtime/download failure, or the configured model-file budget. `--max-model-mib` bounds the **file**, not total runtime RAM; an external admission wrapper must reserve runtime/context overhead. `--models smol135-q4` selects a subset. No network or GPU availability is assumed.

Phones must use existing fleet polite admission (battery/thermal/owner-memory gates, nice/ionice/affinity). No hard cgroup isolation is claimed for PRoot. Never bypass a safety or account quota pause. CPU versus OpenCL paths require separate receipts.
