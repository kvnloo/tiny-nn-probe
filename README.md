# Tiny NN edge probe

Public synthetic code and toy weights only. No private traces or training data.

## Tiny model

A deterministic float32 64→128 ReLU→16 MLP (10,384 parameters), procedural toy weights in `tiny_nn.py`, fixture in `fixture.json`. This is an arithmetic/performance probe, **not a trained classifier**. The 64 ones input has exact expected output `[-7.5,-5,-2.5,0,2.5,5,7.5,10]` repeated twice. NumPy and exported ONNX execute the same graph.

`tiny.onnx` is the checked ONNX copy exported and executed on the public Avrea worker (GitHub Actions job run 40127230851). SHA256: `8d66e916afee77a160222fa4649cd1b70a4127e49212e05e563c272f8fb00522` (41,779 bytes). NumPy and ONNX Runtime both matched the fixture. Regenerate using the command below; CPU one-thread settings make the tiny benchmark independent of BLAS oversubscription.

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

`llama-bench` runs pp128/tg32, three repeats. Classification uses eight public synthetic 3-choice items, temperature0/seed1729, max8 generated tokens, no forced grammar; raw responses and per-request timing are saved. Qwen thinking is explicitly closed. Accuracy is only this tiny diagnostic set, not a production quality estimate. The ladder stops at <5 decode tok/s, runtime/download failure, or an explicitly configured model-file budget. `--max-model-mib` bounds the **file**, not total runtime RAM; account separately for runtime/context overhead under the active worker policy. `--models smol135-q4` selects a subset. No network or GPU availability is assumed.

Use the operator-selected live worker policy; this probe does not impose universal phone RAM, core-affinity, battery, or thermal limits. Read the active configuration instead of treating old measurements as current admission rules. Record actual allowed/online cores, memory allowance, battery and thermal observations, and accelerator execution. No hard cgroup isolation is claimed for PRoot. CPU versus OpenCL paths require separate receipts, with matching benchmark settings for a speedup comparison.

## Native MNN comparison tools

`mnn/inputs.json` pins MNN 3.6.1 source files and the official Android ARM64 CPU/OpenCL shared-library archive by hash. `mnn/build_android.py` cross-links the upstream `llm_bench` / `llm_demo` and this repository's small `edge_mnn` raw-prompt/HTTP executable. Run the build **on an admitted x86-64 Linux worker**, not on the benchmark device or an orchestration host:

```sh
python3 mnn/build_android.py --out results/mnn
```

The build downloads Android NDK r27d and verifies Google's published archive checksum. It writes `results/mnn/build-receipt.json` and `results/mnn/artifact/{bin,lib}`. Model weights are separate inputs; MNN uses its converted model/configuration format, not GGUF.

`edge_mnn MODEL_CONFIG raw` reads JSON lines with raw `prompt` text and reports real engine counters. `edge_mnn MODEL_CONFIG serve PORT MODEL_ID` keeps the selected model loaded on device loopback and exposes `/health`, `/v1/models`, and `/v1/chat/completions`. The latter passes each request's `max_tokens` to the engine, accepts deterministic temperature0, and omits optional usage rather than inventing token counts. SSE responses are buffered completion chunks, not live token streaming. Engine version, loaded backend configuration and thread count are exposed in health; configuration is not proof of individual operator placement.

Compare the same base model, prompt and protocol, recording each engine's actual quantization and device execution. Do not reuse a published vendor speedup as a local measurement.

