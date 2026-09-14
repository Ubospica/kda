# Small-batch RMSNorm for decoder inference (demo)

This is an illustrative task for trying the wishlist pull request template. The use case and desired improvement below are hypothetical; this is not a reported production bottleneck.

## Project and Desired Improvement

Consider a small decoder inference service with hidden size 4096. At each decoding step, it applies RMSNorm (root mean square normalization) to a small batch of token embeddings before the next layer. The request is to reduce per-call latency for batches of 1, 4, 16, and 32 tokens while preserving the current input/output interface.

Use the installed FlashInfer RMSNorm implementation as the comparison baseline. Compare candidate and baseline latency separately for every batch size with the same environment and measurement settings. No speedup target or measured improvement is claimed by this demo.

## Contract and Baseline

- Input: contiguous bfloat16 tensor `input` with shape `[batch_size, 4096]`.
- Weight: contiguous bfloat16 tensor `weight` with shape `[4096]`.
- Epsilon: Python float `eps`, fixed to `1e-6` in the provided workloads.
- Output: a newly allocated contiguous bfloat16 tensor with the input shape. Inputs must not be modified.
- Reference: explicit float32 arithmetic in `definition.json`.
- Workloads: synthetic random inputs for batch sizes 1, 4, 16, and 32; these do not represent a measured traffic distribution.
- Target hardware: NVIDIA B200.
- Correctness: relative and absolute tolerances of `1e-2`; outputs must be finite for the provided inputs.
- Constraints: keep the same Python entry point and input/output contract. Document any extra dependency required by a candidate.

The baseline in `baseline.py` calls [`flashinfer.norm.rmsnorm`](https://docs.flashinfer.ai/generated/flashinfer.norm.rmsnorm.html) from [FlashInfer 0.6.18.post1](https://pypi.org/project/flashinfer-python/0.6.18.post1/). It is the known optimized comparison implementation chosen for this demo, not a claim that all possible implementations have been compared. Its source is Apache-2.0 licensed.

## Validation Status

- The definition and all four workloads pass the FlashInfer Bench data models.
- The baseline imports successfully and can be registered as a Python solution.
- The mathematical reference passes CPU checks against an independent float64 calculation for all four batch sizes, including output shape/type, finite values, preserved inputs, and the zero-input case.
- The benchmark command has been checked for Python syntax and points to this request directory.
- GPU correctness and performance have not been run for this demo. No GPU latency or speedup is claimed. Run the command below on B200 before treating this as a complete optimization request.

## Evaluate with FlashInfer Bench


This request has no `benchmark.py`, so evaluation uses `flashinfer-bench`. Use Python 3.12, a CUDA-enabled PyTorch installation, and a B200/B300-compatible NVIDIA driver and CUDA toolkit. Install the pinned packages in your chosen environment:

```bash
python -m pip install torch==2.11.0 flashinfer-python==0.6.18.post1 flashinfer-bench==0.1.2
```

From the repository root, select an idle supported GPU and run the following command. `baseline.py::run` is registered as a Python solution; this keeps the submission layout simple while using FlashInfer Bench's native evaluator.

```bash
CUDA_VISIBLE_DEVICES=0 REQUEST_DIR=requests/ubospica-rmsnorm-small-batch-demo python -c '
import os
from pathlib import Path
import torch
from flashinfer_bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import BuildSpec, Definition, Solution, SourceFile, Trace, TraceSet

hardware = torch.cuda.get_device_name(0)
assert any(model in hardware.split() for model in ("B200", "B300")), hardware
request = Path(os.environ["REQUEST_DIR"])
definition = Definition.model_validate_json((request / "definition.json").read_text())
workloads = [Trace.model_validate_json(line) for line in
             (request / "workloads.jsonl").read_text().splitlines() if line.strip()]
assert workloads and all(t.definition == definition.name for t in workloads)
baseline = Solution(
    name=f"{definition.name}_baseline", definition=definition.name, author="baseline",
    spec=BuildSpec(language="python", target_hardware=["B200"],
                   entry_point="baseline.py::run", destination_passing_style=False),
    sources=[SourceFile(path="baseline.py", content=(request / "baseline.py").read_text())],
)
dataset = TraceSet(definitions={definition.name: definition},
                   workloads={definition.name: workloads},
                   solutions={definition.name: [baseline]})
benchmark = Benchmark(dataset, BenchmarkConfig(
    warmup_runs=10, iterations=100, num_trials=3, rtol=1e-2, atol=1e-2))
try:
    results = benchmark.run_all(dump_traces=False)
finally:
    benchmark.close()
traces = results.traces.get(definition.name, [])
for trace in traces:
    print(trace.model_dump_json())
assert len(traces) == len(workloads), "Missing evaluation results"
assert all(t.is_successful() for t in traces), "Correctness or execution failed"
'
```

The command validates the baseline against the mathematical reference and measures it with 10 warm-up runs, 100 timed iterations, and 3 trials. It prints evaluation records, including environment and latency, without writing benchmark output into the repository. FlashInfer Bench manages its build cache separately.

The reported `speedup_factor` compares the registered implementation with the mathematical reference. When evaluating a new candidate, compare its latency with the latency of `baseline.py`; speedup over the reference alone does not establish improvement over the best-known implementation. Set tolerances and measurement settings appropriate to your own request.

If your request needs a custom measurement method, include an optional `benchmark.py` and document its command, environment, warm-up, timing, and correctness checks in the request README.


## License

This wrapper and the embedded reference are Apache-2.0 code; this documentation is Creative Commons Attribution 4.0 under the [project license](https://github.com/NVlabs/kda/blob/main/LICENSE). The baseline calls the separately installed [FlashInfer library](https://github.com/flashinfer-ai/flashinfer), which uses Apache-2.0. No external implementation source or dataset is bundled here.
