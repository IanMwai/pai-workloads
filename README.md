# pai-workloads

Workloads to run on a GPU while the PAI hardware pipeline measures it. The
pipeline (`paihardware` on the 3.102 machine, `paihardware-gamma2` on gamma2)
owns acquisition, the dashboard and archiving; this repo only owns what the
GPU is doing while it is being measured. Any workload is run as

```bash
pai hardware --name <run name> -- <workload command>
```

`pai` exports `PAI_RUN_DIR` and `PAI_RUN_ID` to the command, so a workload
that writes its logs under `$PAI_RUN_DIR/workload/` gets them archived with
the power data.

## Install

Standard library only, so any Python 3.9+ works:

```bash
git clone https://github.com/IanMwai/pai-workloads.git
cd pai-workloads
python3 -m venv .venv && .venv/bin/pip install -e .[dev]   # gives `nhpp-load`
```

or run the scripts directly with `python3 pai_workloads/<name>.py`.

## `nhpp-load`: inference under a daily arrival pattern

Drives any OpenAI-compatible server (vLLM, TGI, Triton) with requests arriving
as a nonhomogeneous Poisson process. The rate λ(t) comes from either

* a diurnal sinusoid (`--base-rate` at night, `--peak-rate` at `--peak-hour`), or
* one day of a real trace (`--trace file.csv --day N`): an Azure LLM inference
  trace CSV with `TIMESTAMP, ContextTokens, GeneratedTokens`, binned per
  minute, which also supplies the prompt and output lengths of each request.
  The traces are in Microsoft's `AzurePublicDataset` repo
  (`data/AzureLLMInferenceTrace_conv.csv`, `..._code.csv`, and the 2024
  week-long files).

Arrivals are drawn by Lewis–Shedler thinning and fired on schedule; a busy
server makes requests queue (bounded by `--max-inflight`), as in production.
`--compress N` replays the day's shape N times faster, so a 24 h day fits in
2 h with `--compress 12`. `--dry-run` prints the planned arrivals per hour
without a server. Prompts are random words (no prefix-cache hits) and
`ignore_eos` makes the server generate exactly the requested output length.

Each request is one JSON line in the log (planned, sent and done timestamps,
queueing delay, tokens, latency, status); the first line is a header with the
configuration and the per-minute rate curve. The default log path is
`$PAI_RUN_DIR/workload/nhpp_requests_<stamp>.jsonl`.

```bash
# shell 1: a model on the GPU under test (port 8000 is the pai dashboard)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct --dtype half \
    --max-model-len 4096 --port 8001

# shell 2, in the pipeline repo: a 24 h day replayed in 2 h, as one run
pai hardware --name day_inference -- nhpp-load \
    --url http://127.0.0.1:8001/v1 --model Qwen/Qwen2.5-7B-Instruct \
    --hours 24 --compress 12
# a real day instead:  ... --trace AzureLLMInferenceTrace_conv.csv --day 2 --compress 12
```

Start with a short `--hours 1 --compress 1 --peak-rate 1` run to check the
server keeps up before committing to a full day.

## Adding a workload

One script per workload under `pai_workloads/`, standard library where
possible, with `--dry-run` or a similar no-hardware mode so it has a test
under `tests/`. Write anything worth keeping under `$PAI_RUN_DIR/workload/`.
