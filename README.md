# blindearth

Reproduces the "blind Earth" eval: a text-only model is asked "Land or Water?" for the latitude
and longitude of every grid cell, and the answers are drawn as a map and scored against a land
mask. The runner works with vendor models, any OpenAI-compatible endpoint and local models. Every
run is stored in SQLite, so later comparisons and reports need no new API calls.

The product spec is in `Blind Earth Eval Runner Product Spec.md`. Module contracts are in `INTERFACES.md`.

## Install

Python 3.11 or newer.

```bash
pip install -e .            # runner, CLI, scoring, reports
pip install -e '.[ui]'      # + Gradio web UI (blindearth serve)
pip install -e '.[local]'   # + in-process transformers models
pip install -e '.[dev]'     # + pytest
```

## Registry setup

Declare providers and models once in `providers.yaml`. The CLI looks for this file by default;
pass `--registry PATH` to use another one.

```yaml
providers:
  - id: anthropic
    kind: anthropic
    api_key_env: ANTHROPIC_API_KEY
    use_batch: false          # true = use the batch API when the adapter supports it (about half price)
  - id: my-vllm
    kind: openai_compatible
    base_url: http://localhost:8000/v1
    api_key_env: VLLM_KEY
models:
  - id: opus-5-5
    provider: anthropic
    name: claude-opus-5-5
  - id: qwen-local
    provider: my-vllm
    name: Qwen/Qwen2.5-7B-Instruct
    quant: awq-int4            # recorded with every run, because it changes the map
```

The runner reads keys from the environment variable named in `api_key_env`, or from the OS
keychain. Keys are never written to the results store.

Provider `extra` settings that the runner reads (all optional): `concurrency`, `rpm`, `tpm`,
`breaker_threshold` (default 20 failures), `breaker_window_s` (default 60), `batch_poll_s`
(default 30) and `store_thinking` (store the full thinking text, not only the token count).

Probe each endpoint once. The probe makes 20 calls and records what the endpoint actually supports.
With `extraction.mode: auto`, the planner then picks logprobs where the probe found them:

```bash
blindearth probe anthropic/opus-5-5
blindearth probe my-vllm/qwen-local
```

## Eval spec

```yaml
name: three-models
budget_usd: 25              # hard cap; runs pause when it is reached
eval:
  mask: natural-earth-land
  grid: { step_deg: 4, placement: cell_center }
  coord_format: hemisphere
  prompt: default-land-water
extraction:
  mode: auto                # logprobs if the probe found them, else sample (thinking runs always sample)
  n_samples: 4
  temperature: 1.0
matrix:
  - model: anthropic/opus-5-5
    configs: [ { effort: low }, { effort: high } ]
  - model: my-vllm/qwen-local
    configs: [ {} ]
```

### Masks

`natural-earth-land` and `gshhg` download on first use into `~/.cache/blindearth` (or
`$BLINDEARTH_CACHE`). Downloads are written atomically and checked (not an HTML page, full length,
valid zip with the expected layers). Upstream publishes no sha256, so the first good download's
hash goes into `<cache>/checksums.json`, and later reads must match it. If they don't, delete the
cached file, or set `BLINDEARTH_ACCEPT_NEW_CHECKSUMS=1` once to accept a new upstream version.

`modis-mod44w` reads `<cache>/modis-mod44w/mod44w_global.tif` (or `mask.path`). Alternatively,
set `BLINDEARTH_MODIS_DOWNLOAD=1` together with `EARTHDATA_TOKEN` (or a
`machine urs.earthdata.nasa.gov` entry in `~/.netrc`), and the MOD44W v061 tiles for
`BLINDEARTH_MODIS_YEAR` (default 2021) are downloaded from LP DAAC and mosaicked. This needs
rasterio built with GDAL's HDF4 driver (e.g. from conda-forge).

## CLI

```bash
blindearth run spec.yaml --estimate            # plan + 50-point pilot per uncached cell; nothing else is spent
blindearth run spec.yaml --estimate --no-pilot # zero-spend estimate (heuristic tokens per point)
blindearth run spec.yaml [--budget 25] [-y]    # execute; Ctrl-C pauses runs cleanly
blindearth runs [--status paused] [--model opus-5-5]
blindearth resume RUN_ID                       # missing points first, then failed points
blindearth compare RUN RUN ... --name opus-vs-qwen   # saved runs only; refuses mixed eval specs
blindearth score RUN ... [--mask gshhg|path.png] [--threshold 0.5] [--no-images]
blindearth report opus-vs-qwen --html out.html --png grid.png
blindearth export RUN --format parquet|csv|json --out exports/
blindearth serve [--port 7860]                 # web UI (needs the ui extra)
```

Run ids can be shortened to any unique prefix.

The plan table lists, for each cell, the extraction mode, points, calls, estimated input, output
and thinking tokens, and estimated USD. It also marks cache hits (a finished run with the same
run hash, reused without new calls), runs that will be resumed, and refusals (settings that the
adapter cannot honour, such as an unsupported effort level). Cells whose model forces thinking
are starred. The planner warns when cells mix extraction modes; the leaderboard then ranks on
thresholded accuracy only.

## Workflow: smoke test at 4°, then 2°

1. **Probe** each model: `blindearth probe <provider/model>`.
2. **Write a 4° spec** (`grid: { step_deg: 4 }`, 4,050 points). For a quicker check, add
   `subset_frac: 0.1`.
3. **Estimate with no spend:** `blindearth run smoke.yaml --estimate --no-pilot`. Check the
   matrix, the modes and any refusals.
4. **Estimate with the pilot:** `blindearth run smoke.yaml --estimate`. This spends 50 points per
   cell and replaces the token guesses with measured tokens per point, which matters for thinking
   runs. The real run reuses the pilot answers, so they are not paid for twice.
5. **Run at 4°:** `blindearth run smoke.yaml --budget 5`. Follow the progress bars. Points are
   visited in stratified order, so a partial map is already usable.
6. **Inspect:** `blindearth compare <runs> --name smoke` and then
   `blindearth report smoke --html smoke.html --png smoke.png`.
7. **Scale to 2°:** copy the spec, set `step_deg: 2` (16,200 points), run `--estimate` again and
   check the cost, then run it with a budget. The 2° cells have their own run hashes, so the 4°
   runs stay in the store next to them.
8. **If a run pauses** (budget, Ctrl-C, or the circuit breaker after a burst of errors such as a
   bad key), fix the cause and run `blindearth resume RUN_ID`. Running `blindearth run spec.yaml`
   again also picks up unfinished runs with the same hash.

## Storage

All results live in one SQLite file (`--db`, default `blindearth.db`), in WAL mode. The tables are
`eval_specs`, `models`, `configs`, `runs`, `points`, `metrics`, `comparisons` (plus
`comparison_runs`) and `capabilities`. Once a run is complete it is immutable, enforced by both
the code and database triggers. Running the same spec again creates a new run beside the old one,
unless the plan finds a cache hit. The run hash covers:

- the eval spec (mask hash, truth rule, grid, placement, coordinate format, prompt and system prompt)
- the provider, the model name and the quantization
- the resolved model version reported by the API
- the effective config: effort, temperature, samples, top logprobs, max output tokens, config system prompt and seed
- the extraction mode and the repeat index

## Tests

```bash
pytest
```

The tests for the store, hashing, planner and executor use fake adapters and a fake registry, so
they make no network calls.
