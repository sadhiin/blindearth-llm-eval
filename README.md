# blindearth

Reproduces the "blind Earth" eval: a text-only model is asked "Land or Water?" for the latitude
and longitude of every grid cell, and the answers are drawn as a map and scored against a land
mask. The runner works with vendor models, any OpenAI-compatible endpoint and local models. Every
run is stored in SQLite, so later comparisons and reports need no new API calls.

The product spec is in `Blind Earth Eval Runner Product Spec.md`. Module contracts are in `INTERFACES.md`.

## Status

Every module in the spec is implemented, but the code has only had static checks so far
(everything compiles and every import between modules resolves). The test suite has not been
run yet, and the repo has no CI. Run `pytest` before relying on any results. Some details still
need confirming against the live APIs; each is listed in the docstring of its module:

- MOD44W download: whether LP DAAC accepts an Earthdata bearer token, and the exact name of the
  data layer inside the MODIS files
- whether OpenRouter's `require_parameters` routing also covers the `reasoning` setting
- the Gemini `top_logprobs` cap, kept at a cautious 5
- the no-effort thinking defaults of the newest gpt-6 models

Development happens on `develop`. `main` holds the initial implementation.

## Install

Python 3.11 or newer.

```bash
pip install -e .            # runner, CLI, scoring, reports
pip install -e '.[ui]'      # + Gradio web UI (blindearth serve)
pip install -e '.[local]'   # + in-process transformers models
pip install -e '.[dev]'     # + pytest
pip install -e '.[dev,ui]'  # what the full test suite needs (the UI tests import gradio)
```

A full example registry is in `examples/providers.yaml`. Example specs: `examples/spec.yaml` and
`examples/smoke_4deg.yaml`.

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

Provider kinds: `anthropic`, `openai`, `google`, `openrouter`, `openai_compatible` (vLLM, TGI,
etc.), `ollama`, `llamacpp` and `transformers` (in-process, needs the `local` extra).

Any key on a provider or model that the registry doesn't recognise is kept in that provider's or
model's `extra`. The runner reads these keys (all optional):

| Key | Where | What it does |
|---|---|---|
| `concurrency`, `rpm`, `tpm` | provider | rate limits |
| `breaker_threshold`, `breaker_window_s` | provider | circuit breaker (default 20 failures in 60 s) |
| `batch_poll_s` | provider | batch polling interval (default 30) |
| `store_thinking` | provider | store the full thinking text, not only the token count |
| `timeout_s` | provider | request timeout |
| `effort_map` | provider or model | map an effort level to native request params, e.g. `{low: {reasoning_effort: low}}` |
| `thinks_by_default` | model | `true`/`false`: overrides the built-in table of models that reason when no effort is set |
| `vertexai` | provider (`google`) | use Vertex AI instead of the Gemini Developer API (no batch) |
| `referer`, `title` | provider (`openrouter`) | OpenRouter attribution headers |

Set `forced_thinking: true` on a model whose reasoning cannot be turned off. See
[Thinking and effort](#thinking-and-effort).

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

### Thinking and effort

`effort` is one of `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or unset. Each adapter
maps it to its vendor's native parameters. If the vendor can't honour a level, the adapter refuses
it (the plan shows a refusal) rather than silently sending something else.

- **No effort set:** the run counts as a thinking run when the model reasons by default.
  `blindearth/thinking_defaults.py` holds that table (for example Claude Sonnet 5.5, Gemini 2.5
  Pro/Flash and Gemini 3.x think by default; Gemini 2.5 Flash-Lite and Claude Haiku 4.5 don't).
  Override the table per model with `thinks_by_default`.
- **Models that always think** (registry `forced_thinking`, or the table) refuse `effort: off`
  and are starred in plans and reports.
- **Thinking runs** have a minimum max-output-tokens that depends on effort: 4,096 at `low`,
  8,192 at `medium` or with no effort, 16,384 at `high`, 32,768 at `max`. They always use
  sampling rather than logprobs. Non-thinking runs default to 16 output tokens.
- **Gemini 3.x** accepts only the thinking levels each model documents; for example 3 Pro has no
  `medium`. Gemini 2.5 thinking budgets are range-checked.
- **OpenRouter** sends `reasoning: {effort: ...}`, with `off` sent as `effort: "none"`. It also
  sets `provider: {require_parameters: true}`, so requests aren't routed to an upstream that
  would drop the reasoning or logprob settings. `top_logprobs` is capped at 20.
- **Ollama, llama.cpp, transformers:** `off` is supported. Ollama silently ignores effort levels
  a model doesn't support, so `low`/`medium`/`high` are refused unless an `effort_map` entry
  enables them, e.g. for gpt-oss:

  ```yaml
  - id: gpt-oss-local
    provider: ollama
    name: gpt-oss:20b
    effort_map: { low: { reasoning_effort: low }, high: { reasoning_effort: high } }
  ```

### Batch APIs

With `use_batch: true` on the provider, the Anthropic, OpenAI and Google adapters submit cells
through the vendor batch API (about half price, results within hours). Google batch works only
on the Gemini Developer API, not Vertex. Requests are sent inline, and the adapter splits them
into several batches to stay under the 20 MB limit per request.

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

`compare` (in both the CLI and the web UI) refuses runs from different eval specs. It stores the
fairness check's findings with the comparison, and warns when runs mix extraction modes or include
runs that always think.

### Regions

Reports break accuracy down by continent and ocean, using the Natural Earth 110m
`geography_regions_polys` and `geography_marine_polys` layers. These are downloaded once into the
cache, with the same checksum checks as the masks. A cell that falls in a gap between polygons
takes the nearest polygon within 3°. If the polygons can't be loaded (offline, or shapely
missing), latitude/longitude boxes are used instead. The report states which method it used.
`BLINDEARTH_REGIONS` sets the behaviour:

- `auto` (default): download the polygons if they aren't cached
- `cached`: use the polygons only if they are already cached
- `boxes`: always use the boxes (also the default when running under pytest)

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

Whether a run counts as a thinking run feeds into the effective config through its max output
tokens. Runs on models that think by default, made with no effort set before PR #1, were labeled
non-thinking and capped at 16 output tokens. They now get a different run hash, so they are rerun
rather than reused as cache hits.

## Environment variables

| Variable | What it does |
|---|---|
| `BLINDEARTH_CACHE` | cache directory for masks and region polygons (default `~/.cache/blindearth`) |
| `BLINDEARTH_ACCEPT_NEW_CHECKSUMS=1` | accept a changed upstream download once and record its new hash |
| `BLINDEARTH_MODIS_DOWNLOAD=1` | allow the MOD44W download and mosaic |
| `BLINDEARTH_MODIS_YEAR` | MOD44W year (default 2021) |
| `EARTHDATA_TOKEN` | NASA Earthdata token for MOD44W (or use `~/.netrc`) |
| `BLINDEARTH_REGIONS` | `auto`, `cached` or `boxes` (see [Regions](#regions)) |
| `BLINDEARTH_PRICING` | YAML file whose prices override `blindearth/data/pricing.yaml` |
| API keys | whatever each provider's `api_key_env` names, e.g. `ANTHROPIC_API_KEY` |

Prices in `blindearth/data/pricing.yaml` are defaults and should be checked against the vendors'
current price lists before you trust the cost estimates.

## Tests

```bash
pip install -e '.[dev,ui]'
pytest
```

The tests use fake adapters, a fake registry and mocked HTTP (`respx`), so they make no network
calls and need no API keys. Regions fall back to the boxes under pytest. The mask download tests
build zip archives in memory and patch the downloader.
