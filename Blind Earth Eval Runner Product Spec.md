# Blind Earth Eval Runner: Product Spec

Oct 3, 2026 · @sadhin

## Overview

A runner that reproduces the "blind Earth" eval for any vendor model, any OpenAI-compatible endpoint and any local model, saves every run, and turns the saved runs into side-by-side maps and a leaderboard.

The eval: ask a text-only model "Land or Water?" for the latitude and longitude of every 2° grid cell (16,200 points), then plot the answers as a map. Celeste's chart ran 12 Claude models this way and scored them against a 1-km land mask, area-weighted: 60.5% for Sonnet 4.5 up to 99.0% for Opus 5.5. Karpathy's comment on it: the models know this from compressing the internet.

**Goals**

- One command or one UI flow runs the same eval across a chosen set of models and configurations.
- Two comparison scopes: model vs model inside one vendor, or vendor vs vendor.
- Per-model configuration, mainly thinking effort, as a second selection level.
- Every run is stored immutably and can be re-compared later without new API calls.
- Reports look like the original chart (map grid) plus the analysis the original skips: baseline, confidence intervals, cost, calibration.

**Non-goals**

- Not a general benchmark harness; the task is fixed to land/water probing of geographic knowledge.
- No vision input. Models are blind by design, so the tool sends text only and no tools.

## Eval definition

The eval asks one question per coordinate and renders the answers as an equirectangular map. It needs no tools and no images.

- **Prompt** (Henry's original, which Celeste's run follows): `If this location is over land, say 'Land'. If this location is over water, say 'Water'. Do not say anything else. x° S, y° W`
- **Grid**: 2° steps, 180 × 90 = 16,200 points per model and configuration.
- **Output per point**: a probability of Land. Models that return logprobs give it by a softmax over "Land" and "Water"; others are sampled several times at temperature 1 (Henry used 4 samples).
- **Map**: the post's chart thresholds the answer to white (Land) or black (Water).
- **Score**: accuracy against a 1-km land mask, area-weighted, so crowded polar rows count less.

Why it works as a probe: the exact question is unlikely to appear in training data, and small models produce smooth continent-shaped blobs rather than memorized city spikes, which suggests compressed geographic structure. The caveat raised in the comments: coordinate tables (GeoNames, OpenStreetMap, Natural Earth) are probably in the training data, so this measures what survived compression, not spatial reasoning.

Sources: [https://x.com/celestepoasts/status/2103232383139057950](https://x.com/celestepoasts/status/2103232383139057950)

## Inputs

Before any model is picked, the user selects the eval inputs: the ground-truth image, the grid, and the prompt. These are saved as an eval spec, and every run points to one.

| Input | Options | Default |
| --- | --- | --- |
| Ground-truth mask | Built-in (Natural Earth land polygons, GSHHG shorelines, a satellite water mask such as MODIS MOD44W), or an uploaded equirectangular PNG/GeoTIFF where white = land; uploads are validated and binarized (see Image comparison) | Natural Earth, rasterized at 1 km |
| Truth rule per cell | Value at the cell center, or majority of mask pixels inside the cell | Cell center |
| Grid step | 1° (64,800 points), 2° (16,200), 4° (4,050), 5° (2,592), custom | 2° |
| Point placement | Cell center (lat -89…89, lon -179…179 at 2°) or cell corner | Cell center |
| Subset | All points, or a seeded random subset (e.g. 10%) for smoke tests | All |
| Coordinate format | Hemisphere (`12° S, 45° W`), signed decimal (`-12.0, -45.0`), degrees-minutes-seconds | Hemisphere |
| Prompt | Default land/water prompt, or a custom template using `{lat}` and `{lon}`; optional system prompt | Default |

Three notes on these choices:

- The 1-km mask behind Celeste's numbers is not named in the screenshot, so the mask is pluggable and each report prints which mask produced its scores. Scores are only comparable across reports that share a mask and truth rule.
- Coordinate format and prompt wording change results, so they are part of the eval spec and the run hash, not hidden settings.
- Points are evaluated in a stratified random order, so a half-finished run already shows a usable map.

## Providers and models

Every model sits behind one adapter interface, `classify(prompt, config) → {text, p_land?, usage, latency, raw}`, so vendors, custom endpoints and local models run through the same pipeline.

| Adapter | Covers | Logprobs | Thinking control |
| --- | --- | --- | --- |
| Anthropic | Claude models | None (Henry sampled 4×) | Effort or thinking budget |
| OpenAI | GPT and reasoning models | On most chat models; often unavailable on reasoning models | Reasoning effort on reasoning models |
| Google | Gemini models | Few models | Thinking budget or level |
| OpenRouter | Any model it routes to | Depends on the upstream provider | Passed through where supported |
| OpenAI-compatible | Any `base_url` + key + model name: vLLM, SGLang, LM Studio, Together, Fireworks, Groq, DeepSeek | Server-dependent | Server-dependent |
| Local | Ollama, llama.cpp server (through their OpenAI-compatible endpoints), and Hugging Face transformers loaded in-process | Full-vocabulary logits on transformers (exact probabilities) | None, or prompt-level |

On registration, a capability probe sends 20 test calls and records what the endpoint actually supports: logprobs, top-N limit, effort parameter, max concurrency. Provider docs are not trusted for this, because OpenAI-compatible servers vary widely.

Providers and models are declared once and then selected in the UI or in a spec:

```yaml
providers:
  - id: anthropic
    kind: anthropic
    api_key_env: ANTHROPIC_API_KEY
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
    quant: awq-int4
```

Keys are read from the environment or the OS keychain and never written to the results store. Local models record quantization, because it changes the map.

## Comparison modes

Selection happens in two levels: first which models are compared, then how each one is configured. Their cross product is the run matrix.

**Level 1: scope**

| Mode | What the user picks | Example |
| --- | --- | --- |
| Within a vendor | One vendor, then several of its models | Sonnet 4.5, Opus 4.6, Opus 5.5 |
| Across vendors | Several vendors, then one or more models from each | A Claude, a GPT and a Gemini model, plus a local Llama |
| Mixed matrix | Any models from the registry, including custom and local endpoints | Opus 5.5 vs a self-hosted Qwen |
| Config sweep | One model, several configurations | One model at effort low, medium, high |
| From saved runs | Previously finished runs; nothing is re-called | Last month's run vs today's |

**Level 2: configuration per model**

Each selected model gets one or more configuration variants (thinking effort, temperature, sample count, max tokens; see Run configuration). A variant is named `model@key=value`, for example `opus-5-5@effort=low`, and shows as its own map and row in the report.

**Flow in the UI and CLI**

1. Choose or create the eval spec (mask, grid, prompt).
2. Choose the scope and pick models.
3. Set configuration variants per model, with an "apply to all" shortcut.
4. Review the plan: number of calls, estimated tokens and cost per cell, and which cells reuse cached runs.
5. Run. Cells execute in parallel under per-provider rate limits.

**Fairness rules the tool enforces**

- All cells in one comparison share the same eval spec, so mask, grid, prompt and coordinate format cannot differ.
- Cells that used different extraction modes (logprobs vs sampling) are flagged, and the leaderboard ranks them on thresholded accuracy only, never on probability metrics.
- Cells with forced thinking are marked, as the starred models in Celeste's chart are, and can be filtered into a separate leaderboard.

## Run configuration

A configuration is a small, normalized set of fields. Each adapter maps the normalized fields to the provider's native parameters, and the run stores both.

| Field | Values | Notes |
| --- | --- | --- |
| `effort` | off, low, medium, high, max, or provider-native | The main second-level knob. Mapped to effort levels, thinking budgets or reasoning effort per provider; unsupported levels are refused in the plan step, not silently dropped |
| `temperature` | 0 to 2 | Ignored where the provider fixes it (some thinking modes) |
| `n_samples` | 1 to 16 | Used in sampling mode; 4 matches the original post |
| `top_logprobs` | 1 to provider cap | Logprobs mode only |
| `max_output_tokens` | integer | Auto-raised for thinking runs so the answer is not cut off |
| `system_prompt` | text | Part of the run hash |
| `concurrency`, `rpm`, `tpm` | integers | Per provider, with adaptive back-off on rate-limit errors |
| `seed` | integer | Point order and subset sampling; also passed to providers that accept one |

Example eval spec with a two-level matrix:

```yaml
eval:
  mask: natural-earth-land
  grid: { step_deg: 2, placement: cell_center }
  coord_format: hemisphere
  prompt: default-land-water
extraction:
  mode: auto            # logprobs if supported, else sample
  n_samples: 4
  temperature: 1.0
matrix:
  - model: anthropic/opus-5-5
    configs: [ { effort: low }, { effort: high } ]
  - model: openai/<model>
    configs: [ { effort: off } ]
  - model: my-vllm/qwen-local
    configs: [ {} ]
```

Thinking runs cost far more than plain runs because reasoning tokens bill as output. The planner therefore runs a 50-point pilot per cell first and uses the measured tokens per point in the cost estimate.

## Probability extraction

The extractor turns one model response into a probability of Land, plus a validity flag. The mode is set per run, and `auto` picks logprobs when the capability probe found them.

| Mode | How P(Land) is computed | When it applies |
| --- | --- | --- |
| Logprobs | Softmax over the first-token logprobs of "Land" and "Water" | Endpoints and local models that return them; exact on transformers |
| Sample | Fraction of `n_samples` answers that say Land, at the configured temperature | Providers without logprobs, and any thinking run |
| Greedy | One answer at temperature 0, P is 0 or 1 | Cheap smoke tests |

Rules shared by all modes:

- **Token matching** is by case-insensitive prefix, with and without a leading space. If "Land" tokenizes as "La" + "nd", the extractor looks for "La", as Henry did.
- **Validity mass**: in logprobs mode the run stores P(Land) + P(Water) from the raw distribution. A low value means the model wanted to say something else, and the point is flagged.
- **Parsing** takes the first word of the final answer after any thinking block, strips punctuation, and accepts only Land or Water. Anything else is `invalid`: refusals, explanations, empty output, truncation.
- **Invalid answers** are counted as wrong in the headline accuracy, so refusing cannot raise a score. A second figure, accuracy on valid answers only, is reported next to it.
- **Threshold**: P(Land) greater than 0.5 draws white. The raw probability is always stored, so the threshold can be changed in reports.
- **Thinking runs** store the thinking token count and, optionally, the full thinking text. Default is the count only, to keep the store small.

## Scoring and metrics

The headline number is area-weighted accuracy, which matches the original chart. Everything else exists to stop that one number from misleading.

| Metric | Definition | Why it is there |
| --- | --- | --- |
| Area-weighted accuracy | Share of correct cells, each weighted by cos(latitude) | Headline; same as the post |
| Unweighted accuracy | Share of correct cells | Shows how much the weighting matters |
| Majority baseline | Score of answering Water everywhere; about 71% on the area-weighted mask, since Earth is about 71% water | The post's lowest score, 60.5%, is below this, so a number alone hides that the model is worse than a constant |
| Skill | (accuracy − baseline) / (1 − baseline) | Puts all models on a 0 to 1 scale above the constant answer |
| Land precision, recall, F1, IoU | Standard, on the Land class | Separates "sees land everywhere" from "misses land" |
| Brier score, log loss, calibration error | On P(Land), probabilistic modes only | Whether the stated confidence means anything |
| Coastline error | Mean distance in degrees from each predicted boundary cell to the nearest true coastline | Separates blobby maps from sharp ones at equal accuracy |
| Region accuracy | Per continent, ocean and polar band | Shows where a model fails; Antarctica is reported separately because mask conventions differ |
| Invalid rate | Share of points with an unparsable answer | Keeps refusals visible |
| Cost and speed | Input, output and thinking tokens; USD; latency p50 and p95 | Needed for cost-vs-accuracy comparison |
| 95% interval | Bootstrap over 10° × 10° blocks of cells | Neighbouring cells are correlated, so a plain per-cell interval would be too narrow |

Two runs are called different only if their block-bootstrap intervals on the paired difference exclude zero. Near-ties, such as Fable 5 (97.8%) and Fable 5.1 (98.2%) in the post's chart, are shown as ties unless that interval excludes zero. Stochastic configurations can be repeated (`repeats: 3`) to measure run-to-run spread.

## Image comparison with ground truth

Each run is also scored as an image. The model's answers are rendered into a map at the ground-truth image's resolution and compared with it using standard computer-vision measures, so the eval works with any ground-truth image the user supplies, not only the built-in mask.

**Preparing the two images**

1. **Ground truth.** A built-in mask or an uploaded image. Uploads are validated: declared projection (equirectangular, or Web Mercator reprojected to equirectangular), a 2:1 aspect ratio for equirectangular, land polarity (white = land, with an invert switch), and a binarization threshold (Otsu by default, shown as a preview for the user to confirm). Transparent pixels are flattened onto black.
2. **Prediction.** The grid of P(Land) values is rendered to an image with the same projection, extent and pixel size as the ground truth. Binary maps are upsampled by nearest neighbour, so blockiness counts against the model; probability maps use bilinear upsampling.
3. **Difference image.** Both images and an overlay are stored with the run: red where the model said Land and the truth is water, blue for the reverse. The overlay feeds the error map in reports.

**Metrics**

| Metric | What it measures | Detail |
| --- | --- | --- |
| Pixel IoU and Dice | Overlap of predicted and true land | Weighted by cos(latitude), as the cell metrics are |
| SSIM | Structural similarity of the two images | Computed on the probability image against the mask |
| Boundary F-score | Share of predicted coastline within a tolerance of the true coastline, and the reverse | Tolerance set in degrees, default 2° and 4° |
| Contour distance | Mean and 95th-percentile distance between predicted and true coastlines | Distance transform of the true boundary; reported in degrees and km |
| Connected components | Number and size of predicted landmasses against the true ones; count of spurious specks | Labelled with the ±180° seam treated as continuous, so Russia, Alaska and Fiji are not split |
| Per-landmass IoU | Overlap for each large true landmass, matched to the predicted component with most overlap | Gives per-continent shape quality |
| Probability error | RMSE between the P(Land) image and the mask | Probabilistic runs only |

**Resolution ceiling.** A 2° map cannot match a 1-km mask pixel for pixel, even for a perfect model, because its coastlines are blocky. The tool therefore scores the ground-truth image itself after downsampling to the grid and upsampling back, and reports each model as a share of that ceiling. Without it, IoU and boundary scores at 2° and 5° grids cannot be compared.

These image metrics sit beside the cell-level accuracy and do not replace it. Both are computed from the stored points, so changing the ground-truth image or threshold re-scores old runs without new API calls.

## Persistence and data model

Every run is stored immutably in one SQLite file plus an artifacts folder, so a comparison months later needs no new API calls. A run at 2° is 16,200 rows, so even hundreds of runs stay small.

| Table | Holds |
| --- | --- |
| `eval_specs` | Mask hash, truth rule, grid, placement, coordinate format, prompt, system prompt |
| `models` | Provider, model name as sent, resolved version returned by the API, quantization for local models |
| `configs` | Normalized configuration and the provider-native parameters it mapped to |
| `runs` | Spec, model and config ids, run hash, status, start and end times, runner version, seed, totals for tokens and cost |
| `points` | Per run and cell: lat, lon, truth, p\_land, n\_valid, answer text, finish reason, latency, input, output and thinking tokens, error |
| `metrics` | Computed scores per run, with the threshold and mask used, so they can be recomputed |
| `comparisons` | A named set of runs, with the filters and ordering used in its report |

Behaviour that follows from this:

- **Run hash**: a hash of mask, grid, prompt, coordinate format, resolved model version, normalized config and extraction mode. A matching finished run is reused instead of re-called, and the plan step shows which cells hit the cache.
- **Resumable**: points are written in batches of 100. A crashed or cancelled run resumes at the first missing point and retries only failed points.
- **Status**: queued, running, paused, complete, failed, with partial runs viewable and scored on the points done so far.
- **Provenance**: the model string as sent, the version the API reports back, and the timestamp are stored, because aliases such as "latest" drift. Re-running the same spec later creates a new run beside the old one.
- **Export**: Parquet and CSV for points, JSON for metrics, PNG and SVG for maps.

## Reports and visualization

A report is generated from a comparison and ships as one self-contained HTML file, with the map grid also exportable as a PNG in the style of the original post.

| View | What it shows | Reader's question |
| --- | --- | --- |
| Map grid | Small multiples, one map per model and config, titled with name and accuracy, as in "How 12 blind Claudes see the Earth"; toggles for binary, probability heatmap and error map; optional true-coastline overlay | Who sees the Earth best? |
| Error map | Green where correct, red where wrong, split into false Land and false Water | Where and how does each model fail? |
| Leaderboard | Sortable table: area-weighted accuracy with 95% interval, skill, F1, invalid rate, cost, latency; filters for vendor, thinking on or off, extraction mode | Which is best, and is the gap real? |
| Effort curve | Accuracy against effort level, one line per model | Does more thinking help? |
| Cost vs accuracy | Scatter of cost per run against accuracy, with the Pareto frontier marked | What do I pay for each point of accuracy? |
| Lineage | Accuracy against release date inside one vendor | Do newer models know more geography? |
| Diff view | Two runs side by side with a disagreement map and the paired-difference interval | Where do A and B differ, and by how much? |
| Region table | Models by continent and ocean, heat-coloured | What does the headline number hide? |
| Calibration | Reliability diagram per run that has probabilities | Is the model's confidence meaningful? |
| Run health | Invalid rate, errors, retries, latency histogram | Can I trust this run? |

While a run is in progress the web UI draws the map live. Because points are visited in stratified random order, the picture sharpens from the first minutes instead of filling top to bottom. Each report header lists the eval spec, mask, run dates, runner version and total spend, so a shared report stands on its own.

## Architecture and interfaces

&#91;embedded content: pipeline · 9 components\]

Work follows the path of the arrows: the plan goes out to the providers, answers come back through the extractor into the store, and scoring and reports read only from the store. That is what makes a later comparison free.

| Layer | Choice |
| --- | --- |
| Runner and CLI | Python 3.11+, asyncio and httpx, with official SDKs behind the adapter interface |
| Masks and scoring | numpy, rasterio and shapely for masks and cell truth; scipy distance transform for coastline error |
| Store | SQLite single file, Parquet export |
| Local models | OpenAI-compatible servers first; transformers later for exact logits |
| API and UI | Gradio for the web UI, with the map updating live during a run; a separate API is added later only if needed |
| Charts | matplotlib for static PNG; Plotly or D3 embedded in the HTML report |

CLI sketch (`blindearth` is a working name):

```bash
blindearth run spec.yaml --estimate     # plan, pilot, cost; no spend
blindearth run spec.yaml                # execute, resumable
blindearth compare <run> <run> ...      # saved runs, no API calls
blindearth report <comparison> --html out.html --png grid.png
blindearth serve                        # web UI
```

## Cost, rate limits, reliability

Calls per run scale with the grid and the sampling mode; a 2° run is 16,200 calls in logprobs or greedy mode and 64,800 at 4 samples.

| Grid step | Points | Calls at 1 per point | Calls at 4 samples |
| --- | --- | --- | --- |
| 1° | 64,800 | 64,800 | 259,200 |
| 2° | 16,200 | 16,200 | 64,800 |
| 4° | 4,050 | 4,050 | 16,200 |
| 5° | 2,592 | 2,592 | 10,368 |

Each call is a short prompt (roughly 50 to 60 input tokens) with a one-word answer, so a plain 2° run is on the order of 1M input tokens and little output. Thinking runs reverse that: reasoning tokens dominate and the bill can be tens of times higher, which is why the plan step measures a 50-point pilot before spending.

**Cost controls**

- Plan step shows calls, estimated tokens and USD per cell, plus a hard budget cap per comparison that pauses the run when reached.
- Batch APIs (Anthropic, OpenAI) are used when selected, since the eval has no latency need and batch pricing is typically about half.
- Prompt caching is not expected to help: the shared prefix is too short.
- Start at 4° to check a setup, then run 2° once the plan looks right.

**Reliability**

- Per-provider token-bucket limiter for requests and tokens per minute, with concurrency that drops on rate-limit errors and recovers slowly.
- Retries with exponential backoff and jitter on 429 and 5xx, up to 5 attempts; a failed point is stored as an error and retried at the end of the run.
- A circuit breaker pauses a cell after a burst of failures, so a bad key or a down endpoint does not burn the whole queue.
- Temperature 0 is not deterministic on most providers, so two runs of the same cell can differ slightly; this is why runs can be repeated.

**Validity caveats the report prints**

- At 2°, a cell is about 222 km wide at the equator, so coastal cells are ambiguous and the truth rule matters.
- Coordinate tables are probably in training data, so scores measure what survived compression, not reasoning.
- Thinking and non-thinking runs are not like for like, and local quantized models are not the full-precision model.
- Model aliases drift; the resolved version is stored with every run.

## Milestones and open questions

The build runs in four phases, each ending in something usable.

1. **Core loop**: one eval spec, Anthropic, OpenAI-compatible and local adapters, sampling and logprobs extraction, SQLite store, area-weighted accuracy, and a PNG map grid. Done when one command reproduces a chart like Celeste's for three models at 4°.
2. **Two-level comparison**: provider registry with capability probe, within-vendor and across-vendor modes, per-model config variants, plan step with cost estimate and cache hits, resume. Done when a mixed matrix of 6 or more cells runs unattended at 2°.
3. **Full report**: HTML report with leaderboard, intervals, baseline and skill, image-comparison metrics, effort curve, cost scatter, diff view, region table and calibration.
4. **Web UI**: input picker with mask upload, model and config selection, live map during runs, saved comparisons. Answer: Gradio

**Open questions**

- Which 1-km land mask should be the default, so scores line up with Celeste's chart? Natural Earth is the placeholder.
- CLI-first with a static HTML report, or a hosted UI from the start? The phases above assume CLI first. -> cli first
- Should forced-thinking models get their own leaderboard, or share one with a marker?&#32;
- Is Hugging Face in-process inference needed in the first release, or are OpenAI-compatible local servers enough?
