-- blindearth results store. One SQLite file; every statement is idempotent.
-- Pragmas (WAL, foreign_keys, busy_timeout) are set per connection in store/db.py.
-- API keys are never stored: provider rows keep only id/kind/base_url, and
-- model extra / native params are scrubbed of secret-looking keys before writing.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Eval spec: mask, truth rule, grid, placement, coordinate format, prompt, system prompt.
CREATE TABLE IF NOT EXISTS eval_specs (
    id              TEXT PRIMARY KEY,           -- evalspec.load.spec_hash(spec, mask_hash)
    mask_hash       TEXT NOT NULL,
    mask_source     TEXT NOT NULL,
    mask_id         TEXT NOT NULL,
    truth_rule      TEXT NOT NULL,
    step_deg        REAL NOT NULL,
    placement       TEXT NOT NULL,
    subset_frac     REAL,
    grid_seed       INTEGER NOT NULL,
    coord_format    TEXT NOT NULL,
    prompt_id       TEXT NOT NULL,
    prompt_template TEXT NOT NULL,
    system_prompt   TEXT,
    spec_json       TEXT NOT NULL,              -- EvalSpec.to_dict()
    created_at      TEXT NOT NULL
);

-- Model as sent, plus the version the API last reported back.
CREATE TABLE IF NOT EXISTS models (
    id               TEXT PRIMARY KEY,          -- "<provider>/<registry id>#<hash8>"
    provider_id      TEXT NOT NULL,
    provider_kind    TEXT NOT NULL,
    registry_id      TEXT NOT NULL,
    name             TEXT NOT NULL,             -- model string as sent
    resolved_version TEXT,                      -- last version the API reported
    quant            TEXT,
    vendor           TEXT,
    release_date     TEXT,
    forced_thinking  INTEGER NOT NULL DEFAULT 0,
    base_url         TEXT,
    extra_json       TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS models_ref ON models(provider_id, registry_id);

-- Normalized configuration and the provider-native parameters it mapped to.
CREATE TABLE IF NOT EXISTS configs (
    id              TEXT PRIMARY KEY,           -- sha256(normalized + native)
    normalized_json TEXT NOT NULL,              -- RunConfig.normalized() (hashed fields)
    native_json     TEXT NOT NULL,              -- adapter.map_config(...)
    config_json     TEXT NOT NULL,              -- full RunConfig incl. rate-limit fields
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                     TEXT PRIMARY KEY,
    run_hash               TEXT NOT NULL,
    plan_hash              TEXT,                -- hash computed at plan time (before version known)
    spec_id                TEXT NOT NULL REFERENCES eval_specs(id),
    model_id               TEXT NOT NULL REFERENCES models(id),
    config_id              TEXT NOT NULL REFERENCES configs(id),
    variant                TEXT NOT NULL,
    extraction_mode        TEXT NOT NULL CHECK (extraction_mode IN ('logprobs','sample','greedy')),
    extraction_json        TEXT NOT NULL DEFAULT '{}',   -- ExtractionSpec used
    status                 TEXT NOT NULL DEFAULT 'queued'
                           CHECK (status IN ('queued','running','paused','complete','failed')),
    note                   TEXT,                -- pause / failure reason
    created_at             TEXT NOT NULL,
    started_at             TEXT,
    ended_at               TEXT,
    runner_version         TEXT NOT NULL DEFAULT '',
    seed                   INTEGER NOT NULL DEFAULT 0,
    repeat_idx             INTEGER NOT NULL DEFAULT 0,
    resolved_model_version TEXT,
    forced_thinking        INTEGER NOT NULL DEFAULT 0,
    thinking               INTEGER NOT NULL DEFAULT 0,
    input_tokens           INTEGER NOT NULL DEFAULT 0,
    output_tokens          INTEGER NOT NULL DEFAULT 0,
    thinking_tokens        INTEGER NOT NULL DEFAULT 0,
    cost_usd               REAL,
    n_points_total         INTEGER NOT NULL DEFAULT 0,
    n_points_done          INTEGER NOT NULL DEFAULT 0,
    batch_state_json       TEXT                 -- pending provider batch ids -> point indices
);
CREATE INDEX IF NOT EXISTS runs_hash_status ON runs(run_hash, status);
CREATE INDEX IF NOT EXISTS runs_plan_hash ON runs(plan_hash);
CREATE INDEX IF NOT EXISTS runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS runs_model ON runs(model_id);
CREATE INDEX IF NOT EXISTS runs_spec ON runs(spec_id);

CREATE TABLE IF NOT EXISTS points (
    run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    idx             INTEGER NOT NULL,
    lat             REAL NOT NULL,
    lon             REAL NOT NULL,
    weight          REAL NOT NULL,              -- cos(lat)
    truth           INTEGER,                    -- 1 land, 0 water
    p_land          REAL,
    n_valid         INTEGER NOT NULL DEFAULT 0,
    n_samples       INTEGER NOT NULL DEFAULT 0,
    validity_mass   REAL,
    answer_text     TEXT NOT NULL DEFAULT '',
    finish_reason   TEXT,
    latency_s       REAL NOT NULL DEFAULT 0,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    thinking_tokens INTEGER NOT NULL DEFAULT 0,
    thinking_text   TEXT,
    error           TEXT,
    attempts        INTEGER NOT NULL DEFAULT 1,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (run_id, idx)
);
CREATE INDEX IF NOT EXISTS points_run_idx ON points(run_id, idx);
CREATE INDEX IF NOT EXISTS points_run_failed ON points(run_id, idx) WHERE error IS NOT NULL;

CREATE TABLE IF NOT EXISTS metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    mask_hash    TEXT NOT NULL,
    threshold    REAL NOT NULL,
    metrics_json TEXT NOT NULL,
    computed_at  TEXT NOT NULL,
    UNIQUE (run_id, mask_hash, threshold)
);
CREATE INDEX IF NOT EXISTS metrics_run ON metrics(run_id);

CREATE TABLE IF NOT EXISTS comparisons (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    run_ids_json  TEXT NOT NULL,
    filters_json  TEXT NOT NULL DEFAULT '{}',
    ordering_json TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS comparison_runs (
    comparison_id TEXT NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
    run_id        TEXT NOT NULL REFERENCES runs(id),
    position      INTEGER NOT NULL,
    PRIMARY KEY (comparison_id, run_id)
);

-- Capability probe results per endpoint + model name.
CREATE TABLE IF NOT EXISTS capabilities (
    provider_id TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    caps_json   TEXT NOT NULL,
    probed_at   TEXT,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (provider_id, model_name)
);

-- Immutability of complete runs, enforced in the database as well as in Python.
CREATE TRIGGER IF NOT EXISTS runs_complete_immutable
BEFORE UPDATE ON runs
WHEN OLD.status = 'complete'
BEGIN
    SELECT RAISE(ABORT, 'run is complete and immutable');
END;

CREATE TRIGGER IF NOT EXISTS points_complete_no_insert
BEFORE INSERT ON points
WHEN (SELECT status FROM runs WHERE id = NEW.run_id) = 'complete'
BEGIN
    SELECT RAISE(ABORT, 'run is complete and immutable');
END;

CREATE TRIGGER IF NOT EXISTS points_complete_no_update
BEFORE UPDATE ON points
WHEN (SELECT status FROM runs WHERE id = OLD.run_id) = 'complete'
BEGIN
    SELECT RAISE(ABORT, 'run is complete and immutable');
END;
