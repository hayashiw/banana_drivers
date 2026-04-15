-- ─────────────────────────────────────────────────────────────────────────────
-- run_registry_schema.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- SQLite schema for the banana_drivers run registry. One database file
-- at banana_drivers/registry.db (top of repo, survives $SCRATCH purges)
-- tracks every non-probe run across stage 1, stage 2, and singlestage.
--
-- Weights and other objective-term coefficients are NOT columnized —
-- they live inside inputs_json and participate in the hash. Query them
-- with SQLite's json_extract:
--     SELECT id FROM stage1
--     WHERE json_extract(inputs_json, '$.weights.iota') > 5;
-- This keeps the schema stable as the objective function evolves.
--
-- Design summary (see PLAN.md § Run registry for motivation):
--   * IDs are content-addressed: s01_<6hex> where the hex is the first 6
--     chars of SHA256 over a canonicalized JSON blob of the stage inputs
--     plus the git commit. Same inputs + same code → same ID, everywhere,
--     without consulting the database.
--   * Inputs are hashed from a whitelist of config fields (see
--     utils/run_registry.py:STAGE{1,2,3}_INPUT_KEYS) rather than the
--     whole config.yaml, so unrelated config additions do not bump IDs.
--   * Each row owns one output directory: $OUT_DIR/{stage}/{id}/. The
--     driver writes all artifacts (wout, boozmn, bsurf, SLURM logs,
--     diagnostics) into that directory; reruns append to slurm_logs/
--     without nuking prior artifacts.
--   * Status lifecycle: pending → running → success | failed | stale
--     (stale = SLURM task died before updating; detected by sweep).
--   * Parent/child relationships are enforced with foreign keys: stage2
--     rows must reference an existing stage1, singlestage rows must
--     reference an existing stage2.
--   * Metrics are columnar (one column per canonical metric) so queries
--     are fast and typed. Free-form stage-specific detail goes in
--     metrics_extra_json.
-- ─────────────────────────────────────────────────────────────────────────────

-- foreign_keys is per-connection (set in RunRegistry._conn); no need here.
PRAGMA journal_mode = WAL;  -- concurrent writers (Pareto sweep). Persistent.

-- ─────────────────────────────────────────────────────────────────────────────
-- stage1 — VMEC fixed-boundary QA optimization
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stage1 (
    -- Identity
    id              TEXT    PRIMARY KEY,           -- e.g. "s01_a3f921"
    input_hash      TEXT    NOT NULL,              -- full SHA256 hex (64 chars)
    git_commit      TEXT    NOT NULL,              -- short SHA of banana_drivers HEAD
    inputs_json     TEXT    NOT NULL,              -- canonical whitelisted config blob
    cold_start      INTEGER NOT NULL,              -- 0 warm, 1 cold (near-axis seed)

    -- Lifecycle
    status          TEXT    NOT NULL CHECK(status IN
                    ('pending','running','success','failed','stale')),
    error_code      TEXT,                          -- short enum, NULL unless failed
    error_message   TEXT,                          -- one-line detail, truncated to 500
    created_at      TEXT    NOT NULL,              -- ISO8601 UTC
    updated_at      TEXT    NOT NULL,
    started_at      TEXT,                          -- set when status→running
    finished_at     TEXT,                          -- set when status→{success,failed}
    run_attempts    INTEGER NOT NULL DEFAULT 0,    -- incremented on each rerun
    last_slurm_job  TEXT,                          -- most recent SLURM job id

    -- SLURM request metadata (populated on submit)
    slurm_qos          TEXT,                       -- 'debug' | 'regular' | ...
    slurm_partition    TEXT,
    slurm_time_limit_s INTEGER,                    -- requested wall seconds
    slurm_ntasks       INTEGER,
    slurm_cpus_per_task INTEGER,

    -- SLURM actual wall time (populated on finish, differs from runtime_s
    -- which is just the driver's minimize() loop — slurm_wall_s includes
    -- module load, MPI bootstrap, save, teardown)
    slurm_wall_s       REAL,

    -- Standard metrics (NULL while pending/running/failed)
    final_iota_axis    REAL,
    final_iota_edge    REAL,
    final_aspect       REAL,
    final_volume       REAL,
    final_qs_metric    REAL,                       -- summed QS residual²
    nfev               INTEGER,                    -- total VMEC evals across ramp
    runtime_s          REAL,

    -- Escape hatch for non-standard metrics (JSON object)
    metrics_extra_json TEXT,

    -- Enforce one row per (canonical inputs, commit). Different commit
    -- with same inputs → different row (different ID via the hash).
    UNIQUE(input_hash, git_commit)
);

CREATE INDEX IF NOT EXISTS idx_stage1_status ON stage1(status);
CREATE INDEX IF NOT EXISTS idx_stage1_hash   ON stage1(input_hash);

-- ─────────────────────────────────────────────────────────────────────────────
-- stage2 — coil-only optimization (SquaredFlux + penalties)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stage2 (
    id              TEXT    PRIMARY KEY,           -- e.g. "s02_4c1e08"
    input_hash      TEXT    NOT NULL,
    git_commit      TEXT    NOT NULL,
    inputs_json     TEXT    NOT NULL,

    -- Parent stage 1 run. stage1_id is PART of the hashed inputs — the
    -- same stage2 config on a different stage1 warm start gets a
    -- different hash, hence a different id.
    stage1_id       TEXT    NOT NULL REFERENCES stage1(id),

    -- Lifecycle (same as stage1)
    status          TEXT    NOT NULL CHECK(status IN
                    ('pending','running','success','failed','stale')),
    error_code      TEXT,
    error_message   TEXT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    run_attempts    INTEGER NOT NULL DEFAULT 0,
    last_slurm_job  TEXT,

    slurm_qos           TEXT,
    slurm_partition     TEXT,
    slurm_time_limit_s  INTEGER,
    slurm_ntasks        INTEGER,
    slurm_cpus_per_task INTEGER,
    slurm_wall_s        REAL,

    -- Standard metrics
    final_sqflx           REAL,
    final_max_curvature   REAL,
    final_max_length      REAL,
    final_min_cc_dist     REAL,
    final_min_cs_dist     REAL,
    final_banana_current  REAL,   -- Amperes, signed
    runtime_s             REAL,

    metrics_extra_json    TEXT,

    UNIQUE(input_hash, git_commit)
);

CREATE INDEX IF NOT EXISTS idx_stage2_status    ON stage2(status);
CREATE INDEX IF NOT EXISTS idx_stage2_hash      ON stage2(input_hash);
CREATE INDEX IF NOT EXISTS idx_stage2_stage1_id ON stage2(stage1_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- singlestage — joint coil + surface optimization (BoozerLS)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS singlestage (
    id              TEXT    PRIMARY KEY,           -- e.g. "s03_912bff"
    input_hash      TEXT    NOT NULL,
    git_commit      TEXT    NOT NULL,
    inputs_json     TEXT    NOT NULL,

    stage2_id       TEXT    NOT NULL REFERENCES stage2(id),

    status          TEXT    NOT NULL CHECK(status IN
                    ('pending','running','success','failed','stale')),
    error_code      TEXT,
    error_message   TEXT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    run_attempts    INTEGER NOT NULL DEFAULT 0,
    last_slurm_job  TEXT,

    slurm_qos           TEXT,
    slurm_partition     TEXT,
    slurm_time_limit_s  INTEGER,
    slurm_ntasks        INTEGER,
    slurm_cpus_per_task INTEGER,
    slurm_wall_s        REAL,

    -- Standard metrics
    final_iota              REAL,
    final_qs_metric         REAL,
    final_boozer_residual   REAL,
    final_sqflx             REAL,
    final_max_curvature     REAL,
    final_min_cc_dist       REAL,
    final_min_cs_dist       REAL,
    final_max_length        REAL,
    final_banana_current    REAL,
    runtime_s               REAL,

    metrics_extra_json      TEXT,

    UNIQUE(input_hash, git_commit)
);

CREATE INDEX IF NOT EXISTS idx_singlestage_status    ON singlestage(status);
CREATE INDEX IF NOT EXISTS idx_singlestage_hash      ON singlestage(input_hash);
CREATE INDEX IF NOT EXISTS idx_singlestage_stage2_id ON singlestage(stage2_id);
