from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import psycopg2

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    step: int
    name: str
    status: str                   # "success" | "failed" | "skipped"
    rows_affected: Optional[int]
    duration_ms: float
    error: Optional[str] = None

    def __str__(self) -> str:
        rows = f"{self.rows_affected} rows" if self.rows_affected is not None else "n/a"
        return (
            f"  Step {self.step:>2}  [{self.status.upper():<7}]  "
            f"{self.name:<55}  {rows:<12}  {self.duration_ms:>8.1f} ms"
        )


@dataclass
class PipelineResult:
    iteration_id: int
    overall_status: str           # "success" | "failed"
    total_duration_ms: float
    steps: list[StepResult] = field(default_factory=list)

    # ── Pretty summary ────────────────────────────────────────────────────────
    def log_summary(self) -> None:
        sep = "=" * 100
        logger.info(sep)
        logger.info("  CAP PREDICTION PIPELINE  –  SUMMARY (6 steps)")
        logger.info(sep)
        logger.info(f"  Iteration ID   : {self.iteration_id}")
        logger.info(f"  Overall status : {self.overall_status.upper()}")
        logger.info(f"  Total duration : {self.total_duration_ms:.1f} ms")
        logger.info("-" * 100)
        logger.info(
            f"  {'Step':>4}  {'Status':<9}  {'Name':<55}  {'Rows':<12}  {'Duration':>10}"
        )
        logger.info("-" * 100)
        for s in self.steps:
            logger.info(str(s))
        logger.info(sep)

        # Surface any errors
        failed = [s for s in self.steps if s.status == "failed"]
        if failed:
            logger.error(f"  {len(failed)} step(s) failed:")
            for s in failed:
                logger.error(f"    Step {s.step} – {s.name}")
                if s.error:
                    first_line = s.error.strip().splitlines()[-1]
                    logger.error(f"      {first_line}")
            logger.info(sep)


# ── SQL definitions ────────────────────────────────────────────────────────────
def _build_queries(iteration_id: int) -> list[dict]:
    """Return all pipeline SQL statements in execution order.

    Steps removed vs. the original 8-step version
    -----------------------------------------------
    Old Step 2 – "Insert CAP predictions from SKU table"
        cap_sku_mapper (Python, called in visicooler.py) already inserts caps
        into temp.cap_prediction_temp with prod_class_id pre-filled via
        4-phase brand+package matching.  Copying SKU rows a second time here
        created ghost rows.

    Old Steps 3 & 4 – SQL vertical-match / centroid-distance remapping
        The Python mapper is brand+package-aware and IoU-deduped; the SQL
        fallback is no longer needed.

    Old Step 5 – "Remove small caps inside SKU bounding box (< 15 % area)"
        With Steps 2-4 gone, cap rows are *only* cap-model detections.
        The Python IoU-dedup in cap_sku_mapper already handles overlapping
        duplicate caps before insertion.

    Remaining pipeline (renumbered 1-6):
        1  Create unique index          (guard against any race-condition dups)
        2  Remove duplicate rows        (clean-up safety net)
        3  Populate coolermetricsmaster
        4  Populate coolermetricstransaction
        5  Update caserid on coolermetricsmaster
        6  Insert structural-count extra rows  ← BUG FIXED HERE
    """
    iid = int(iteration_id)          # guard against injection
    return [
        # ── 1 ─────────────────────────────────────────────────────────────────
        {
            "step": 1,
            "name": "Create unique index on cap_prediction_temp",
            "sql": """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_cap_box
                ON temp.cap_prediction_temp (
                    store_id, image_file_name, s3path_annotated_file,
                    iteration_id, cap_class_id, x1, y1, x2, y2
                );
            """,
        },
        # ── 2 ─────────────────────────────────────────────────────────────────
        {
            "step": 2,
            "name": "Remove duplicate rows",
            "sql": """
                DELETE FROM temp.cap_prediction_temp c
                USING (
                    SELECT
                        store_id, image_file_name, s3path_annotated_file,
                        iteration_id, cap_class_id, x1, y1,
                        MIN(ctid) AS keep_ctid
                    FROM temp.cap_prediction_temp
                    GROUP BY
                        store_id, image_file_name, s3path_annotated_file,
                        iteration_id, cap_class_id, x1, y1
                    HAVING COUNT(*) > 1
                ) d
                WHERE c.store_id              = d.store_id
                  AND c.image_file_name       = d.image_file_name
                  AND c.s3path_annotated_file = d.s3path_annotated_file
                  AND c.iteration_id          = d.iteration_id
                  AND c.cap_class_id          = d.cap_class_id
                  AND c.x1                   = d.x1
                  AND c.y1                   = d.y1
                  AND c.ctid                 <> d.keep_ctid;
            """,
        },
        # ── 3 ─────────────────────────────────────────────────────────────────
        {
            "step": 3,
            "name": "Populate orgi.coolermetricsmaster",
            "sql": f"""
                WITH image_map AS (
                    SELECT DISTINCT
                        cpt.iteration_id  AS iterationid,
                        cpt.store_id      AS storeid,
                        cpt.image_file_name,
                        cpt.s3path_annotated_file,
                        DENSE_RANK() OVER (
                            PARTITION BY cpt.iteration_id
                            ORDER BY
                                cpt.store_id,
                                cpt.image_file_name,
                                cpt.s3path_annotated_file
                        ) AS iterationtranid
                    FROM temp.cap_prediction_temp cpt
                    WHERE cpt.iteration_id = {iid}
                )
                INSERT INTO orgi.coolermetricsmaster (
                    iterationid, iterationtranid, storeid,
                    caserid, modelrun, processed_flag
                )
                SELECT
                    i.iterationid,
                    i.iterationtranid,
                    i.storeid,
                    pm.caserid,
                    NOW(),
                    'N'
                FROM image_map i
                JOIN orgi.puritymapping pm ON pm.caserid IS NOT NULL
                ON CONFLICT (iterationid, iterationtranid) DO NOTHING;
            """,
        },
        # ── 4 ─────────────────────────────────────────────────────────────────
        {
            "step": 4,
            "name": "Populate orgi.coolermetricstransaction",
            "sql": f"""
                WITH image_map AS (
                    SELECT DISTINCT
                        cpt.iteration_id  AS iterationid,
                        cpt.store_id      AS storeid,
                        cpt.image_file_name,
                        cpt.s3path_annotated_file,
                        DENSE_RANK() OVER (
                            PARTITION BY cpt.iteration_id
                            ORDER BY
                                cpt.store_id,
                                cpt.image_file_name,
                                cpt.s3path_annotated_file
                        ) AS iterationtranid
                    FROM temp.cap_prediction_temp cpt
                    WHERE cpt.iteration_id = {iid}
                )
                INSERT INTO orgi.coolermetricstransaction (
                    iterationid, iterationtranid, shelfnumber,
                    productsequenceno, productclassid,
                    x1, y1, x2, y2, confidence,
                    imagefilename, s3path_actual_file, s3path_annotated_file
                )
                SELECT
                    cpt.iteration_id,
                    im.iterationtranid,
                    cpt.shelfnumber,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            cpt.iteration_id,
                            cpt.store_id,
                            cpt.image_file_name,
                            cpt.s3path_annotated_file,
                            cpt.shelfnumber
                        ORDER BY cpt.prod_class_id, cpt.x1
                    ) AS productsequenceno,
                    cpt.prod_class_id,
                    cpt.x1, cpt.y1, cpt.x2, cpt.y2,
                    NULL,
                    cpt.image_file_name,
                    'april_store_images/' || cpt.image_file_name,
                    cpt.s3path_annotated_file
                FROM temp.cap_prediction_temp cpt
                JOIN image_map im
                  ON  im.iterationid           = cpt.iteration_id
                 AND im.storeid                = cpt.store_id
                 AND im.image_file_name        = cpt.image_file_name
                 AND im.s3path_annotated_file  = cpt.s3path_annotated_file
                WHERE cpt.iteration_id = {iid};
            """,
        },
        # ── 5 ─────────────────────────────────────────────────────────────────
        {
            "step": 5,
            "name": "Update caserid on coolermetricsmaster",
            "sql": f"""
                UPDATE orgi.coolermetricsmaster c
                SET caserid = p.caserid
                FROM orgi.storemaster s
                JOIN orgi.puritymapping p
                  ON lower(p.casername) = lower(s.cooler)
                WHERE s.storeid      = c.storeid
                  AND c.iterationid  = {iid};
            """,
        },
        # ── 6 ─────────────────────────────────────────────────────────────────
        # BUG FIX: Extra rows for structural counts were being duplicated across
        # multiple images of the same store.
        #
        # Root cause (old code)
        # ─────────────────────
        # The old Step 6 joined structural_count_temp directly to
        # coolermetricsmaster ON (iterationid, storeid) with NO image_file_name
        # filter.  A store with 2 images has 2 rows in coolermetricsmaster
        # (iterationtranid=1 and iterationtranid=2).  Every row in
        # structural_count_temp (which is per-image) therefore matched BOTH
        # coolermetricsmaster rows and generated extra_count rows × 2.
        #
        # Example (this run):
        #   Coca-Cola img2 → extra_count=4, matched iterationtranid 1 AND 2
        #   → 8 extra rows attempted → some survived ON CONFLICT → DB got 24
        #   instead of the correct 20.
        #
        #   Sprite img1   → extra_count=7, matched iterationtranid 1 AND 2
        #   → 14 rows attempted → productsequenceno collisions → some dropped
        #   → DB got 55 instead of the correct 56.
        #
        # Fix
        # ───
        # Resolve iterationtranid per image by joining structural_count_temp
        # through cap_prediction_temp (which carries image_file_name and the
        # same DENSE_RANK ordering used in Steps 3/4).  This guarantees exactly
        # one iterationtranid per structural_count_temp row regardless of how
        # many images the store has.
        #
        # productsequenceno offset
        # ────────────────────────
        # MAX() is intentionally NOT filtered by productclassid.  The PK of
        # coolermetricstransaction is (iterationid, iterationtranid, shelfnumber,
        # productsequenceno) with no productclassid column, so we must offset
        # past the global max for that (iterationtranid, shelfnumber=0) slot to
        # avoid collisions with rows already written by Step 4 for other SKUs.
        {
            "step": 6,
            "name": "Insert structural-count extra rows",
            "sql": f"""
                INSERT INTO orgi.coolermetricstransaction (
                    iterationid,
                    iterationtranid,
                    shelfnumber,
                    productsequenceno,
                    productclassid,
                    imagefilename,
                    s3path_actual_file,
                    s3path_annotated_file
                )
                SELECT
                    sc.iteration_id,
                    im.iterationtranid,
                    0                           AS shelfnumber,
                    ROW_NUMBER() OVER (
                        PARTITION BY sc.iteration_id, im.iterationtranid
                        ORDER BY sc.prod_class_id
                    ) + COALESCE((
                        SELECT MAX(t2.productsequenceno)
                        FROM orgi.coolermetricstransaction t2
                        WHERE t2.iterationid     = sc.iteration_id
                          AND t2.iterationtranid = im.iterationtranid
                          AND t2.shelfnumber     = 0
                        -- Intentionally NO productclassid filter here.
                        -- The PK includes productsequenceno but NOT productclassid,
                        -- so we must offset past the global max for this
                        -- (iterationtranid, shelfnumber) slot to avoid collisions
                        -- with rows inserted for other SKUs in Step 4.
                    ), 0)                       AS productsequenceno,
                    sc.prod_class_id,
                    sc.image_file_name,
                    'april_store_images/' || sc.image_file_name,
                    NULL                        AS s3path_annotated_file
                FROM temp.structural_count_temp sc
                -- ── FIX: resolve iterationtranid per image ───────────────────
                -- Join through cap_prediction_temp to get the DENSE_RANK value
                -- for this specific image_file_name, matching exactly what
                -- Steps 3 and 4 computed.  Without image_file_name in the join
                -- a store with N images fans out to N iterationtranid matches
                -- and inserts N × extra_count rows instead of extra_count.
                JOIN (
                    SELECT DISTINCT
                        cpt.iteration_id,
                        cpt.store_id,
                        cpt.image_file_name,
                        DENSE_RANK() OVER (
                            PARTITION BY cpt.iteration_id
                            ORDER BY
                                cpt.store_id,
                                cpt.image_file_name,
                                cpt.s3path_annotated_file
                        ) AS iterationtranid
                    FROM temp.cap_prediction_temp cpt
                    WHERE cpt.iteration_id = {iid}
                ) im
                  ON  im.iteration_id    = sc.iteration_id
                 AND  im.store_id::TEXT  = sc.store_id::TEXT
                 AND  im.image_file_name = sc.image_file_name
                -- ── belt-and-braces: verify master row exists ─────────────────
                JOIN orgi.coolermetricsmaster cm
                  ON  cm.iterationid     = im.iteration_id
                 AND  cm.iterationtranid = im.iterationtranid
                CROSS JOIN generate_series(1, sc.extra_count)
                WHERE sc.iteration_id = {iid}
                  AND EXISTS (
                        SELECT 1 FROM temp.structural_count_temp
                        WHERE iteration_id = {iid}
                  )
                ON CONFLICT (iterationid, iterationtranid, shelfnumber, productsequenceno)
                DO NOTHING;
            """,
        },
    ]


# ── Core runner ────────────────────────────────────────────────────────────────
def run_cap_pipeline(db_config: dict, iteration_id: int) -> PipelineResult:
    """
    Execute all 6 CAP post-processing steps sequentially.

    Parameters
    ----------
    db_config : dict
        Database credentials from config.json  (keys: host, port, database,
        user, password).
    iteration_id : int
        The iteration ID used in the current pipeline run (same value passed
        to visicooler / ollama steps).

    Returns
    -------
    PipelineResult
        Dataclass containing per-step outcomes and the overall status.
        Always logs a formatted summary table to the application logger.
    """
    pipeline_start = time.monotonic()
    step_results: list[StepResult] = []
    overall_status = "success"

    logger.info("=" * 100)
    logger.info(f"  CAP PREDICTION PIPELINE  –  START  (iteration_id={iteration_id})")
    logger.info("=" * 100)

    # ── Open connection ────────────────────────────────────────────────────────
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            dbname=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
        )
        conn.autocommit = False
        logger.info(
            f"  Connected to {db_config['host']}:{db_config['port']} / {db_config['database']}"
        )
    except Exception as exc:
        logger.error(f"  Database connection failed: {exc}")
        result = PipelineResult(
            iteration_id=iteration_id,
            overall_status="failed",
            total_duration_ms=round((time.monotonic() - pipeline_start) * 1000, 2),
            steps=[
                StepResult(
                    step=0,
                    name="Database connection",
                    status="failed",
                    rows_affected=None,
                    duration_ms=0.0,
                    error=str(exc),
                )
            ],
        )
        result.log_summary()
        return result

    # ── Execute steps ──────────────────────────────────────────────────────────
    try:
        with conn:
            with conn.cursor() as cur:
                for query in _build_queries(iteration_id):
                    step_start = time.monotonic()
                    step_num = query["step"]
                    step_name = query["name"]

                    logger.info(f"  ▶  Step {step_num}/6 – {step_name} …")

                    try:
                        cur.execute(query["sql"])
                        rows = cur.rowcount if cur.rowcount >= 0 else None
                        duration = round((time.monotonic() - step_start) * 1000, 2)

                        step_results.append(
                            StepResult(
                                step=step_num,
                                name=step_name,
                                status="success",
                                rows_affected=rows,
                                duration_ms=duration,
                            )
                        )
                        rows_label = f"{rows} rows" if rows is not None else ""
                        logger.info(
                            f"     ✓  Completed in {duration:.1f} ms  {rows_label}"
                        )

                    except Exception as exc:
                        duration = round((time.monotonic() - step_start) * 1000, 2)
                        err_trace = traceback.format_exc()

                        step_results.append(
                            StepResult(
                                step=step_num,
                                name=step_name,
                                status="failed",
                                rows_affected=None,
                                duration_ms=duration,
                                error=err_trace,
                            )
                        )
                        logger.error(
                            f"     ✗  FAILED in {duration:.1f} ms – rolling back transaction"
                        )
                        logger.error(err_trace)

                        overall_status = "failed"
                        conn.rollback()
                        break  # abort remaining steps

    finally:
        try:
            conn.close()
        except Exception:
            pass

    # ── Assemble and return result ─────────────────────────────────────────────
    total_ms = round((time.monotonic() - pipeline_start) * 1000, 2)

    result = PipelineResult(
        iteration_id=iteration_id,
        overall_status=overall_status,
        total_duration_ms=total_ms,
        steps=step_results,
    )
    result.log_summary()
    return result