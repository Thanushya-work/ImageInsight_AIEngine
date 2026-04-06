import matplotlib
matplotlib.use('Agg')

import traceback
import logging
import os
import sys
import tempfile
import time

from app.config_loader import load_config
from app.s3_handler import S3Handler
from app.activation import run_activation_detection, insert_activation_results
from app.file_uploader import FileUploader
from app.db_handler import initialize_db_connection, close_db_connection
from app.visicooler import run_visicooler_analysis, check_visibilitydetails_schema
from cap_pipeline_runner import run_cap_pipeline


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

STALE_TIMEOUT_MINUTES = 60


def mark_batch_failed(db_config, pod_id):
    """
    Marks files in this pod batch as FAILED so they won't retry forever
    """

    try:

        conn, cur = initialize_db_connection(db_config)

        cur.execute("""
            UPDATE orgi.fileupload
            SET processed_flag = 'E'
            WHERE podid = %s
              AND processed_flag = 'I'
        """, (pod_id,))

        failed_count = cur.rowcount

        conn.commit()

        logger.error(f"Marked {failed_count} files as FAILED for {pod_id}")

        close_db_connection(conn, cur)

    except Exception as e:
        logger.error(f"Failed to mark batch as FAILED: {e}")


def get_unprocessed_store_count(conn):

    try:

        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(DISTINCT storeid)
            FROM orgi.fileupload
            WHERE (processed_flag IN ('P','0.0') OR processed_flag IS NULL)
                AND storeid IS NOT NULL
                AND subcategory_id != 3
        """)

        count = cur.fetchone()[0]

        cur.close()

        return count

    except Exception as e:

        logger.error(f"Failed to get unprocessed store count: {e}")

        return 0


def reset_stale_batches(conn, stale_timeout_minutes):

    try:

        cur = conn.cursor()

        cur.execute("""
            UPDATE orgi.fileupload
            SET processed_flag = 'P', podid = NULL
            WHERE processed_flag = 'I'
              AND uploadtimestamp < NOW() - INTERVAL '%s minutes'
        """ % stale_timeout_minutes)

        reset_count = cur.rowcount

        conn.commit()

        cur.close()

        if reset_count > 0:
            logger.warning(
                f"Reset {reset_count} stale files (stuck >{stale_timeout_minutes}m)"
            )

        return reset_count

    except Exception as e:

        logger.error(f"Failed to reset stale batches: {e}")

        conn.rollback()

        return 0


def assign_stores_to_pod(conn, store_count, pod_id):

    try:

        cur = conn.cursor()

        cur.execute("""
            SELECT DISTINCT storeid
            FROM orgi.fileupload
            WHERE (processed_flag IN ('P','0.0') OR processed_flag IS NULL)
                AND storeid IS NOT NULL
                AND subcategory_id != 3
            ORDER BY storeid
            LIMIT %s
        """, (store_count,))

        selected_stores = [row[0] for row in cur.fetchall()]

        if not selected_stores:

            cur.close()

            return 0, 0

        cur.execute("""
            UPDATE orgi.fileupload
            SET processed_flag = 'I', podid = %s
            WHERE storeid = ANY(%s)
                AND (processed_flag IN ('P','0.0') OR processed_flag IS NULL)
                AND subcategory_id != 3
        """, (pod_id, selected_stores))

        assigned_files = cur.rowcount
        assigned_stores = len(selected_stores)

        conn.commit()

        cur.close()

        logger.info(
            f"Assigned {assigned_stores} stores ({assigned_files} images) to {pod_id}"
        )

        return assigned_stores, assigned_files

    except Exception as e:

        logger.error(f"Failed to assign stores to pod: {e}")

        conn.rollback()

        return 0, 0


def get_or_create_iterationid(conn):

    try:

        cur = conn.cursor()

        cur.execute("SELECT COALESCE(MAX(iteration_id),0) FROM temp.cap_prediction_temp")

        max_iteration = cur.fetchone()[0]

        iterationid = max_iteration + 1

        cur.close()

        logger.info(f"Using iterationid: {iterationid} for this pipeline run")

        return iterationid

    except Exception as e:

        logger.error(f"Failed to get iterationid: {e}")

        return 1


# --------------------------------------------------------


def execute_models(pod_id, iterationid, stagingid, batch_number):

    conn = None
    cur = None

    try:

        logger.info(f"Batch {batch_number} → Starting execution for {pod_id}")

        config = load_config('config.json')

        s3_config = config['s3_config']
        db_config = config['db_config']
        visicooler_config = config['visicooler_config']

        conn, cur = initialize_db_connection(db_config)

        if not check_visibilitydetails_schema(cur):

            logger.error("Schema validation failed for orgi.visibilitydetails")

            return False

        s3_handler = S3Handler(s3_config, db_config)

        with tempfile.TemporaryDirectory() as temp_dir:

            logger.info(f"Downloading images for {pod_id}...")

            image_paths, failed_files = s3_handler.download_images_from_s3(temp_dir, pod_id)

            logger.info(f"Batch {batch_number} → Downloaded {len(image_paths)} images")

            if not image_paths:

                logger.warning(f"No images downloaded for {pod_id}")

                return False

            cur.execute("""
                SELECT GREATEST(
                    COALESCE((SELECT MAX(cyclecountid) FROM orgi.visibilitydetails),0),
                    COALESCE((SELECT MAX(stagingid) FROM orgi.visibilityitemsstaging),0)
                )
            """)

            row = cur.fetchone()

            cyclecountid = (row[0] if row and row[0] else 0) + 1

            logger.info(f"Batch {batch_number} → Running visicooler analysis")

            run_visicooler_analysis(
                image_paths=image_paths,
                config=config,
                s3_handler=s3_handler,
                conn=conn,
                cur=cur,
                output_folder_path=visicooler_config['output_folder_path'],
                cyclecountid=cyclecountid,
                iterationid=iterationid
            )

            conn = None
            cur = None

            logger.info(f"Batch {batch_number} → Running activation detection")

            activation_results = run_activation_detection(
                image_paths,
                config,
                s3_handler,
                stagingid
            )

            insert_activation_results(
                db_config=db_config,
                activation_results=activation_results,
                stagingid=stagingid
            )

            conn, cur = initialize_db_connection(db_config)

            file_uploader = FileUploader(None)

            file_uploader.update_processed_flag(conn, image_paths)

            logger.info(f"Batch {batch_number} → Processed flag updated successfully")

        return True

    except Exception as e:

        logger.error(f"Error in execute_models: {e}")

        logger.error(traceback.format_exc())

        return False


# --------------------------------------------------------


def main():

    if len(sys.argv) < 2:

        logger.error("Usage: python main.py <pod-id>")

        sys.exit(1)

    pod_id = sys.argv[1]

    config = load_config('config.json')

    db_config = config['db_config']

    conn, cur = initialize_db_connection(db_config)

    iterationid = get_or_create_iterationid(conn)

    cur.execute("SELECT MAX(stagingid) FROM orgi.visibilityitemsstaging")

    result = cur.fetchone()

    stagingid = (result[0] if result[0] else 0) + 1

    close_db_connection(conn, cur)

    logger.info("="*60)
    logger.info(f"PIPELINE STARTED")
    logger.info(f"Iteration ID: {iterationid}")
    logger.info(f"Staging ID: {stagingid}")
    logger.info("="*60)

    batch_size = None
    batch_number = 0

    max_batch_retries = 3
    max_assignment_retries = 5
    max_loop_failures = 3

    assignment_retry_count = 0
    loop_failure_count = 0

    while True:

        try:

            conn, cur = initialize_db_connection(db_config)

            reset_stale_batches(conn, STALE_TIMEOUT_MINUTES)

            unprocessed_stores = get_unprocessed_store_count(conn)

            if unprocessed_stores == 0:

                logger.info("All stores processed successfully")

                close_db_connection(conn, cur)

                run_cap_pipeline(
                    db_config=db_config,
                    iteration_id=iterationid,
                    config=config
                )

                logger.info("="*60)
                logger.info("PIPELINE COMPLETED")
                logger.info(f"Iteration ID: {iterationid}")
                logger.info(f"Staging ID: {stagingid}")
                logger.info(f"Total Batches: {batch_number}")
                logger.info("="*60)

                break

            if batch_size is None:

                while True:

                    try:

                        batch_input = input(
                            f"Enter batch size (stores) for {pod_id} (1-{unprocessed_stores}): "
                        ).strip()

                        batch_size = int(batch_input)

                        if 1 <= batch_size <= unprocessed_stores:

                            break

                    except ValueError:

                        print("Enter a valid number")

            assigned_stores, assigned_files = assign_stores_to_pod(
                conn,
                batch_size,
                pod_id
            )

            close_db_connection(conn, cur)

            if assigned_stores == 0:

                assignment_retry_count += 1

                if assignment_retry_count >= max_assignment_retries:

                    logger.error("Maximum assignment retries reached.")

                    break

                logger.warning(
                    f"No stores assigned. Retry {assignment_retry_count}/{max_assignment_retries}"
                )

                time.sleep(10)

                continue

            assignment_retry_count = 0

            batch_number += 1

            logger.info(f"Starting Batch {batch_number}")

            current_batch_retries = 0

            while current_batch_retries < max_batch_retries:

                success = execute_models(pod_id, iterationid, stagingid, batch_number)

                if success:

                    logger.info(f"Batch {batch_number} completed successfully")

                    break

                current_batch_retries += 1

                if current_batch_retries >= max_batch_retries:

                    logger.error(
                        f"Batch {batch_number} failed after {max_batch_retries} retries."
                    )

                    mark_batch_failed(db_config, pod_id)

                    break

                logger.warning(
                    f"Batch {batch_number} failed. Retry {current_batch_retries}/{max_batch_retries}"
                )

                time.sleep(10)

            loop_failure_count = 0

            time.sleep(5)

        except KeyboardInterrupt:

            logger.info("Pipeline interrupted by user")

            break

        except Exception as e:

            loop_failure_count += 1

            logger.error(f"Error in main loop: {e}")

            logger.error(traceback.format_exc())

            if loop_failure_count >= max_loop_failures:

                logger.error("Main loop crashed repeatedly. Stopping pipeline.")

                break

            time.sleep(10)

    logger.info(f"Pipeline execution completed for {pod_id}")


if __name__ == "__main__":

    main()