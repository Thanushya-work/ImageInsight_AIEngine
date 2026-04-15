import logging
import cv2
import os
import tempfile
from app.db_handler import initialize_db_connection, close_db_connection
from app.s3_handler import S3Handler
from app.sovi_pipeline_runner import run_sovi_post_pipeline
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def get_last_iteration_id(conn):
    """Fetch the maximum iterationid from orgi.coolermetricsmaster."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT MAX(iterationid) FROM orgi.coolermetricsmaster")
        result = cur.fetchone()
        return result[0] if result and result[0] is not None else None
    finally:
        cur.close()

def get_sovi_records(conn):
    """Fetch records from orgi.fileupload where processed_flag = 'S' and category_id in (1,2)."""
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT filesequenceid, storeid, filename
            FROM orgi.fileupload
            WHERE processed_flag = 'S' AND category_id IN (1, 2)
        """)
        return cur.fetchall()
    finally:
        cur.close()

def update_processed_flag(conn, file_id, status):
    """Update processed_flag to 'Y' or 'E' for a given file_id."""
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE orgi.fileupload
            SET processed_flag = %s
            WHERE filesequenceid = %s
        """, (status, file_id))
        conn.commit()
    finally:
        cur.close()

def build_sovi_annotated_path(staging_id, image_name):
    """Build the SOVI annotated path for the given staging_id and image_name."""
    return f"ModelResults/Visicooler_{staging_id}/segmented_{image_name}"


def download_single_image(s3_handler, s3_path, local_path):
    s3_handler.s3_client.download_file(
        s3_handler.bucket_name,
        s3_path,
        local_path
    )


def insert_into_temp_tables(conn, results, store_id, image_name, iteration_id, s3_annotated):
    """Insert YOLO detections into temp.sku_prediction_temp_sovi and temp.cap_prediction_temp_sovi."""
    cur = conn.cursor()
    try:
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            shelfnumber = 0
            brand_name = 'UNKNOWN'

            # Insert into temp.sku_prediction_temp_sovi
            cur.execute("""
                INSERT INTO temp.sku_prediction_temp_sovi
                (store_id, image_file_name, iteration_id, prod_class_id, x1, y1, x2, y2, shelfnumber, brand_name, s3path_annotated_file)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (store_id, image_name, iteration_id, class_id, x1, y1, x2, y2, shelfnumber, brand_name, s3_annotated))

            # Insert into temp.cap_prediction_temp_sovi
            cur.execute("""
                INSERT INTO temp.cap_prediction_temp_sovi
                (store_id, image_file_name, iteration_id, cap_class_id, prod_class_id, x1, y1, x2, y2, shelfnumber, brand_name, s3path_annotated_file)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (store_id, image_name, iteration_id, class_id, None, x1, y1, x2, y2, shelfnumber, brand_name, s3_annotated))

        conn.commit()
    finally:
        cur.close()

def run_sovi_pipeline(config, iteration_id):
    # 1. Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 2. Connect to DB using existing get_db_connection
    db_config = config["db_config"]
    conn, cur = initialize_db_connection(db_config)

    try:
        logger.info(f"Using iteration_id: {iteration_id}")

        # 3. Fetch records from orgi.fileupload where processed_flag = 'S' AND category_id IN (1,2)
        records = get_sovi_records(conn)

        # 5. If no records found, log and exit
        if not records:
            logger.info("No records found with processed_flag='S' and category_id in (1,2). Exiting.")
            return

        # 6. Load YOLO model using config["sovi_model"]["model_path"]
        model_path = config["sovi_model"]["model_path"]
        model = YOLO(model_path)
        logger.info(f"Loaded YOLO model from {model_path}")

        # Initialize S3 handler
        s3_handler = S3Handler(config["s3_config"], config["db_config"])

        staging_counter = 1

        # 7. Loop through each record and process images
        for record in records:
            filesequenceid, storeid, filename = record
            logger.info(f"Processing {filename}")

            success = False
            for attempt in range(3):
                try:
                    # 1. Download image
                    temp_dir = tempfile.gettempdir()
                    local_image_path = os.path.join(temp_dir, f"input_{filename}")
                    s3path = config["s3_config"]["image_folder_s3"] + filename
                    download_single_image(s3_handler, s3path, local_image_path)

                    # 2. Run YOLO model
                    confidence = config["sovi_model"]["confidence"]
                    results = model.predict(local_image_path, conf=confidence)

                    staging_id = staging_counter
                    staging_counter += 1

                    # 4. Build annotated S3 path
                    annotated_s3_path = build_sovi_annotated_path(staging_id, filename)

                    # Insert into temp tables
                    insert_into_temp_tables(conn, results, storeid, filename, iteration_id, annotated_s3_path)

                    # 5. Save annotated image locally
                    annotated_image = results[0].plot()
                    local_annotated_path = os.path.join(temp_dir, f"annotated_{filename}")
                    cv2.imwrite(local_annotated_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                    # 6. Upload annotated image to S3
                    s3_handler.upload_file_to_s3(local_annotated_path, annotated_s3_path)

                    # Mark as processed
                    update_processed_flag(conn, filesequenceid, 'Y')
                    logger.info(f"Successfully processed {filename}")
                    success = True
                    break

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {filename}: {e}")
                    if attempt == 2:
                        logger.error(f"Failed to process {filename} after 3 attempts")
                        update_processed_flag(conn, filesequenceid, 'E')

            if not success:
                continue

        # Run post-processing pipeline
        logger.info("Starting SOVI post-processing pipeline")
        post_result = run_sovi_post_pipeline(config["db_config"], iteration_id, config)
        logger.info(f"SOVI post-processing completed with status: {post_result.overall_status}")

    finally:
        close_db_connection(conn, cur)