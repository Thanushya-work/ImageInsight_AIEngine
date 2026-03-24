# import logging
# import os
# from datetime import datetime

# import cv2
# import pg8000.dbapi as pg
# from ultralytics import YOLO

# from app.db_handler import close_db_connection, get_classtext

# logger = logging.getLogger(__name__)

# # Load model only once
# _ACTIVATION_MODEL = None

# # Categories excluded from activation detection
# VISICOOLER_CATEGORIES = {601, 602, 603, 604, 605}

# # Activation YOLO class → visibility class ID mapping
# ACTIVATION_MAPPINGS = {
#     "bar_signage": 1024,
#     "signage": 1024,
#     "combo_board": 1022,
#     "dps": 1053,
#     "menu_board": 1023,
#     "pillar_branding": 1040,
#     "poster": 1019,
#     "rgb_crate_stacking": 1056,
#     "shelf_display": 1064,
#     "table_sticker": 1027,
#     "streamer": 1020
# }


# def get_activation_model(model_path):

#     global _ACTIVATION_MODEL

#     if _ACTIVATION_MODEL is None:
#         logger.info(f"Loading activation YOLO model: {model_path}")
#         _ACTIVATION_MODEL = YOLO(model_path)

#     return _ACTIVATION_MODEL


# def run_activation_detection(image_paths, config, s3_handler, stagingid):

#     activation_cfg = config["activation_config"]

#     model_path = activation_cfg["activation_model_path"]
#     conf_threshold = activation_cfg["activation_conf_threshold"]
#     output_folder = activation_cfg.get("annotated_output_folder", "activation_annotated")

#     model = get_activation_model(model_path)

#     results = []

#     logger.info("Starting activation YOLO detection")

#     # Read S3 config
#     bucket = config["s3_config"]["bucket_name"]
#     folder = config["s3_config"].get("image_folder_s3", "")

#     valid_images = []
#     metadata = []

#     # Collect valid images first
#     for img in image_paths:

#         fileseqid, storename, filename, local_path, s3_key, storeid, subcategory = img

#         if subcategory in VISICOOLER_CATEGORIES:
#             continue

#         if not os.path.exists(local_path):
#             continue

#         valid_images.append(local_path)
#         metadata.append((filename, storename, storeid))

#     # Safety check
#     if not valid_images:
#         logger.info("No images eligible for activation detection")
#         return results

#     try:

#         # Run YOLO on ALL images at once (batch inference)
#         detections = model(valid_images, conf=conf_threshold, verbose=False)

#         os.makedirs(output_folder, exist_ok=True)

#         for i, result in enumerate(detections):

#             filename, storename, storeid = metadata[i]

#             # Process detections
#             activation_found = False

#             for box in result.boxes:

#                 cls_id = int(box.cls[0])
#                 class_name = model.names[cls_id]

#                 detected_name = class_name.lower().replace(" ", "_")

#                 if detected_name not in ACTIVATION_MAPPINGS:
#                     continue

#                 activation_found = True

#                 classid = ACTIVATION_MAPPINGS[detected_name]

#                 annotated_filename = f"annotated_{filename}"
#                 annotated_local_path = os.path.join(output_folder, annotated_filename)

#                 annotated_s3_path = f"ModelResults/VisibleItem_{stagingid}/{annotated_filename}"

#                 s3_path = f"{bucket}/{folder}{filename}"

#                 record = {
#                     "imagefilename": filename,
#                     "classid": classid,
#                     "classtext": class_name,
#                     "storeid": storeid,
#                     "storename": storename,
#                     "s3path_actual_file": s3_path,
#                     "s3path_annotated_file": annotated_s3_path
#                 }

#                 results.append(record)

#                 logger.info(
#                     f"Activation mapped '{detected_name}' → class {classid} | image: {filename}"
#                 )

#             # Only generate annotated image if activation detected
#             if activation_found:

#                 rendered = result.plot()

#                 annotated_filename = f"annotated_{filename}"
#                 annotated_local_path = os.path.join(output_folder, annotated_filename)

#                 cv2.imwrite(annotated_local_path, rendered)

#                 annotated_s3_path = f"ModelResults/VisibleItem_{stagingid}/{annotated_filename}"

#                 s3_handler.upload_file_to_s3(
#                     annotated_local_path,
#                     annotated_s3_path
#                 )

#     except Exception as e:

#         logger.error(f"Activation detection batch failed: {e}")

#     logger.info(f"Activation detection completed. Total detections: {len(results)}")

#     return results


# def insert_activation_results(db_config, activation_results, stagingid):

#     if not activation_results:
#         logger.info("No activation results to insert")
#         return

#     conn = pg.connect(
#         host=db_config['host'],
#         port=db_config['port'],
#         database=db_config['database'],
#         user=db_config['user'],
#         password=db_config['password']
#     )

#     cur = conn.cursor()

#     logger.info("Database connection established for activation insert")

#     # Cache classtext values
#     classtext_cache = {}

#     for cid in range(1018, 1065):
#         try:
#             classtext_cache[cid] = get_classtext(cur, cid)
#         except Exception:
#             classtext_cache[cid] = "Unknown"

#     # Get max existing rowid
#     cur.execute(
#         "SELECT COALESCE(MAX(rowid),0) FROM orgi.visibilityitemsstaging WHERE stagingid=%s",
#         (stagingid,)
#     )

#     max_rowid = int(cur.fetchone()[0])

#     now = datetime.now()

#     records = []
#     rowid = max_rowid + 1

#     for r in activation_results:

#         records.append(
#             (
#                 rowid,
#                 stagingid,
#                 "activation_yolo",
#                 r["imagefilename"],
#                 r["classid"],
#                 classtext_cache.get(r["classid"], r["classtext"]),
#                 "Y",
#                 1.0,
#                 now,
#                 "N",
#                 r["storeid"],
#                 r["storename"],
#                 r["s3path_actual_file"],
#                 r["s3path_annotated_file"]
#             )
#         )

#         rowid += 1

#     insert_query = """
#     INSERT INTO orgi.visibilityitemsstaging
#     (
#         rowid,
#         stagingid,
#         modelname,
#         imagefilename,
#         classid,
#         classtext,
#         value,
#         inference,
#         modelrun,
#         processed_flag,
#         storeid,
#         storename,
#         s3path_actual_file,
#         s3path_annotated_file
#     )
#     VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
#     """

#     cur.executemany(insert_query, records)

#     conn.commit()

#     close_db_connection(conn, cur)

#     logger.info(f"Inserted {len(records)} activation results into visibilityitemsstaging")


import logging
import os
from datetime import datetime

import cv2
import pg8000.dbapi as pg
from ultralytics import YOLO

from app.db_handler import close_db_connection, get_classtext

logger = logging.getLogger(__name__)

# Load model only once
_ACTIVATION_MODEL = None

# Categories excluded from activation detection
VISICOOLER_CATEGORIES = {601, 602, 603, 604, 605}

# Activation YOLO class → visibility class ID mapping
ACTIVATION_MAPPINGS = {
    "bar_signage": 1024,
    "signage": 1024,
    "combo_board": 1022,
    "dps": 1053,
    "menu_board": 1023,
    "pillar_branding": 1040,
    "poster": 1019,
    "rgb_crate_stacking": 1056,
    "shelf_display": 1064,
    "table_sticker": 1027,
    "streamer": 1020
}


def get_activation_model(model_path):

    global _ACTIVATION_MODEL

    if _ACTIVATION_MODEL is None:
        logger.info(f"Loading activation YOLO model: {model_path}")
        _ACTIVATION_MODEL = YOLO(model_path)

    return _ACTIVATION_MODEL


def run_activation_detection(image_paths, config, s3_handler, stagingid):

    activation_cfg = config["activation_config"]

    model_path = activation_cfg["activation_model_path"]
    conf_threshold = activation_cfg["activation_conf_threshold"]
    output_folder = activation_cfg.get("annotated_output_folder", "activation_annotated")

    model = get_activation_model(model_path)

    results = []

    logger.info("Starting activation YOLO detection")

    valid_images = []
    metadata = []

    # Collect valid images first
    for img in image_paths:

        fileseqid, storename, filename, local_path, s3_key, storeid, subcategory = img

        if subcategory in VISICOOLER_CATEGORIES:
            continue

        if not os.path.exists(local_path):
            continue

        valid_images.append(local_path)

        # include s3_key so we know exact source path
        metadata.append((filename, storename, storeid, s3_key))

    # Safety check
    if not valid_images:
        logger.info("No images eligible for activation detection")
        return results

    try:

        # Run YOLO on ALL images at once (batch inference)
        detections = model(valid_images, conf=conf_threshold, verbose=False)

        os.makedirs(output_folder, exist_ok=True)

        for i, result in enumerate(detections):

            filename, storename, storeid, s3_key = metadata[i]

            activation_found = False

            for box in result.boxes:

                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                detected_name = class_name.lower().replace(" ", "_")

                if detected_name not in ACTIVATION_MAPPINGS:
                    continue

                activation_found = True

                classid = ACTIVATION_MAPPINGS[detected_name]

                annotated_filename = f"annotated_{filename}"
                annotated_local_path = os.path.join(output_folder, annotated_filename)

                annotated_s3_path = f"ModelResults/VisibleItem_{stagingid}/{annotated_filename}"

                # Use actual S3 key that worked during download
                s3_path = s3_key

                record = {
                    "imagefilename": filename,
                    "classid": classid,
                    "classtext": class_name,
                    "storeid": storeid,
                    "storename": storename,
                    "s3path_actual_file": s3_path,
                    "s3path_annotated_file": annotated_s3_path
                }

                results.append(record)

                logger.info(
                    f"Activation mapped '{detected_name}' → class {classid} | image: {filename}"
                )

            # Only generate annotated image if activation detected
            if activation_found:

                rendered = result.plot()

                annotated_filename = f"annotated_{filename}"
                annotated_local_path = os.path.join(output_folder, annotated_filename)

                cv2.imwrite(annotated_local_path, rendered)

                annotated_s3_path = f"ModelResults/VisibleItem_{stagingid}/{annotated_filename}"

                s3_handler.upload_file_to_s3(
                    annotated_local_path,
                    annotated_s3_path
                )

    except Exception as e:

        logger.error(f"Activation detection batch failed: {e}")

    logger.info(f"Activation detection completed. Total detections: {len(results)}")

    return results


def insert_activation_results(db_config, activation_results, stagingid):

    if not activation_results:
        logger.info("No activation results to insert")
        return

    conn = pg.connect(
        host=db_config['host'],
        port=db_config['port'],
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password']
    )

    cur = conn.cursor()

    logger.info("Database connection established for activation insert")

    # Cache classtext values
    classtext_cache = {}

    for cid in range(1018, 1065):
        try:
            classtext_cache[cid] = get_classtext(cur, cid)
        except Exception:
            classtext_cache[cid] = "Unknown"

    # Get max existing rowid
    cur.execute(
        "SELECT COALESCE(MAX(rowid),0) FROM orgi.visibilityitemsstaging WHERE stagingid=%s",
        (stagingid,)
    )

    max_rowid = int(cur.fetchone()[0])

    now = datetime.now()

    records = []
    rowid = max_rowid + 1

    for r in activation_results:

        records.append(
            (
                rowid,
                stagingid,
                "activation_yolo",
                r["imagefilename"],
                r["classid"],
                classtext_cache.get(r["classid"], r["classtext"]),
                "Y",
                1.0,
                now,
                "N",
                r["storeid"],
                r["storename"],
                r["s3path_actual_file"],
                r["s3path_annotated_file"]
            )
        )

        rowid += 1

    insert_query = """
    INSERT INTO orgi.visibilityitemsstaging
    (
        rowid,
        stagingid,
        modelname,
        imagefilename,
        classid,
        classtext,
        value,
        inference,
        modelrun,
        processed_flag,
        storeid,
        storename,
        s3path_actual_file,
        s3path_annotated_file
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    cur.executemany(insert_query, records)

    conn.commit()

    close_db_connection(conn, cur)

    logger.info(f"Inserted {len(records)} activation results into visibilityitemsstaging")