import logging
import pg8000.dbapi as pg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_db_connection(db_config):
    # Initialize database connection with 30 second timeout
    try:
        conn = pg.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password'],
            timeout=60
        )
        cur = conn.cursor()
        logger.info("Database connection established.")
        return conn, cur
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {e}")
        raise


def close_db_connection(conn, cur):
    # Close database connection safely
    try:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
        logger.info("Database connection closed.")
    except Exception as e:
        # Check if it's just an already closed connection
        error_msg = str(e).lower()
        if 'closed' in error_msg or 'invalid' in error_msg:
            logger.debug(f"Connection was already closed: {e}")
        else:
            logger.error(f"Failed to close database connection: {e}")

def get_max_stagingid(cur):
    """Get the maximum stagingid from orgi.visibilityitemsstaging."""
    try:
        cur.execute("SELECT MAX(stagingid) FROM orgi.visibilityitemsstaging")
        result = cur.fetchone()
        return result[0] if result[0] is not None else 0
    except Exception as e:
        logger.error(f"Failed to get max stagingid: {e}")
        raise

def get_classtext(cur, classid):
    """Get classtext from orgi.productmaster or predefined mappings."""
    predefined_mappings = {
        1001: "Visicooler - Visicooler available",
        1002: "Visicooler - Brand of Visicooler",
        1003: "Coca-Cola Company visicooler - Size of Visicooler",
        1004: "Coca-Cola Company visicooler - Visicooler working",
        1005: "Coca-Cola Company visicooler - Visicooler pics allowed",
        1006: "Coca-Cola Company visicooler - Visicooler visibility from street",
        1007: "Coca-Cola Company visicooler - Visicooler visibility in shop Partial/Fully/Not visible",
        1008: "Coca-Cola Company visicooler - Visicooler accessible",
        1009: "Coca-Cola Company visicooler - Brand shelf strip",
        1010: "Coca-Cola Company visicooler - As per planogram",
        1011: "Coca-Cola Company visicooler - Cooler Purity",
        1012: "Coca-Cola Company visicooler - Number of shelves in cooler",
        1013: "Coca-Cola Company visicooler - Number of pure shelves",
        1014: "Coca-Cola Company visicooler - RGB space occupied ",
        1018: "Other Elements branded by Coca-Cola - Bottle Neck Ringer",
        1019: "Other Elements branded by Coca-Cola - Poster",
        1020: "Other Elements branded by Coca-Cola - Streamer",
        1021: "Other Elements branded by Coca-Cola - Combo board Non-digital",
        1022: "Other Elements branded by Coca-Cola - Digital Combo board",
        1023: "Other Elements branded by Coca-Cola - Menu board",
        1024: "Other Elements branded by Coca-Cola - Bar Signage",
        1025: "Other Elements branded by Coca-Cola - Menu Book",
        1026: "Other Elements branded by Coca-Cola - Table Menu",
        1027: "Other Elements branded by Coca-Cola - Table Sticker",
        1028: "Other Elements branded by Coca-Cola - Bar Mat",
        1029: "Other Elements branded by Coca-Cola - Napkin Holder",
        1030: "Other Elements branded by Coca-Cola - Tent Card",
        1031: "Other Elements branded by Coca-Cola - Tray",
        1032: "Other Elements branded by Coca-Cola - Shop Board",
        1033: "Other Elements branded by Coca-Cola - Umbrella",
        1034: "Other Elements branded by Coca-Cola - Neon Signage",
        1035: "Other Elements branded by Coca-Cola - Coke Glass",
        1036: "Other Elements branded by Coca-Cola - Waiter Apron",
        1037: "Other Elements branded by Coca-Cola - One pager Combo menu",
        1038: "Other Elements branded by Coca-Cola - Wall Menu",
        1039: "Other Elements branded by Coca-Cola - Rack shelf branding",
        1040: "Other Elements branded by Coca-Cola - Pillar Branding",
        1041: "Other Elements branded by Coca-Cola - Window Branding (One Way Vision)",
        1042: "Other Elements branded by Coca-Cola - Offer Communication (Printed Communication Material)",
        1043: "Other Elements branded by Coca-Cola - Aerial Hanger",
        1044: "Other Elements branded by Coca-Cola - Digital Display Ad",
        1045: "Other Elements branded by Coca-Cola - Coke logo Cup",
        1046: "Other Elements branded by Coca-Cola - Price Communication",
        1047: "All SKUs of Coca Cola Company and Pepsico - Present Y/N",
        1048: "All SKUs of Coca Cola Company and Pepsico - Number of Chilled Facings",
        1049: "All SKUs of Coca Cola Company and Pepsico - Number of Warm Facings",
        1050: "All SKUs of Coca Cola Company and Pepsico - Share of Chilled Facings",
        1051: "All SKUs of Coca Cola Company and Pepsico - Share of Warm Facings",
        1052: "All SKUs of Coca Cola Company and Pepsico - Present, but no facings",
        1053: "Other Elements branded by Coca-Cola - DPS"
    }
    if classid in predefined_mappings:
        return predefined_mappings[classid]
    try:
        cur.execute("SELECT productname FROM orgi.productmaster WHERE productclassid = %s", (classid,))
        result = cur.fetchone()
        return result[0] if result else 'Unknown'
    except Exception as e:
        logger.error(f"Failed to get classtext for classid {classid}: {e}")
        return 'Unknown'

def insert_ollama_results(cur, stagingid, results, modelname, s3_annotated_folder, image_paths,
                          conn=None, db_config=None, chunk_size=20):
    """Insert Ollama results into orgi.visibilityitemsstaging table.

    Inserts in chunks of `chunk_size` rows to avoid TCP timeout on long inserts.
    If a connection error occurs mid-insert, reconnects automatically (requires
    conn and db_config to be passed in) and retries the failed chunk up to 3 times.

    Note: stagingid is shared across all batches in a single pipeline run,
    so multiple inserts under the same stagingid are expected and correct.
    """
    insert_query = """
    INSERT INTO orgi.visibilityitemsstaging
    (stagingid, rowid, modelname, imagefilename, classid, classtext, value, inference,
     modelrun, processed_flag, storeid, storename, s3path_actual_file, s3path_annotated_file)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    formatted_records = [
        (
            stagingid,
            result['rowid'],
            modelname,
            result['imagefilename'],
            result['classid'],
            result['classtext'],
            result['value'],
            result['inference'],
            result['modelrun'],
            result['processed_flag'],
            result['storeid'],
            result['storename'],
            result['s3path_actual_file'],
            result['s3path_annotated_file'],
        )
        for result in results
    ]

    total = len(formatted_records)
    inserted = 0

    # Split into chunks so each round-trip is short and the TCP connection stays alive
    for chunk_start in range(0, total, chunk_size):
        chunk = formatted_records[chunk_start: chunk_start + chunk_size]
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                cur.executemany(insert_query, chunk)
                # Commit after every chunk so partial progress is saved
                if conn is not None:
                    conn.commit()
                inserted += len(chunk)
                logger.info(
                    f"Inserted rows {chunk_start + 1}–{chunk_start + len(chunk)} "
                    f"of {total} (stagingid {stagingid})"
                )
                break  # chunk succeeded, move on

            except Exception as e:
                error_msg = str(e).lower()
                network_keywords = [
                    'network', 'connection', 'timeout', 'broken pipe',
                    'reset by peer', 'timed out', '10053', '10054', '10060',
                    'winerror', 'connection was aborted', 'connection attempt failed',
                    'host has failed to respond',
                ]
                is_network_error = any(kw in error_msg for kw in network_keywords)

                if is_network_error and conn is not None and db_config is not None and attempt < max_attempts:
                    wait = 5 * attempt
                    logger.warning(
                        f"Network error on chunk {chunk_start}–{chunk_start + len(chunk)} "
                        f"(attempt {attempt}/{max_attempts}): {e}. "
                        f"Reconnecting in {wait}s…"
                    )
                    import time
                    time.sleep(wait)
                    try:
                        close_db_connection(conn, cur)
                    except Exception:
                        pass
                    # Reconnect and update the caller's cursor/connection references
                    new_conn, new_cur = initialize_db_connection(db_config)
                    # Replace references in-place so the rest of the insert uses them
                    conn.__dict__.update(new_conn.__dict__)
                    cur.__dict__.update(new_cur.__dict__)
                    logger.info("Reconnected successfully, retrying chunk…")
                else:
                    logger.error(
                        f"Failed to insert chunk {chunk_start}–{chunk_start + len(chunk)} "
                        f"into orgi.visibilityitemsstaging: {e}"
                    )
                    raise

    logger.info(f"Inserted {inserted} rows into orgi.visibilityitemsstaging with stagingid {stagingid}.")