import oracledb

# DSN for Oracle connection
dsn = "(DESCRIPTION = (ADDRESS = (PROTOCOL = TCP)(HOST = chip-vm02-scan.educ.gov.bc.ca)(PORT = 1521))(CONNECT_DATA=(SERVER = DEDICATED)(SERVICE_NAME = oltpd1.world)))"

def check_db_time():
    """Check current database time."""
    try:
        conn = oracledb.connect(user="", password="", dsn=dsn)
        cursor = conn.cursor()
        cursor.execute("SELECT sysdate FROM dual")
        db_time = cursor.fetchone()[0]
        print(f"Database time: {db_time}")
    except Exception as e:
        print("Failed to check DB time:", e)
    finally:
        if 'conn' in locals():
            conn.close()

def fetch_one_record():
    """Fetch one real data record from a table."""
    try:
        conn = oracledb.connect(user="", password="", dsn=dsn)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM student FETCH FIRST 1 ROWS ONLY")
        record = cursor.fetchone()
        print(f"One record: {record}")
    except Exception as e:
        print("Failed to fetch record:", e)
    finally:
        if 'conn' in locals():
            conn.close()
    

if __name__ == "__main__":
    print("Testing Oracle DB connection...")
    check_db_time()
    fetch_one_record()