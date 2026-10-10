
import os
from pathlib import Path

from databricks import sql
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def load_customer_data():
    """Load customer churn data from the Databricks SQL table."""
    connection = sql.connect(
        server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM workspace.default.telco_customer_churn"
            )
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()

        import pandas as pd
        return pd.DataFrame(rows, columns=columns)
    finally:
        connection.close()
