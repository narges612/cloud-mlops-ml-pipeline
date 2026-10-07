from pathlib import Path
import urllib.request

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = (
    "https://raw.githubusercontent.com/"
    "IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)

OUTPUT_FILE = DATA_DIR / "telco_customer_churn.csv"


def download_dataset():
    if OUTPUT_FILE.exists():
        print(f"Dataset already exists: {OUTPUT_FILE}")
        return

    urllib.request.urlretrieve(URL, OUTPUT_FILE)
    print(f"Dataset downloaded to: {OUTPUT_FILE}")


if __name__ == "__main__":
    download_dataset()