import os
import requests
import pandas as pd
import duckdb
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Constants
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
RAW_DIR = "data/raw/"
INPUT_PARQUET = "data/forecast_input.parquet"
OUTPUT_PARQUET = "data/forecast_output.parquet"

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def check_remote_parquet_exists(month_str):
    fname = f"yellow_tripdata_{month_str}.parquet"
    url = TLC_BASE_URL + fname
    resp = requests.head(url)
    return resp.status_code == 200

def download_parquet(month_str):
    fname = f"yellow_tripdata_{month_str}.parquet"
    url = TLC_BASE_URL + fname
    ensure_dir(RAW_DIR)
    local_path = os.path.join(RAW_DIR, fname)

    if os.path.exists(local_path):
        log(f"[INFO] File already downloaded: {fname}")
        return local_path

    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        log(f"[INFO] Downloaded {fname}")
        return local_path
    else:
        raise Exception(f"[ERROR] Could not download {url}. Status code {resp.status_code}")

def summarize_month_to_df(parquet_path, month_str):
    y, m = month_str.split("-")
    con = duckdb.connect()
    query = f"""
        SELECT
            CAST(tpep_pickup_datetime AS DATE) AS trip_date,
            COUNT(*) AS total_rides
        FROM read_parquet('{parquet_path}')
        WHERE CAST(tpep_pickup_datetime AS DATE) BETWEEN DATE '{month_str}-01'
              AND (DATE '{month_str}-01' + INTERVAL 1 MONTH - INTERVAL 1 DAY)
        GROUP BY 1 ORDER BY 1
    """
    df = con.execute(query).fetch_df()
    con.close()
    df["trip_date"] = pd.to_datetime(df["trip_date"])
    return df

def append_and_save(df_month):
    if os.path.exists(INPUT_PARQUET):
        df_existing = pd.read_parquet(INPUT_PARQUET)
        df_all = pd.concat([df_existing, df_month], ignore_index=True)
    else:
        df_all = df_month

    df_all = df_all.drop_duplicates(subset="trip_date", keep="last")
    df_all = df_all.sort_values("trip_date")
    df_all.to_parquet(INPUT_PARQUET, index=False)
    log(f"[DONE] Appended new data — total rows now: {len(df_all)}")

def prime_forecast_output_if_needed():
    if not os.path.exists(OUTPUT_PARQUET):
        df = pd.read_parquet(INPUT_PARQUET)
        df = df.rename(columns={"trip_date": "ds", "total_rides": "y"})
        df["ds"] = pd.to_datetime(df["ds"]).dt.normalize()
        df = df.dropna(subset=["ds", "y"]).sort_values("ds")
        df["yhat"] = df["yhat_lower"] = df["yhat_upper"] = df["y"]
        df["type"] = "actual"
        df.drop(columns="y", inplace=True)
        df.to_parquet(OUTPUT_PARQUET, index=False)
        log(f"[INIT] Created forecast_output.parquet with {len(df)} rows of actuals.")

def get_available_months():
    """
    Find all missing months between your local Parquet and today.
    """
    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError("Input Parquet does not exist — can't determine next month.")

    df = pd.read_parquet(INPUT_PARQUET)
    latest_date = pd.to_datetime(df["trip_date"]).max()
    next_month = (latest_date + pd.offsets.MonthBegin(1)).replace(day=1)

    today = datetime.today()
    this_month = datetime(today.year, today.month, 1)

    months = []
    current = next_month

    while current <= this_month:
        month_str = current.strftime("%Y-%m")
        if check_remote_parquet_exists(month_str):
            months.append(month_str)
        current += relativedelta(months=1)

    return months

def main():
    try:
        new_months = get_available_months()
        if not new_months:
            log(f"[SKIP] No new remote files available.")
            return

        log(f"[INFO] Found {len(new_months)} new month(s): {', '.join(new_months)}")

        for month_str in new_months:
            path = download_parquet(month_str)
            df_month = summarize_month_to_df(path, month_str)
            append_and_save(df_month)
            os.remove(path)
            log(f"[CLEANUP] Removed raw file: {path}")

        prime_forecast_output_if_needed()

        # Sanity check
        df = pd.read_parquet(INPUT_PARQUET)
        log(f"[CHECK] Local Parquet now covers: {df['trip_date'].min().date()} — {df['trip_date'].max().date()}")

    except Exception as e:
        log(f"[ERROR] Ingestion failed: {e}")

if __name__ == "__main__":
    main()
