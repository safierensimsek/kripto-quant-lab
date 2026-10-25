import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# ==============================================================================
# SABİT AYARLAR
# ==============================================================================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVALS = ["1h", "4h"]
DATA_DIR = "data"

BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
FUNDING_ENDPOINT = "/fapi/v1/fundingRate"

# 1 Ocak 2019 00:00:00 UTC
START_TIME_MS = int(datetime(2019, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines(symbol: str, interval: str, start_ms: int) -> pd.DataFrame:
    """Sadece KAPANMIŞ mumları (OHLCV) Binance Futures'tan indirir."""
    print(f"[{symbol} - {interval}] Mum verileri kontrol ediliyor/indiriliyor...")
    all_rows = []
    current_start = start_ms
    now_ms = int(time.time() * 1000)

    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": current_start, "limit": 1500}
        try:
            resp = requests.get(BASE_URL + KLINES_ENDPOINT, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Bağlantı hatası: {e}. 5 sn bekleniyor...")
            time.sleep(5)
            continue

        if not data:
            break

        for row in data:
            open_time = int(row[0])
            close_time = int(row[6])
            # KAPANMAMIŞ CANLI MUMU ALMA (Look-ahead engelleme)
            if close_time > now_ms:
                continue
            all_rows.append({
                "open_time": open_time,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": close_time
            })

        last_dt = datetime.fromtimestamp(data[-1][0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
        print(f"  -> {symbol} ({interval}): {last_dt} tarihine kadar çekildi ({len(all_rows)} mum)")

        if len(data) < 1500:
            break
        current_start = int(data[-1][0]) + 1
        time.sleep(0.15)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def fetch_funding_rates(symbol: str, start_ms: int) -> pd.DataFrame:
    """Binance'ten gerçek geçmiş Funding Rate verilerini indirir."""
    print(f"[{symbol}] Gerçek Funding Rate verileri indiriliyor...")
    all_rows = []
    current_start = start_ms

    while True:
        params = {"symbol": symbol, "startTime": current_start, "limit": 1000}
        try:
            resp = requests.get(BASE_URL + FUNDING_ENDPOINT, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Bağlantı hatası: {e}. 5 sn bekleniyor...")
            time.sleep(5)
            continue

        if not data:
            break

        for row in data:
            raw_time = int(row["fundingTime"])
            clean_time = round(raw_time / 60000) * 60000
            all_rows.append({
                "funding_time": clean_time,
                "funding_rate": float(row["fundingRate"])
            })

        if len(data) < 1000:
            break
        current_start = int(data[-1]["fundingTime"]) + 1
        time.sleep(0.15)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["funding_time"]).sort_values("funding_time").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
    return df


def download_all_data(force_update=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    summary = []

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            path = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")
            if os.path.exists(path) and not force_update:
                df_k = pd.read_csv(path)
                print(f"[HAZIR] {path} ({len(df_k)} satır)")
            else:
                df_k = fetch_klines(symbol, interval, START_TIME_MS)
                df_k.to_csv(path, index=False)
            summary.append({
                "Parite": symbol, "Tip": f"OHLCV {interval}", "Satır": len(df_k),
                "Başlangıç": str(df_k["datetime"].iloc[0])[:16] if not df_k.empty else "-",
                "Bitiş": str(df_k["datetime"].iloc[-1])[:16] if not df_k.empty else "-"
            })

        f_path = os.path.join(DATA_DIR, f"{symbol}_funding.csv")
        if os.path.exists(f_path) and not force_update:
            df_f = pd.read_csv(f_path)
            print(f"[HAZIR] {f_path} ({len(df_f)} satır)")
        else:
            df_f = fetch_funding_rates(symbol, START_TIME_MS)
            df_f.to_csv(f_path, index=False)
        summary.append({
            "Parite": symbol, "Tip": "Funding", "Satır": len(df_f),
            "Başlangıç": str(df_f["datetime"].iloc[0])[:16] if not df_f.empty else "-",
            "Bitiş": str(df_f["datetime"].iloc[-1])[:16] if not df_f.empty else "-"
        })

    print("\n" + "=" * 75)
    print(pd.DataFrame(summary).to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    download_all_data()