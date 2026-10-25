import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# STRATEJİ VE RİSK SABİTLERİ (OPTİMİZE EDİLMEZ, SABİTTİR)
# ==============================================================================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DATA_DIR = "data"
RESULTS_DIR = "results"

INITIAL_CAPITAL = 1000.0
RISK_PER_TRADE = 0.01          # Her işlemde kasanın %1'i
MAX_PORTFOLIO_RISK = 0.03      # Aynı anda açık tüm işlemlerin toplam riski max %3 (Max 3 parite)
MAX_LEVERAGE = 3.0             # Kaldıraç tavanı 3x

TAKER_FEE = 0.0005             # %0.05 Taker komisyonu
SLIPPAGE = 0.0003              # %0.03 Kayma (Slippage)


# ==============================================================================
# 1. VERİ HAZIRLAMA VE İNDİKATÖR HESAPLAMA (LOOK-AHEAD KORUMALI)
# ==============================================================================
def prepare_symbol_data(symbol: str, base_interval: str = "1h", donchian_entry: int = 20, donchian_exit: int = 10) -> pd.DataFrame:
    """
    1H (veya 4H) veriyi okur, indikatörleri hesaplar, 4H EMA200 ve Funding verisini
    GELECEĞİ GÖRME (Look-ahead) hatası yapmadan birleştirir.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_{base_interval}.csv"))
    df4h = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_4h.csv"))
    df_fund = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_funding.csv"))

    df = df.sort_values("open_time").reset_index(drop=True)
    df4h = df4h.sort_values("open_time").reset_index(drop=True)

    # --- 4H EMA(200) Hesabı ve Look-Ahead Olmadan Birleştirme ---
    df4h["ema200_4h"] = df4h["close"].ewm(span=200, adjust=False).mean()
    df4h["trend_4h"] = np.where(df4h["close"] > df4h["ema200_4h"], 1, -1)

    # KRİTİK KURAL: 4H mumunun KAPANMA zamanı (close_time), 1H mumunun kapanma zamanından
    # küçük veya eşit olmalı. Böylece henüz kapanmamış 4H mumu ASLA kullanılmaz.
    df = pd.merge_asof(
        df.sort_values("close_time"),
        df4h[["close_time", "ema200_4h", "trend_4h"]].sort_values("close_time"),
        on="close_time",
        direction="backward"
    ).sort_values("open_time").reset_index(drop=True)

    # --- Funding Rate Birleştirme ---
    # Funding ödemesi mumun açılış anında (00:00, 08:00, 16:00 UTC) gerçekleşir
    df = pd.merge(
        df,
        df_fund[["funding_time", "funding_rate"]],
        left_on="open_time",
        right_on="funding_time",
        how="left"
    )
    df["funding_rate"] = df["funding_rate"].fillna(0.0)

    # --- ATR(14) ve ATR SMA(50) ---
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
    df["atr14"] = df["tr"].rolling(window=14).mean()
    df["atr_sma50"] = df["atr14"].rolling(window=50).mean()

    # --- Donchian Kanalları (Mevcut mum HARİÇ -> .shift(1) kullanıldı) ---
    # İndeks m'deki donchian_high_20, [m-20 ... m-1] mumlarının en yükseğidir.
    df["donchian_high_entry"] = df["high"].shift(1).rolling(window=donchian_entry).max()
    df["donchian_low_entry"] = df["low"].shift(1).rolling(window=donchian_entry).min()

    df["donchian_high_exit"] = df["high"].shift(1).rolling(window=donchian_exit).max()
    df["donchian_low_exit"] = df["low"].shift(1).rolling(window=donchian_exit).min()

    # --- Hacim Ortalaması (Mevcut kırılım mumu HARİÇ son 20 mumun ortalaması) ---
    df["vol_sma20"] = df["volume"].shift(1).rolling(window=20).mean()

    # --- Sıkışma Filtresi ---
    # Kırılım mumunda (m) DEĞİL, kırılımdan bir önceki kapanmış mumda (m-1) kontrol edilir!
    df["squeeze_prev_candle"] = df["atr14"].shift(1) < df["atr_sma50"].shift(1)

    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def split_in_sample_and_locked(df: pd.DataFrame, locked_months: int = 12):
    """Son 12 ayı 'Kilitli Test Verisi' (Out-of-Sample) olarak ayırır."""
    max_date = df["datetime"].max()
    cutoff_date = max_date - pd.DateOffset(months=locked_months)
    df_dev = df[df["datetime"] <= cutoff_date].copy().reset_index(drop=True)
    df_locked = df[df["datetime"] > cutoff_date].copy().reset_index(drop=True)
    return df_dev, df_locked


# ==============================================================================
# 2. TEKİL PARİTE BACKTEST MOTORU (V0, V1, V1-Rastgele, V2, V3)
# ==============================================================================
def run_single_symbol_backtest(df: pd.DataFrame, version: str = "V1", initial_capital: float = INITIAL_CAPITAL, seed: int = 42):
    """
    İndeks Açıklaması (Look-ahead olmaması için):
      - Döngü 'i' indeksinde çalışır. 'i' mevcut AÇILAN mumdur (İşlem open[i] fiyatından yapılır).
      - 's = i - 1' ise az önce KAPANMIŞ olan sinyal mumudur. Tüm sinyaller 's' mumunda kontrol edilir.
      - 's - 1 = i - 2' ise kırılımdan bir önceki mumdur (Sıkışma filtresi burada kontrol edilir).
    """
    df = df.dropna(subset=["atr_sma50", "ema200_4h", "donchian_high_entry"]).reset_index(drop=True)
    if len(df) < 10:
        return {}, pd.Series(dtype=float), pd.DataFrame()

    # V0: Sadece Al ve Tut (Buy & Hold Benchmark)
    if version == "V0":
        first_open = df["open"].iloc[0] * (1 + TAKER_FEE + SLIPPAGE)
        qty = initial_capital / first_open
        equity_curve = qty * df["close"]
        equity_curve.index = df["datetime"]
        metrics = calculate_metrics(equity_curve, [], initial_capital, 0.0, 0.0)
        return metrics, equity_curve, pd.DataFrame()

    rng = np.random.default_rng(seed)
    cash = initial_capital
    pos = 0              # 1: Long, -1: Short, 0: Yok
    qty = 0.0
    entry_price = 0.0
    stop_price = 0.0
    liq_price = 0.0
    entry_time = None

    total_fees = 0.0
    total_funding = 0.0
    trades = []
    equity_values = np.zeros(len(df))
    equity_values[0] = initial_capital

    # NumPy dizilerine çevir (Python döngüsünü 20 kat hızlandırır)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    vols = df["volume"].values
    vol_sma20 = df["vol_sma20"].values
    atr14 = df["atr14"].values
    d_high_in = df["donchian_high_entry"].values
    d_low_in = df["donchian_low_entry"].values
    d_high_out = df["donchian_high_exit"].values
    d_low_out = df["donchian_low_exit"].values
    trend_4h = df["trend_4h"].values
    squeeze_prev = df["squeeze_prev_candle"].values
    fund_rates = df["funding_rate"].values
    dts = df["datetime"].values

    for i in range(1, len(df)):
        s = i - 1  # KAPANMIŞ SİNYAL MUMU İNDEKSİ

        # ------------------------------------------------------------------
        # A) AÇIK POZİSYON VARSA: FUNDING, ÇIKIŞ SİNYALİ VE STOP KONTROLÜ
        # ------------------------------------------------------------------
        if pos != 0:
            # 1. Gerçek Funding Rate Uygulaması (Mum açılışında 00:00, 08:00, 16:00 UTC)
            if fund_rates[i] != 0.0:
                notional = qty * opens[i]
                # Long isek ve funding pozitifse öderiz (-), negatifse alırız (+)
                # Short isek ve funding pozitifse alırız (+), negatifse öderiz (-)
                funding_payment = -1.0 * pos * notional * fund_rates[i]
                cash += funding_payment
                total_funding += abs(funding_payment)

            # 2. Önceki kapanmış mumda (s) Donchian(10) çıkış sinyali oluştu mu?
            exit_signal = False
            if pos == 1 and closes[s] < d_low_out[s]:
                exit_signal = True
            elif pos == -1 and closes[s] > d_high_out[s]:
                exit_signal = True

            if exit_signal:
                # Yeni mumun (i) açılış fiyatından çık
                raw_exit = opens[i]
                exec_exit = raw_exit * (1 - SLIPPAGE) if pos == 1 else raw_exit * (1 + SLIPPAGE)
                fee = qty * exec_exit * TAKER_FEE
                pnl = pos * qty * (exec_exit - entry_price) - fee
                cash += pnl
                total_fees += fee
                trades.append({
                    "entry_time": entry_time, "exit_time": dts[i], "dir": "LONG" if pos == 1 else "SHORT",
                    "entry": entry_price, "exit": exec_exit, "pnl": pnl, "reason": "Donchian10_Exit"
                })
                pos = 0
                qty = 0.0

            # 3. Pozisyon hala açıksa: Mum içinde Stop-Loss (2xATR) veya Likidasyon kontrolü
            if pos != 0:
                stopped = False
                if pos == 1 and lows[i] <= stop_price:
                    # Eğer mum stop fiyatının da altında boşluklu (gap) açıldıysa open[i] al
                    raw_exit = min(opens[i], stop_price)
                    exec_exit = raw_exit * (1 - SLIPPAGE)
                    stopped = True
                elif pos == -1 and highs[i] >= stop_price:
                    raw_exit = max(opens[i], stop_price)
                    exec_exit = raw_exit * (1 + SLIPPAGE)
                    stopped = True

                if stopped:
                    fee = qty * exec_exit * TAKER_FEE
                    pnl = pos * qty * (exec_exit - entry_price) - fee
                    cash += pnl
                    total_fees += fee
                    trades.append({
                        "entry_time": entry_time, "exit_time": dts[i], "dir": "LONG" if pos == 1 else "SHORT",
                        "entry": entry_price, "exit": exec_exit, "pnl": pnl, "reason": "Stop_Loss"
                    })
                    pos = 0
                    qty = 0.0

        # ------------------------------------------------------------------
        # B) POZİSYON YOKSA: KAPANMIŞ 's' MUMUNDA GİRİŞ SİNYALİ KONTROLÜ
        # ------------------------------------------------------------------
        if pos == 0 and cash > 10.0:
            signal = 0

            if version == "V1_RANDOM":
                # V1 ile ortalama aynı sıklıkta (~%2.5 olasılıkla) rastgele Long/Short girer
                r = rng.random()
                if r < 0.0125:
                    signal = 1
                elif r < 0.025:
                    signal = -1
            else:
                # Ham Donchian(20) Kırılımı (Kapanmış s mumunda)
                long_break = closes[s] > d_high_in[s]
                short_break = closes[s] < d_low_in[s]

                if version == "V1":
                    if long_break:
                        signal = 1
                    elif short_break:
                        signal = -1

                elif version == "V2":
                    # V1 + 4H EMA(200) Trend Filtresi
                    if long_break and trend_4h[s] == 1:
                        signal = 1
                    elif short_break and trend_4h[s] == -1:
                        signal = -1

                elif version == "V3":
                    # V2 + Sıkışma (s-1 mumunda) + Hacim (s mumunda > 1.5x)
                    vol_ok = vols[s] >= 1.5 * vol_sma20[s]
                    sqz_ok = bool(squeeze_prev[s])
                    if long_break and trend_4h[s] == 1 and sqz_ok and vol_ok:
                        signal = 1
                    elif short_break and trend_4h[s] == -1 and sqz_ok and vol_ok:
                        signal = -1

            # Sinyal varsa 'i' mumunun açılışından (opens[i]) pozisyona gir
            if signal != 0 and atr14[s] > 0:
                raw_entry = opens[i]
                exec_entry = raw_entry * (1 + SLIPPAGE) if signal == 1 else raw_entry * (1 - SLIPPAGE)
                stop_dist = 2.0 * atr14[s]

                # Risk bazlı pozisyon büyüklüğü: (Kasa x 0.01) / |Giriş - Stop|
                risk_usdt = cash * RISK_PER_TRADE
                desired_qty = risk_usdt / stop_dist

                # Maksimum 3x Kaldıraç Sınırı
                max_qty = (cash * MAX_LEVERAGE) / exec_entry
                final_qty = min(desired_qty, max_qty)

                if final_qty * exec_entry > 5.0:
                    pos = signal
                    qty = final_qty
                    entry_price = exec_entry
                    stop_price = entry_price - stop_dist if pos == 1 else entry_price + stop_dist
                    # Isolated Margin: Stop mesafesi 2xATR, kaldıraç max 3x olduğu için
                    # likidasyon fiyatı (~%33 ters hareket) stop noktasının çok ötesindedir.
                    liq_price = entry_price * (1 - 1 / MAX_LEVERAGE) if pos == 1 else entry_price * (1 + 1 / MAX_LEVERAGE)
                    entry_time = dts[i]

                    fee = qty * entry_price * TAKER_FEE
                    cash -= fee
                    total_fees += fee

        # Mum sonu anlık kasa (Mark-to-Market Equity)
        unrealized = pos * qty * (closes[i] - entry_price) if pos != 0 else 0.0
        equity_values[i] = max(0.0, cash + unrealized)

    equity_series = pd.Series(equity_values, index=df["datetime"])
    trades_df = pd.DataFrame(trades)
    metrics = calculate_metrics(equity_series, trades, initial_capital, total_fees, total_funding)
    return metrics, equity_series, trades_df


# ==============================================================================
# 3. BİRLEŞİK PORTFÖY MOTORU (ORTAK KASA VE MAX %3 PORTFÖY RİSKİ TAVANI)
# ==============================================================================
def run_portfolio_backtest(data_dict: dict, version: str = "V1", initial_capital: float = INITIAL_CAPITAL, seed: int = 42):
    """
    5 pariteyi ortak 1000 USDT kasa ile aynı zaman çizelgesinde (saat saat) koşturur.
    Aynı anda açık tüm pozisyonların toplam riski kasanın %3'ünü (3 pozisyonu) geçemez.
    """
    if version == "V0":
        # Eşit ağırlıklı (her coine 200 USDT) Buy & Hold Portföyü
        combined_eq = None
        per_coin_cap = initial_capital / len(data_dict)
        for sym, df in data_dict.items():
            _, eq, _ = run_single_symbol_backtest(df, version="V0", initial_capital=per_coin_cap)
            combined_eq = eq if combined_eq is None else combined_eq.add(eq, fill_value=per_coin_cap)
        combined_eq = combined_eq.dropna()
        metrics = calculate_metrics(combined_eq, [], initial_capital, 0.0, 0.0)
        return metrics, combined_eq, pd.DataFrame()

    # Tüm pariteleri ortak saatlik zaman indeksine hizala
    all_times = sorted(list(set().union(*(df["open_time"].values for df in data_dict.values()))))
    indexed_dfs = {sym: df.set_index("open_time") for sym, df in data_dict.items()}

    rng = np.random.default_rng(seed)
    cash = initial_capital
    positions = {}  # sym -> {pos, qty, entry_price, stop_price, entry_time}
    trades = []
    total_fees = 0.0
    total_funding = 0.0
    equity_list = []
    time_list = []

    for idx in range(1, len(all_times)):
        t_curr = all_times[idx]
        t_prev = all_times[idx - 1]

        # 1. Açık Pozisyonları Güncelle
        for sym in list(positions.keys()):
            df_sym = indexed_dfs[sym]
            if t_curr not in df_sym.index or t_prev not in df_sym.index:
                continue

            row_curr = df_sym.loc[t_curr]
            row_prev = df_sym.loc[t_prev]
            p_info = positions[sym]
            pos = p_info["pos"]
            qty = p_info["qty"]

            # Funding
            f_rate = row_curr["funding_rate"]
            if f_rate != 0.0:
                f_pay = -1.0 * pos * (qty * row_curr["open"]) * f_rate
                cash += f_pay
                total_funding += abs(f_pay)

            # Çıkış Sinyali (Önceki kapanmış mumda)
            exit_sig = (pos == 1 and row_prev["close"] < row_prev["donchian_low_exit"]) or \
                       (pos == -1 and row_prev["close"] > row_prev["donchian_high_exit"])

            if exit_sig:
                exec_exit = row_curr["open"] * (1 - SLIPPAGE) if pos == 1 else row_curr["open"] * (1 + SLIPPAGE)
                fee = qty * exec_exit * TAKER_FEE
                pnl = pos * qty * (exec_exit - p_info["entry_price"]) - fee
                cash += pnl
                total_fees += fee
                trades.append({"symbol": sym, "entry_time": p_info["entry_time"], "exit_time": row_curr["datetime"],
                               "dir": "LONG" if pos == 1 else "SHORT", "pnl": pnl, "reason": "Donchian10_Exit"})
                del positions[sym]
                continue

            # Stop-Loss Kontrolü
            stopped = False
            if pos == 1 and row_curr["low"] <= p_info["stop_price"]:
                raw_exit = min(row_curr["open"], p_info["stop_price"])
                exec_exit = raw_exit * (1 - SLIPPAGE)
                stopped = True
            elif pos == -1 and row_curr["high"] >= p_info["stop_price"]:
                raw_exit = max(row_curr["open"], p_info["stop_price"])
                exec_exit = raw_exit * (1 + SLIPPAGE)
                stopped = True

            if stopped:
                fee = qty * exec_exit * TAKER_FEE
                pnl = pos * qty * (exec_exit - p_info["entry_price"]) - fee
                cash += pnl
                total_fees += fee
                trades.append({"symbol": sym, "entry_time": p_info["entry_time"], "exit_time": row_curr["datetime"],
                               "dir": "LONG" if pos == 1 else "SHORT", "pnl": pnl, "reason": "Stop_Loss"})
                del positions[sym]

        # 2. Yeni Giriş Sinyalleri (Toplam Açık Risk < %3 ise izin ver)
        max_open_positions = int(round(MAX_PORTFOLIO_RISK / RISK_PER_TRADE))  # 3 pozisyon
        for sym in SYMBOLS:
            if sym in positions or len(positions) >= max_open_positions:
                continue
            df_sym = indexed_dfs[sym]
            if t_curr not in df_sym.index or t_prev not in df_sym.index:
                continue

            row_curr = df_sym.loc[t_curr]
            row_prev = df_sym.loc[t_prev]
            if pd.isna(row_prev["atr_sma50"]) or pd.isna(row_prev["ema200_4h"]) or pd.isna(row_prev["donchian_high_entry"]):
                continue

            signal = 0
            if version == "V1_RANDOM":
                r = rng.random()
                signal = 1 if r < 0.0125 else (-1 if r < 0.025 else 0)
            else:
                l_brk = row_prev["close"] > row_prev["donchian_high_entry"]
                s_brk = row_prev["close"] < row_prev["donchian_low_entry"]
                if version == "V1":
                    signal = 1 if l_brk else (-1 if s_brk else 0)
                elif version == "V2":
                    if l_brk and row_prev["trend_4h"] == 1: signal = 1
                    elif s_brk and row_prev["trend_4h"] == -1: signal = -1
                elif version == "V3":
                    vol_ok = row_prev["volume"] >= 1.5 * row_prev["vol_sma20"]
                    sqz_ok = bool(row_prev["squeeze_prev_candle"])
                    if l_brk and row_prev["trend_4h"] == 1 and sqz_ok and vol_ok: signal = 1
                    elif s_brk and row_prev["trend_4h"] == -1 and sqz_ok and vol_ok: signal = -1

            if signal != 0 and row_prev["atr14"] > 0:
                exec_entry = row_curr["open"] * (1 + SLIPPAGE) if signal == 1 else row_curr["open"] * (1 - SLIPPAGE)
                stop_dist = 2.0 * row_prev["atr14"]
                desired_qty = (cash * RISK_PER_TRADE) / stop_dist
                # Portföyde her coine kasanın max 1x-3x'i kadar marjin ayrılabilir (Toplam 3x tavanı)
                max_qty = (cash * (MAX_LEVERAGE / max_open_positions)) / exec_entry
                final_qty = min(desired_qty, max_qty)

                if final_qty * exec_entry > 5.0:
                    stop_p = exec_entry - stop_dist if signal == 1 else exec_entry + stop_dist
                    fee = final_qty * exec_entry * TAKER_FEE
                    cash -= fee
                    total_fees += fee
                    positions[sym] = {
                        "pos": signal, "qty": final_qty, "entry_price": exec_entry,
                        "stop_price": stop_p, "entry_time": row_curr["datetime"]
                    }

        # Anlık Portföy Değeri (Equity)
        unrealized = 0.0
        for sym, p_info in positions.items():
            if t_curr in indexed_dfs[sym].index:
                c_price = indexed_dfs[sym].loc[t_curr]["close"]
                unrealized += p_info["pos"] * p_info["qty"] * (c_price - p_info["entry_price"])

        equity_list.append(max(0.0, cash + unrealized))
        time_list.append(pd.to_datetime(t_curr, unit="ms", utc=True))

    eq_series = pd.Series(equity_list, index=time_list)
    metrics = calculate_metrics(eq_series, trades, initial_capital, total_fees, total_funding)
    return metrics, eq_series, pd.DataFrame(trades)


# ==============================================================================
# 4. PERFORMANS METRİKLERİ HESAPLAMA
# ==============================================================================
def calculate_metrics(equity: pd.Series, trades: list, initial_cap: float, fees: float, funding: float) -> dict:
    if equity.empty:
        return {}
    final_eq = float(equity.iloc[-1])
    total_ret = (final_eq / initial_cap - 1.0) * 100.0

    days = max(1.0, (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0)
    years = days / 365.25
    cagr = ((final_eq / initial_cap) ** (1.0 / years) - 1.0) * 100.0 if final_eq > 0 else -100.0

    # Max Drawdown
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    max_dd = float(dd.min()) * 100.0

    # Günlük getiri üzerinden Yıllıklandırılmış Sharpe Oranı
    daily_eq = equity.resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if len(daily_ret) > 2 and daily_ret.std() > 0 else 0.0

    num_trades = len(trades)
    if num_trades > 0:
        pnls = np.array([t["pnl"] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        win_rate = (len(wins) / num_trades) * 100.0
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    else:
        win_rate = profit_factor = win_loss_ratio = 0.0

    return {
        "Son_Kasa($)": round(final_eq, 1),
        "Toplam_Getiri(%)": round(total_ret, 2),
        "Yillik_CAGR(%)": round(cagr, 2),
        "Max_DD(%)": round(max_dd, 2),
        "Sharpe": round(float(sharpe), 2),
        "Profit_Factor": round(float(profit_factor), 2),
        "Islem_Sayisi": num_trades,
        "Kazanma_Orani(%)": round(win_rate, 1),
        "Ort_Kazanc/Kayip": round(float(win_loss_ratio), 2),
        "Komisyon($)": round(fees, 1),
        "Funding($)": round(funding, 1)
    }


# ==============================================================================
# 5. ADIM 4: WALK-FORWARD (6 AYLIK DİLİMLER) VE PARAMETRE HASSASİYET TESTİ
# ==============================================================================
def run_walk_forward_and_sensitivity(data_dict: dict, version: str = "V2"):
    print(f"\n" + "=" * 85)
    print(f"ADIM 4A: WALK-FORWARD DOĞRULAMA (6 AYLIK DİLİMLER - {version} BİRLEŞİK PORTFÖY)")
    print("=" * 85)

    btc_df = data_dict["BTCUSDT"]
    start_dt = btc_df["datetime"].min()
    end_dt = btc_df["datetime"].max()

    wf_rows = []
    curr = start_dt
    while curr < end_dt:
        nxt = curr + pd.DateOffset(months=6)
        sliced_dict = {}
        for sym, df in data_dict.items():
            sub = df[(df["datetime"] >= curr) & (df["datetime"] < nxt)].copy().reset_index(drop=True)
            if len(sub) > 60:
                sliced_dict[sym] = sub

        if sliced_dict:
            m, _, _ = run_portfolio_backtest(sliced_dict, version=version)
            if m:
                wf_rows.append({
                    "Donem": f"{str(curr)[:10]} -> {str(nxt)[:10]}",
                    "Getiri(%)": m["Toplam_Getiri(%)"],
                    "Max_DD(%)": m["Max_DD(%)"],
                    "Sharpe": m["Sharpe"],
                    "Profit_Factor": m["Profit_Factor"],
                    "Islem_Sayisi": m["Islem_Sayisi"]
                })
        curr = nxt

    print(pd.DataFrame(wf_rows).to_string(index=False))

    print(f"\n" + "=" * 85)
    print(f"ADIM 4B: PARAMETRE HASSASİYET TESTİ (KOMŞU DONCHIAN DEĞERLERİ - {version})")
    print("=" * 85)
    sens_rows = []
    for d_in, d_out in [(15, 8), (20, 10), (25, 12)]:
        temp_dict = {}
        for sym in SYMBOLS:
            df_full = prepare_symbol_data(sym, base_interval="1h", donchian_entry=d_in, donchian_exit=d_out)
            df_dev, _ = split_in_sample_and_locked(df_full, locked_months=12)
            temp_dict[sym] = df_dev
        m, _, _ = run_portfolio_backtest(temp_dict, version=version)
        m["Parametre"] = f"Giriş:{d_in} / Çıkış:{d_out}"
        sens_rows.append(m)

    cols = ["Parametre", "Toplam_Getiri(%)", "Yillik_CAGR(%)", "Max_DD(%)", "Sharpe", "Profit_Factor", "Islem_Sayisi"]
    print(pd.DataFrame(sens_rows)[cols].to_string(index=False))


# ==============================================================================
# 6. TÜM LABORATUVARI ÇALIŞTIRMA VE GRAFİK ÇİZME
# ==============================================================================
def run_full_comparison(run_locked_test: bool = False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    versions = ["V0", "V1", "V1_RANDOM", "V2", "V3"]

    dev_dict_1h = {}
    locked_dict_1h = {}
    dev_dict_4h = {}

    print("\nVeriler yükleniyor ve indikatörler (Look-ahead korumalı) hesaplanıyor...")
    for sym in SYMBOLS:
        df_1h = prepare_symbol_data(sym, base_interval="1h")
        d_dev, d_lock = split_in_sample_and_locked(df_1h, locked_months=12)
        dev_dict_1h[sym] = d_dev
        locked_dict_1h[sym] = d_lock

        df_4h = prepare_symbol_data(sym, base_interval="4h")
        d_dev4, _ = split_in_sample_and_locked(df_4h, locked_months=12)
        dev_dict_4h[sym] = d_dev4

    # 1. Parite Bazlı Ayrı Sonuçlar (1H)
    for sym in SYMBOLS:
        print(f"\n--- {sym} (1H - GELİŞTİRME VERİSİ, SON 12 AY HARİÇ) ---")
        sym_rows = []
        for v in versions:
            m, _, _ = run_single_symbol_backtest(dev_dict_1h[sym], version=v)
            m["Versiyon"] = v
            sym_rows.append(m)
        df_res = pd.DataFrame(sym_rows)
        cols = ["Versiyon", "Son_Kasa($)", "Toplam_Getiri(%)", "Yillik_CAGR(%)", "Max_DD(%)",
                "Sharpe", "Profit_Factor", "Islem_Sayisi", "Kazanma_Orani(%)", "Ort_Kazanc/Kayip", "Komisyon($)", "Funding($)"]
        print(df_res[cols].to_string(index=False))

    # 2. Birleşik Portföy Karşılaştırması (1H vs 4H)
    print("\n" + "=" * 100)
    print("BİRLEŞİK PORTFÖY SONUÇLARI (ORTAK 1000$ KASA, MAX %3 PORTFÖY RİSKİ - 1H ANA TETİKLEYİCİ)")
    print("=" * 100)
    port_rows = []
    plt.figure(figsize=(12, 6))

    for v in versions:
        m, eq, _ = run_portfolio_backtest(dev_dict_1h, version=v)
        m["Versiyon"] = f"{v} (1H)"
        port_rows.append(m)
        if not eq.empty:
            plt.plot(eq.index, eq.values, label=f"{v} (1H)")

    # Ayrıca 4H Tetikleyici ile V1 ve V2 Karşılaştırması
    for v in ["V1", "V2"]:
        m4, _, _ = run_portfolio_backtest(dev_dict_4h, version=v)
        m4["Versiyon"] = f"{v} (4H)"
        port_rows.append(m4)

    df_port = pd.DataFrame(port_rows)
    print(df_port[cols].to_string(index=False))
    df_port.to_csv(os.path.join(RESULTS_DIR, "portfolio_comparison.csv"), index=False)

    plt.title("Birleşik Portföy Equity Eğrisi (Geliştirme Dönemi)")
    plt.xlabel("Tarih")
    plt.ylabel("Kasa (USDT)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    chart_path = os.path.join(RESULTS_DIR, "equity_curves.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[GRAFİK KAYDEDİLDİ] -> {chart_path} (VS Code sol menüden tıklayıp bakabilirsin)")

    # 3. Kilitli Son 12 Ay Testi (Eğer İstenirse)
    if run_locked_test:
        print("\n" + "=" * 100)
        print("ADIM 5: KİLİTLİ SON 12 AYLIK VERİDE (OUT-OF-SAMPLE) TEK SEFERLİK FİNAL TESTİ")
        print("=" * 100)
        lock_rows = []
        for v in versions:
            ml, _, _ = run_portfolio_backtest(locked_dict_1h, version=v)
            ml["Versiyon"] = v
            lock_rows.append(ml)
        print(pd.DataFrame(lock_rows)[cols].to_string(index=False))