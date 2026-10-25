import pandas as pd
from step1_data_downloader import download_all_data
from step2_backtest_engine import (
    SYMBOLS, prepare_symbol_data, split_in_sample_and_locked,
    run_full_comparison, run_portfolio_backtest
)
from step3_paper_trader import evaluate_live_signals, run_hourly_paper_loop


def run_step4_4h_analysis():
    """
    4H verisi üzerinde:
    1) V0, V1, V1_RANDOM, V2 ve V3 (4H) karşılaştırmasını,
    2) 6 Aylık Walk-Forward doğrulamasını (V2 - 4H),
    3) Donchian 15/20/25 Parametre Hassasiyet testini (V2 ve V3 - 4H) çalıştırır.
    """
    print("\n4H verileri yükleniyor ve indikatörler hesaplanıyor...")
    dev_dict_4h = {}
    for sym in SYMBOLS:
        df_4h = prepare_symbol_data(sym, base_interval="4h")
        d_dev, _ = split_in_sample_and_locked(df_4h, locked_months=12)
        dev_dict_4h[sym] = d_dev

    cols = [
        "Versiyon", "Son_Kasa($)", "Toplam_Getiri(%)", "Yillik_CAGR(%)",
        "Max_DD(%)", "Sharpe", "Profit_Factor", "Islem_Sayisi",
        "Kazanma_Orani(%)", "Ort_Kazanc/Kayip", "Komisyon($)", "Funding($)"
    ]

    # --- 1. 4H V0, V1, V1_RANDOM, V2, V3 TAM KARŞILAŞTIRMA ---
    print("\n" + "=" * 100)
    print("4 SAATLİK (4H) BİRLEŞİK PORTFÖY KARŞILAŞTIRMASI (V0, V1, V1_RANDOM, V2, V3)")
    print("=" * 100)
    rows_4h = []
    for v in ["V0", "V1", "V1_RANDOM", "V2", "V3"]:
        m, _, _ = run_portfolio_backtest(dev_dict_4h, version=v)
        m["Versiyon"] = f"{v} (4H)"
        rows_4h.append(m)
    print(pd.DataFrame(rows_4h)[cols].to_string(index=False))

    # --- 2. 6 AYLIK WALK-FORWARD DOĞRULAMA (V2 - 4H) ---
    print("\n" + "=" * 100)
    print("ADIM 4A: WALK-FORWARD DOĞRULAMA (6 AYLIK DİLİMLER - V2 4H PORTFÖY)")
    print("=" * 100)
    btc_df = dev_dict_4h["BTCUSDT"]
    start_dt = btc_df["datetime"].min()
    end_dt = btc_df["datetime"].max()

    wf_rows = []
    curr = start_dt
    while curr < end_dt:
        nxt = curr + pd.DateOffset(months=6)
        sliced_dict = {}
        active_coins = 0
        for sym, df in dev_dict_4h.items():
            sub = df[(df["datetime"] >= curr) & (df["datetime"] < nxt)].copy().reset_index(drop=True)
            # KRİTİK DÜZELTME: O 6 aylık dönemde henüz piyasaya çıkmamış coinler (örn. 2019'da SOL)
            # için boş DataFrame gönderiyoruz ki KeyError vermesin, motor o coini otomatik pas geçsin.
            sliced_dict[sym] = sub
            if len(sub) > 30:
                active_coins += 1

        if active_coins > 0:
            m, _, _ = run_portfolio_backtest(sliced_dict, version="V2")
            if m and m.get("Islem_Sayisi", 0) > 0:
                wf_rows.append({
                    "Donem": f"{str(curr)[:10]} -> {str(nxt)[:10]}",
                    "Aktif_Coin": active_coins,
                    "Getiri(%)": m["Toplam_Getiri(%)"],
                    "Max_DD(%)": m["Max_DD(%)"],
                    "Sharpe": m["Sharpe"],
                    "Profit_Factor": m["Profit_Factor"],
                    "Islem_Sayisi": m["Islem_Sayisi"],
                    "Kazanma(%)": m["Kazanma_Orani(%)"]
                })
        curr = nxt
    print(pd.DataFrame(wf_rows).to_string(index=False))

    # --- 3. PARAMETRE HASSASİYET TESTİ (15/8, 20/10, 25/12 - V2 ve V3 4H) ---
    print("\n" + "=" * 100)
    print("ADIM 4B: PARAMETRE HASSASİYET TESTİ (KOMŞU DONCHIAN DEĞERLERİ - 4H PORTFÖY)")
    print("=" * 100)
    sens_rows = []
    for ver in ["V2", "V3"]:
        for d_in, d_out in [(15, 8), (20, 10), (25, 12)]:
            temp_dict = {}
            for sym in SYMBOLS:
                df_full = prepare_symbol_data(sym, base_interval="4h", donchian_entry=d_in, donchian_exit=d_out)
                df_dev, _ = split_in_sample_and_locked(df_full, locked_months=12)
                temp_dict[sym] = df_dev
            m, _, _ = run_portfolio_backtest(temp_dict, version=ver)
            m["Versiyon_Parametre"] = f"{ver} (4H) | Giriş:{d_in} / Çıkış:{d_out}"
            sens_rows.append(m)

    s_cols = [
        "Versiyon_Parametre", "Son_Kasa($)", "Toplam_Getiri(%)",
        "Yillik_CAGR(%)", "Max_DD(%)", "Sharpe", "Profit_Factor", "Islem_Sayisi"
    ]
    print(pd.DataFrame(sens_rows)[s_cols].to_string(index=False))


def run_step5_locked_4h():
    """Son 12 aylık kilitli veride (Out-of-Sample) 4H stratejilerini tek seferlik test eder."""
    print("\n" + "=" * 100)
    print("ADIM 5: KİLİTLİ SON 12 AYLIK VERİDE (OUT-OF-SAMPLE) FİNAL TESTİ (4H PORTFÖY)")
    print("=" * 100)
    locked_dict_4h = {}
    for sym in SYMBOLS:
        df_4h = prepare_symbol_data(sym, base_interval="4h")
        _, d_lock = split_in_sample_and_locked(df_4h, locked_months=12)
        locked_dict_4h[sym] = d_lock

    cols = [
        "Versiyon", "Son_Kasa($)", "Toplam_Getiri(%)", "Yillik_CAGR(%)",
        "Max_DD(%)", "Sharpe", "Profit_Factor", "Islem_Sayisi",
        "Kazanma_Orani(%)", "Ort_Kazanc/Kayip", "Komisyon($)", "Funding($)"
    ]
    lock_rows = []
    for v in ["V0", "V1", "V1_RANDOM", "V2", "V3"]:
        ml, _, _ = run_portfolio_backtest(locked_dict_4h, version=v)
        ml["Versiyon"] = f"{v} (4H - KİLİTLİ 12 AY)"
        lock_rows.append(ml)
    print(pd.DataFrame(lock_rows)[cols].to_string(index=False))


def main():
    print("\n" + "=" * 65)
    print("   KRİPTO QUANT ARAŞTIRMA VE BACKTEST LABORATUVARI")
    print("=" * 65)
    print("1 - [ADIM 1] Verileri İndir / Kontrol Et (OHLCV 1H, 4H + Gerçek Funding)")
    print("2 - [ADIM 2 & 3] 1H ve 4H Genel Backtest + Equity Grafiği")
    print("3 - [ADIM 4] 4H Tam Analiz + Walk-Forward (6 Ay) + Parametre Hassasiyeti")
    print("4 - [ADIM 5] Kilitli Son 12 Aylık Veride (Out-of-Sample) Final Testi")
    print("5 - [ADIM 6] Paper Trading (Canlı Veriyle Tek Seferlik Sinyal Testi & CSV Log)")
    print("6 - [ADIM 6] Paper Trading (7/24 Saatlik Cron Döngüsü Başlat)")
    print("=" * 65)

    secim = input("Çalıştırmak istediğin adımın numarasını gir (1-6): ").strip()

    if secim == "1":
        download_all_data(force_update=False)

    elif secim == "2":
        run_full_comparison(run_locked_test=False)

    elif secim == "3":
        run_step4_4h_analysis()

    elif secim == "4":
        onay = input("DİKKAT: Kilitli son 12 ay verisi sadece 1 kez test edilmelidir. Onaylıyor musun? (e/h): ")
        if onay.lower() == "e":
            run_step5_locked_4h()

    elif secim == "5":
        evaluate_live_signals(version="V2")

    elif secim == "6":
        run_hourly_paper_loop(version="V2")

    else:
        print("Geçersiz seçim.")


if __name__ == "__main__":
    main()