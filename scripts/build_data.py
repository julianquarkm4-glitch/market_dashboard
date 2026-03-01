#!/usr/bin/env python3
"""
EVC Terminal Data Pipeline
Adapted from traderwillhu/market_dashboard

Runs daily at 16:30 ET via GitHub Actions

Outputs:
  data/screener.json  — Jeff Sun momentum scans + quant metrics
  data/sectors.json   — Sector/industry/country ETF tracking
  data/breadth.json   — Market breadth indicators
  data/calendar.json  — Upcoming economic events
  data/meta.json      — Build timestamp and metadata
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

SCAN_UNIVERSE = [
    "NVDA", "PLTR", "COIN", "HOOD", "MSTR", "APP", "AFRM", "SMCI",
    "RKLB", "IONQ", "AXON", "CRWD", "NET", "DDOG", "SNOW", "MDB",
    "SHOP", "SQ", "MELI", "SE", "GRAB", "NU", "SOFI", "UPST",
    "RDDT", "DUOL", "HIMS", "CELH", "DECK", "ON", "ANET", "ARM",
    "AVGO", "TSM", "ASML", "KLAC", "LRCX", "AMAT", "MRVL", "QCOM",
    "AMD", "INTC", "MU", "DELL", "HPE", "PANW", "FTNT", "ZS",
    "OKTA", "MNDY", "TEAM", "NOW", "ADBE", "CRM", "ORCL", "SAP",
    "UBER", "LYFT", "DASH", "ABNB", "BKNG", "EXPE", "MAR",
    "LLY", "NVO", "REGN", "VRTX", "ISRG", "DXCM", "PODD",
    "GEV", "VST", "CEG", "OKLO", "SMR", "NNE", "LEU",
    "GOOG", "META", "AMZN", "MSFT", "AAPL", "NFLX", "TSLA",
]

SECTOR_ETFS = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Healthcare", "XLI": "Industrials", "XLC": "Comm Services",
    "XLY": "Cons Discretionary", "XLP": "Cons Staples", "XLB": "Materials",
    "XLRE": "Real Estate", "XLU": "Utilities",
    "SMH": "Semiconductors", "XBI": "Biotech", "ARKK": "Innovation",
    "TAN": "Solar", "KWEB": "China Tech", "GDX": "Gold Miners",
    "XOP": "Oil & Gas E&P", "ITB": "Homebuilders", "JETS": "Airlines",
}

COUNTRY_ETFS = {
    "EWJ": "Japan", "EWG": "Germany", "EWU": "UK", "FXI": "China",
    "EWZ": "Brazil", "INDA": "India", "EWY": "South Korea",
    "EWT": "Taiwan", "EWA": "Australia", "EWC": "Canada",
}

INDUSTRY_ETFS = {
    "SOXX": "Semiconductors", "IGV": "Software", "HACK": "Cybersecurity",
    "ROBO": "Robotics & AI", "CIBR": "Cybersecurity", "CLOU": "Cloud Computing",
    "FINX": "Fintech", "ARKG": "Genomics", "LIT": "Lithium & Battery",
    "URA": "Uranium", "COPX": "Copper Miners",
}


# ══════════════════════════════════════════════════════════
# CORE CALCULATIONS
# ══════════════════════════════════════════════════════════

def calc_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calc_adr_pct(df, period=20):
    daily_range_pct = (df["High"] - df["Low"]) / df["Close"] * 100
    return daily_range_pct.rolling(window=period).mean()

def calc_atr_extension_from_sma(df, sma_period=50, atr_period=14):
    sma = df["Close"].rolling(window=sma_period).mean()
    atr = calc_atr(df, atr_period)
    return (df["Close"] - sma) / atr

def calc_vars(df, spy_df, period=21):
    stock_ret = df["Close"].pct_change(period)
    spy_ret = spy_df["Close"].pct_change(period)
    stock_vol = df["Close"].pct_change().rolling(window=period).std() * np.sqrt(252)
    vars_score = (stock_ret - spy_ret) / stock_vol.replace(0, np.nan)
    return vars_score

def calc_relative_strength(df, spy_df, period=21):
    stock_ret = df["Close"].pct_change(period).iloc[-1]
    spy_ret = spy_df["Close"].pct_change(period).iloc[-1]
    if spy_ret == 0 or pd.isna(spy_ret):
        return stock_ret * 100 if not pd.isna(stock_ret) else 0
    return stock_ret / abs(spy_ret)

def calc_relative_volume(df, period=20):
    avg_vol = df["Volume"].rolling(window=period).mean()
    return df["Volume"] / avg_vol

def get_ma_status(df):
    close = df["Close"].iloc[-1]
    ma10 = df["Close"].rolling(10).mean().iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    ma50 = df["Close"].rolling(50).mean().iloc[-1]
    ma200 = df["Close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else None
    pct_from_10ma = ((close - ma10) / ma10) * 100
    if abs(pct_from_10ma) <= 3 and close > ma20:
        status = "Touch"
    elif close > ma10:
        status = "Above"
    else:
        status = "Below"
    return {
        "ma_status": status,
        "pct_from_10ma": round(pct_from_10ma, 2),
        "above_20ma": close > ma20,
        "above_50ma": close > ma50,
        "above_200ma": close > ma200 if ma200 else None,
        "ma200_declining": (ma200 < df["Close"].rolling(200).mean().iloc[-20]) if (ma200 and len(df) >= 220) else None,
    }

def get_performance_periods(df):
    close = df["Close"]
    now = close.iloc[-1]
    periods = {}
    for label, days in [("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126)]:
        if len(close) > days:
            prev = close.iloc[-days-1]
            periods[label] = round(((now - prev) / prev) * 100, 2)
        else:
            periods[label] = None
    return periods

def classify_scan_tier(perf):
    tiers = []
    if perf.get("1W") and perf["1W"] >= 20: tiers.append("1W")
    if perf.get("1M") and perf["1M"] >= 30: tiers.append("1M")
    if perf.get("3M") and perf["3M"] >= 50: tiers.append("3M")
    if perf.get("6M") and perf["6M"] >= 100: tiers.append("6M")
    return tiers

def is_cve_ready(ma_info, adr_pct):
    return (
        ma_info["ma_status"] == "Touch"
        and ma_info["above_20ma"]
        and adr_pct >= 5.0
    )

def get_warnings(ma_info, atr_ext, adr_pct):
    warnings = []
    if ma_info.get("ma200_declining"):
        warnings.append("⚠ DECLINING 200-MA")
    if atr_ext and atr_ext > 3:
        warnings.append("⚠ EXTENDED >3 ATR")
    if adr_pct and adr_pct < 5:
        warnings.append("LOW ADR%")
    return warnings


# ══════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════

def fetch_history(ticker, period="1y"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df.empty:
            print(f"  WARNING: No data for {ticker}")
            return None
        return df
    except Exception as e:
        print(f"  ERROR fetching {ticker}: {e}")
        return None

def build_screener_data(out_dir):
    print("\n=== BUILDING SCREENER DATA ===")
    print("  Fetching SPY benchmark...")
    spy_df = fetch_history("SPY", period="1y")
    if spy_df is None:
        print("  FATAL: Cannot fetch SPY data")
        return []

    results = []
    total = len(SCAN_UNIVERSE)
    for i, ticker in enumerate(SCAN_UNIVERSE):
        print(f"  [{i+1}/{total}] Processing {ticker}...")
        df = fetch_history(ticker, period="1y")
        if df is None or len(df) < 50:
            continue
        try:
            close = df["Close"].iloc[-1]
            perf = get_performance_periods(df)
            tiers = classify_scan_tier(perf)
            atr_series = calc_atr(df, 14)
            atr_val = atr_series.iloc[-1] if not pd.isna(atr_series.iloc[-1]) else 0
            atr_pct = (atr_val / close * 100) if close > 0 else 0
            adr_pct_series = calc_adr_pct(df, 20)
            adr_pct = adr_pct_series.iloc[-1] if not pd.isna(adr_pct_series.iloc[-1]) else 0
            atr_ext_series = calc_atr_extension_from_sma(df, 50, 14)
            atr_ext = atr_ext_series.iloc[-1] if not pd.isna(atr_ext_series.iloc[-1]) else 0
            vars_series = calc_vars(df, spy_df, 21)
            vars_val = vars_series.iloc[-1] if not pd.isna(vars_series.iloc[-1]) else 0
            rs_vs_spy = calc_relative_strength(df, spy_df, 21)
            rvol_series = calc_relative_volume(df, 20)
            rvol = rvol_series.iloc[-1] if not pd.isna(rvol_series.iloc[-1]) else 0
            ma_info = get_ma_status(df)
            cve_ready = is_cve_ready(ma_info, adr_pct)
            warnings = get_warnings(ma_info, atr_ext, adr_pct)
            avg_vol = df["Volume"].rolling(20).mean().iloc[-1]

            results.append({
                "symbol": ticker,
                "close": round(close, 2),
                "avg_volume": int(avg_vol) if not pd.isna(avg_vol) else 0,
                "pct_1w": perf.get("1W"),
                "pct_1m": perf.get("1M"),
                "pct_3m": perf.get("3M"),
                "pct_6m": perf.get("6M"),
                "scan_tiers": tiers,
                "adr_pct": round(adr_pct, 2),
                "atr_pct": round(atr_pct, 2),
                "atr_ext_50sma": round(atr_ext, 2),
                "vars_1m": round(vars_val, 3),
                "rs_vs_spy": round(rs_vs_spy, 2),
                "rvol": round(rvol, 2),
                "ma_status": ma_info["ma_status"],
                "pct_from_10ma": ma_info["pct_from_10ma"],
                "above_20ma": ma_info["above_20ma"],
                "above_50ma": ma_info["above_50ma"],
                "above_200ma": ma_info["above_200ma"],
                "ma200_declining": ma_info["ma200_declining"],
                "cve_ready": cve_ready,
                "warnings": warnings,
                "c_grade": "",
                "v_grade": "",
                "e_grade": "",
                "notes": "",
                "watchlist": False,
            })
        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")
            continue

    results.sort(key=lambda x: x.get("vars_1m", 0), reverse=True)
    print(f"  Screener complete: {len(results)} stocks processed")
    return results

def build_sector_data(out_dir):
    print("\n=== BUILDING SECTOR DATA ===")
    spy_df = fetch_history("SPY", period="1y")
    if spy_df is None:
        return {}

    all_etfs = {}
    all_etfs.update(SECTOR_ETFS)
    all_etfs.update(COUNTRY_ETFS)
    all_etfs.update(INDUSTRY_ETFS)

    results = {"sectors": [], "countries": [], "industries": []}
    for ticker, name in all_etfs.items():
        print(f"  Processing {ticker} ({name})...")
        df = fetch_history(ticker, period="1y")
        if df is None or len(df) < 50:
            continue
        try:
            close = df["Close"].iloc[-1]
            perf = get_performance_periods(df)
            atr_ext_series = calc_atr_extension_from_sma(df, 50, 14)
            atr_ext = atr_ext_series.iloc[-1] if not pd.isna(atr_ext_series.iloc[-1]) else 0
            vars_series = calc_vars(df, spy_df, 21)
            vars_val = vars_series.iloc[-1] if not pd.isna(vars_series.iloc[-1]) else 0
            rs = calc_relative_strength(df, spy_df, 21)
            entry = {
                "symbol": ticker, "name": name, "close": round(close, 2),
                "pct_1w": perf.get("1W"), "pct_1m": perf.get("1M"),
                "pct_3m": perf.get("3M"), "pct_6m": perf.get("6M"),
                "atr_ext_50sma": round(atr_ext, 2),
                "vars_1m": round(vars_val, 3),
                "rs_vs_spy": round(rs, 2),
            }
            if ticker in SECTOR_ETFS:
                results["sectors"].append(entry)
            elif ticker in COUNTRY_ETFS:
                results["countries"].append(entry)
            elif ticker in INDUSTRY_ETFS:
                results["industries"].append(entry)
        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")
            continue

    for cat in results:
        results[cat].sort(key=lambda x: x.get("vars_1m", 0), reverse=True)
    print(f"  Sectors: {len(results['sectors'])}, Countries: {len(results['countries'])}, Industries: {len(results['industries'])}")
    return results

def build_breadth_data(out_dir):
    print("\n=== BUILDING BREADTH DATA ===")
    result = {}

    vix_df = fetch_history("^VIX", period="1mo")
    if vix_df is not None and len(vix_df) > 0:
        result["vix"] = round(vix_df["Close"].iloc[-1], 2)
        result["vix_5d_ago"] = round(vix_df["Close"].iloc[-5], 2) if len(vix_df) >= 5 else None
        result["vix_status"] = "GREEN" if result["vix"] < 20 else "YELLOW" if result["vix"] < 30 else "RED"

    rsp_df = fetch_history("RSP", period="3mo")
    spy_df = fetch_history("SPY", period="3mo")
    if rsp_df is not None and spy_df is not None:
        rsp_1m = rsp_df["Close"].pct_change(21).iloc[-1] * 100
        spy_1m = spy_df["Close"].pct_change(21).iloc[-1] * 100
        result["rsp_1m_ret"] = round(rsp_1m, 2)
        result["spy_1m_ret"] = round(spy_1m, 2)
        result["rsp_spy_divergence"] = round(rsp_1m - spy_1m, 2)

    iwm_df = fetch_history("IWM", period="3mo")
    if iwm_df is not None:
        iwm_1m = iwm_df["Close"].pct_change(21).iloc[-1] * 100
        result["iwm_1m_ret"] = round(iwm_1m, 2)

    print("  Calculating breadth from scan universe...")
    above_20ma_count = 0
    above_50ma_count = 0
    total_checked = 0
    breadth_tickers = list(SECTOR_ETFS.keys()) + list(INDUSTRY_ETFS.keys())
    for ticker in breadth_tickers:
        df = fetch_history(ticker, period="3mo")
        if df is not None and len(df) >= 50:
            total_checked += 1
            close = df["Close"].iloc[-1]
            ma20 = df["Close"].rolling(20).mean().iloc[-1]
            ma50 = df["Close"].rolling(50).mean().iloc[-1]
            if close > ma20: above_20ma_count += 1
            if close > ma50: above_50ma_count += 1

    if total_checked > 0:
        result["pct_above_20ma"] = round((above_20ma_count / total_checked) * 100, 1)
        result["pct_above_50ma"] = round((above_50ma_count / total_checked) * 100, 1)

    vix = result.get("vix", 20)
    pct20 = result.get("pct_above_20ma", 50)
    if vix < 20 and pct20 > 60:
        result["regime"] = "RISK ON"
        result["regime_color"] = "green"
    elif vix > 30 or pct20 < 30:
        result["regime"] = "SIT ON HANDS"
        result["regime_color"] = "red"
    else:
        result["regime"] = "CAUTIOUS"
        result["regime_color"] = "yellow"

    print(f"  Breadth complete: VIX={result.get('vix')}, Above 20MA={result.get('pct_above_20ma')}%, Regime={result.get('regime')}")
    return result

def build_calendar_data(out_dir):
    print("\n=== BUILDING CALENDAR DATA ===")
    events = []
    watchlist = ["NVDA", "PLTR", "COIN", "HOOD", "MSTR", "APP"]
    for ticker in watchlist:
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is not None and not cal.empty if isinstance(cal, pd.DataFrame) else cal is not None:
                if isinstance(cal, dict):
                    if "Earnings Date" in cal:
                        dates = cal["Earnings Date"]
                        for d in (dates if isinstance(dates, list) else [dates]):
                            events.append({
                                "type": "earnings", "symbol": ticker,
                                "text": f"{ticker} Earnings", "date": str(d), "impact": "high",
                            })
                elif isinstance(cal, pd.DataFrame):
                    for col in cal.columns:
                        if "Earnings" in str(col):
                            for d in cal[col].dropna():
                                events.append({
                                    "type": "earnings", "symbol": ticker,
                                    "text": f"{ticker} Earnings", "date": str(d), "impact": "high",
                                })
        except Exception as e:
            print(f"  Could not get calendar for {ticker}: {e}")
    print(f"  Calendar: {len(events)} events found")
    return events


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EVC Terminal Data Pipeline")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"EVC Terminal Data Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {out_dir}")

    screener = build_screener_data(out_dir)
    sectors = build_sector_data(out_dir)
    breadth = build_breadth_data(out_dir)
    calendar = build_calendar_data(out_dir)

    def write_json(filename, data):
        path = out_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Wrote {path} ({os.path.getsize(path)} bytes)")

    write_json("screener.json", screener)
    write_json("sectors.json", sectors)
    write_json("breadth.json", breadth)
    write_json("calendar.json", calendar)
    write_json("meta.json", {
        "built_at": datetime.now().isoformat(),
        "universe_size": len(SCAN_UNIVERSE),
        "screener_results": len(screener),
        "version": "1.0.0",
        "source": "yfinance",
        "note": "ATR extension and VARS formulas adapted from traderwillhu/market_dashboard"
    })

    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"  {len(screener)} stocks in screener")
    print(f"  {sum(len(v) for v in sectors.values())} ETFs tracked")
    print(f"  Breadth regime: {breadth.get('regime', 'UNKNOWN')}")

if __name__ == "__main__":
    main()
