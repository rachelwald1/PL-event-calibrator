"""
ingest/kalshi_collect.py

Collect Kalshi market data for Premier League (or any keyword-filtered set of markets).

Strategy:
1) List markets (paginated) via /markets
2) Filter locally for Premier League using keyword matching over title/subtitle/ticker
3) For matching tickers, fetch:
   - batch candlesticks (/markets/candlesticks)
   - optional orderbook snapshots (/markets/{ticker}/orderbook)

Outputs (default under data/kalshi/):
- pl_markets_<stamp>.json         (filtered market metadata)
- pl_candles_<stamp>.parquet      (candlesticks flattened, prob units [0,1])
- pl_orderbooks_<stamp>.parquet   (optional, microstructure snapshot features)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass(frozen=True)
class KalshiConfig:
    base_url: str = DEFAULT_BASE_URL
    timeout_s: int = 20
    max_retries: int = 3
    retry_backoff_s: float = 1.0


class KalshiClient:
    def __init__(self, cfg: KalshiConfig):
        self.cfg = cfg

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + path
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                r = requests.get(url, params=params, timeout=self.cfg.timeout_s)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_s * attempt)
                else:
                    raise RuntimeError(f"GET failed: {url} params={params}") from last_err
        raise RuntimeError("unreachable")

    def list_markets(self, status: str = "open", limit: int = 100, cursor: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"status": status, "limit": int(limit)}
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets", params=params)

    def batch_candlesticks(
        self,
        tickers: List[str],
        start_ts: int,
        end_ts: int,
        period_interval: str = "5m",
        include_latest_before_start: bool = True,
    ) -> Dict[str, Any]:
        params = {
            "tickers": ",".join(tickers),
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "period_interval": period_interval,
            "include_latest_before_start": str(include_latest_before_start).lower(),
        }
        return self._get("/markets/candlesticks", params=params)

    def get_orderbook(self, ticker: str, depth: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if depth is not None:
            params["depth"] = int(depth)
        return self._get(f"/markets/{ticker}/orderbook", params=params)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _market_text(m: Dict[str, Any]) -> str:
    # Defensive: fields vary; combine common textual fields
    parts = []
    for k in ["ticker", "title", "subtitle", "event_title", "series_title", "category", "tags"]:
        v = m.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            parts.append(" ".join(str(x) for x in v))
        else:
            parts.append(str(v))
    return " ".join(parts).lower()


def filter_markets_by_keywords(markets: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    kw = [k.lower() for k in keywords if k.strip()]
    out = []
    for m in markets:
        txt = _market_text(m)
        if any(k in txt for k in kw):
            out.append(m)
    return out


def _parse_candle_row(ticker: Optional[str], c: Dict[str, Any]) -> Dict[str, Any]:
    ts = c.get("ts") or c.get("start_ts") or c.get("t")
    open_c = c.get("open") or c.get("open_price")
    high_c = c.get("high") or c.get("high_price")
    low_c = c.get("low") or c.get("low_price")
    close_c = c.get("close") or c.get("close_price")

    def cents_to_prob(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            return float(x) / 100.0
        except Exception:
            return None

    return {
        "ticker": ticker or c.get("ticker") or c.get("market_ticker"),
        "ts": ts,
        "open_p": cents_to_prob(open_c),
        "high_p": cents_to_prob(high_c),
        "low_p": cents_to_prob(low_c),
        "close_p": cents_to_prob(close_c),
        "volume": c.get("volume"),
    }


def flatten_batch_candles(resp: Dict[str, Any]) -> pd.DataFrame:
    """
    Flatten batch candlestick response into:
      ticker, ts, open_p, high_p, low_p, close_p, volume
    Prices are in probability units [0,1] (cents / 100).
    """
    rows: List[Dict[str, Any]] = []

    groups = resp.get("candlesticks")
    if isinstance(groups, list):
        for g in groups:
            ticker = g.get("ticker") or g.get("market_ticker")
            candles = g.get("candlesticks") or g.get("data") or []
            for c in candles:
                rows.append(_parse_candle_row(ticker, c))
    elif isinstance(groups, dict):
        for ticker, candles in groups.items():
            for c in candles or []:
                rows.append(_parse_candle_row(ticker, c))
    else:
        # best-effort fallback
        maybe = resp.get("markets") or {}
        if isinstance(maybe, dict):
            for ticker, candles in maybe.items():
                if isinstance(candles, list):
                    for c in candles:
                        rows.append(_parse_candle_row(ticker, c))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.dropna(subset=["ticker", "ts", "close_p"])
    df["ts"] = df["ts"].astype(int)
    for col in ["open_p", "high_p", "low_p", "close_p"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0.0, 1.0)
    return df


def flatten_orderbook_snapshots(obs: List[Tuple[str, Dict[str, Any]]], top_k: int = 10) -> pd.DataFrame:
    """
    Convert YES/NO bids to a snapshot feature row per ticker.

    Note: Kalshi orderbook returns bids for YES and NO. An implied YES ask can be
    computed from the best NO bid: ask_yes = 100 - best_no_bid.
    """
    now_ts = int(time.time())
    rows: List[Dict[str, Any]] = []

    for ticker, ob in obs:
        yes = ob.get("yes") or ob.get("orderbook", {}).get("yes") or []
        no = ob.get("no") or ob.get("orderbook", {}).get("no") or []

        def best_bid(levels):
            if not levels:
                return None
            return max((lvl[0] for lvl in levels if isinstance(lvl, list) and len(lvl) >= 2), default=None)

        def sum_qty(levels, k):
            qtys = [lvl[1] for lvl in levels if isinstance(lvl, list) and len(lvl) >= 2]
            return float(sum(qtys[:k])) if qtys else 0.0

        yes_best = best_bid(yes)  # cents
        no_best = best_bid(no)    # cents
        yes_ask = (100 - no_best) if no_best is not None else None

        yes_bid_p = (yes_best / 100.0) if yes_best is not None else None
        yes_ask_p = (yes_ask / 100.0) if yes_ask is not None else None

        mid_p = None
        spread_p = None
        if yes_bid_p is not None and yes_ask_p is not None:
            mid_p = 0.5 * (yes_bid_p + yes_ask_p)
            spread_p = yes_ask_p - yes_bid_p

        yes_qty_k = sum_qty(yes, top_k)
        no_qty_k = sum_qty(no, top_k)
        imbalance = None
        denom = yes_qty_k + no_qty_k
        if denom > 0:
            imbalance = (yes_qty_k - no_qty_k) / denom

        rows.append({
            "ticker": ticker,
            "ts": now_ts,
            "yes_bid_p": yes_bid_p,
            "yes_ask_p": yes_ask_p,
            "mid_p": mid_p,
            "spread_p": spread_p,
            "yes_qty_topk": yes_qty_k,
            "no_qty_topk": no_qty_k,
            "imbalance_topk": imbalance,
        })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Kalshi markets filtered for Premier League (keyword-based).")
    ap.add_argument("--outdir", type=str, default="data/kalshi", help="Output directory")
    ap.add_argument("--status", type=str, default="open", help="Market status (open/closed)")
    ap.add_argument("--pages", type=int, default=5, help="How many /markets pages to scan (100 each)")
    ap.add_argument("--keywords", type=str, default="premier league,epl,english premier league",
                    help="Comma-separated keywords used to filter markets by title/subtitle/ticker")
    ap.add_argument("--n-max", type=int, default=60, help="Max tickers to collect after filtering")
    ap.add_argument("--hours", type=int, default=72, help="How far back to fetch candlesticks")
    ap.add_argument("--period", type=str, default="5m", help="Candlestick interval (e.g. 1m,5m,1h,1d)")
    ap.add_argument("--include-latest-before-start", action="store_true",
                    help="Include latest candle before start for continuity")
    ap.add_argument("--with-orderbooks", action="store_true", help="Also fetch orderbook snapshots")
    ap.add_argument("--orderbook-depth", type=int, default=50, help="Depth passed to orderbook endpoint")
    ap.add_argument("--orderbook-topk", type=int, default=10, help="Top K levels to summarize")
    ap.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Kalshi API base URL")
    args = ap.parse_args()

    _ensure_dir(args.outdir)
    client = KalshiClient(KalshiConfig(base_url=args.base_url))

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    # 1) Scan markets pages and filter locally
    all_markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    for _ in range(args.pages):
        resp = client.list_markets(status=args.status, limit=100, cursor=cursor)
        markets = resp.get("markets") or []
        if not markets:
            break
        all_markets.extend(markets)
        cursor = resp.get("cursor")
        if not cursor:
            break

    pl_markets = filter_markets_by_keywords(all_markets, keywords)
    # limit tickers
    pl_markets = pl_markets[: args.n_max]
    tickers = [m.get("ticker") for m in pl_markets if m.get("ticker")]

    stamp = _utc_stamp()

    # Save filtered market metadata
    meta_path = os.path.join(args.outdir, f"pl_markets_{stamp}.json")
    _save_json(meta_path, {
        "status": args.status,
        "keywords": keywords,
        "n_scanned": len(all_markets),
        "n_matched": len(pl_markets),
        "markets": pl_markets,
    })
    print(f"[kalshi_collect] Saved {meta_path} (matched={len(pl_markets)})")

    if not tickers:
        print("[kalshi_collect] No Premier League-like tickers found with given keywords.")
        print("Try widening --keywords, increasing --pages, or inspect markets_open.json to see naming.")
        return

    # 2) Download candlesticks for the matching tickers
    end_ts = int(time.time())
    start_ts = end_ts - int(args.hours) * 3600

    # API typically supports up to 100 tickers in a batch request; keep it safe.
    tickers_batch = tickers[: min(len(tickers), 100)]
    candles_resp = client.batch_candlesticks(
        tickers=tickers_batch,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=args.period,
        include_latest_before_start=args.include_latest_before_start,
    )
    candles_df = flatten_batch_candles(candles_resp)

    candles_path = os.path.join(args.outdir, f"pl_candles_{stamp}.parquet")
    candles_df.to_parquet(candles_path, index=False)
    print(f"[kalshi_collect] Saved {candles_path} rows={len(candles_df)} tickers={len(tickers_batch)}")

    # 3) Optional orderbooks
    if args.with_orderbooks:
        obs: List[Tuple[str, Dict[str, Any]]] = []
        for t in tickers[: min(len(tickers), 50)]:  # orderbooks are heavier; keep modest
            ob = client.get_orderbook(t, depth=args.orderbook_depth)
            obs.append((t, ob))
        ob_df = flatten_orderbook_snapshots(obs, top_k=args.orderbook_topk)
        ob_path = os.path.join(args.outdir, f"pl_orderbooks_{stamp}.parquet")
        ob_df.to_parquet(ob_path, index=False)
        print(f"[kalshi_collect] Saved {ob_path} rows={len(ob_df)} tickers={len(obs)}")


if __name__ == "__main__":
    main()
