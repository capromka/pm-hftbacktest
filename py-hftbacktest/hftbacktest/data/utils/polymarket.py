import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing import Any

from ...types import (
    BUY_EVENT,
    DEPTH_CLEAR_EVENT,
    DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
    event_dtype,
)
from ..validation import correct_event_order

DEFAULT_FALLBACK_LATENCY_NS = 20_000_000
HBT_COLS = ["ev", "exch_ts", "local_ts", "px", "qty", "order_id", "ival", "fval"]


def _ts_ns_expr(df: pl.DataFrame, col: str = "timestamp") -> pl.Expr:
    """Converts a timestamp column to nanoseconds as int64."""
    dtype = df.schema[col]
    if hasattr(dtype, "time_unit") or str(dtype).startswith("Datetime"):
        return pl.col(col).dt.epoch("ns").cast(pl.Int64)
    return pl.col(col).cast(pl.Int64) * 1_000_000


def _local_ts_ns_expr(
    df: pl.DataFrame,
    exch_ts_expr: pl.Expr,
    fallback_latency_ns: int,
) -> pl.Expr:
    if "local_timestamp" not in df.columns:
        return exch_ts_expr + fallback_latency_ns
    return _ts_ns_expr(df, "local_timestamp")


def _make_row(ev: int, exch_ts: int, local_ts: int, px: float, qty: float) -> NDArray:
    row = np.zeros(1, dtype=event_dtype)
    row[0]["ev"] = ev
    row[0]["exch_ts"] = exch_ts
    row[0]["local_ts"] = local_ts
    row[0]["px"] = px
    row[0]["qty"] = qty
    return row


def polymarket_to_hbt(
    l2_df: Any,
    fallback_latency_ns: int = DEFAULT_FALLBACK_LATENCY_NS,
) -> NDArray:
    r"""
    Converts a Polymarket L2 DataFrame into an HftBacktest event array.

    Args:
        l2_df: DataFrame containing the Polymarket L2 data.
        fallback_latency_ns: Latency used when the input has no local timestamp
                             data. Defaults to 20ms.
    """
    df = pl.DataFrame(l2_df)

    ts_expr = _ts_ns_expr(df)
    local_ts_expr = _local_ts_ns_expr(df, ts_expr, fallback_latency_ns)
    parts: list[NDArray] = []

    books = df.filter(pl.col("event_type") == "book")
    if len(books) > 0:
        books = books.with_columns(
            ts_expr.alias("ts"),
            local_ts_expr.alias("local_ts"),
        ).sort("ts")
        for row in books.iter_rows(named=True):
            book_ts = int(row["ts"])
            book_local_ts = int(row["local_ts"])

            for px_col, qty_col, side_flag in [
                ("bid_prices", "bid_sizes", BUY_EVENT),
                ("ask_prices", "ask_sizes", SELL_EVENT),
            ]:
                prices = row.get(px_col) or []
                sizes = row.get(qty_col) or []
                if not prices:
                    continue

                clear_px = min(prices) if side_flag == BUY_EVENT else max(prices)
                parts.append(
                    _make_row(
                        DEPTH_CLEAR_EVENT | side_flag,
                        book_ts,
                        book_local_ts,
                        float(clear_px),
                        0.0,
                    )
                )

                n = min(len(prices), len(sizes))
                if n == 0:
                    continue
                snapshot = np.zeros(n, dtype=event_dtype)
                for i in range(n):
                    snapshot[i]["ev"] = DEPTH_SNAPSHOT_EVENT | side_flag
                    snapshot[i]["exch_ts"] = book_ts
                    snapshot[i]["local_ts"] = book_local_ts
                    snapshot[i]["px"] = float(prices[i])
                    snapshot[i]["qty"] = float(sizes[i])
                parts.append(snapshot)

    trades = df.filter(
        (pl.col("event_type") == "last_trade_price")
        & pl.col("trade_price").is_not_null()
    )
    if len(trades) > 0:
        arr = (
            trades.with_columns(
                pl.when(pl.col("trade_side") == "BUY")
                .then(pl.lit(TRADE_EVENT | BUY_EVENT))
                .otherwise(pl.lit(TRADE_EVENT | SELL_EVENT))
                .cast(pl.UInt64)
                .alias("ev"),
                ts_expr.alias("exch_ts"),
                local_ts_expr.alias("local_ts"),
                pl.col("trade_price").cast(pl.Float64).alias("px"),
                pl.col("trade_size").cast(pl.Float64).alias("qty"),
                pl.lit(0).cast(pl.UInt64).alias("order_id"),
                pl.lit(0).cast(pl.Int64).alias("ival"),
                pl.lit(0.0).alias("fval"),
            )
            .select(HBT_COLS)
            .to_numpy(structured=True)
        )
        out = np.zeros(len(arr), dtype=event_dtype)
        for col in event_dtype.names:
            out[col] = arr[col]
        parts.append(out)

    price_changes = df.filter(
        (pl.col("event_type") == "price_change") & pl.col("pc_price").is_not_null()
    )
    if len(price_changes) > 0:
        arr = (
            price_changes.with_columns(
                pl.when(pl.col("pc_side") == "BUY")
                .then(pl.lit(DEPTH_EVENT | BUY_EVENT))
                .otherwise(pl.lit(DEPTH_EVENT | SELL_EVENT))
                .cast(pl.UInt64)
                .alias("ev"),
                ts_expr.alias("exch_ts"),
                local_ts_expr.alias("local_ts"),
                pl.col("pc_price").cast(pl.Float64).alias("px"),
                pl.col("pc_size").cast(pl.Float64).alias("qty"),
                pl.lit(0).cast(pl.UInt64).alias("order_id"),
                pl.lit(0).cast(pl.Int64).alias("ival"),
                pl.lit(0.0).alias("fval"),
            )
            .select(HBT_COLS)
            .to_numpy(structured=True)
        )
        out = np.zeros(len(arr), dtype=event_dtype)
        for col in event_dtype.names:
            out[col] = arr[col]
        parts.append(out)

    if not parts:
        return np.zeros(0, dtype=event_dtype)

    data = np.concatenate(parts)
    data = data[np.argsort(data["exch_ts"], kind="mergesort")]
    return correct_event_order(
        data,
        np.argsort(data["exch_ts"], kind="mergesort"),
        np.argsort(data["local_ts"], kind="mergesort"),
    )
