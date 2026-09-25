"""Synthetic dataset schemas and value distributions.

Everything here is invented. It is meant to look like the general shape of
large event/reporting data (skewed keys, low-cardinality dimensions, a few
nullable measures) rather than any real system's schema.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

# Logical types are what readers (DuckDB, cuDF, pyarrow) see. Low-cardinality
# strings are declared as plain `string`; the Parquet writer dictionary-encodes
# them on disk anyway, and keeping the logical type simple avoids reader-specific
# categorical/dictionary behaviour.
EVENT_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.uint64(), nullable=False),
        pa.field("timestamp", pa.timestamp("us"), nullable=False),  # naive, interpreted as UTC
        pa.field("user_id", pa.uint64(), nullable=False),
        pa.field("vendor_id", pa.int32(), nullable=False),
        pa.field("product_id", pa.int32(), nullable=False),
        pa.field("region", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("quantity", pa.int32(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
        pa.field("latency_ms", pa.float32(), nullable=True),
    ]
)

PRODUCT_SCHEMA = pa.schema(
    [
        pa.field("product_id", pa.int32(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("vendor_id", pa.int32(), nullable=False),
        pa.field("base_price", pa.float64(), nullable=False),
        pa.field("brand", pa.string(), nullable=False),
    ]
)

REGIONS = [
    ("na-east", 0.22), ("na-west", 0.16), ("na-central", 0.10), ("sa-east", 0.06),
    ("eu-west", 0.14), ("eu-central", 0.10), ("ap-north", 0.10), ("ap-south", 0.08),
    ("af-south", 0.04),
]  # fmt: skip

CATEGORIES = [
    ("electronics", 0.12), ("books", 0.08), ("home", 0.08), ("kitchen", 0.06),
    ("toys", 0.05), ("apparel", 0.09), ("beauty", 0.05), ("grocery", 0.07),
    ("sports", 0.05), ("automotive", 0.04), ("garden", 0.03), ("office", 0.04),
    ("pet", 0.04), ("health", 0.05), ("jewelry", 0.02), ("music", 0.02),
    ("video_games", 0.04), ("tools", 0.03), ("baby", 0.02), ("outdoors", 0.03),
]  # fmt: skip

# Refunds carry a negative `value`, so aggregations see mixed signs.
EVENT_TYPES = [
    ("view", 0.55), ("search", 0.15), ("add_to_cart", 0.14), ("purchase", 0.13),
    ("refund", 0.03),
]  # fmt: skip
REFUND = "refund"

# Relative event volume by hour of day (UTC), so hourly workloads have some shape.
DIURNAL = [
    0.5, 0.4, 0.3, 0.3, 0.3, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2, 1.2,
    1.3, 1.3, 1.2, 1.2, 1.3, 1.4, 1.6, 1.7, 1.6, 1.3, 1.0, 0.7,
]  # fmt: skip

# Relative event volume by calendar month (Nov/Dec peak), plus year-over-year growth.
SEASONALITY = [0.90, 0.85, 0.95, 0.95, 1.00, 1.00, 1.00, 1.05, 1.00, 1.05, 1.30, 1.50]
ANNUAL_GROWTH = 1.15

# Zipf-like exponents: product popularity, and how many products each vendor owns.
PRODUCT_POPULARITY_EXPONENT = 0.8
VENDOR_SIZE_EXPONENT = 0.7


def names(pairs: list[tuple[str, float]]) -> list[str]:
    return [name for name, _ in pairs]


def cdf(weights) -> np.ndarray:
    """Cumulative distribution for inverse-CDF sampling via np.searchsorted."""
    w = np.asarray(weights, dtype=np.float64)
    c = np.cumsum(w / w.sum())
    c[-1] = 1.0  # guard against round-off so every uniform draw in [0, 1) maps to a bucket
    return c


def zipf_cdf(n: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """CDF over n items where item popularity follows 1/rank^exponent.

    Ranks are shuffled so popular items are scattered across the id space instead
    of clustering at low ids (which would make id-range filters unrealistically
    selective).
    """
    weights = 1.0 / np.arange(1, n + 1, dtype=np.float64) ** exponent
    return cdf(rng.permutation(weights))
