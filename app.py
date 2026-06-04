import os
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from openelectricity import OEClient
from openelectricity.types import MarketMetric

from bands import STORAGE_BANDS

# ── Constants ──────────────────────────────────────────────────────────────────

REGIONS = {
    "NSW": "NSW1",
    "VIC": "VIC1",
    "QLD": "QLD1",
    "SA":  "SA1",
    "TAS": "TAS1",
}

_today = date.today()

PRESETS: dict[str, tuple[date, date]] = {
    "Full Year 2025":     (date(2025, 1, 1),           date(2025, 12, 31)),
    "H1 2025 (Jan–Jun)":  (date(2025, 1, 1),           date(2025, 6, 30)),
    "H2 2025 (Jul–Dec)":  (date(2025, 7, 1),           date(2025, 12, 31)),
    "Last 30 days":       (_today - timedelta(days=30), _today),
    "Last 90 days":       (_today - timedelta(days=90), _today),
    "Last 12 months":     (_today - timedelta(days=365), _today),
    "Custom":             (date(2025, 1, 1),            _today),
}

INTERVAL_MINUTES = 30  # 30-minute trading intervals

# ── Data fetching ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(region_code: str, date_start: date, date_end: date) -> pd.Series:
    """Return a Series of spot prices ($/MWh) at 30-minute trading intervals."""
    api_key = os.environ.get("OPENELECTRICITY_API_KEY", "")
    with OEClient(api_key=api_key) as client:
        response = client.get_market(
            network_code="NEM",
            metrics=[MarketMetric.PRICE],
            interval="1h",
            date_start=datetime.combine(date_start, datetime.min.time()),
            date_end=datetime.combine(date_end, datetime.max.time()),
            network_region=region_code,
        )

    df = response.to_pandas()

    # Locate the price column — handle varying column naming from the client
    price_col = next(
        (c for c in df.columns if "price" in str(c).lower()),
        None,
    )
    if price_col is None:
        numeric = df.select_dtypes(include="number").columns.tolist()
        if not numeric:
            raise ValueError(f"Cannot find price column. Got columns: {list(df.columns)}")
        price_col = numeric[0]

    return df[price_col].dropna()


# ── Band statistics ────────────────────────────────────────────────────────────

def compute_band_stats(prices: pd.Series) -> list[dict]:
    total = len(prices)
    stats = []
    for band in STORAGE_BANDS:
        if band.min_price is None:
            mask = prices < band.max_price
        elif band.max_price is None:
            mask = prices >= band.min_price
        else:
            mask = (prices >= band.min_price) & (prices < band.max_price)

        subset = prices[mask]
        count = int(len(subset))
        hours = count * INTERVAL_MINUTES / 60
        pct = count / total * 100 if total > 0 else 0.0
        avg = float(subset.mean()) if count > 0 else None

        stats.append(dict(band=band, count=count, hours=hours, pct=pct, avg=avg))
    return stats


# ── HTML table renderer ────────────────────────────────────────────────────────

_CSS = """
<style>
.pbt-wrap { overflow-x: auto; }
.pbt {
  width: 100%;
  border-collapse: collapse;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 14px;
}
.pbt th {
  text-align: left;
  padding: 8px 14px;
  font-size: 11px;
  font-weight: 600;
  color: #9ca3af;
  text-transform: uppercase;
  letter-spacing: .06em;
  border-bottom: 2px solid #e5e7eb;
  white-space: nowrap;
}
.pbt td {
  padding: 12px 14px;
  border-bottom: 1px solid #f3f4f6;
  vertical-align: middle;
}
.pbt tr:last-child td { border-bottom: none; }
.pbt tbody tr:hover td { background: #fafafa; }
.badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 22px;
  border-radius: 5px;
  font-size: 11px;
  font-weight: 700;
  color: #fff;
  letter-spacing: .03em;
}
.bar-wrap {
  background: #f3f4f6;
  border-radius: 4px;
  height: 14px;
  width: 200px;
  overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 4px; }
.mono { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 12.5px; }
.dim  { color: #9ca3af; }
.strong { font-weight: 600; }
</style>
"""


def _fmt_avg(avg: float | None) -> str:
    if avg is None:
        return "–"
    if avg < 0:
        return f"-${abs(avg):,.0f}"
    return f"${avg:,.0f}"


def _fmt_hours(hours: float) -> str:
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h:,}h {m:02d}m"


def render_band_table(stats: list[dict]) -> None:
    max_pct = max(s["pct"] for s in stats) or 1.0

    rows = ""
    for s in stats:
        band = s["band"]
        bar_w = int(s["pct"] / max_pct * 100)

        rows += f"""
        <tr>
          <td><span class="badge" style="background:{band.color}">{band.id}</span></td>
          <td class="strong">{band.name}</td>
          <td class="dim" style="font-size:13px">{band.action}</td>
          <td class="mono">{band.range_label}</td>
          <td class="strong">{_fmt_avg(s['avg'])}</td>
          <td class="dim">{s['count']:,}</td>
          <td class="dim">{_fmt_hours(s['hours'])}</td>
          <td class="strong" style="color:{band.color}">{s['pct']:.1f}%</td>
          <td>
            <div class="bar-wrap">
              <div class="bar-fill" style="width:{bar_w}%;background:{band.color}"></div>
            </div>
          </td>
        </tr>"""

    st.markdown(
        _CSS + f"""
        <div class="pbt-wrap">
        <table class="pbt">
          <thead><tr>
            <th>Band</th><th>Name</th><th>Action</th>
            <th>Price Range</th><th>Avg Price</th>
            <th>Intervals</th><th>Hours</th><th>%</th>
            <th>Distribution</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Page layout ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NEM Price Bands",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ NEM Price Band Analyser")
st.caption("Spot price distribution across 5-minute dispatch intervals — optimised for utility-scale storage.")

# Controls
c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
with c1:
    region_name = st.selectbox("Region", list(REGIONS.keys()), index=0)
    region_code = REGIONS[region_name]
with c2:
    preset = st.selectbox("Timeframe", list(PRESETS.keys()), index=0)

if preset == "Custom":
    with c3:
        date_start = st.date_input("From", PRESETS["Custom"][0])
    with c4:
        date_end = st.date_input("To", PRESETS["Custom"][1])
else:
    date_start, date_end = PRESETS[preset]
    date_end = min(date_end, _today)

st.divider()

# Fetch data
status_slot = st.empty()
status_slot.info(f"Fetching {region_name} 5-minute spot prices for {date_start:%d %b %Y} – {date_end:%d %b %Y}…")

try:
    prices = fetch_prices(region_code, date_start, date_end)
    status_slot.empty()
except Exception as exc:
    status_slot.error(f"API error: {exc}")
    st.stop()

if prices.empty:
    status_slot.warning("No data returned for this selection. Try a different region or timeframe.")
    st.stop()

# Summary metrics
total_hours = len(prices) * INTERVAL_MINUTES / 60
avg_price   = float(prices.mean())
neg_pct     = float((prices < 0).mean() * 100)
high_pct    = float((prices >= 300).mean() * 100)
spike_pct   = float((prices >= 1000).mean() * 100)
max_price   = float(prices.max())

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Avg Spot Price",       f"${avg_price:,.0f} /MWh")
m2.metric("Total Hours",          f"{total_hours:,.0f}h")
m3.metric("Negative Price",       f"{neg_pct:.1f}%",  help="% of intervals with price < $0 (charging opportunity)")
m4.metric("High Price (≥$300)",   f"{high_pct:.1f}%", help="% of intervals with price ≥ $300/MWh")
m5.metric("Spike (≥$1k)",         f"{spike_pct:.2f}%", help="% of intervals with price ≥ $1,000/MWh")
m6.metric("Max Observed Price",   f"${max_price:,.0f} /MWh")

st.markdown("&nbsp;")

# Band table
stats = compute_band_stats(prices)
render_band_table(stats)

st.markdown("&nbsp;")
st.caption(
    f"Source: Open Electricity API · 1-hour intervals · "
    f"{len(prices):,} intervals · {date_start:%d %b %Y} – {date_end:%d %b %Y}"
)
