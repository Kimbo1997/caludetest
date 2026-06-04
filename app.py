import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
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
    "Full Year 2025":     (date(2025, 1, 1),            date(2025, 12, 31)),
    "H1 2025 (Jan–Jun)":  (date(2025, 1, 1),            date(2025, 6, 30)),
    "H2 2025 (Jul–Dec)":  (date(2025, 7, 1),            date(2025, 12, 31)),
    "Last 30 days":       (_today - timedelta(days=30),  _today),
    "Last 90 days":       (_today - timedelta(days=90),  _today),
    "Last 12 months":     (_today - timedelta(days=365), _today),
    "Custom":             (date(2025, 1, 1),             _today),
}

INTERVAL_MINUTES = 60
CHUNK_DAYS = 30

CHART_FREQS: dict[str, str] = {
    "1 Week":    "W",
    "1 Month":   "ME",
    "1 Quarter": "QE",
    "1 Year":    "YE",
}


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_chunk(client: OEClient, region_code: str, chunk_start: date, chunk_end: date) -> pd.Series:
    response = client.get_market(
        network_code="NEM",
        metrics=[MarketMetric.PRICE],
        interval="1h",
        date_start=datetime.combine(chunk_start, datetime.min.time()),
        date_end=datetime.combine(chunk_end + timedelta(days=1), datetime.min.time()),
        network_region=region_code,
    )
    df = response.to_pandas()

    # Ensure datetime index for resampling
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = next(
            (c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()),
            None,
        )
        if date_col:
            df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index)

    price_col = next((c for c in df.columns if "price" in str(c).lower()), None)
    if price_col is None:
        numeric = df.select_dtypes(include="number").columns.tolist()
        if not numeric:
            raise ValueError(f"No price column found. Columns: {list(df.columns)}")
        price_col = numeric[0]

    return df[price_col].dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(region_code: str, date_start: date, date_end: date) -> pd.Series:
    """Fetch hourly spot prices, chunked into 30-day windows to respect API limits."""
    api_key = os.environ.get("OPENELECTRICITY_API_KEY", "")

    chunks: list[tuple[date, date]] = []
    cursor = date_start
    while cursor <= date_end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), date_end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)

    series_list: list[pd.Series] = []
    progress = st.progress(0, text=f"Fetching data… (0 / {len(chunks)} chunks)")

    with OEClient(api_key=api_key) as client:
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            for attempt in range(3):
                try:
                    series_list.append(_fetch_chunk(client, region_code, chunk_start, chunk_end))
                    break
                except Exception as exc:
                    # Retry on rate-limit (429); re-raise immediately on auth (403)
                    if "429" in str(exc) and attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise
            progress.progress(
                (i + 1) / len(chunks),
                text=f"Fetching data… ({i + 1} / {len(chunks)} chunks)",
            )

    progress.empty()

    if not series_list:
        return pd.Series(dtype=float)

    combined = pd.concat(series_list)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


# ── Band statistics ────────────────────────────────────────────────────────────

def _band_mask(prices: pd.Series, band) -> pd.Series:
    if band.min_price is None:
        return prices < band.max_price
    elif band.max_price is None:
        return prices >= band.min_price
    return (prices >= band.min_price) & (prices < band.max_price)


def compute_band_stats(prices: pd.Series) -> list[dict]:
    total = len(prices)
    stats = []
    for band in STORAGE_BANDS:
        subset = prices[_band_mask(prices, band)]
        count = int(len(subset))
        hours = count * INTERVAL_MINUTES / 60
        pct = count / total * 100 if total > 0 else 0.0
        avg = float(subset.mean()) if count > 0 else None
        stats.append(dict(band=band, count=count, hours=hours, pct=pct, avg=avg))
    return stats


def compute_timeseries(prices: pd.Series, freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample prices by freq; return (% in band df, avg price in band df)."""
    freq_rows: list[dict] = []
    avg_rows: list[dict] = []

    for period, group in prices.groupby(pd.Grouper(freq=freq)):
        if len(group) == 0:
            continue
        total = len(group)
        freq_row: dict = {"date": period}
        avg_row: dict = {"date": period}
        for band in STORAGE_BANDS:
            subset = group[_band_mask(group, band)]
            freq_row[band.id] = len(subset) / total * 100
            avg_row[band.id] = float(subset.mean()) if len(subset) > 0 else None
        freq_rows.append(freq_row)
        avg_rows.append(avg_row)

    def _to_df(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("date")

    return _to_df(freq_rows), _to_df(avg_rows)


# ── HTML band table ────────────────────────────────────────────────────────────

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
.bar-wrap { background: #f3f4f6; border-radius: 4px; height: 14px; width: 200px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 4px; }
.mono  { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 12.5px; }
.dim   { color: #9ca3af; }
.strong { font-weight: 600; }
</style>
"""


def _fmt_avg(avg: float | None) -> str:
    if avg is None:
        return "–"
    return f"-${abs(avg):,.0f}" if avg < 0 else f"${avg:,.0f}"


def _fmt_hours(hours: float) -> str:
    h, m = int(hours), int((hours % 1) * 60)
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

    st.markdown(_CSS, unsafe_allow_html=True)
    st.html(f"""
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
    """)


# ── Plotly charts ──────────────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    height=400,
    margin=dict(t=20, b=40, l=60, r=20),
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
    xaxis=dict(showgrid=False, showline=True, linecolor="#e5e7eb"),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", showline=False, zeroline=True, zerolinecolor="#e5e7eb"),
)


def render_frequency_chart(freq_df: pd.DataFrame) -> None:
    fig = go.Figure()
    for band in STORAGE_BANDS:
        if band.id not in freq_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=freq_df.index, y=freq_df[band.id],
            name=f"{band.id} · {band.name}",
            line=dict(color=band.color, width=2),
            mode="lines+markers", marker=dict(size=4),
        ))
    fig.update_layout(
        **_CHART_LAYOUT,
        yaxis=dict(**_CHART_LAYOUT["yaxis"], title="% of Hours", ticksuffix="%", rangemode="tozero"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_avg_price_chart(avg_df: pd.DataFrame) -> None:
    fig = go.Figure()
    for band in STORAGE_BANDS:
        if band.id not in avg_df.columns or avg_df[band.id].isna().all():
            continue
        fig.add_trace(go.Scatter(
            x=avg_df.index, y=avg_df[band.id],
            name=f"{band.id} · {band.name}",
            line=dict(color=band.color, width=2),
            mode="lines+markers", marker=dict(size=4),
        ))
    fig.update_layout(
        **_CHART_LAYOUT,
        yaxis=dict(**_CHART_LAYOUT["yaxis"], title="Avg Price ($/MWh)", tickprefix="$"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NEM Price Bands", page_icon="⚡", layout="wide")

st.title("⚡ NEM Price Band Analyser")
st.caption("Spot price distribution across hourly intervals — optimised for utility-scale storage.")

# Controls row
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

# Fetch
status_slot = st.empty()
status_slot.info(f"Fetching {region_name} spot prices for {date_start:%d %b %Y} – {date_end:%d %b %Y}…")

try:
    prices = fetch_prices(region_code, date_start, date_end)
    status_slot.empty()
except Exception as exc:
    msg = str(exc)
    if "403" in msg:
        status_slot.error(
            "API key rejected (403). Check that OPENELECTRICITY_API_KEY in your .env file is correct "
            "and that your account is active at platform.openelectricity.org.au"
        )
    else:
        status_slot.error(f"API error: {msg}")
    st.stop()

if prices.empty:
    status_slot.warning("No data returned for this selection.")
    st.stop()

# Summary metrics
avg_price = float(prices.mean())
neg_pct   = float((prices < 0).mean() * 100)
high_pct  = float((prices >= 300).mean() * 100)
spike_pct = float((prices >= 1000).mean() * 100)
max_price = float(prices.max())

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Avg Spot Price",     f"${avg_price:,.0f} /MWh")
m2.metric("Total Hours",        f"{len(prices):,}h")
m3.metric("Negative Price",     f"{neg_pct:.1f}%",   help="% of hours with price < $0")
m4.metric("High (≥$300)",       f"{high_pct:.1f}%",  help="% of hours ≥ $300/MWh")
m5.metric("Spike (≥$1k)",       f"{spike_pct:.2f}%", help="% of hours ≥ $1,000/MWh")
m6.metric("Max Price",          f"${max_price:,.0f} /MWh")

st.markdown("&nbsp;")

# Band table
stats = compute_band_stats(prices)
render_band_table(stats)

st.divider()

# Time series charts
st.subheader("Price Band Trends Over Time")
chart_freq_label = st.radio(
    "X-axis interval",
    list(CHART_FREQS.keys()),
    horizontal=True,
    index=1,
)
chart_freq = CHART_FREQS[chart_freq_label]

freq_df, avg_df = compute_timeseries(prices, chart_freq)

if freq_df.empty:
    st.info("Not enough data to plot trends at this interval. Try a wider timeframe or smaller interval.")
else:
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Band Frequency** — % of hours in each band per period")
        render_frequency_chart(freq_df)
    with col_right:
        st.markdown("**Average Band Price** — avg $/MWh within each band per period")
        render_avg_price_chart(avg_df)

st.markdown("&nbsp;")
st.caption(
    f"Source: Open Electricity API · 1-hour intervals · "
    f"{len(prices):,} data points · {date_start:%d %b %Y} – {date_end:%d %b %Y}"
)
