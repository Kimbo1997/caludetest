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

REGION_COLORS = {
    "NSW": "#2563eb",
    "VIC": "#7c3aed",
    "QLD": "#b45309",
    "SA":  "#dc2626",
    "TAS": "#059669",
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

    if "interval" in df.columns:
        df = df.set_index("interval")
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
    progress = st.progress(0, text=f"Fetching {region_code}… (0 / {len(chunks)} chunks)")

    with OEClient(api_key=api_key) as client:
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            for attempt in range(3):
                try:
                    series_list.append(_fetch_chunk(client, region_code, chunk_start, chunk_end))
                    break
                except Exception as exc:
                    if "429" in str(exc) and attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise
            progress.progress(
                (i + 1) / len(chunks),
                text=f"Fetching {region_code}… ({i + 1} / {len(chunks)} chunks)",
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


def compute_timeseries(
    prices: pd.Series, freq: str
) -> tuple[pd.DataFrame, pd.DataFrame, "date | None"]:
    """Resample prices by freq; return (% in band df, avg price in band df, last complete date)."""
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

    freq_rows = freq_rows[:-1]
    avg_rows  = avg_rows[:-1]

    def _to_df(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("date")

    freq_df  = _to_df(freq_rows)
    avg_df   = _to_df(avg_rows)
    last_date = freq_df.index[-1].date() if not freq_df.empty else None
    return freq_df, avg_df, last_date


def compute_threshold_timeseries(
    prices: pd.Series, threshold: float, freq: str
) -> tuple[pd.DataFrame, "date | None"]:
    """Return % hours below/above threshold per period, dropping the trailing incomplete period."""
    rows: list[dict] = []
    for period, group in prices.groupby(pd.Grouper(freq=freq)):
        if len(group) == 0:
            continue
        total = len(group)
        below = float((group < threshold).sum() / total * 100)
        rows.append({"date": period, "below": below, "above": 100.0 - below})

    rows = rows[:-1]
    if not rows:
        return pd.DataFrame(), None

    df = pd.DataFrame(rows).set_index("date")
    last_date = df.index[-1].date()
    return df, last_date


# ── HTML band table ────────────────────────────────────────────────────────────

_CSS = """
<style>
.pbt-wrap { overflow-x: auto; }
.pbt {
  width: 100%;
  border-collapse: collapse;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 13px;
}
.pbt th {
  text-align: left;
  padding: 6px 10px;
  font-size: 10px;
  font-weight: 600;
  color: #9ca3af;
  text-transform: uppercase;
  letter-spacing: .06em;
  border-bottom: 2px solid #e5e7eb;
  white-space: nowrap;
}
.pbt td {
  padding: 9px 10px;
  border-bottom: 1px solid #f3f4f6;
  vertical-align: middle;
}
.pbt tr:last-child td { border-bottom: none; }
.pbt tbody tr:hover td { background: #fafafa; }
.badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 30px; height: 20px; border-radius: 4px;
  font-size: 10px; font-weight: 700; color: #fff; letter-spacing: .03em;
}
.bar-wrap { background: #f3f4f6; border-radius: 3px; height: 12px; width: 120px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 3px; }
.mono  { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 11.5px; }
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
          <td class="mono">{band.range_label}</td>
          <td class="strong">{_fmt_avg(s['avg'])}</td>
          <td class="dim">{_fmt_hours(s['hours'])}</td>
          <td class="strong" style="color:{band.color}">{s['pct']:.1f}%</td>
          <td>
            <div class="bar-wrap">
              <div class="bar-fill" style="width:{bar_w}%;background:{band.color}"></div>
            </div>
          </td>
        </tr>"""

    st.html(f"""
        <div class="pbt-wrap">
        <table class="pbt">
          <thead><tr>
            <th>Band</th><th>Name</th><th>Range</th>
            <th>Avg Price</th><th>Hours</th><th>%</th><th>Distribution</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        </div>
    """)


# ── Plotly charts ──────────────────────────────────────────────────────────────

_CHART_BASE = dict(
    margin=dict(t=30, b=60, l=60, r=20),
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    xaxis=dict(showgrid=False, showline=True, linecolor="#e5e7eb"),
)

_YAXIS_BASE = dict(
    showgrid=True, gridcolor="#f0f0f0",
    showline=False, zeroline=True, zerolinecolor="#e5e7eb",
)


def _add_data_through_annotation(fig: go.Figure, last_date: "date | None") -> None:
    if last_date is None:
        return
    fig.add_annotation(
        text=f"Data through {last_date:%d %b %Y}",
        xref="paper", yref="paper", x=1, y=1,
        xanchor="right", yanchor="bottom",
        showarrow=False, font=dict(size=11, color="#9ca3af"),
    )


def render_spot_price_chart(all_prices: dict[str, pd.Series], use_log: bool) -> None:
    fig = go.Figure()
    for rname, prices in all_prices.items():
        fig.add_trace(go.Scatter(
            x=prices.index, y=prices.values,
            name=rname,
            line=dict(color=REGION_COLORS.get(rname, "#6b7280"), width=1),
            mode="lines",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        ))
    yaxis_type = "log" if use_log else "linear"
    tick_fmt = dict(tickformat="$,.0f") if use_log else dict(tickprefix="$")
    fig.update_layout(
        **_CHART_BASE,
        height=300,
        yaxis=dict(**_YAXIS_BASE, title="Spot Price ($/MWh)", type=yaxis_type, **tick_fmt),
    )
    if use_log:
        st.caption("ℹ Negative prices hidden in log scale — switch to linear to see them")
    st.plotly_chart(fig, use_container_width=True)


def render_frequency_chart(freq_df: pd.DataFrame, last_date: "date | None") -> None:
    fig = go.Figure()
    for band in STORAGE_BANDS:
        if band.id not in freq_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=freq_df.index, y=freq_df[band.id],
            name=f"{band.id} · {band.name}",
            line=dict(color=band.color, width=2),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate="%{y:.0f}%<extra></extra>",
        ))
    fig.update_layout(
        **_CHART_BASE,
        height=380,
        yaxis=dict(**_YAXIS_BASE, title="% of Hours", ticksuffix="%", rangemode="tozero"),
    )
    _add_data_through_annotation(fig, last_date)
    st.plotly_chart(fig, use_container_width=True)


def render_avg_price_chart(
    avg_df: pd.DataFrame,
    last_date: "date | None",
    use_log: bool = True,
    y_cap: float | None = None,
) -> None:
    fig = go.Figure()
    for band in STORAGE_BANDS:
        if band.id not in avg_df.columns or avg_df[band.id].isna().all():
            continue
        actual = avg_df[band.id]
        # B1 has negative averages — on log scale plot abs() with dashed line
        y_vals = actual.abs() if (use_log and band.id == "B1") else actual
        dash = "dash" if (use_log and band.id == "B1") else "solid"
        fig.add_trace(go.Scatter(
            x=avg_df.index, y=y_vals,
            customdata=actual,
            name=f"{band.id} · {band.name}",
            line=dict(color=band.color, width=2, dash=dash),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate="$%{customdata:,.0f}<extra></extra>",
        ))
    yaxis_type = "log" if use_log else "linear"
    tick_fmt = dict(tickformat="$,.0f") if use_log else dict(tickprefix="$")
    y_range = None
    if not use_log and y_cap is not None:
        y_range = [None, y_cap]
    fig.update_layout(
        **_CHART_BASE,
        height=380,
        yaxis=dict(
            **_YAXIS_BASE,
            title="Avg Price ($/MWh)",
            type=yaxis_type,
            range=y_range,
            **tick_fmt,
        ),
    )
    _add_data_through_annotation(fig, last_date)
    if use_log:
        st.caption("ⓘ B1 · Negative band shown as |value|, dashed — hover to see actual price")
    st.plotly_chart(fig, use_container_width=True)


def render_threshold_chart(
    df: pd.DataFrame, threshold: float, last_date: "date | None"
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["below"],
        name=f"Below ${threshold:,.0f}",
        stackgroup="one",
        fillcolor="rgba(22, 163, 74, 0.75)",
        line=dict(width=0),
        hovertemplate="%{y:.0f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["above"],
        name=f"Above ${threshold:,.0f}",
        stackgroup="one",
        fillcolor="rgba(220, 38, 38, 0.75)",
        line=dict(width=0),
        hovertemplate="%{y:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        **_CHART_BASE,
        height=380,
        yaxis=dict(**_YAXIS_BASE, title="% of Hours", ticksuffix="%", range=[0, 100]),
    )
    _add_data_through_annotation(fig, last_date)
    st.plotly_chart(fig, use_container_width=True)


# ── Page layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NEM Price Bands", page_icon="⚡", layout="wide")
st.markdown(_CSS, unsafe_allow_html=True)

st.title("⚡ NEM Price Band Analyser")
st.caption("Spot price distribution across hourly intervals — optimised for utility-scale storage.")

# Controls row
c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
with c1:
    region_names: list[str] = st.multiselect(
        "Regions", list(REGIONS.keys()), default=["NSW"]
    )
with c2:
    preset = st.selectbox("Timeframe", list(PRESETS.keys()), index=0)

if not region_names:
    st.warning("Select at least one region to continue.")
    st.stop()

if preset == "Custom":
    with c3:
        date_start = st.date_input("From", PRESETS["Custom"][0])
    with c4:
        date_end = st.date_input("To", PRESETS["Custom"][1])
else:
    date_start, date_end = PRESETS[preset]
    date_end = min(date_end, _today)

st.divider()

# ── Fetch all selected regions ─────────────────────────────────────────────────

all_prices: dict[str, pd.Series] = {}
for rname in region_names:
    try:
        p = fetch_prices(REGIONS[rname], date_start, date_end)
        if p.empty:
            st.warning(f"No data returned for {rname}.")
        else:
            all_prices[rname] = p
    except Exception as exc:
        msg = str(exc)
        if "403" in msg:
            st.error(
                f"API key rejected for {rname} (403). Check OPENELECTRICITY_API_KEY in .env "
                "and that your account is active at platform.openelectricity.org.au"
            )
        else:
            st.error(f"API error fetching {rname}: {msg}")

if not all_prices:
    st.stop()

# ── Summary metrics (compact, per region) ──────────────────────────────────────

metric_cols = st.columns(len(all_prices))
for col, (rname, prices) in zip(metric_cols, all_prices.items()):
    with col:
        st.markdown(f"**{rname}**")
        ma, mb, mc = st.columns(3)
        ma.metric("Avg Price",    f"${prices.mean():,.0f}/MWh")
        mb.metric("Negative",     f"{(prices < 0).mean()*100:.1f}%")
        mc.metric("High (≥$300)", f"{(prices >= 300).mean()*100:.1f}%")

st.markdown("&nbsp;")

# ── Band tables ────────────────────────────────────────────────────────────────

table_cols = st.columns(len(all_prices))
for col, (rname, prices) in zip(table_cols, all_prices.items()):
    with col:
        st.markdown(f"##### {rname} — Band Distribution")
        render_band_table(compute_band_stats(prices))

st.divider()

# ── Spot price chart ───────────────────────────────────────────────────────────

st.subheader("Spot Price")
sc1, sc2 = st.columns([4, 1])
with sc2:
    use_log_spot = st.checkbox("Log scale", value=False, key="log_spot")
render_spot_price_chart(all_prices, use_log_spot)

st.divider()

# ── Price Band Trends ──────────────────────────────────────────────────────────

st.subheader("Price Band Trends Over Time")

tr1, tr2, tr3 = st.columns([3, 1, 1])
with tr1:
    chart_freq_label = st.radio(
        "X-axis interval", list(CHART_FREQS.keys()), horizontal=True, index=1
    )
    chart_freq = CHART_FREQS[chart_freq_label]
with tr2:
    use_log_avg = st.checkbox("Log scale (avg price)", value=True, key="log_avg")
with tr3:
    if not use_log_avg:
        y_cap = float(st.number_input("Cap Y-axis ($/MWh)", value=2000, step=500, min_value=100, key="y_cap"))
    else:
        y_cap = None

for rname, prices in all_prices.items():
    st.markdown(f"##### {rname}")
    freq_df, avg_df, last_date = compute_timeseries(prices, chart_freq)
    if freq_df.empty:
        st.info(f"Not enough data for {rname} at this interval.")
        continue
    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Band Frequency** — % of hours in each band per period")
        render_frequency_chart(freq_df, last_date)
    with cr:
        st.markdown("**Average Band Price** — avg $/MWh within each band per period")
        render_avg_price_chart(avg_df, last_date, use_log_avg, y_cap)

st.divider()

# ── Above / Below Threshold ────────────────────────────────────────────────────

st.subheader("Above / Below Price Threshold")
th1, _ = st.columns([1, 3])
with th1:
    threshold = float(st.number_input(
        "Threshold price ($/MWh)", value=150, step=50, min_value=-1000, max_value=20000
    ))

for rname, prices in all_prices.items():
    st.markdown(f"##### {rname}")
    thr_df, thr_last_date = compute_threshold_timeseries(prices, threshold, chart_freq)
    if thr_df.empty:
        st.info(f"Not enough data for {rname} at this interval.")
    else:
        render_threshold_chart(thr_df, threshold, thr_last_date)

st.markdown("&nbsp;")
st.caption(
    f"Source: Open Electricity API · 1-hour intervals · "
    f"{date_start:%d %b %Y} – {date_end:%d %b %Y} · "
    f"Regions: {', '.join(all_prices.keys())}"
)
