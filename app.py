import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

US_ISOS: dict[str, dict] = {
    # cls: gridstatus class name  call: dispatch strategy  market: market string (None = not used)
    "CAISO":  {"cls": "CAISO",  "call": "std",    "market": "DAY_AHEAD_HOURLY"},
    "ERCOT":  {"cls": "Ercot",  "call": "ercot",  "market": None},            # no market param; class is Ercot
    "NYISO":  {"cls": "NYISO",  "call": "std",    "market": "REAL_TIME_HOURLY"},
    "PJM":    {"cls": "PJM",    "call": "std",    "market": "REAL_TIME_HOURLY"},  # needs PJM_API_KEY
    "MISO":   {"cls": "MISO",   "call": "std",    "market": "REAL_TIME_HOURLY_FINAL"},
    "ISO-NE": {"cls": "ISONE",  "call": "std",    "market": "REAL_TIME_HOURLY"},
    "SPP":    {"cls": "SPP",    "call": "spp_da", "market": None},            # no get_lmp(); uses get_lmp_day_ahead_hourly()
}

US_COLORS: dict[str, str] = {
    "CAISO":  "#2563eb",
    "ERCOT":  "#dc2626",
    "NYISO":  "#7c3aed",
    "PJM":    "#d97706",
    "MISO":   "#059669",
    "ISO-NE": "#0891b2",
    "SPP":    "#b45309",
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


# ── Data fetching — Australia ──────────────────────────────────────────────────

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
    """Fetch hourly NEM spot prices, chunked into 30-day windows."""
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


# ── Data fetching — USA ────────────────────────────────────────────────────────

def _iso_call_lmp(iso_key: str, iso, cs: date, ce: date):
    """Dispatch the correct get_lmp call per ISO — APIs vary significantly."""
    cfg = US_ISOS[iso_key]
    date_str = cs.isoformat()
    end_str  = (ce + timedelta(days=1)).isoformat()
    call = cfg["call"]

    if call == "ercot":
        # ERCOT: no market param; location_type filters to settlement points
        return iso.get_lmp(date=date_str, end=end_str, location_type="Settlement Point", verbose=False)
    elif call == "spp_da":
        # SPP: no get_lmp(); use the day-ahead hourly specific method
        return iso.get_lmp_day_ahead_hourly(date=date_str, end=end_str, verbose=False)
    else:
        # Standard: get_lmp with market string, no location_type
        return iso.get_lmp(date=date_str, end=end_str, market=cfg["market"], verbose=False)


def _lmp_df_to_hourly_series(iso_key: str, df) -> pd.Series:
    """Normalise a gridstatus LMP DataFrame to an hourly pd.Series (average across locations)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Time column priority: gridstatus consistently uses "Interval Start"
    time_col = next(
        (c for c in ["Interval Start", "Time", "SCED Timestamp"] if c in df.columns),
        None,
    )
    if time_col is None:
        time_col = next(
            (c for c in df.columns if "interval" in c.lower() or "time" in c.lower()),
            df.columns[0],
        )

    # LMP column
    price_col = "LMP" if "LMP" in df.columns else next(
        (c for c in df.columns if "lmp" in c.lower()), None
    )
    if price_col is None:
        raise ValueError(f"No LMP column in {iso_key} response. Columns: {list(df.columns)}")

    # Build series — multiple rows per timestamp (one per location); resample averages them
    s = df.set_index(time_col)[price_col].copy()
    s.index = pd.to_datetime(s.index, utc=True)
    return s.resample("1h").mean().dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices_gridstatus(iso_key: str, date_start: date, date_end: date) -> pd.Series:
    """Fetch hourly LMP prices for a US ISO via gridstatus, averaged across all returned locations."""
    import gridstatus as gs

    # PJM requires a free API key — give a clear message if missing
    if iso_key == "PJM" and not os.environ.get("PJM_API_KEY"):
        raise ValueError(
            "PJM requires a free API key. Register at dataminer2.pjm.com → "
            "sign in → My Account → API Keys, then add PJM_API_KEY to your .env file."
        )

    cfg = US_ISOS[iso_key]

    chunks: list[tuple[date, date]] = []
    cursor = date_start
    while cursor <= date_end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), date_end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)

    progress = st.progress(0, text=f"Fetching {iso_key}… (0 / {len(chunks)} chunks)")
    results: dict[int, pd.Series] = {}
    completed_count = 0
    lock = threading.Lock()

    def _fetch_one(idx: int, cs: date, ce: date) -> None:
        nonlocal completed_count
        # Create a fresh ISO instance per thread — gridstatus objects are not thread-safe
        iso = getattr(gs, cfg["cls"])()
        for attempt in range(3):
            try:
                df = _iso_call_lmp(iso_key, iso, cs, ce)
                series = _lmp_df_to_hourly_series(iso_key, df)
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
        with lock:
            results[idx] = series
            completed_count += 1
            progress.progress(
                completed_count / len(chunks),
                text=f"Fetching {iso_key}… ({completed_count} / {len(chunks)} chunks)",
            )

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_fetch_one, i, cs, ce) for i, (cs, ce) in enumerate(chunks)]
        for f in as_completed(futures):
            f.result()  # re-raise any exception from the worker thread

    progress.empty()

    if not results:
        return pd.Series(dtype=float)

    combined = pd.concat([results[i] for i in range(len(chunks))])
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


def compute_spot_timeseries(prices: pd.Series, freq: str) -> tuple[pd.Series, "date | None"]:
    resampled = prices.resample(freq).mean().dropna()
    if len(resampled) <= 1:
        return pd.Series(dtype=float), None
    resampled = resampled.iloc[:-1]
    last_date = resampled.index[-1].date()
    return resampled, last_date


def compute_threshold_timeseries(
    prices: pd.Series, threshold: float, freq: str
) -> tuple[pd.DataFrame, "date | None"]:
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


# ── HTML components ────────────────────────────────────────────────────────────

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

@media (prefers-color-scheme: dark) {
  .pbt th { border-bottom: 2px solid #374151; color: #6b7280; }
  .pbt td { border-bottom: 1px solid #1f2937; }
  .pbt tbody tr:hover td { background: #111827; }
  .bar-wrap { background: #374151; }
  .dim { color: #6b7280; }
}
</style>
"""


def _fmt_avg(avg: float | None, currency: str = "$") -> str:
    if avg is None:
        return "–"
    return f"-{currency}{abs(avg):,.0f}" if avg < 0 else f"{currency}{avg:,.0f}"


def _fmt_band_range(band, currency: str = "$") -> str:
    if band.min_price is None:
        return f"< {currency}0"
    elif band.max_price is None:
        return f"> {currency}{band.min_price:,.0f}"
    return f"{currency}{band.min_price:,.0f} – {currency}{band.max_price:,.0f}"


def _fmt_hours(hours: float) -> str:
    h, m = int(hours), int((hours % 1) * 60)
    return f"{h:,}h {m:02d}m"


def _metric_card(label: str, value: str, accent: str) -> str:
    return (
        f'<div style="padding:12px 14px;border-radius:8px;border-left:3px solid {accent};'
        f'background:rgba(0,0,0,0.03);margin-bottom:4px">'
        f'<div style="font-size:10px;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:.06em;color:#9ca3af;margin-bottom:4px">{label}</div>'
        f'<div style="font-size:20px;font-weight:700">{value}</div>'
        f"</div>"
    )


def _region_pill(rname: str, color_map: dict) -> None:
    color = color_map.get(rname, "#6b7280")
    st.markdown(
        f'<span style="display:inline-block;padding:4px 12px;border-radius:12px;'
        f'background:{color};color:#fff;font-size:13px;font-weight:600;'
        f'letter-spacing:.04em;margin-bottom:6px">{rname}</span>',
        unsafe_allow_html=True,
    )


def render_band_table(stats: list[dict], currency: str = "$") -> None:
    max_pct = max(s["pct"] for s in stats) or 1.0
    rows = ""
    for s in stats:
        band = s["band"]
        bar_w = int(s["pct"] / max_pct * 100)
        rows += f"""
        <tr>
          <td><span class="badge" style="background:{band.color}">{band.id}</span></td>
          <td class="mono">{_fmt_band_range(band, currency)}</td>
          <td class="strong">{_fmt_avg(s['avg'], currency)}</td>
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
            <th>Band</th><th>Range</th>
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
    xaxis=dict(
        showgrid=False, showline=True, linecolor="#e5e7eb",
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#9ca3af", spikethickness=1, spikedash="dot",
        tickangle=0,
    ),
)

_YAXIS_BASE = dict(
    showgrid=True, gridcolor="#f0f0f0",
    showline=False, zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=2,
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


def render_spot_price_chart(
    all_prices: dict[str, pd.Series], freq: str, use_log: bool,
    color_map: dict, currency: str = "$",
) -> None:
    fig = go.Figure()
    last_date = None
    for rname, prices in all_prices.items():
        resampled, ld = compute_spot_timeseries(prices, freq)
        if resampled.empty:
            continue
        if last_date is None:
            last_date = ld
        fig.add_trace(go.Scatter(
            x=resampled.index, y=resampled.values,
            name=rname,
            line=dict(color=color_map.get(rname, "#6b7280"), width=2),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate=f"{currency}%{{y:,.0f}}<extra></extra>",
        ))
    yaxis_type = "log" if use_log else "linear"
    tick_fmt = dict(tickformat=f"{currency},.0f") if use_log else dict(tickprefix=currency)
    fig.update_layout(
        **_CHART_BASE,
        height=340,
        yaxis=dict(**_YAXIS_BASE, title=f"Avg Spot Price ({currency}/MWh)", type=yaxis_type, **tick_fmt),
    )
    _add_data_through_annotation(fig, last_date)
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
            name=band.id,
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
    currency: str = "$",
) -> None:
    fig = go.Figure()
    for band in STORAGE_BANDS:
        if band.id not in avg_df.columns or avg_df[band.id].isna().all():
            continue
        actual = avg_df[band.id]
        y_vals = actual.abs() if (use_log and band.id == "B1") else actual
        dash = "dash" if (use_log and band.id == "B1") else "solid"
        fig.add_trace(go.Scatter(
            x=avg_df.index, y=y_vals,
            customdata=actual,
            name=band.id,
            line=dict(color=band.color, width=2, dash=dash),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate=f"{currency}%{{customdata:,.0f}}<extra></extra>",
        ))
    yaxis_type = "log" if use_log else "linear"
    tick_fmt = dict(tickformat=f"{currency},.0f") if use_log else dict(tickprefix=currency)
    y_range = None
    if not use_log and y_cap is not None:
        y_range = [None, y_cap]
    fig.update_layout(
        **_CHART_BASE,
        height=380,
        yaxis=dict(
            **_YAXIS_BASE,
            title=f"Avg Price ({currency}/MWh)",
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
    df: pd.DataFrame, threshold: float, last_date: "date | None", currency: str = "$"
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["below"],
        name=f"Below {currency}{threshold:,.0f}",
        stackgroup="one",
        fillcolor="rgba(22, 163, 74, 0.75)",
        line=dict(width=0),
        hovertemplate="%{y:.0f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["above"],
        name=f"Above {currency}{threshold:,.0f}",
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

# ── Sidebar controls ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚡ Price Band Analyser")

    market = st.radio("Market", ["Australia 🇦🇺", "USA 🇺🇸"], horizontal=True)
    is_australia = market.startswith("Australia")

    st.subheader("Data")
    if is_australia:
        region_names: list[str] = st.multiselect(
            "Regions", list(REGIONS.keys()), default=["NSW"]
        )
        currency = "$"
        color_map = REGION_COLORS
        data_label = "NEM · Real-time spot price"
    else:
        region_names = st.multiselect(
            "ISOs", list(US_ISOS.keys()), default=["CAISO"]
        )
        currency = "$"
        color_map = US_COLORS
        data_label = "Real-time hourly LMP"

    preset = st.selectbox("Timeframe", list(PRESETS.keys()), index=3)

    if preset == "Custom":
        date_start = st.date_input("From", PRESETS["Custom"][0])
        date_end   = st.date_input("To",   PRESETS["Custom"][1])
    else:
        date_start, date_end = PRESETS[preset]
        date_end = min(date_end, _today)

    st.divider()
    st.subheader("Chart Options")
    chart_freq_label = st.radio("X-axis interval", list(CHART_FREQS.keys()), index=1)
    chart_freq = CHART_FREQS[chart_freq_label]

    st.divider()
    st.subheader("Spot Price")
    use_log_spot = st.checkbox("Logarithmic scale", value=False, key="log_spot")

    st.divider()
    st.subheader("Avg Band Price")
    use_log_avg = st.checkbox("Logarithmic scale", value=True, key="log_avg")
    if not use_log_avg:
        y_cap = float(st.number_input("Cap Y-axis ($/MWh)", value=2000, step=500, min_value=100, key="y_cap"))
    else:
        y_cap = None

    st.divider()
    st.subheader("Price Threshold")
    threshold = float(st.number_input(
        "Threshold ($/MWh)", value=150, step=50, min_value=-1000, max_value=20000
    ))

if not region_names:
    st.warning("Select at least one region in the sidebar to continue.")
    st.stop()

# ── Main content ───────────────────────────────────────────────────────────────

st.title("Price Band Analyser")
st.caption(
    f"{data_label} · 1-hour intervals · "
    f"{date_start:%d %b %Y} – {date_end:%d %b %Y} · "
    f"{', '.join(region_names)}"
)

st.divider()

# ── Fetch all selected regions ─────────────────────────────────────────────────

all_prices: dict[str, pd.Series] = {}
for rname in region_names:
    try:
        if is_australia:
            p = fetch_prices(REGIONS[rname], date_start, date_end)
        else:
            p = fetch_prices_gridstatus(rname, date_start, date_end)

        if p.empty:
            st.warning(f"No data returned for {rname}.")
        else:
            all_prices[rname] = p

    except Exception as exc:
        msg = str(exc)
        if "ERCOT" in rname and ("401" in msg or "403" in msg or "key" in msg.lower()):
            st.error(
                "ERCOT requires a free API key. Register at https://apiexplorer.ercot.com → "
                "Products → Public API → Subscribe, then add ERCOT_API_KEY to your .env file."
            )
        elif "not found" in msg.lower() or "hub" in msg.lower():
            st.error(f"{rname}: {msg}")
        elif "403" in msg:
            st.error(
                f"API key rejected for {rname} (403). Check your credentials in .env."
            )
        else:
            # Include the exception type so blank-message errors are still identifiable
            err_detail = msg or f"({type(exc).__name__} — no message)"
            st.error(f"Error fetching {rname}: {err_detail}")

if not all_prices:
    st.stop()

# ── Summary metrics ────────────────────────────────────────────────────────────

metric_cols = st.columns(len(all_prices))
for col, (rname, prices) in zip(metric_cols, all_prices.items()):
    with col:
        avg_price = prices.mean()
        neg_pct   = (prices < 0).mean() * 100
        high_pct  = (prices >= 300).mean() * 100

        avg_color  = "#16a34a" if avg_price < 0 else "#2563eb"
        neg_color  = "#16a34a" if neg_pct > 5 else "#6b7280"
        high_color = "#dc2626" if high_pct > 10 else "#6b7280"

        avg_str = f"-{currency}{abs(avg_price):,.0f}" if avg_price < 0 else f"{currency}{avg_price:,.0f}"

        _region_pill(rname, color_map)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.html(_metric_card("Avg Price", f"{avg_str}/MWh", avg_color))
        with mc2:
            st.html(_metric_card("Negative", f"{neg_pct:.1f}%", neg_color))
        with mc3:
            st.html(_metric_card("High ≥$300", f"{high_pct:.1f}%", high_color))

st.divider()

# ── Band tables ────────────────────────────────────────────────────────────────

st.subheader("Band Distribution")
table_cols = st.columns(len(all_prices))
for col, (rname, prices) in zip(table_cols, all_prices.items()):
    with col:
        _region_pill(rname, color_map)
        render_band_table(compute_band_stats(prices), currency)

st.divider()

# ── Spot price chart ───────────────────────────────────────────────────────────

st.subheader("Spot Price")
render_spot_price_chart(all_prices, chart_freq, use_log_spot, color_map, currency)

st.divider()

# ── Price Band Trends ──────────────────────────────────────────────────────────

st.subheader("Price Band Trends Over Time")

for rname, prices in all_prices.items():
    _region_pill(rname, color_map)
    freq_df, avg_df, last_date = compute_timeseries(prices, chart_freq)
    if freq_df.empty:
        st.info(f"Not enough data for {rname} at this interval.")
        continue
    cl, cr = st.columns(2)
    with cl:
        st.caption("Band Frequency — % of hours in each band per period")
        render_frequency_chart(freq_df, last_date)
    with cr:
        st.caption(f"Average Band Price — avg {currency}/MWh within each band per period")
        render_avg_price_chart(avg_df, last_date, use_log_avg, y_cap, currency)

st.divider()

# ── Above / Below Threshold ────────────────────────────────────────────────────

st.subheader("Above / Below Price Threshold")
st.caption(f"Threshold set to {currency}{threshold:,.0f}/MWh — adjust in the sidebar")

for rname, prices in all_prices.items():
    _region_pill(rname, color_map)
    below_pct = float((prices < threshold).mean() * 100)
    mc1, mc2 = st.columns(2)
    with mc1:
        st.html(_metric_card(
            f"Avg below {currency}{threshold:,.0f}/MWh",
            f"{below_pct:.1f}% of hours",
            "#16a34a",
        ))
    with mc2:
        st.html(_metric_card(
            f"Avg above {currency}{threshold:,.0f}/MWh",
            f"{100 - below_pct:.1f}% of hours",
            "#dc2626",
        ))
    thr_df, thr_last_date = compute_threshold_timeseries(prices, threshold, chart_freq)
    if thr_df.empty:
        st.info(f"Not enough data for {rname} at this interval.")
    else:
        render_threshold_chart(thr_df, threshold, thr_last_date, currency)

st.caption(
    f"Source: {'Open Electricity API' if is_australia else 'gridstatus / ISO public data'} · "
    f"1-hour intervals · {date_start:%d %b %Y} – {date_end:%d %b %Y}"
)
