import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
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
    "CAISO":  {"cls": "CAISO",  "call": "std",    "market": "REAL_TIME_15_MIN"},
    "ERCOT":  {"cls": "Ercot",  "call": "ercot",  "market": None},
    "NYISO":  {"cls": "NYISO",  "call": "std",    "market": "REAL_TIME_HOURLY"},
    "PJM":    {"cls": "PJM",    "call": "std",    "market": "REAL_TIME_HOURLY"},
    "MISO":   {"cls": "MISO",   "call": "std",    "market": "REAL_TIME_HOURLY_FINAL"},
    "ISO-NE": {"cls": "ISONE",  "call": "std",    "market": "REAL_TIME_HOURLY"},
    "SPP":    {"cls": "SPP",    "call": "spp_rt", "market": None},
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

EU_COUNTRIES: dict[str, str] = {
    "Germany/Lux": "DE_LU",
    "France":      "FR",
    "Spain":       "ES",
    "Netherlands": "NL",
    "Belgium":     "BE",
    "Austria":     "AT",
    "Italy":       "IT_NORD",
    "Sweden":      "SE_3",
    "Norway":      "NO_2",
    "Denmark":     "DK_1",
    "Finland":     "FI",
    "Poland":      "PL",
    "Portugal":    "PT",
    "Czech Rep.":  "CZ",
}

EU_COLORS: dict[str, str] = {
    "Germany/Lux": "#2563eb",
    "France":      "#dc2626",
    "Spain":       "#d97706",
    "Netherlands": "#059669",
    "Belgium":     "#7c3aed",
    "Austria":     "#0891b2",
    "Italy":       "#16a34a",
    "Sweden":      "#1d4ed8",
    "Norway":      "#9333ea",
    "Denmark":     "#ef4444",
    "Finland":     "#0ea5e9",
    "Poland":      "#b45309",
    "Portugal":    "#10b981",
    "Czech Rep.":  "#f59e0b",
}

_today = date.today()

PRESETS: dict[str, tuple[date, date]] = {
    "Last 24 hours":      (_today - timedelta(days=1),   _today),
    "Last 7 days":        (_today - timedelta(days=7),   _today),
    "Last 30 days":       (_today - timedelta(days=30),  _today),
    "Last 90 days":       (_today - timedelta(days=90),  _today),
    "Last 12 months":     (_today - timedelta(days=365), _today),
    "Full Year 2025":     (date(2025, 1, 1),            date(2025, 12, 31)),
    "H1 2025 (Jan–Jun)":  (date(2025, 1, 1),            date(2025, 6, 30)),
    "H2 2025 (Jul–Dec)":  (date(2025, 7, 1),            date(2025, 12, 31)),
    "Custom":             (date(2025, 1, 1),             _today),
}

INTERVAL_MINUTES = 60
CHUNK_DAYS = 30
CHUNK_DAYS_ERCOT = 7
CHUNK_DAYS_SPP = 1
CHUNK_DAYS_EU = 90

CHART_FREQS: dict[str, str] = {
    "1 Hour":    "1h",
    "5 Hours":   "5h",
    "1 Day":     "D",
    "1 Week":    "W",
    "1 Month":   "ME",
    "1 Quarter": "QE",
    "1 Year":    "YE",
}

_FREQ_TICK: dict[str, dict] = {
    "1h": dict(nticks=12, tickformat="%d %b %H:%M"),
    "5h": dict(nticks=12, tickformat="%d %b %H:%M"),
    "D":  dict(nticks=14, tickformat="%d %b"),
}

# Per-region IANA timezone (None = AU data already in AEST-naive, no conversion)
_REGION_TZ: dict[str, str | None] = {
    "NSW": None, "VIC": None, "QLD": None, "SA": None, "TAS": None,
    "CAISO":  "America/Los_Angeles",
    "ERCOT":  "America/Chicago",
    "NYISO":  "America/New_York",
    "PJM":    "America/New_York",
    "MISO":   "America/Chicago",
    "ISO-NE": "America/New_York",
    "SPP":    "America/Chicago",
    "Germany/Lux": "Europe/Berlin",
    "France":      "Europe/Paris",
    "Spain":       "Europe/Madrid",
    "Netherlands": "Europe/Amsterdam",
    "Belgium":     "Europe/Brussels",
    "Austria":     "Europe/Vienna",
    "Italy":       "Europe/Rome",
    "Sweden":      "Europe/Stockholm",
    "Norway":      "Europe/Oslo",
    "Denmark":     "Europe/Copenhagen",
    "Finland":     "Europe/Helsinki",
    "Poland":      "Europe/Warsaw",
    "Portugal":    "Europe/Lisbon",
    "Czech Rep.":  "Europe/Prague",
}

_REGION_TZ_ABBR: dict[str, str] = {
    "NSW": "AEST", "VIC": "AEST", "QLD": "AEST", "SA": "ACST", "TAS": "AEST",
    "CAISO": "PT", "ERCOT": "CT", "NYISO": "ET", "PJM": "ET",
    "MISO": "CT", "ISO-NE": "ET", "SPP": "CT",
    "Germany/Lux": "CET", "France": "CET", "Spain": "CET",
    "Netherlands": "CET", "Belgium": "CET", "Austria": "CET",
    "Italy": "CET", "Sweden": "CET", "Norway": "CET",
    "Denmark": "CET", "Finland": "EET", "Poland": "CET",
    "Portugal": "WET", "Czech Rep.": "CET",
}

# Unified region registry — market: "australia" | "usa" | "europe"
ALL_REGIONS: dict[str, dict] = {}
for _k, _v in REGIONS.items():
    ALL_REGIONS[_k] = {"market": "australia", "code": _v,
                       "color": REGION_COLORS[_k], "currency": "$", "price_type": "spot"}
for _k in US_ISOS:
    ALL_REGIONS[_k] = {"market": "usa", "code": _k,
                       "color": US_COLORS[_k], "currency": "$", "price_type": "spot"}
for _k, _v in EU_COUNTRIES.items():
    ALL_REGIONS[_k] = {"market": "europe", "code": _v,
                       "color": EU_COLORS[_k], "currency": "€", "price_type": "day_ahead"}


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
    """Dispatch the correct call per ISO — APIs vary significantly."""
    cfg = US_ISOS[iso_key]
    date_str = cs.isoformat()
    end_str  = (ce + timedelta(days=1)).isoformat()
    call = cfg["call"]

    if call == "ercot":
        from gridstatus import Markets
        return iso.get_spp(
            date=date_str, end=end_str,
            market=Markets.REAL_TIME_15_MIN,
            verbose=False,
        )
    elif call == "spp_rt":
        # Fetch SPP daily file directly — bypasses gridstatus @support_date_range
        # decorator which crashes with ValueError when pd.concat([]) is called on
        # an empty list (when all daily files return 404 for recent/unavailable dates).
        from gridstatus import Markets
        url = (
            "https://portal.spp.org/file-browser-api/download/"
            f"rtbm-lmp-by-location?path=/{cs.strftime('%Y/%m')}"
            f"/By_Day/RTBM-LMP-DAILY-SL-{cs.strftime('%Y%m%d')}.csv"
        )
        try:
            df = pd.read_csv(url)
            df.columns = df.columns.str.strip()
            df = df.rename(columns={
                "GMT Interval": "GMTIntervalEnd",
                "Settlement Location Name": "Settlement Location",
                "PNODE Name": "PNode",
            })
            return iso._finalize_spp_df(
                df, market=Markets.REAL_TIME_5_MIN, location_type="Hub",
            )
        except Exception:
            return pd.DataFrame(columns=["Interval Start", "LMP"])
    else:
        return iso.get_lmp(date=date_str, end=end_str, market=cfg["market"], verbose=False)


def _lmp_df_to_hourly_series(iso_key: str, df) -> pd.Series:
    """Normalise a gridstatus LMP DataFrame to an hourly pd.Series (average across locations)."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    time_col = next(
        (c for c in ["Interval Start", "Time", "SCED Timestamp"] if c in df.columns),
        None,
    )
    if time_col is None:
        time_col = next(
            (c for c in df.columns if "interval" in c.lower() or "time" in c.lower()),
            df.columns[0],
        )

    # get_lmp returns "LMP"; get_spp (ERCOT) returns "SPP"; SPP RT also returns "LMP"
    price_col = next(
        (c for c in ["LMP", "SPP"] if c in df.columns),
        None,
    ) or next(
        (c for c in df.columns if "lmp" in c.lower() or "spp" in c.lower()),
        None,
    )
    if price_col is None:
        raise ValueError(f"No LMP/SPP column in {iso_key} response. Columns: {list(df.columns)}")

    s = df.set_index(time_col)[price_col].copy()
    s.index = pd.to_datetime(s.index, utc=True)
    return s.resample("1h").mean().dropna()


_CHUNK_TIMEOUT = 120  # seconds per chunk before giving up

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices_gridstatus(iso_key: str, date_start: date, date_end: date) -> pd.Series:
    """Fetch hourly LMP prices for a US ISO via gridstatus, averaged across all returned locations."""
    import gridstatus as gs
    for _log in ("gridstatus", "urllib3", "requests"):
        logging.getLogger(_log).setLevel(logging.WARNING)

    if iso_key == "PJM" and not os.environ.get("PJM_API_KEY"):
        raise ValueError(
            "PJM requires a free API key. Register at dataminer2.pjm.com → "
            "sign in → My Account → API Keys, then add PJM_API_KEY to your .env file."
        )

    cfg = US_ISOS[iso_key]
    if iso_key == "ERCOT":
        chunk_days = CHUNK_DAYS_ERCOT
    elif iso_key == "SPP":
        chunk_days = CHUNK_DAYS_SPP
    else:
        chunk_days = CHUNK_DAYS

    chunks: list[tuple[date, date]] = []
    cursor = date_start
    while cursor <= date_end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), date_end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)

    progress = st.progress(0, text=f"Fetching {iso_key}… (0 / {len(chunks)} chunks)")
    results: dict[int, pd.Series] = {}

    def _fetch_one(idx: int, cs: date, ce: date) -> None:
        if iso_key == "ERCOT":
            ercot_key = os.environ.get("ERCOT_API_KEY")
            try:
                iso = gs.Ercot(api_key=ercot_key) if ercot_key else gs.Ercot()
            except TypeError:
                iso = gs.Ercot()
        else:
            iso = getattr(gs, cfg["cls"])()
        for attempt in range(3):
            try:
                df = _iso_call_lmp(iso_key, iso, cs, ce)
                results[idx] = _lmp_df_to_hourly_series(iso_key, df)
                return
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

    max_workers = 5 if iso_key == "SPP" else 3
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_one, i, cs, ce) for i, (cs, ce) in enumerate(chunks)]
        for done, f in enumerate(as_completed(futures), start=1):
            try:
                f.result(timeout=_CHUNK_TIMEOUT)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"{iso_key} chunk fetch timed out after {_CHUNK_TIMEOUT}s. "
                    "Try a shorter period or a different ISO."
                )
            progress.progress(
                done / len(chunks),
                text=f"Fetching {iso_key}… ({done} / {len(chunks)} chunks)",
            )

    progress.empty()

    if not results:
        return pd.Series(dtype=float)

    combined = pd.concat([results[i] for i in range(len(chunks))])
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


# ── Data fetching — Europe ────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices_entsoe(area_code: str, date_start: date, date_end: date) -> pd.Series:
    """Fetch hourly day-ahead prices for a European bidding zone via ENTSO-E."""
    from entsoe import EntsoePandasClient

    api_key = os.environ.get("ENTSOE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ENTSO-E API key missing. Add ENTSOE_API_KEY to your .env file. "
            "Register at transparency.entsoe.eu → My Account Settings → Web API Security Token."
        )

    client = EntsoePandasClient(api_key=api_key)

    chunks: list[tuple[date, date]] = []
    cursor = date_start
    while cursor <= date_end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS_EU - 1), date_end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)

    progress = st.progress(0, text=f"Fetching {area_code}… (0 / {len(chunks)} chunks)")
    series_list: list[pd.Series] = []

    for i, (cs, ce) in enumerate(chunks):
        start_ts = pd.Timestamp(cs, tz="UTC")
        end_ts   = pd.Timestamp(ce + timedelta(days=1), tz="UTC")
        for attempt in range(3):
            try:
                s = client.query_day_ahead_prices(area_code, start=start_ts, end=end_ts)
                series_list.append(s)
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
        progress.progress(
            (i + 1) / len(chunks),
            text=f"Fetching {area_code}… ({i + 1} / {len(chunks)} chunks)",
        )

    progress.empty()

    if not series_list:
        return pd.Series(dtype=float)

    combined = pd.concat(series_list)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    combined = combined.resample("1h").mean().dropna()
    if combined.index.tz is not None:
        combined.index = combined.index.tz_convert("UTC").tz_localize(None)
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


def _band_label(band, currency: str = "$") -> str:
    return f"{band.id}: {_fmt_band_range(band, currency)}"


def _to_local_tz(prices: pd.Series, rname: str) -> pd.Series:
    """Convert UTC-indexed price series to local grid time (returned as naive index)."""
    tz = _REGION_TZ.get(rname)
    if tz is None or prices.empty:
        return prices  # AU: already AEST-naive; no conversion needed
    idx = prices.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return prices.set_axis(idx.tz_convert(tz).tz_localize(None))


def _get_tz_abbr(rname: str) -> str:
    return _REGION_TZ_ABBR.get(rname, "UTC")


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


def _region_pill(rname: str, color_map: dict, price_type: str = "spot") -> None:
    color = color_map.get(rname, "#6b7280")
    label = "Day-ahead" if price_type == "day_ahead" else "Real-time spot"
    badge_color = "#d97706" if price_type == "day_ahead" else "#059669"
    st.markdown(
        f'<span style="display:inline-block;padding:4px 12px;border-radius:12px;'
        f'background:{color};color:#fff;font-size:13px;font-weight:600;'
        f'letter-spacing:.04em;margin-bottom:6px">{rname}</span>'
        f'<span style="font-size:11px;color:{badge_color};margin-left:6px;font-weight:600">'
        f'{label}</span>',
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
    margin=dict(t=40, b=90, l=65, r=20),
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, font=dict(size=13)),
    xaxis=dict(
        showgrid=False, showline=True, linecolor="#e5e7eb",
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#9ca3af", spikethickness=1, spikedash="dot",
        tickangle=0,
    ),
)

_YAXIS_BASE = dict(
    showgrid=True, gridcolor="rgba(128,128,128,0.15)",
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


def _base_layout(**extra) -> dict:
    """_CHART_BASE without xaxis so callers can merge tick overrides cleanly."""
    return {k: v for k, v in _CHART_BASE.items() if k != "xaxis"} | extra


def render_spot_price_chart(
    all_prices: dict[str, pd.Series], freq: str, use_log: bool,
    color_map: dict, currency: str = "$",
    trace_labels: "dict[str, str] | None" = None,
    tz_label: str = "",
) -> None:
    fig = go.Figure()
    last_date = None
    for rname, prices in all_prices.items():
        resampled, ld = compute_spot_timeseries(prices, freq)
        if resampled.empty:
            continue
        if last_date is None:
            last_date = ld
        name = trace_labels[rname] if trace_labels and rname in trace_labels else rname
        fig.add_trace(go.Scatter(
            x=resampled.index, y=resampled.values,
            name=name,
            line=dict(color=color_map.get(rname, "#6b7280"), width=2),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate=f"{name}: {currency}%{{y:,.0f}}<extra></extra>",
        ))
    yaxis_type = "log" if use_log else "linear"
    tick_fmt = dict(tickprefix=currency, tickformat=",.0f")
    x_tick = _FREQ_TICK.get(freq, {})
    fig.update_layout(
        **_base_layout(height=400),
        xaxis=dict(**_CHART_BASE["xaxis"], **x_tick, **({"title": tz_label} if tz_label else {})),
        yaxis=dict(**_YAXIS_BASE, title=f"Avg Price ({currency}/MWh)", type=yaxis_type, **tick_fmt),
    )
    _add_data_through_annotation(fig, last_date)
    if use_log:
        st.caption("ℹ Negative prices hidden in log scale — switch to linear to see them")
    st.plotly_chart(fig, use_container_width=True)


def render_frequency_chart(
    freq_df: pd.DataFrame, last_date: "date | None",
    band_ids: "list[str] | None" = None,
    currency: str = "$",
    chart_freq: str = "",
    tz_label: str = "",
) -> None:
    fig = go.Figure()
    for band in STORAGE_BANDS:
        if band.id not in freq_df.columns:
            continue
        if band_ids is not None and band.id not in band_ids:
            continue
        fig.add_trace(go.Scatter(
            x=freq_df.index, y=freq_df[band.id],
            name=_band_label(band, currency),
            line=dict(color=band.color, width=2),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate="%{y:.0f}%<extra></extra>",
        ))
    x_tick = _FREQ_TICK.get(chart_freq, {})
    fig.update_layout(
        **_base_layout(height=420),
        xaxis=dict(**_CHART_BASE["xaxis"], **x_tick, **({"title": tz_label} if tz_label else {})),
        yaxis=dict(**_YAXIS_BASE, title="% of Hours", ticksuffix="%", rangemode="tozero"),
    )
    _add_data_through_annotation(fig, last_date)
    st.plotly_chart(fig, use_container_width=True)


def render_avg_price_chart(
    avg_df: pd.DataFrame,
    last_date: "date | None",
    use_log: bool = True,
    currency: str = "$",
    band_ids: "list[str] | None" = None,
    chart_freq: str = "",
    tz_label: str = "",
) -> None:
    fig = go.Figure()
    for band in STORAGE_BANDS:
        if band.id not in avg_df.columns or avg_df[band.id].isna().all():
            continue
        if band_ids is not None and band.id not in band_ids:
            continue
        actual = avg_df[band.id]
        fig.add_trace(go.Scatter(
            x=avg_df.index, y=actual,
            name=_band_label(band, currency),
            line=dict(color=band.color, width=2),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate=f"{currency}%{{y:,.0f}}<extra></extra>",
        ))
    yaxis_type = "log" if use_log else "linear"
    tick_fmt = dict(tickprefix=currency, tickformat=",.0f")
    x_tick = _FREQ_TICK.get(chart_freq, {})
    fig.update_layout(
        **_base_layout(height=420),
        xaxis=dict(**_CHART_BASE["xaxis"], **x_tick, **({"title": tz_label} if tz_label else {})),
        yaxis=dict(**_YAXIS_BASE, title=f"Avg Price ({currency}/MWh)", type=yaxis_type, **tick_fmt),
    )
    _add_data_through_annotation(fig, last_date)
    st.plotly_chart(fig, use_container_width=True)


def render_threshold_chart(
    df: pd.DataFrame, threshold: float, last_date: "date | None",
    currency: str = "$", chart_freq: str = "", tz_label: str = "",
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
    x_tick = _FREQ_TICK.get(chart_freq, {})
    fig.update_layout(
        **_base_layout(height=420),
        xaxis=dict(**_CHART_BASE["xaxis"], **x_tick, **({"title": tz_label} if tz_label else {})),
        yaxis=dict(**_YAXIS_BASE, title="% of Hours", ticksuffix="%", range=[0, 100]),
    )
    _add_data_through_annotation(fig, last_date)
    st.plotly_chart(fig, use_container_width=True)


def render_band_distribution_chart(
    freq_df: pd.DataFrame, last_date: "date | None",
    band_ids: list[str], currency: str = "$", chart_freq: str = "", tz_label: str = "",
) -> None:
    fig = go.Figure()
    for band in reversed(STORAGE_BANDS):
        if band.id not in band_ids or band.id not in freq_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=freq_df.index, y=freq_df[band.id],
            name=_band_label(band, currency),
            stackgroup="one",
            groupnorm="percent",
            line=dict(width=0, color=band.color),
            fillcolor=band.color,
            hovertemplate="%{y:.1f}%<extra></extra>",
        ))
    x_tick = _FREQ_TICK.get(chart_freq, {})
    fig.update_layout(
        **_base_layout(height=420),
        xaxis=dict(**_CHART_BASE["xaxis"], **x_tick, **({"title": tz_label} if tz_label else {})),
        yaxis=dict(**_YAXIS_BASE, title="% of Hours", ticksuffix="%", range=[0, 100]),
    )
    _add_data_through_annotation(fig, last_date)
    st.plotly_chart(fig, use_container_width=True)


# ── Page layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Electricity Price Bands", page_icon="⚡", layout="wide")
st.markdown(_CSS, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚡ Price Band Analyser")

    # ── Markets & Regions ──────────────────────────────────────────────────────
    st.subheader("Markets")
    show_aus = st.checkbox("Australia 🇦🇺", value=True)
    show_usa = st.checkbox("USA 🇺🇸",       value=False)
    show_eu  = st.checkbox("Europe 🇪🇺",    value=False)

    _market_order = {"australia": 0, "usa": 1, "europe": 2}
    _market_flag  = {"australia": "🇦🇺", "usa": "🇺🇸", "europe": "🇪🇺"}
    available = sorted(
        [r for r, info in ALL_REGIONS.items()
         if (show_aus and info["market"] == "australia")
         or (show_usa and info["market"] == "usa")
         or (show_eu  and info["market"] == "europe")],
        key=lambda r: _market_order[ALL_REGIONS[r]["market"]],
    )

    region_names: list[str] = st.multiselect(
        "Regions / ISOs / Countries", available,
        default=[available[0]] if available else [],
        format_func=lambda r: f"{_market_flag[ALL_REGIONS[r]['market']]} {r}",
    )

    currencies = {ALL_REGIONS[r]["currency"] for r in region_names} if region_names else {"$"}
    currency = "€" if currencies == {"€"} else "$"
    color_map = {r: ALL_REGIONS[r]["color"] for r in ALL_REGIONS}

    # ── Timeframe ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Timeframe")
    preset = st.selectbox("Period", list(PRESETS.keys()), index=2, label_visibility="collapsed")

    if preset == "Custom":
        date_start = st.date_input("From", PRESETS["Custom"][0])
        date_end   = st.date_input("To",   PRESETS["Custom"][1])
    else:
        date_start, date_end = PRESETS[preset]
        date_end = min(date_end, _today)

    # ── Chart Display ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Chart Display")
    chart_freq_label = st.radio("X-axis interval", list(CHART_FREQS.keys()), index=3)
    chart_freq = CHART_FREQS[chart_freq_label]

    _lcol, _rcol = st.columns(2)
    with _lcol:
        use_log_spot = st.checkbox("Log: spot", value=False, key="log_spot")
    with _rcol:
        use_log_avg = st.checkbox("Log: avg", value=True, key="log_avg")

    if len(region_names) > 1:
        view_mode = st.radio(
            "Multi-region view", ["Tabs", "All Regions"],
            horizontal=True, key="view_mode",
        )
    else:
        view_mode = "Tabs"

    # ── Band Visibility ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Band Visibility")
    _band_opts = [_band_label(b, currency) for b in STORAGE_BANDS]
    _band_sel = st.multiselect(
        "Visible bands", _band_opts, default=_band_opts,
        key=f"all_bands_{currency}",
        help="Applies to Band Frequency, Avg Price, and Distribution charts.",
    )
    band_ids_all = [b.id for b in STORAGE_BANDS if _band_label(b, currency) in _band_sel]

    # ── Threshold / Distribution ────────────────────────────────────────────────
    st.divider()
    st.subheader("Threshold Chart")
    threshold_mode = st.radio(
        "Chart type", ["Price Threshold", "Band Distribution"],
        key="thr_mode", horizontal=True, label_visibility="collapsed",
    )
    if threshold_mode == "Price Threshold":
        threshold = float(st.number_input(
            f"Threshold ({currency}/MWh)", value=150, step=50, min_value=-1000, max_value=20000
        ))
    else:
        threshold = 0.0

# ── Guard: require at least one region ────────────────────────────────────────

if not region_names:
    st.title("⚡ Electricity Price Band Analyser")
    st.info(
        "Select one or more markets and regions in the sidebar to load price data. "
        "You can compare across Australia (NEM), US ISOs, and European countries simultaneously."
    )
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────

st.title("⚡ Electricity Price Band Analyser")

_source_map = {
    "australia": "Open Electricity API",
    "usa":       "gridstatus / ISO public data",
    "europe":    "ENTSO-E Transparency Platform",
}
_markets_in_sel = {ALL_REGIONS[r]["market"] for r in region_names}
_sources = " · ".join(_source_map[m] for m in ["australia", "usa", "europe"] if m in _markets_in_sel)

col_title_l, col_title_r = st.columns([3, 1])
with col_title_l:
    st.caption(f"{', '.join(region_names)}")
with col_title_r:
    st.caption(f"{date_start:%d %b %Y} – {date_end:%d %b %Y}  |  {_sources}", unsafe_allow_html=False)

st.divider()

# Day-ahead warning
da_regions = [r for r in region_names if ALL_REGIONS[r]["price_type"] == "day_ahead"]
if da_regions:
    st.warning(
        f"⚠️ **Day-ahead prices** (not real-time spot): {', '.join(da_regions)}. "
        "Day-ahead prices are set the evening before delivery and may differ from real-time spot prices."
    )

# Mixed-currency note
if len(currencies) > 1:
    st.info(
        "Mixed currencies: Australian prices in AUD (A\\$), US prices in USD (US\\$), "
        "EU prices in EUR (€). Values plotted on the same axis — "
        "cross-currency comparisons are indicative only."
    )

# ── Fetch all selected regions ─────────────────────────────────────────────────

all_prices: dict[str, pd.Series] = {}
for rname in region_names:
    info = ALL_REGIONS[rname]
    try:
        if info["market"] == "australia":
            p = fetch_prices(info["code"], date_start, date_end)
        elif info["market"] == "usa":
            if rname == "MISO":
                eff_end = min(date_end, _today - timedelta(days=1))
                p = fetch_prices_gridstatus(rname, date_start, eff_end)
            elif rname == "SPP":
                # SPP daily files are published with a 2–3 day lag
                eff_end = min(date_end, _today - timedelta(days=3))
                if eff_end < date_start:
                    st.warning("SPP: no data available yet — daily files are published with a 2–3 day lag.")
                    continue
                p = fetch_prices_gridstatus(rname, date_start, eff_end)
                if eff_end < date_end:
                    st.caption(f"ℹ SPP data available through {eff_end:%d %b %Y} (files published with ~3 day lag).")
            else:
                p = fetch_prices_gridstatus(rname, date_start, date_end)
        else:
            p = fetch_prices_entsoe(info["code"], date_start, date_end)

        if p.empty:
            if rname == "ERCOT":
                st.warning(
                    "ERCOT returned no data. The ERCOT public MIS endpoint may be temporarily "
                    "unavailable or the date range has no published data. Try a different period."
                )
            else:
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
        elif "ENTSOE_API_KEY" in msg or "Web API Security Token" in msg:
            st.error(msg)
        elif "401" in msg and info["market"] == "europe":
            st.error(
                f"ENTSO-E API key rejected (401) for {rname}. "
                "Check ENTSOE_API_KEY in your .env file."
            )
        elif "not found" in msg.lower() or "hub" in msg.lower():
            st.error(f"{rname}: {msg}")
        elif "403" in msg:
            st.error(f"API key rejected for {rname} (403). Check your credentials in .env.")
        else:
            err_detail = msg or f"({type(exc).__name__} — no message)"
            st.error(f"Error fetching {rname}: {err_detail}")

if not all_prices:
    st.stop()

# Trace labels for spot price chart: append currency suffix when mixing markets
_mkt_suffix = {"australia": "(A$)", "usa": "(US$)", "europe": "(€)"}
trace_labels: "dict[str, str] | None" = None
if len(currencies) > 1:
    trace_labels = {r: f"{r} {_mkt_suffix[ALL_REGIONS[r]['market']]}" for r in all_prices}

# Tab labels for multi-region sections
_tab_labels = (
    [trace_labels[r] for r in all_prices] if trace_labels
    else list(all_prices.keys())
)

# ── Overview: metrics + band tables ───────────────────────────────────────────

st.subheader("📊 Overview")

ov_cols = st.columns(len(all_prices))
for col, (rname, prices) in zip(ov_cols, all_prices.items()):
    with col:
        avg_price = prices.mean()
        neg_pct   = (prices < 0).mean() * 100
        peak_pct  = (prices >= 100).mean() * 100  # B6+ threshold — meaningful for BESS dispatch

        avg_color  = "#16a34a" if avg_price < 0 else "#2563eb"
        neg_color  = "#16a34a" if neg_pct > 5 else "#6b7280"
        peak_color = "#dc2626" if peak_pct > 5 else "#6b7280"

        r_currency = ALL_REGIONS[rname]["currency"]
        avg_str = (
            f"-{r_currency}{abs(avg_price):,.0f}" if avg_price < 0
            else f"{r_currency}{avg_price:,.0f}"
        )

        _region_pill(rname, color_map, ALL_REGIONS[rname]["price_type"])
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.html(_metric_card("Avg Price", f"{avg_str}/MWh", avg_color))
        with mc2:
            st.html(_metric_card("Negative hrs", f"{neg_pct:.1f}%", neg_color))
        with mc3:
            st.html(_metric_card(f"Peak ≥{r_currency}100", f"{peak_pct:.1f}%", peak_color))

        render_band_table(compute_band_stats(prices), r_currency)

st.divider()

# ── Spot Price ─────────────────────────────────────────────────────────────────

st.subheader("📈 Spot Price")
if len(all_prices) == 1:
    _sp_rname = next(iter(all_prices))
    _sp_prices = _to_local_tz(all_prices[_sp_rname], _sp_rname)
    render_spot_price_chart(
        {_sp_rname: _sp_prices}, chart_freq, use_log_spot, color_map, currency,
        trace_labels=trace_labels, tz_label=_get_tz_abbr(_sp_rname),
    )
else:
    render_spot_price_chart(
        all_prices, chart_freq, use_log_spot, color_map, currency,
        trace_labels=trace_labels, tz_label="UTC",
    )

st.divider()

# ── Band Trends ────────────────────────────────────────────────────────────────

st.subheader("🎯 Band Trends")

def _render_band_trends(rname: str, prices: pd.Series) -> None:
    prices_local = _to_local_tz(prices, rname)
    tz_abbr = _get_tz_abbr(rname)
    freq_df, avg_df, last_date = compute_timeseries(prices_local, chart_freq)
    if freq_df.empty:
        st.info("Not enough data at this interval — try a coarser X-axis setting.")
        return
    r_currency = ALL_REGIONS[rname]["currency"]
    cl, cr = st.columns(2)
    with cl:
        st.caption("Band Frequency — % of hours in each price band per period")
        render_frequency_chart(
            freq_df, last_date, band_ids=band_ids_all,
            currency=r_currency, chart_freq=chart_freq, tz_label=tz_abbr,
        )
    with cr:
        st.caption(f"Avg Band Price — mean {r_currency}/MWh within each band per period")
        render_avg_price_chart(
            avg_df, last_date, use_log_avg, r_currency,
            band_ids=band_ids_all, chart_freq=chart_freq, tz_label=tz_abbr,
        )

if len(all_prices) > 1:
    if view_mode == "All Regions":
        for _i, (rname, prices) in enumerate(all_prices.items()):
            _region_pill(rname, color_map, ALL_REGIONS[rname]["price_type"])
            _render_band_trends(rname, prices)
            if _i < len(all_prices) - 1:
                st.divider()
    else:
        trend_tabs = st.tabs(_tab_labels)
        for tab, rname in zip(trend_tabs, all_prices.keys()):
            with tab:
                _region_pill(rname, color_map, ALL_REGIONS[rname]["price_type"])
                _render_band_trends(rname, all_prices[rname])
else:
    rname = next(iter(all_prices))
    _render_band_trends(rname, all_prices[rname])

st.divider()

# ── Price Threshold / Band Distribution ────────────────────────────────────────

if threshold_mode == "Price Threshold":
    st.subheader("⬆️ Price Threshold")
    st.caption(f"% of hours above / below {currency}{threshold:,.0f}/MWh — adjust in sidebar")
else:
    st.subheader("📊 Band Distribution Over Time")
    st.caption("100% stacked area — share of hours in each price band per period")

def _render_threshold_section(rname: str, prices: pd.Series) -> None:
    prices_local = _to_local_tz(prices, rname)
    tz_abbr = _get_tz_abbr(rname)
    r_currency = ALL_REGIONS[rname]["currency"]
    if threshold_mode == "Price Threshold":
        below_pct = float((prices < threshold).mean() * 100)
        st.html(_metric_card(
            f"Hours below {r_currency}{threshold:,.0f}/MWh",
            f"{below_pct:.1f}%",
            "#16a34a",
        ))
        thr_df, thr_last_date = compute_threshold_timeseries(prices_local, threshold, chart_freq)
        if thr_df.empty:
            st.info("Not enough data at this interval.")
        else:
            render_threshold_chart(thr_df, threshold, thr_last_date, r_currency, chart_freq=chart_freq, tz_label=tz_abbr)
    else:
        freq_df, _, last_date = compute_timeseries(prices_local, chart_freq)
        if freq_df.empty:
            st.info("Not enough data at this interval.")
        else:
            render_band_distribution_chart(freq_df, last_date, band_ids_all, r_currency, chart_freq=chart_freq, tz_label=tz_abbr)

if len(all_prices) > 1:
    if view_mode == "All Regions":
        for _i, (rname, prices) in enumerate(all_prices.items()):
            _region_pill(rname, color_map, ALL_REGIONS[rname]["price_type"])
            _render_threshold_section(rname, prices)
            if _i < len(all_prices) - 1:
                st.divider()
    else:
        thr_tabs = st.tabs(_tab_labels)
        for tab, rname in zip(thr_tabs, all_prices.keys()):
            with tab:
                _region_pill(rname, color_map, ALL_REGIONS[rname]["price_type"])
                _render_threshold_section(rname, all_prices[rname])
else:
    rname = next(iter(all_prices))
    _region_pill(rname, color_map, ALL_REGIONS[rname]["price_type"])
    _render_threshold_section(rname, all_prices[rname])

# ── Footer ─────────────────────────────────────────────────────────────────────

_markets_used = {ALL_REGIONS[r]["market"] for r in all_prices}
_source = " · ".join(_source_map[m] for m in ["australia", "usa", "europe"] if m in _markets_used)
st.caption(
    f"Source: {_source} · 1-hour intervals · {date_start:%d %b %Y} – {date_end:%d %b %Y}"
)
