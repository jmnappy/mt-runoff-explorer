"""
Western Montana Runoff Explorer
================================
Hydrologic decision tool for snowmelt runoff timing,
peak flow tracking, and late-summer recession analysis.

pip install dash plotly pandas numpy dataretrieval scipy dash-bootstrap-components requests
python streamflow_dev.py
"""

import dash
from dash import dcc, html, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dataretrieval.nwis as nwis
from scipy import stats
import requests
import warnings
warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# STATION CATALOG
# ═════════════════════════════════════════════════════════════════════════════
STATIONS = {
    "12340500": {"name": "Clark Fork at Missoula", "basin": "Clark Fork", "lat": 46.8701, "lon": -114.0894},
    "12334550": {"name": "Clark Fork at Turah Bridge", "basin": "Clark Fork", "lat": 46.7953, "lon": -113.7756},
    "12323800": {"name": "Clark Fork near Deer Lodge", "basin": "Clark Fork", "lat": 46.3584, "lon": -112.7453},
    "12324680": {"name": "Clark Fork at Goldcreek", "basin": "Clark Fork", "lat": 46.5515, "lon": -112.9562},
    "12331800": {"name": "Clark Fork near Drummond", "basin": "Clark Fork", "lat": 46.6712, "lon": -113.1567},
    "12353000": {"name": "Clark Fork below Missoula", "basin": "Clark Fork", "lat": 47.0404, "lon": -114.2536},
    "12340000": {"name": "Blackfoot River near Bonner", "basin": "Blackfoot", "lat": 46.8687, "lon": -113.8828},
    "12344000": {"name": "Bitterroot River near Missoula", "basin": "Bitterroot", "lat": 46.8526, "lon": -114.0903},
    "12342500": {"name": "West Fork Bitterroot near Conner", "basin": "Bitterroot", "lat": 45.9262, "lon": -114.0703},
    "12342000": {"name": "East Fork Bitterroot near Conner", "basin": "Bitterroot", "lat": 45.8984, "lon": -113.9203},
    "12341000": {"name": "Bitterroot River at Bell Crossing", "basin": "Bitterroot", "lat": 46.2773, "lon": -114.1067},
    "12334510": {"name": "Rock Creek near Clinton", "basin": "Clark Fork", "lat": 46.7621, "lon": -113.6956},
    "12362500": {"name": "S Fork Flathead nr Columbia Falls", "basin": "Flathead", "lat": 48.2151, "lon": -114.0514},
    "12355500": {"name": "N Fork Flathead nr Columbia Falls", "basin": "Flathead", "lat": 48.4932, "lon": -114.0889},
    "12358500": {"name": "Middle Fork Flathead nr West Glacier", "basin": "Flathead", "lat": 48.4965, "lon": -113.9806},
    "12363000": {"name": "Flathead River at Columbia Falls", "basin": "Flathead", "lat": 48.3596, "lon": -114.1842},
    "12370000": {"name": "Flathead River at Polson", "basin": "Flathead", "lat": 47.6901, "lon": -114.1592},
    "12372000": {"name": "Flathead River near Perma", "basin": "Flathead", "lat": 47.3579, "lon": -114.4169},
    "12301933": {"name": "Kootenai River below Libby Dam", "basin": "Kootenai", "lat": 48.4076, "lon": -115.3125},
    "06043500": {"name": "Gallatin River near Gateway", "basin": "Missouri HW", "lat": 45.4790, "lon": -111.2112},
    "06052500": {"name": "Gallatin River at Logan", "basin": "Missouri HW", "lat": 45.8865, "lon": -111.4290},
    "06041000": {"name": "Madison River below Ennis Lake", "basin": "Missouri HW", "lat": 45.6654, "lon": -111.5959},
    "06038500": {"name": "Madison River nr West Yellowstone", "basin": "Missouri HW", "lat": 44.6582, "lon": -111.0973},
    "06036650": {"name": "Jefferson River near Three Forks", "basin": "Missouri HW", "lat": 45.9215, "lon": -111.5520},
    "06026500": {"name": "Jefferson River near Twin Bridges", "basin": "Missouri HW", "lat": 45.5337, "lon": -112.3209},
    "06018500": {"name": "Beaverhead River near Twin Bridges", "basin": "Missouri HW", "lat": 45.5331, "lon": -112.3345},
}

BASIN_COLORS = {
    "Clark Fork": "#1f77b4", "Blackfoot": "#2ca02c", "Bitterroot": "#d62728",
    "Flathead": "#9467bd", "Kootenai": "#8c564b", "Missouri HW": "#ff7f0e",
    "Search Result": "#17becf",
}

DROPDOWN_OPTIONS = [
    {"label": f"{v['name']} ({k})", "value": k}
    for k, v in sorted(STATIONS.items(), key=lambda x: x[1]["name"])
]

# ═════════════════════════════════════════════════════════════════════════════
# STYLING
# ═════════════════════════════════════════════════════════════════════════════
BG       = "#2c3038"
CARD_BG  = "#353a44"
PLOT_BG  = "#2e333c"
GRID_CLR = "#3e4450"
TEXT_CLR = "#d4d8e0"
TEXT_DIM = "#8b919c"
HEADER_BG = "#1e2228"
ACCENT   = "#3b82f6"

GRAPH_CFG = {"displayModeBar": True, "scrollZoom": True,
             "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"],
             "displaylogo": False}
MAP_CFG = {"displayModeBar": False, "scrollZoom": True}

PLTLY = dict(layout=go.Layout(
    paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=TEXT_CLR),
    xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
))

C_Q  = {"fill": "96,165,250", "focus": "#ef4444", "label": "Q (cfs)"}
C_WT = {"fill": "251,146,60", "focus": "#ea580c", "label": "Water °F"}
C_AT = {"fill": "74,222,128", "focus": "#22c55e", "label": "Air °F"}
C_ROC = {"pos": "#22d3ee", "neg": "#f87171", "hist": "148,163,184"}


# ═════════════════════════════════════════════════════════════════════════════
# DATA FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _find_param_col(df, param_code):
    for c in df.columns:
        cs = str(c)
        if param_code in cs and "_cd" not in cs.lower() and "qualification" not in cs.lower():
            return c
    return None


def fetch_usgs(site_no, start="1980-01-01", end=None):
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    try:
        df, _ = nwis.get_dv(sites=site_no, parameterCd="00060,00010",
                            start=start, end=end)
    except Exception as e:
        print(f"  NWIS error: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    q_col = _find_param_col(df, "00060")
    t_col = _find_param_col(df, "00010")
    if q_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    out["q"] = pd.to_numeric(df[q_col], errors="coerce")
    if t_col:
        out["water_temp_c"] = pd.to_numeric(df[t_col], errors="coerce")
        out.loc[(out["water_temp_c"] < -5) | (out["water_temp_c"] > 40), "water_temp_c"] = np.nan
        out["water_temp_f"] = out["water_temp_c"] * 9.0 / 5.0 + 32.0
    else:
        out["water_temp_c"] = np.nan
        out["water_temp_f"] = np.nan
    out = out.dropna(subset=["q"])
    out = out[out["q"] >= 0].copy().sort_values("date").reset_index(drop=True)
    print(f"  USGS: {len(out):,} Q | water temp: {out['water_temp_c'].notna().sum():,}")
    return out


def fetch_air_temp(lat, lon, start="1980-01-01", end=None):
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    frames = []
    chunk_start = start_dt
    archive_end = min(end_dt, pd.to_datetime(datetime.now()) - pd.DateOffset(days=7))
    while chunk_start < archive_end:
        chunk_end = min(chunk_start + pd.DateOffset(years=10), archive_end)
        try:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": lat, "longitude": lon,
                "start_date": chunk_start.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
                "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
                "temperature_unit": "fahrenheit", "timezone": "America/Denver",
            }, timeout=30)
            r.raise_for_status()
            d = r.json().get("daily", {})
            if d and d.get("time"):
                frames.append(pd.DataFrame({
                    "date": pd.to_datetime(d["time"]),
                    "air_temp_max_f": d.get("temperature_2m_max"),
                    "air_temp_min_f": d.get("temperature_2m_min"),
                    "air_temp_mean_f": d.get("temperature_2m_mean"),
                }))
                print(f"  Air temp {chunk_start.year}–{chunk_end.year}: {len(d['time'])} days")
        except Exception as e:
            print(f"  Air temp FAIL ({chunk_start.year}–{chunk_end.year}): {e}")
        chunk_start = chunk_end + pd.DateOffset(days=1)
    try:
        past_days = min(92, (pd.to_datetime(datetime.now()) - archive_end).days + 14)
        if past_days > 0:
            r = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
                "temperature_unit": "fahrenheit", "timezone": "America/Denver",
                "past_days": past_days, "forecast_days": 1,
            }, timeout=30)
            r.raise_for_status()
            d = r.json().get("daily", {})
            if d and d.get("time"):
                frames.append(pd.DataFrame({
                    "date": pd.to_datetime(d["time"]),
                    "air_temp_max_f": d.get("temperature_2m_max"),
                    "air_temp_min_f": d.get("temperature_2m_min"),
                    "air_temp_mean_f": d.get("temperature_2m_mean"),
                }))
    except Exception as e:
        print(f"  Air temp forecast FAIL: {e}")
    if not frames:
        return pd.DataFrame(columns=["date", "air_temp_max_f", "air_temp_min_f", "air_temp_mean_f"])
    result = pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    result = result.dropna(subset=["air_temp_mean_f"], how="all")
    print(f"  Air temp TOTAL: {len(result):,}")
    return result


def fetch_all(site_no, start="1980-01-01"):
    print(f"\n{'='*60}\nFetching {site_no}\n{'='*60}")
    usgs = fetch_usgs(site_no, start=start)
    if usgs.empty:
        return pd.DataFrame()
    info = STATIONS.get(site_no, {})
    lat, lon = info.get("lat"), info.get("lon")
    if lat and lon:
        air = fetch_air_temp(lat, lon, start=start)
        if not air.empty:
            usgs = usgs.merge(air, on="date", how="left")
        else:
            for c in ["air_temp_max_f", "air_temp_min_f", "air_temp_mean_f"]:
                usgs[c] = np.nan
    else:
        for c in ["air_temp_max_f", "air_temp_min_f", "air_temp_mean_f"]:
            usgs[c] = np.nan
    usgs["year"] = usgs["date"].dt.year
    usgs["month"] = usgs["date"].dt.month
    usgs["doy"] = usgs["date"].dt.dayofyear
    print(f"  FINAL: {len(usgs):,} rows | wt:{usgs['water_temp_f'].notna().sum():,} | at:{usgs['air_temp_mean_f'].notna().sum():,}")
    return usgs


def search_sites(query_text, state="MT"):
    try:
        if query_text.strip().isdigit():
            info, _ = nwis.get_info(sites=query_text.strip())
            if info is not None and len(info) > 0:
                return info
        sites, _ = nwis.get_info(stateCd=state, parameterCd="00060",
                                  siteType="ST", siteStatus="active", hasDataTypeCd="dv")
        if sites is not None and len(sites) > 0:
            return sites[sites["station_nm"].str.contains(query_text, case=False, na=False)]
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def sname(site):
    return STATIONS.get(site, {}).get("name", site)


# ═════════════════════════════════════════════════════════════════════════════
# DERIVED ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_derived(df):
    """Add rate-of-change and cumulative degree-day columns."""
    df = df.sort_values("date").copy()
    # 7-day rate of change in cfs/day
    df["q_smooth"] = df["q"].rolling(7, center=True, min_periods=3).mean()
    df["dq_7d"] = df["q_smooth"].diff(periods=7) / 7.0
    # Cumulative degree-days above 32°F (0°C) — reset each water year (Oct 1)
    if "air_temp_mean_f" in df.columns:
        df["dd_above_32"] = (df["air_temp_mean_f"] - 32.0).clip(lower=0)
        # Water year starts Oct 1
        df["wy"] = df["date"].apply(lambda d: d.year + 1 if d.month >= 10 else d.year)
        df["cum_dd"] = df.groupby("wy")["dd_above_32"].cumsum()
    return df


def compute_timing_metrics(df, focus_year):
    """Compute runoff timing metrics for the focus year vs historical."""
    metrics = {}
    cur = df[df["year"] == focus_year].sort_values("doy").copy()
    hist = df[df["year"] != focus_year].copy()

    if cur.empty or hist.empty:
        return metrics

    # --- Latest values ---
    latest = cur.iloc[-1]
    metrics["latest_q"] = float(latest["q"])
    metrics["latest_date"] = latest["date"]

    # --- 24hr change ---
    if len(cur) >= 2:
        prev_q = cur.iloc[-2]["q"]
        dq_24h = float(latest["q"] - prev_q)
        metrics["dq_24h"] = dq_24h
        if prev_q > 0:
            metrics["dq_24h_pct"] = float(dq_24h / prev_q * 100)

    # --- 7-day max ---
    last7 = cur.tail(7)
    if not last7.empty:
        max_row = last7.loc[last7["q"].idxmax()]
        metrics["q_7d_max"] = float(max_row["q"])
        metrics["q_7d_max_date"] = max_row["date"]

    # --- Current air temp ---
    if "air_temp_mean_f" in cur.columns:
        at_vals = cur.dropna(subset=["air_temp_mean_f"])
        if not at_vals.empty:
            latest_at = at_vals.iloc[-1]
            metrics["air_temp"] = float(latest_at["air_temp_mean_f"])
            if "air_temp_max_f" in at_vals.columns:
                metrics["air_temp_max"] = float(latest_at.get("air_temp_max_f", np.nan))
                metrics["air_temp_min"] = float(latest_at.get("air_temp_min_f", np.nan))

    # --- Median peak DOY historically ---
    hist_peaks = hist.groupby("year").apply(
        lambda g: g.loc[g["q"].idxmax(), "doy"] if len(g) > 60 else np.nan
    ).dropna()
    if len(hist_peaks) > 3:
        metrics["hist_peak_doy"] = int(hist_peaks.median())

    # --- Focus year peak so far ---
    if not cur.empty:
        peak_row = cur.loc[cur["q"].idxmax()]
        metrics["focus_peak_q"] = float(peak_row["q"])
        metrics["focus_peak_doy"] = int(peak_row["doy"])
        metrics["focus_peak_date"] = peak_row["date"]

    # --- Runoff onset: first DOY with 7-day avg dQ > 0 sustained for 5+ days ---
    if "dq_7d" in cur.columns:
        cur_roc = cur.dropna(subset=["dq_7d"])
        if not cur_roc.empty:
            rising = (cur_roc["dq_7d"] > 0).astype(int)
            consec = rising.groupby((rising != rising.shift()).cumsum()).cumsum()
            onset_mask = consec >= 5
            if onset_mask.any():
                first_idx = onset_mask.idxmax()
                onset_doy = cur_roc.loc[first_idx, "doy"]
                metrics["focus_onset_doy"] = int(onset_doy)

        # Historical median onset
        onset_doys = []
        for yr, grp in hist.groupby("year"):
            g = grp.sort_values("doy").copy()
            if "dq_7d" not in g.columns:
                continue
            g_roc = g.dropna(subset=["dq_7d"])
            if g_roc.empty:
                continue
            rising = (g_roc["dq_7d"] > 0).astype(int)
            consec = rising.groupby((rising != rising.shift()).cumsum()).cumsum()
            mask = consec >= 5
            if mask.any():
                onset_doys.append(int(g_roc.loc[mask.idxmax(), "doy"]))
        if onset_doys:
            metrics["hist_onset_doy"] = int(np.median(onset_doys))

    # --- Current trajectory: recent 14-day avg dQ ---
    if "dq_7d" in cur.columns:
        recent = cur.tail(14)["dq_7d"].dropna()
        if len(recent) > 3:
            metrics["recent_dq"] = float(recent.mean())

    # --- Percentile of current Q ---
    latest_doy = cur["doy"].max()
    latest_q = cur[cur["doy"] == latest_doy]["q"].values
    if len(latest_q) > 0:
        hist_at_doy = hist[(hist["doy"] >= latest_doy - 3) & (hist["doy"] <= latest_doy + 3)]["q"]
        if len(hist_at_doy) > 10:
            pctile = (hist_at_doy < latest_q[0]).sum() / len(hist_at_doy) * 100
            metrics["current_pctile"] = float(pctile)

    # --- Historical median Q at current DOY ---
    hist_now = hist[(hist["doy"] >= latest_doy - 3) & (hist["doy"] <= latest_doy + 3)]["q"]
    if len(hist_now) > 5:
        metrics["hist_median_q"] = float(hist_now.median())

    return metrics


def doy_to_label(doy):
    try:
        return (datetime(2024, 1, 1) + timedelta(days=int(doy) - 1)).strftime("%b %d")
    except Exception:
        return str(doy)


# ═════════════════════════════════════════════════════════════════════════════
# SEASONAL AUTO-FRAMING
# ═════════════════════════════════════════════════════════════════════════════

def get_seasonal_range():
    """Return (start_doy, end_doy) for auto-zoom based on current month.
    Tight windows keep y-axis readable for the current period."""
    m = datetime.now().month
    if m in [12, 1, 2]:           # Winter: Jan–Apr (baseflow → early rising limb)
        return 1, 121
    elif m in [3, 4]:              # Early spring: Feb–Jun (onset → approaching peak)
        return 32, 182
    elif m == 5:                   # Late spring: Mar–Jul (rising limb → peak)
        return 60, 213
    elif m in [6, 7]:              # Summer: Apr–Sep (peak → early recession)
        return 91, 274
    elif m in [8, 9]:              # Late summer: Jun–Nov (recession)
        return 152, 335
    elif m in [10, 11]:            # Fall: Aug–Dec (late recession → baseflow)
        return 213, 366


def doy_to_ref_date(doy):
    return pd.to_datetime("2024-01-01") + pd.to_timedelta(int(doy) - 1, unit="D")


# ═════════════════════════════════════════════════════════════════════════════
# MAP
# ═════════════════════════════════════════════════════════════════════════════

def build_station_map(selected_site=None):
    fig = go.Figure()
    for basin, bcolor in BASIN_COLORS.items():
        sids = [k for k, v in STATIONS.items() if v["basin"] == basin]
        if not sids:
            continue
        fig.add_trace(go.Scattermapbox(
            lat=[STATIONS[s]["lat"] for s in sids],
            lon=[STATIONS[s]["lon"] for s in sids],
            mode="markers",
            marker=dict(
                size=[18 if s == selected_site else 10 for s in sids],
                color=["#facc15" if s == selected_site else bcolor for s in sids],
                opacity=[1.0 if s == selected_site else 0.85 for s in sids],
            ),
            text=[f"<b>{STATIONS[s]['name']}</b><br>{s}" for s in sids],
            customdata=sids,
            hovertemplate="%{text}<extra>" + basin + "</extra>",
            name=basin,
        ))
    fig.update_layout(
        mapbox=dict(style="open-street-map",
                    center=dict(lat=46.8, lon=-113.3), zoom=5.3),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=CARD_BG,
        uirevision="static",
        legend=dict(orientation="h", y=1.01, x=0,
                    bgcolor="rgba(53,58,68,0.9)",
                    font=dict(size=12, color=TEXT_CLR)),
        clickmode="event",
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def empty_fig(msg="Select a station and click Load Data"):
    return go.Figure().update_layout(
        template=PLTLY,
        annotations=[dict(text=msg, showarrow=False, xref="paper", yref="paper",
                          x=0.5, y=0.5, font=dict(size=14, color=TEXT_DIM))],
    )


def _today_vlines(fig, today_rd, n_rows):
    """Add 'Today' vertical line to all subplot rows using shapes (avoids Timestamp bug)."""
    today_str = today_rd.isoformat()
    for r in range(1, n_rows + 1):
        xref = "x" if r == 1 else f"x{r}"
        yref = f"y{r}" if r > 1 else "y"
        fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                      xref=xref, yref=f"{yref} domain",
                      line=dict(color="#fbbf24", width=1.5, dash="dot"))
    # Annotation on row 1 only
    fig.add_annotation(x=today_str, y=1, xref="x", yref="y domain",
                       text="Today", showarrow=False,
                       font=dict(color="#fbbf24", size=11),
                       xanchor="left", yanchor="bottom", xshift=4)


def _add_percentile_band(fig, hist, param, row, fill_rgb, focus_color,
                          cur=None, ylabel="", fmt=",.0f", rangemode="normal"):
    """Percentile envelope + focus year overlay."""
    h = hist.dropna(subset=[param])
    if h.empty:
        fig.update_yaxes(title_text=ylabel, row=row, col=1)
        return
    pcts = h.groupby("doy")[param].agg(
        p05=lambda x: x.quantile(0.05), p10=lambda x: x.quantile(0.10),
        p25=lambda x: x.quantile(0.25), p50=lambda x: x.quantile(0.50),
        p75=lambda x: x.quantile(0.75), p90=lambda x: x.quantile(0.90),
        p95=lambda x: x.quantile(0.95),
    ).reset_index()
    for col in pcts.columns[1:]:
        pcts[col] = pcts[col].rolling(7, center=True, min_periods=1).mean()
    pcts["rd"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(pcts["doy"] - 1, unit="D")

    # Outer band 5–95
    fig.add_trace(go.Scatter(
        x=pd.concat([pcts["rd"], pcts["rd"][::-1]]),
        y=pd.concat([pcts["p95"], pcts["p05"][::-1]]),
        fill="toself", fillcolor=f"rgba({fill_rgb},0.05)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"), row=row, col=1)
    # Mid band 10–90
    fig.add_trace(go.Scatter(
        x=pd.concat([pcts["rd"], pcts["rd"][::-1]]),
        y=pd.concat([pcts["p90"], pcts["p10"][::-1]]),
        fill="toself", fillcolor=f"rgba({fill_rgb},0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"), row=row, col=1)
    # IQR band 25–75
    fig.add_trace(go.Scatter(
        x=pd.concat([pcts["rd"], pcts["rd"][::-1]]),
        y=pd.concat([pcts["p75"], pcts["p25"][::-1]]),
        fill="toself", fillcolor=f"rgba({fill_rgb},0.14)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"), row=row, col=1)

    # Median line
    fig.add_trace(go.Scatter(
        x=pcts["rd"], y=pcts["p50"],
        line=dict(color=f"rgba({fill_rgb},0.85)", width=2),
        name="Median", showlegend=(row == 1),
        hovertemplate=f"Median: %{{y:{fmt}}}<extra></extra>"), row=row, col=1)

    # Focus year
    if cur is not None:
        c = cur.dropna(subset=[param]).sort_values("doy").copy()
        if not c.empty:
            c["rd"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(c["doy"] - 1, unit="D")
            c["smooth"] = c[param].rolling(5, center=True, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=c["rd"], y=c["smooth"],
                line=dict(color=focus_color, width=3),
                name=str(c["year"].iloc[0]),
                showlegend=(row == 1),
                hovertemplate=f"%{{y:{fmt}}}<extra></extra>"), row=row, col=1)

    fig.update_yaxes(title_text=ylabel, gridcolor=GRID_CLR,
                     rangemode=rangemode, row=row, col=1)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: RUNOFF TIMING (auto-framed, Q + dQ + temp)
# ═════════════════════════════════════════════════════════════════════════════

def plot_runoff_timing(df, name, focus_year):
    hist = df[df["year"] != focus_year].copy()
    cur = df[df["year"] == focus_year].copy()
    if hist.empty:
        return empty_fig("Not enough historical data")

    has_at = df["air_temp_mean_f"].notna().sum() > 100
    has_wt = df["water_temp_f"].notna().sum() > 100
    has_roc = "dq_7d" in df.columns

    # Build panel list: Q always, then dQ, then temps
    n_rows = 2  # Q + dQ always
    if has_at or has_wt:
        n_rows = 3
    heights = {2: [0.55, 0.45], 3: [0.36, 0.30, 0.34]}[n_rows]

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=heights)

    # ── Row 1: Discharge percentile envelope ──
    _add_percentile_band(fig, hist, "q", 1, C_Q["fill"], C_Q["focus"],
                          cur=cur, ylabel="Discharge (cfs)", fmt=",.0f", rangemode="tozero")

    # Today marker — added after layout via helper
    today_doy = datetime.now().timetuple().tm_yday
    today_rd = doy_to_ref_date(today_doy)

    # ── Row 2: Rate of change ──
    if has_roc:
        # Historical median dQ
        h_roc = hist.dropna(subset=["dq_7d"])
        if not h_roc.empty:
            roc_pcts = h_roc.groupby("doy")["dq_7d"].agg(
                p25=lambda x: x.quantile(0.25),
                p50=lambda x: x.quantile(0.50),
                p75=lambda x: x.quantile(0.75),
            ).reset_index()
            for col in ["p25", "p50", "p75"]:
                roc_pcts[col] = roc_pcts[col].rolling(7, center=True, min_periods=1).mean()
            roc_pcts["rd"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(roc_pcts["doy"] - 1, unit="D")

            # IQR band
            fig.add_trace(go.Scatter(
                x=pd.concat([roc_pcts["rd"], roc_pcts["rd"][::-1]]),
                y=pd.concat([roc_pcts["p75"], roc_pcts["p25"][::-1]]),
                fill="toself", fillcolor=f"rgba({C_ROC['hist']},0.10)",
                line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=roc_pcts["rd"], y=roc_pcts["p50"],
                line=dict(color=f"rgba({C_ROC['hist']},0.5)", width=1.5, dash="dash"),
                name="Hist Median ΔQ", showlegend=True,
                hovertemplate="Median ΔQ: %{y:+,.0f} cfs/d<extra></extra>"), row=2, col=1)

        # Focus year dQ — colored by sign
        c_roc = cur.dropna(subset=["dq_7d"]).sort_values("doy").copy()
        if not c_roc.empty:
            c_roc["rd"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(c_roc["doy"] - 1, unit="D")
            c_roc["dq_smooth"] = c_roc["dq_7d"].rolling(5, center=True, min_periods=1).mean()

            # Positive bars (rising)
            pos = c_roc[c_roc["dq_smooth"] >= 0]
            fig.add_trace(go.Bar(
                x=pos["rd"], y=pos["dq_smooth"],
                marker_color=C_ROC["pos"], opacity=0.7, width=86400000,
                name="Rising", showlegend=True,
                hovertemplate="ΔQ: %{y:+,.0f} cfs/d<extra></extra>"), row=2, col=1)

            # Negative bars (falling)
            neg = c_roc[c_roc["dq_smooth"] < 0]
            fig.add_trace(go.Bar(
                x=neg["rd"], y=neg["dq_smooth"],
                marker_color=C_ROC["neg"], opacity=0.7, width=86400000,
                name="Falling", showlegend=True,
                hovertemplate="ΔQ: %{y:+,.0f} cfs/d<extra></extra>"), row=2, col=1)

        # Zero line
        fig.add_hline(y=0, line=dict(color=TEXT_DIM, width=0.8), row=2, col=1)

    fig.update_yaxes(title_text="ΔQ (cfs/day)", gridcolor=GRID_CLR, row=2, col=1)

    # ── Row 3: Temperature (air + water if available) ──
    if n_rows == 3:
        if has_at:
            _add_percentile_band(fig, hist, "air_temp_mean_f", n_rows,
                                  C_AT["fill"], C_AT["focus"],
                                  cur=cur, ylabel="Temp (°F)", fmt=".0f")
        if has_wt:
            # Overlay water temp focus year on same panel
            wt_cur = cur.dropna(subset=["water_temp_f"]).sort_values("doy").copy()
            if not wt_cur.empty:
                wt_cur["rd"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(wt_cur["doy"] - 1, unit="D")
                wt_cur["smooth"] = wt_cur["water_temp_f"].rolling(7, center=True, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=wt_cur["rd"], y=wt_cur["smooth"],
                    line=dict(color=C_WT["focus"], width=2, dash="dash"),
                    name=f"Water Temp {focus_year}",
                    hovertemplate="Water: %{y:.1f}°F<extra></extra>"), row=n_rows, col=1)

        # Freezing line — use plain hline without annotation to avoid Timestamp bug
        fig.add_hline(y=32, line=dict(color="#94a3b8", width=0.8, dash="dot"), row=n_rows, col=1)
        yref_t = f"y{n_rows}" if n_rows > 1 else "y"
        fig.add_annotation(x=0, y=32, xref="x domain", yref=yref_t,
                           text="32°F", showarrow=False,
                           font=dict(color="#94a3b8", size=10),
                           xanchor="left", yanchor="bottom", xshift=4)

    # Add today lines to all rows
    _today_vlines(fig, today_rd, n_rows)

    # ── Auto-frame x-axis + cap y-axis to visible window ──
    s_doy, e_doy = get_seasonal_range()
    x0 = doy_to_ref_date(s_doy)
    x1 = doy_to_ref_date(min(e_doy, 366))

    # Cap y-axis on Q panel to 95th percentile within visible DOY window
    vis_hist = hist[(hist["doy"] >= s_doy) & (hist["doy"] <= e_doy)]
    vis_cur = cur[(cur["doy"] >= s_doy) & (cur["doy"] <= e_doy)] if not cur.empty else pd.DataFrame()
    if not vis_hist.empty:
        q_cap = vis_hist["q"].quantile(0.97)
        if not vis_cur.empty:
            q_cap = max(q_cap, vis_cur["q"].max() * 1.1)
        fig.update_yaxes(range=[0, q_cap * 1.05], row=1, col=1)

    # Cap dQ y-axis to visible range
    if has_roc and not vis_hist.empty and "dq_7d" in vis_hist.columns:
        dq_vis = vis_hist["dq_7d"].dropna()
        if not dq_vis.empty:
            dq_cap = max(abs(dq_vis.quantile(0.03)), abs(dq_vis.quantile(0.97))) * 1.3
            fig.update_yaxes(range=[-dq_cap, dq_cap], row=2, col=1)

    yr_range = f"{hist['year'].min()}–{hist['year'].max()}"
    fig.update_layout(
        title=dict(text=f"<b>{name}</b> — Runoff Timing · {focus_year} vs Historical ({yr_range})",
                   font=dict(size=15, color=TEXT_CLR), y=0.98, yanchor="top"),
        template=PLTLY,
        legend=dict(orientation="h", y=1.0, yanchor="bottom",
                    font=dict(size=11, color=TEXT_CLR),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=58, b=32, l=65, r=12),
        hovermode="x unified", dragmode="zoom", barmode="relative",
    )
    fig.update_xaxes(tickformat="%b %d", gridcolor=GRID_CLR,
                     range=[x0, x1], row=n_rows, col=1)
    for r in range(1, n_rows):
        fig.update_xaxes(showticklabels=False, gridcolor=GRID_CLR, row=r, col=1)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: FULL YEAR COMPARISON (original percentile view, full year)
# ═════════════════════════════════════════════════════════════════════════════

def plot_full_year(df, name, focus_year):
    hist = df[df["year"] != focus_year].copy()
    cur = df[df["year"] == focus_year].copy()
    if hist.empty:
        return empty_fig("Not enough historical data")

    has_wt = df["water_temp_f"].notna().sum() > 100
    has_at = df["air_temp_mean_f"].notna().sum() > 100

    panels = [("q", C_Q, ",.0f", "tozero", "Discharge (cfs)")]
    if has_wt:
        panels.append(("water_temp_f", C_WT, ".1f", "normal", "Water Temp (°F)"))
    if has_at:
        panels.append(("air_temp_mean_f", C_AT, ".0f", "normal", "Air Temp (°F)"))
    n = len(panels)
    heights = {1: [1.0], 2: [0.55, 0.45], 3: [0.40, 0.30, 0.30]}[n]

    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=heights)

    for i, (param, colors, fmt, rm, ylabel) in enumerate(panels):
        _add_percentile_band(fig, hist, param, i + 1, colors["fill"], colors["focus"],
                              cur=cur, ylabel=ylabel, fmt=fmt, rangemode=rm)

    # Today marker on all rows
    today_rd = doy_to_ref_date(datetime.now().timetuple().tm_yday)
    _today_vlines(fig, today_rd, n)

    yr_range = f"{hist['year'].min()}–{hist['year'].max()}"
    fig.update_layout(
        title=dict(text=f"<b>{name}</b> — Full Year · {focus_year} vs Historical ({yr_range})",
                   font=dict(size=15, color=TEXT_CLR), y=0.98, yanchor="top"),
        template=PLTLY,
        legend=dict(orientation="h", y=1.0, yanchor="bottom",
                    font=dict(size=11, color=TEXT_CLR),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=58, b=32, l=65, r=12),
        hovermode="x unified", dragmode="zoom",
    )
    fig.update_xaxes(tickformat="%b", dtick="M1", gridcolor=GRID_CLR, row=n, col=1)
    for r in range(1, n):
        fig.update_xaxes(showticklabels=False, gridcolor=GRID_CLR, row=r, col=1)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: DAILY RECORD + TREND
# ═════════════════════════════════════════════════════════════════════════════

def plot_daily_trend(df, name):
    has_wt = df["water_temp_f"].notna().sum() > 100
    has_at = df["air_temp_mean_f"].notna().sum() > 100
    n = 1 + int(has_wt) + int(has_at)
    if n == 3:
        heights = [0.38, 0.30, 0.32]
    elif n == 2:
        heights = [0.55, 0.45]
    else:
        heights = [1.0]

    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=heights)
    df_s = df.sort_values("date").copy()
    row = 1

    # Discharge
    fig.add_trace(go.Scatter(
        x=df_s["date"], y=df_s["q"], mode="lines",
        line=dict(color="#93c5fd", width=0.8), opacity=0.55,
        name="Daily Q", hovertemplate="%{x|%b %d, %Y}: %{y:,.0f} cfs<extra></extra>"), row=row, col=1)
    df_s["q30"] = df_s["q"].rolling(30, center=True, min_periods=7).mean()
    fig.add_trace(go.Scatter(
        x=df_s["date"], y=df_s["q30"], mode="lines",
        line=dict(color="#3b82f6", width=2.5), name="30-Day Avg Q"), row=row, col=1)

    annual = df.groupby("year")["q"].mean().reset_index()
    annual["mid"] = pd.to_datetime(annual["year"].astype(str) + "-07-01")
    if len(annual) > 5:
        x_n = annual["year"].values.astype(float)
        slope, intercept, r, p, se = stats.linregress(x_n, annual["q"].values)
        sign = "+" if slope > 0 else ""
        fig.add_trace(go.Scatter(
            x=annual["mid"], y=slope * x_n + intercept,
            line=dict(color="#ef4444", width=2, dash="dash"),
            name=f"Trend ({sign}{slope:.1f} cfs/yr, p={p:.3f})"), row=row, col=1)

    fig.update_yaxes(title_text="Discharge (cfs)", rangemode="tozero",
                     gridcolor=GRID_CLR, row=row, col=1)
    row += 1

    if has_wt:
        s = df_s.dropna(subset=["water_temp_f"]).copy()
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["water_temp_f"], mode="lines",
            line=dict(color="#fdba74", width=0.8), opacity=0.55, name="Daily Water Temp"), row=row, col=1)
        s["avg30"] = s["water_temp_f"].rolling(30, center=True, min_periods=7).mean()
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["avg30"], mode="lines",
            line=dict(color="#ea580c", width=2.5), name="30-Day Water Temp"), row=row, col=1)
        fig.update_yaxes(title_text="Water Temp (°F)", gridcolor=GRID_CLR, row=row, col=1)
        row += 1

    if has_at:
        s = df_s.dropna(subset=["air_temp_mean_f"]).copy()
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["air_temp_mean_f"], mode="lines",
            line=dict(color="#86efac", width=0.8), opacity=0.55, name="Daily Air Temp"), row=row, col=1)
        s["avg30"] = s["air_temp_mean_f"].rolling(30, center=True, min_periods=7).mean()
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["avg30"], mode="lines",
            line=dict(color="#16a34a", width=2.5), name="30-Day Air Temp"), row=row, col=1)
        # Max/min band
        at = df_s.dropna(subset=["air_temp_max_f", "air_temp_min_f"]).sort_values("date")
        if not at.empty:
            fig.add_trace(go.Scatter(
                x=pd.concat([at["date"], at["date"][::-1]]),
                y=pd.concat([at["air_temp_max_f"].rolling(7, min_periods=1).mean(),
                             at["air_temp_min_f"].rolling(7, min_periods=1).mean().iloc[::-1]]),
                fill="toself", fillcolor="rgba(22,163,74,0.06)",
                line=dict(width=0), showlegend=False, hoverinfo="skip"), row=row, col=1)
        fig.update_yaxes(title_text="Air Temp (°F)", gridcolor=GRID_CLR, row=row, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{name}</b> — Daily Record with Trend",
                   font=dict(size=15, color=TEXT_CLR), y=0.98, yanchor="top"),
        template=PLTLY,
        legend=dict(orientation="h", y=1.0, yanchor="bottom",
                    font=dict(size=11, color=TEXT_CLR),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=58, b=40, l=65, r=12),
        hovermode="x unified", dragmode="zoom",
    )
    fig.update_xaxes(
        gridcolor=GRID_CLR,
        rangeselector=dict(buttons=[
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ], bgcolor="#454b56", activecolor="#3b82f6", font=dict(size=12, color=TEXT_CLR)),
        rangeslider=dict(visible=True, thickness=0.04, bgcolor=CARD_BG),
        row=n, col=1,
    )
    for r in range(1, n):
        fig.update_xaxes(showticklabels=False, gridcolor=GRID_CLR, row=r, col=1)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY METRICS — runoff-focused
# ═════════════════════════════════════════════════════════════════════════════

def build_summary(df, name, focus_year, metrics):
    if df.empty:
        return html.Div("No data loaded", style={"fontSize": "13px", "padding": "10px",
                                                   "color": TEXT_DIM, "textAlign": "center"})

    KPI_BG = "#2a2f38"
    KPI_BORDER = "#3a4050"

    def kpi_card(children):
        return html.Div(children, style={
            "backgroundColor": KPI_BG, "borderRadius": "6px",
            "border": f"1px solid {KPI_BORDER}", "padding": "8px 10px",
            "flex": "1", "minWidth": "115px",
        })

    def pctile_bar(pct):
        """Mini horizontal percentile bar."""
        if pct is None:
            return None
        # Color based on percentile
        if pct < 10:
            clr = "#ef4444"
        elif pct < 25:
            clr = "#f97316"
        elif pct < 75:
            clr = "#22d3ee"
        elif pct < 90:
            clr = "#3b82f6"
        else:
            clr = "#8b5cf6"
        return html.Div([
            html.Div(style={
                "width": f"{min(pct, 100):.0f}%", "height": "4px",
                "backgroundColor": clr, "borderRadius": "2px",
                "transition": "width 0.5s ease",
            }),
        ], style={
            "width": "100%", "height": "4px", "backgroundColor": "#1e2228",
            "borderRadius": "2px", "marginTop": "5px",
        })

    cards = []

    # ── 1. CURRENT FLOW ──
    lq = metrics.get("latest_q")
    pct = metrics.get("current_pctile")
    med_q = metrics.get("hist_median_q")
    if lq is not None:
        pct_color = "#ef4444" if pct and pct < 10 else "#f97316" if pct and pct < 25 else \
                    "#22d3ee" if pct and pct < 75 else "#3b82f6" if pct and pct < 90 else \
                    "#8b5cf6" if pct else "#22d3ee"
        sub_parts = []
        if pct is not None:
            sub_parts.append(f"{pct:.0f}th percentile")
        if med_q is not None:
            sub_parts.append(f"Median: {med_q:,.0f}")
        cards.append(kpi_card([
            html.Div("Current flow", style={"fontSize": "10px", "color": TEXT_DIM,
                                              "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div([
                html.Span(f"{lq:,.0f}", style={"fontSize": "22px", "fontWeight": "700",
                                                  "color": pct_color}),
                html.Span(" cfs", style={"fontSize": "11px", "color": TEXT_DIM, "marginLeft": "3px"}),
            ]),
            html.Div(f"As of {metrics['latest_date']:%b %d}",
                      style={"fontSize": "9px", "color": TEXT_DIM, "marginTop": "2px"}),
            pctile_bar(pct),
        ]))

    # ── 2. 24HR CHANGE ──
    dq_24h = metrics.get("dq_24h")
    dq_pct = metrics.get("dq_24h_pct")
    if dq_24h is not None:
        if dq_24h > 1:
            arrow = "↑"
            chg_color = "#22d3ee"
            label = "increase"
        elif dq_24h < -1:
            arrow = "↓"
            chg_color = "#f87171"
            label = "decrease"
        else:
            arrow = "→"
            chg_color = "#fbbf24"
            label = "no change"
        cards.append(kpi_card([
            html.Div("24hr change", style={"fontSize": "10px", "color": TEXT_DIM,
                                             "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div([
                html.Span(f"{arrow}", style={"fontSize": "16px", "marginRight": "3px"}),
                html.Span(f"{abs(dq_pct):.1f}" if dq_pct else "0",
                           style={"fontSize": "22px", "fontWeight": "700", "color": chg_color}),
                html.Span("%", style={"fontSize": "13px", "color": chg_color}),
            ]),
            html.Div(f"{abs(dq_24h):,.0f} cfs {label}",
                      style={"fontSize": "9px", "color": TEXT_DIM, "marginTop": "2px"}),
        ]))

    # ── 3. 14-DAY TREND ──
    rdq = metrics.get("recent_dq")
    if rdq is not None:
        if rdq > 5:
            traj = "Rising"
            arrow = "▲"
            tc = "#22d3ee"
        elif rdq < -5:
            traj = "Falling"
            arrow = "▼"
            tc = "#f87171"
        else:
            traj = "Stable"
            arrow = "●"
            tc = "#fbbf24"
        # Mini trend indicator dots
        trend_dots = html.Div([
            html.Div(style={
                "width": "6px", "height": "6px", "borderRadius": "50%",
                "backgroundColor": tc, "opacity": "1.0" if i == 2 else "0.3",
                "display": "inline-block", "margin": "0 1px",
            }) for i in range(3)
        ], style={"marginTop": "4px"})
        cards.append(kpi_card([
            html.Div("14-day trend", style={"fontSize": "10px", "color": TEXT_DIM,
                                              "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div([
                html.Span(arrow, style={"fontSize": "14px", "marginRight": "4px", "color": tc}),
                html.Span(traj, style={"fontSize": "20px", "fontWeight": "700", "color": tc}),
            ]),
            html.Div(f"{rdq:+,.0f} cfs/day avg",
                      style={"fontSize": "9px", "color": TEXT_DIM, "marginTop": "2px"}),
            trend_dots,
        ]))

    # ── 4. 7-DAY MAX ──
    q7max = metrics.get("q_7d_max")
    q7date = metrics.get("q_7d_max_date")
    if q7max is not None:
        cards.append(kpi_card([
            html.Div("7-day max", style={"fontSize": "10px", "color": TEXT_DIM,
                                           "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div([
                html.Span(f"{q7max:,.0f}", style={"fontSize": "22px", "fontWeight": "700",
                                                     "color": "#60a5fa"}),
                html.Span(" cfs", style={"fontSize": "11px", "color": TEXT_DIM, "marginLeft": "3px"}),
            ]),
            html.Div(f"Reached {q7date:%b %d}" if hasattr(q7date, 'strftime') else "",
                      style={"fontSize": "9px", "color": TEXT_DIM, "marginTop": "2px"}),
        ]))

    # ── 5. RUNOFF ONSET ──
    fo = metrics.get("focus_onset_doy")
    ho = metrics.get("hist_onset_doy")
    if fo is not None or ho is not None:
        if fo is not None:
            onset_val = doy_to_label(fo)
            onset_color = TEXT_CLR
            sub = ""
            if ho is not None:
                d = fo - ho
                if d > 5:
                    onset_color = "#f87171"
                    sub = f"{d}d late vs median ({doy_to_label(ho)})"
                elif d < -5:
                    onset_color = "#22d3ee"
                    sub = f"{abs(d)}d early vs median ({doy_to_label(ho)})"
                else:
                    sub = f"Near median ({doy_to_label(ho)})"
        else:
            onset_val = "Not yet"
            onset_color = TEXT_DIM
            sub = f"Median: {doy_to_label(ho)}" if ho else ""
        cards.append(kpi_card([
            html.Div("Runoff onset", style={"fontSize": "10px", "color": TEXT_DIM,
                                              "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div(onset_val, style={"fontSize": "20px", "fontWeight": "700",
                                        "color": onset_color}),
            html.Div(sub, style={"fontSize": "9px", "color": TEXT_DIM, "marginTop": "2px"}) if sub else None,
        ]))

    # ── 6. AIR TEMP ──
    at = metrics.get("air_temp")
    if at is not None and not np.isnan(at):
        at_max = metrics.get("air_temp_max")
        at_min = metrics.get("air_temp_min")
        if at > 50:
            at_color = "#f97316"
        elif at > 32:
            at_color = "#fbbf24"
        else:
            at_color = "#38bdf8"
        # Freezing indicator
        above_freezing = at > 32
        freeze_dot = html.Div(style={
            "width": "8px", "height": "8px", "borderRadius": "50%",
            "backgroundColor": "#22c55e" if above_freezing else "#38bdf8",
            "display": "inline-block", "marginRight": "5px", "verticalAlign": "middle",
        })
        sub_t = ""
        if at_max is not None and at_min is not None and not np.isnan(at_max):
            sub_t = f"High {at_max:.0f}° / Low {at_min:.0f}°"
        cards.append(kpi_card([
            html.Div("Air temp", style={"fontSize": "10px", "color": TEXT_DIM,
                                          "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div([
                freeze_dot,
                html.Span(f"{at:.0f}", style={"fontSize": "22px", "fontWeight": "700",
                                                "color": at_color}),
                html.Span("°F", style={"fontSize": "13px", "color": at_color}),
            ]),
            html.Div(sub_t, style={"fontSize": "9px", "color": TEXT_DIM, "marginTop": "2px"}) if sub_t else None,
            html.Div("Above freezing" if above_freezing else "Below freezing",
                      style={"fontSize": "9px", "color": "#22c55e" if above_freezing else "#38bdf8",
                             "marginTop": "2px"}),
        ]))

    return html.Div(cards, style={
        "display": "flex", "flexWrap": "wrap", "gap": "5px",
        "padding": "6px", "overflowY": "auto", "maxHeight": "100%",
    })


# ═════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server  # Expose for gunicorn: gunicorn streamflow_dev:server

app.layout = html.Div(style={
    "backgroundColor": BG, "height": "100vh", "margin": 0, "padding": 0,
    "fontFamily": "Inter, Segoe UI, sans-serif",
    "display": "flex", "flexDirection": "column", "overflow": "hidden",
    "color": TEXT_CLR,
}, children=[

    # HEADER
    html.Div(style={
        "backgroundColor": HEADER_BG, "padding": "8px 16px",
        "borderBottom": f"2px solid {ACCENT}", "flexShrink": "0",
    }, children=[
        dbc.Row([
            dbc.Col(html.Div("Western MT Runoff Explorer",
                              style={"color": "#fff", "fontWeight": 700,
                                     "fontSize": "17px", "whiteSpace": "nowrap"}), width="auto"),
            dbc.Col(dcc.Dropdown(id="station-dd", options=DROPDOWN_OPTIONS,
                                  value="12340500", clearable=False,
                                  style={"fontSize": "13px", "minWidth": "280px"},
                                  className="dash-dark"), width="auto"),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Year", style={"fontSize": "13px", "padding": "3px 6px"}),
                html.Div(dcc.Dropdown(id="year-dd", value=datetime.now().year, clearable=False,
                                      style={"fontSize": "13px", "width": "85px"},
                                      className="dash-dark"), style={"width": "85px"}),
            ], size="sm"), width="auto"),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("From", style={"fontSize": "13px", "padding": "3px 6px"}),
                dbc.Input(id="start-year", type="number", value=1980,
                          min=1900, max=2026, size="sm",
                          style={"width": "72px", "fontSize": "13px"}),
            ], size="sm"), width="auto"),
            dbc.Col(dbc.Button("Load Data", id="load-btn", color="primary", size="sm",
                               style={"fontWeight": 600, "fontSize": "13px"}), width="auto"),
            dbc.Col(html.Div(style={"width": "10px"}), width="auto"),
            dbc.Col(dbc.InputGroup([
                dbc.Input(id="search-input", placeholder="Search USGS site...", size="sm",
                          style={"fontSize": "13px"}),
                dbc.Button("Search", id="search-btn", outline=True, color="secondary", size="sm",
                           style={"fontSize": "13px"}),
            ], size="sm"), width="auto"),
            dbc.Col(html.Div(style={"width": "6px"}), width="auto"),
            dbc.Col(dbc.Button("🗺", id="map-toggle-btn", outline=True, color="secondary",
                               size="sm", title="Toggle map panel",
                               style={"fontSize": "15px", "padding": "2px 8px"}), width="auto"),
        ], align="center", className="g-2 flex-nowrap"),
    ]),

    # STATUS
    html.Div(style={
        "backgroundColor": "#272c34", "padding": "3px 16px",
        "borderBottom": f"1px solid {GRID_CLR}", "flexShrink": "0",
    }, children=[
        html.Div(id="status-msg", style={"fontSize": "13px", "color": TEXT_DIM}),
    ]),

    # MAIN
    html.Div(style={
        "flex": "1", "minHeight": "0", "padding": "6px 8px 2px 8px",
        "display": "flex", "gap": "6px", "overflow": "hidden",
    }, children=[

        # LEFT: map + summary metrics
        html.Div(id="map-panel", style={
            "width": "25%", "minWidth": "260px", "display": "flex",
            "flexDirection": "column", "gap": "5px",
            "transition": "all 0.3s ease", "overflow": "hidden",
        }, children=[
            html.Div(style={
                "backgroundColor": CARD_BG, "borderRadius": "6px",
                "border": f"1px solid {GRID_CLR}", "overflow": "hidden",
                "flex": "1", "minHeight": "0",
            }, children=[
                dcc.Graph(id="station-map", figure=build_station_map("12340500"),
                          config=MAP_CFG, responsive=True,
                          style={"height": "100%", "width": "100%"}),
            ]),
            html.Div(id="summary-card", style={
                "backgroundColor": CARD_BG, "borderRadius": "6px",
                "border": f"1px solid {GRID_CLR}", "flexShrink": "0",
                "maxHeight": "45%", "overflowY": "auto",
            }),
        ]),

        # RIGHT: tabbed plots
        html.Div(style={
            "flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column",
        }, children=[
            html.Div(style={
                "backgroundColor": CARD_BG, "borderRadius": "6px",
                "border": f"1px solid {GRID_CLR}", "overflow": "hidden",
                "flex": "1", "minHeight": "0",
                "display": "flex", "flexDirection": "column",
            }, children=[
                html.Div(style={
                    "flex": "1", "minHeight": "0",
                    "display": "flex", "flexDirection": "column",
                }, children=[
                    dbc.Tabs(id="plot-tabs", active_tab="t1",
                             style={"padding": "2px 6px", "flexShrink": "0"}, children=[
                        dbc.Tab(label="⏱ Runoff Timing",        tab_id="t1"),
                        dbc.Tab(label="📊 Full Year Comparison", tab_id="t2"),
                        dbc.Tab(label="📈 Daily Record & Trend", tab_id="t3"),
                    ]),
                    html.Div(id="graph-wrap", style={
                        "flex": "1", "minHeight": "0", "position": "relative",
                    }, children=[
                        dcc.Graph(id="active-plot", config=GRAPH_CFG,
                                  responsive=True,
                                  style={"position": "absolute", "top": 0,
                                         "left": 0, "right": 0, "bottom": 0}),
                    ]),
                ]),
            ]),
        ]),
    ]),

    # Stores
    dcc.Store(id="data-store"),
    dcc.Store(id="fig-t1"),
    dcc.Store(id="fig-t2"),
    dcc.Store(id="fig-t3"),
    dcc.Store(id="map-visible", data=True),
])


# ═════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("map-visible", "data"),
    Output("map-panel", "style"),
    Input("map-toggle-btn", "n_clicks"),
    State("map-visible", "data"),
    prevent_initial_call=True,
)
def toggle_map(n, visible):
    new_vis = not visible
    if new_vis:
        return True, {"width": "25%", "minWidth": "260px", "display": "flex",
                       "flexDirection": "column", "gap": "5px",
                       "transition": "all 0.3s ease", "overflow": "hidden", "opacity": "1"}
    return False, {"width": "0px", "minWidth": "0px", "display": "flex",
                    "flexDirection": "column", "gap": "5px",
                    "transition": "all 0.3s ease", "overflow": "hidden",
                    "opacity": "0", "padding": "0"}


@app.callback(
    Output("station-dd", "value", allow_duplicate=True),
    Input("station-map", "clickData"),
    prevent_initial_call=True,
)
def map_click(click_data):
    if click_data is None:
        return no_update
    sid = click_data["points"][0].get("customdata")
    return sid if sid else no_update


@app.callback(Output("station-map", "figure"), Input("station-dd", "value"))
def sync_map(site):
    return build_station_map(site)


@app.callback(
    Output("data-store", "data"),
    Output("status-msg", "children"),
    Output("year-dd", "options"),
    Output("year-dd", "value"),
    Input("load-btn", "n_clicks"),
    State("station-dd", "value"),
    State("start-year", "value"),
    State("year-dd", "value"),
    prevent_initial_call=True,
)
def load_data(n, site, start_yr, current_focus):
    df = fetch_all(site, start=f"{start_yr}-01-01")
    if df.empty:
        return None, "⚠️ No data returned.", [], None
    # Compute derived columns
    df = compute_derived(df)
    years = sorted(df["year"].unique())
    yr_opts = [{"label": str(y), "value": y} for y in years]
    focus = current_focus if current_focus in years else years[-1]
    name = sname(site)
    has_wt = df["water_temp_f"].notna().sum() > 0
    has_at = df["air_temp_mean_f"].notna().sum() > 0
    flags = []
    if has_wt:
        flags.append("water temp ✓")
    if has_at:
        flags.append("air temp ✓")
    extra = " · ".join(flags)
    msg = f"✓  {name}  ·  {len(df):,} records  ·  {years[0]}–{years[-1]} ({len(years)} yrs)  ·  {extra}"
    return df.to_json(date_format="iso"), msg, yr_opts, focus


@app.callback(
    Output("fig-t1", "data"),
    Output("fig-t2", "data"),
    Output("fig-t3", "data"),
    Output("summary-card", "children"),
    Input("data-store", "data"),
    Input("year-dd", "value"),
    State("station-dd", "value"),
)
def compute_figs(json_data, focus_year, site):
    ef = empty_fig()
    if json_data is None:
        return ef.to_json(), ef.to_json(), ef.to_json(), build_summary(pd.DataFrame(), "", None, {})
    df = pd.read_json(json_data)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear
    df = df[df["q"] >= 0].copy()
    # Recompute derived (lost in JSON round-trip)
    df = compute_derived(df)
    name = sname(site)
    metrics = compute_timing_metrics(df, focus_year)
    f1 = plot_runoff_timing(df, name, focus_year)
    f2 = plot_full_year(df, name, focus_year)
    f3 = plot_daily_trend(df, name)
    return f1.to_json(), f2.to_json(), f3.to_json(), build_summary(df, name, focus_year, metrics)


@app.callback(
    Output("active-plot", "figure"),
    Input("plot-tabs", "active_tab"),
    Input("fig-t1", "data"),
    Input("fig-t2", "data"),
    Input("fig-t3", "data"),
)
def swap_tab(tab, j1, j2, j3):
    lookup = {"t1": j1, "t2": j2, "t3": j3}
    j = lookup.get(tab)
    if j:
        return pio.from_json(j)
    return empty_fig()


@app.callback(
    Output("search-results", "children", allow_duplicate=True) if False else Output("status-msg", "children", allow_duplicate=True),
    Output("station-dd", "options"),
    Input("search-btn", "n_clicks"),
    State("search-input", "value"),
    State("station-dd", "options"),
    prevent_initial_call=True,
)
def do_search(n, query, existing_opts):
    if not query:
        return "", existing_opts
    results = search_sites(query)
    if results is None or (hasattr(results, "empty") and results.empty):
        return "No results found", existing_opts
    new_opts = list(existing_opts)
    existing_vals = {o["value"] for o in existing_opts}
    msgs = []
    for _, row in results.head(8).iterrows():
        sid, nm = row.get("site_no", ""), row.get("station_nm", "")
        lat, lon = row.get("dec_lat_va"), row.get("dec_long_va")
        if sid and sid not in existing_vals:
            new_opts.append({"label": f"{nm} ({sid})", "value": sid})
            existing_vals.add(sid)
            if sid not in STATIONS and lat and lon:
                STATIONS[sid] = {"name": nm, "basin": "Search Result",
                                 "lat": float(lat), "lon": float(lon)}
        msgs.append(f"{nm} ({sid})")
    return f"Found {len(results)} — " + " · ".join(msgs[:3]), new_opts


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("RENDER") is None
    print(f"\n  Western MT Runoff Explorer")
    print(f"  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
