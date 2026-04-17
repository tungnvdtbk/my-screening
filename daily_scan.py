"""
daily_scan.py — Headless daily stock scan for all VN100 symbols.

Runs all 6 scanners, builds an HTML report, and sends it via Telegram.
Can also be run locally: `python daily_scan.py` (saves HTML without sending).

Environment variables (set as GitHub secrets for CI):
  TELEGRAM_BOT_TOKEN  — from @BotFather
  TELEGRAM_CHAT_ID    — your chat/group ID
"""

import io
import json
import os
import sys
import types
import traceback
import urllib.request
from datetime import datetime


# ── Stub Streamlit before importing app ──────────────────────────────
class _CM:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, _): return lambda *a, **kw: _CM()
    def __iter__(self): return iter([])

def _noop(*a, **kw): return _CM()

_st = types.ModuleType("streamlit")
_st.cache_data    = lambda **kw: (lambda f: f)
_st.session_state = {}
_st.button        = lambda *a, **kw: False
_st.checkbox      = lambda *a, **kw: True
_st.columns       = lambda n, *a, **kw: [_CM() for _ in (range(n) if isinstance(n, int) else n)]
_st.sidebar       = _CM()
_st.spinner       = _CM
_st.tabs          = lambda labels: [_CM() for _ in labels]
for _attr in [
    "set_page_config", "title", "caption", "info", "warning", "error",
    "subheader", "markdown", "metric", "dataframe", "selectbox",
    "slider", "download_button", "stop", "rerun", "header", "write",
    "divider", "plotly_chart", "image", "text_input",
]:
    setattr(_st, _attr, _noop)
sys.modules["streamlit"] = _st
sys.modules.setdefault("vnstock3", types.ModuleType("vnstock3"))

# Now safe to import app
from app import (                   # noqa: E402
    VN100_STOCKS,
    get_vnindex_data,
    run_scan,
    run_swing_scan,
    run_pa_scan,
    run_mr_scan,
    run_climax_scan,
    run_pinbar_4h_scan,
)

# ── Scanner definitions ──────────────────────────────────────────────
SCANNERS = [
    ("Breakout / NR7 / Gap / PinBar / TrendFilter", "main"),
    ("Swing Filter",                        "swing"),
    ("Price Action",                        "pa"),
    ("Mean Reversion",                      "mr"),
    ("Climax Reversal",                     "climax"),
    ("Pin Bar 4H",                          "pinbar4h"),
]

TIER_FIELDS = {
    "main":      lambda r: r.get("bo_tier") or r.get("nr7_tier") or r.get("gap_tier") or r.get("pin_tier") or r.get("tf_tier") or "",
    "swing":     lambda r: r.get("sw_tier", ""),
    "pa":        lambda r: r.get("pa_tier", ""),
    "mr":        lambda r: r.get("mr_tier", ""),
    "climax":    lambda r: r.get("cx_tier", ""),
    "pinbar4h":  lambda r: r.get("pin_tier", ""),
}


# ── Run all scans ────────────────────────────────────────────────────
def run_all_scans() -> dict:
    print("Fetching VNINDEX data...")
    vnindex_df = get_vnindex_data()

    results = {
        "main": [], "swing": [], "pa": [], "mr": [],
        "climax": [], "pinbar4h": [], "market_down": False, "errors": [],
    }

    # 1. Main scan (breakout, gap, NR7, pin bar, trend filter)
    try:
        print("Running main scan (Breakout/Gap/NR7/PinBar/TrendFilter)...")
        sigs, mkt_down = run_scan(VN100_STOCKS, use_cache=True, vnindex_df=vnindex_df)
        results["main"] = sigs
        results["market_down"] = mkt_down
        print(f"  -> {len(sigs)} signals")
    except Exception as e:
        results["errors"].append(f"Main scan: {e}")
        traceback.print_exc()

    # 2. Swing
    try:
        print("Running Swing Filter scan...")
        results["swing"] = run_swing_scan(VN100_STOCKS, use_cache=True, vnindex_df=vnindex_df)
        print(f"  -> {len(results['swing'])} signals")
    except Exception as e:
        results["errors"].append(f"Swing scan: {e}")
        traceback.print_exc()

    # 3. Price Action
    try:
        print("Running Price Action scan...")
        results["pa"] = run_pa_scan(VN100_STOCKS, use_cache=True, vnindex_df=vnindex_df)
        print(f"  -> {len(results['pa'])} signals")
    except Exception as e:
        results["errors"].append(f"PA scan: {e}")
        traceback.print_exc()

    # 4. Mean Reversion
    try:
        print("Running Mean Reversion scan...")
        results["mr"] = run_mr_scan(VN100_STOCKS, use_cache=True, vnindex_df=vnindex_df)
        print(f"  -> {len(results['mr'])} signals")
    except Exception as e:
        results["errors"].append(f"MR scan: {e}")
        traceback.print_exc()

    # 5. Climax
    try:
        print("Running Climax Reversal scan...")
        results["climax"] = run_climax_scan(VN100_STOCKS, use_cache=True, vnindex_df=vnindex_df)
        print(f"  -> {len(results['climax'])} signals")
    except Exception as e:
        results["errors"].append(f"Climax scan: {e}")
        traceback.print_exc()

    # 6. Pin Bar 4H
    try:
        print("Running Pin Bar 4H scan...")
        results["pinbar4h"] = run_pinbar_4h_scan(VN100_STOCKS, vnindex_df=vnindex_df)
        print(f"  -> {len(results['pinbar4h'])} signals")
    except Exception as e:
        results["errors"].append(f"Pin Bar 4H scan: {e}")
        traceback.print_exc()

    return results


# ── HTML report builder ──────────────────────────────────────────────
def _signal_row(sig: dict, key: str) -> str:
    tier = TIER_FIELDS[key](sig)
    tier_color = "#00e676" if tier in ("A", "Tier A") else "#ffca28" if tier in ("B", "Tier B") else "#888"
    symbol = sig.get("symbol", "?").replace(".VN", "")
    # Pin bar quality extras
    extras = ""
    ps = sig.get("pin_score")
    if ps is not None:
        extras += f' <span style="color:#90caf9">[Q{ps}/13]</span>'
        sd = sig.get("score_detail", "")
        if sd:
            extras += f' <span style="color:#666;font-size:12px">{sd}</span>'
    return (
        f'<tr>'
        f'<td style="font-weight:bold">{symbol}</td>'
        f'<td>{sig.get("signal", "")}{extras}</td>'
        f'<td style="color:{tier_color}">{tier}</td>'
        f'<td style="text-align:right">{sig.get("close", 0):.2f}</td>'
        f'<td style="text-align:right">{sig.get("sl", 0):.2f}</td>'
        f'<td style="text-align:right">{sig.get("tp", 0):.2f}</td>'
        f'<td style="text-align:right">{sig.get("rr", 0):.1f}</td>'
        f'<td>{sig.get("vol_tier", "")}</td>'
        f'</tr>'
    )


def build_html_report(results: dict) -> str:
    date_str = datetime.now().strftime("%A, %d %B %Y")
    market_label = "BEARISH — market gate active" if results["market_down"] else "BULLISH"
    market_color = "#ef5350" if results["market_down"] else "#00e676"

    total = sum(len(results[k]) for k in ("main", "swing", "pa", "mr", "climax", "pinbar4h"))

    # Table header
    th = (
        '<tr style="border-bottom:2px solid #333">'
        '<th style="text-align:left">Symbol</th>'
        '<th style="text-align:left">Signal</th>'
        '<th style="text-align:left">Tier</th>'
        '<th style="text-align:right">Close</th>'
        '<th style="text-align:right">SL</th>'
        '<th style="text-align:right">TP</th>'
        '<th style="text-align:right">R:R</th>'
        '<th style="text-align:left">Vol Tier</th>'
        '</tr>'
    )

    sections = ""
    for label, key in SCANNERS:
        sigs = results.get(key, [])
        count = len(sigs)
        if count == 0:
            sections += (
                f'<h2 style="color:#90caf9;margin-top:28px">{label}</h2>'
                f'<p style="color:#888">No signals today.</p>'
            )
        else:
            rows = "".join(_signal_row(s, key) for s in sigs)
            sections += (
                f'<h2 style="color:#90caf9;margin-top:28px">{label} ({count})</h2>'
                f'<table style="width:100%;border-collapse:collapse;font-size:14px">'
                f'{th}{rows}</table>'
            )

    # Errors
    err_html = ""
    if results.get("errors"):
        err_items = "".join(f"<li>{e}</li>" for e in results["errors"])
        err_html = (
            f'<h2 style="color:#ef5350;margin-top:28px">Errors</h2>'
            f'<ul style="color:#ef5350">{err_items}</ul>'
        )

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="background:#0a0e17;color:#e0e0e0;font-family:Consolas,'Courier New',monospace;
             padding:20px;max-width:900px;margin:0 auto">
  <h1 style="color:#fff;margin-bottom:4px">VN Stock Daily Scan</h1>
  <p style="color:#aaa;margin-top:0">{date_str}</p>
  <p>
    Market: <strong style="color:{market_color}">{market_label}</strong>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Total signals: <strong>{total}</strong>
  </p>
  <hr style="border-color:#333">
  {sections}
  {err_html}
  <hr style="border-color:#333;margin-top:32px">
  <p style="color:#666;font-size:12px">
    Generated by daily_scan.py — VN Stock Screener
  </p>
</body>
</html>"""


# ── Telegram summary (text for chat message) ────────────────────────
def build_telegram_summary(results: dict) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d (%A)")
    market = "BEARISH" if results["market_down"] else "BULLISH"
    total = sum(len(results[k]) for k in ("main", "swing", "pa", "mr", "climax", "pinbar4h"))

    lines = [
        f"<b>VN Stock Daily Scan — {date_str}</b>",
        f"Market: <b>{market}</b> | Signals: <b>{total}</b>",
        "",
    ]

    for label, key in SCANNERS:
        sigs = results.get(key, [])
        if not sigs:
            lines.append(f"<b>{label}</b> — no signals")
            continue
        lines.append(f"<b>{label} ({len(sigs)})</b>")
        for s in sigs:
            sym = s.get("symbol", "?").replace(".VN", "")
            tier = TIER_FIELDS[key](s)
            rr = s.get("rr", 0)
            sig_type = s.get("signal", "")
            # Pin bar quality score
            pb_extra = ""
            ps = s.get("pin_score")
            if ps is not None:
                pb_extra = f"  Q{ps}/13"
                sd = s.get("score_detail", "")
                if sd:
                    pb_extra += f" ({sd})"
            lines.append(f"  <code>{sym:6s}</code> {sig_type}  {tier}  R:R {rr:.1f}{pb_extra}")
        lines.append("")

    if results.get("errors"):
        lines.append("<b>Errors</b>")
        for e in results["errors"]:
            lines.append(f"  {e}")

    return "\n".join(lines)


# ── Telegram sender ──────────────────────────────────────────────────
def _tg_api(method: str, data: dict = None, files: dict = None) -> dict:
    """Call Telegram Bot API. Uses urllib (stdlib) — no extra deps."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    url = f"https://api.telegram.org/bot{token}/{method}"
    print(f"[Telegram] Calling {method}...")

    if files:
        boundary = "----PythonFormBoundary"
        body = io.BytesIO()
        for k, v in (data or {}).items():
            body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for field, (filename, content, ctype) in files.items():
            body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n".encode())
            body.write(content)
            body.write(b"\r\n")
        body.write(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(url, data=body.getvalue(),
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    else:
        payload = json.dumps(data or {}).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            print(f"[Telegram] {method} OK: {result.get('ok')}")
            return result
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"[Telegram] {method} FAILED: HTTP {e.code} — {error_body}")
        raise
    except Exception as e:
        print(f"[Telegram] {method} FAILED: {e}")
        raise


def send_telegram(summary: str, html_path: str) -> None:
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    print(f"[Telegram] Sending to chat_id={chat_id}")
    print(f"[Telegram] BOT_TOKEN present: {bool(os.environ.get('TELEGRAM_BOT_TOKEN'))}")
    print(f"[Telegram] Summary length: {len(summary)} chars")

    # 1. Send summary message
    _tg_api("sendMessage", {
        "chat_id": chat_id,
        "text": summary,
        "parse_mode": "HTML",
    })
    print("[Telegram] Summary message sent.")

    # 2. Send full HTML report as document
    with open(html_path, "rb") as f:
        html_bytes = f.read()
    print(f"[Telegram] Report file size: {len(html_bytes)} bytes")
    date_str = datetime.now().strftime("%Y-%m-%d")
    _tg_api("sendDocument", {
        "chat_id": chat_id,
        "caption": f"Full report — {date_str}",
    }, files={
        "document": (f"vn_scan_{date_str}.html", html_bytes, "text/html"),
    })
    print("[Telegram] Report document sent.")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_all_scans()
    html = build_html_report(results)
    report_path = "daily_report.html"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved to {report_path}")

    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        try:
            summary = build_telegram_summary(results)
            send_telegram(summary, report_path)
            print(f"\nTelegram message sent to chat {os.environ['TELEGRAM_CHAT_ID']}")
        except Exception as e:
            print(f"\nERROR sending Telegram: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nTelegram not configured — skipping send.")
        print(f"  TELEGRAM_BOT_TOKEN set: {bool(os.environ.get('TELEGRAM_BOT_TOKEN'))}")
        print(f"  TELEGRAM_CHAT_ID set: {bool(os.environ.get('TELEGRAM_CHAT_ID'))}")
