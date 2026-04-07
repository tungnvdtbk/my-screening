import sys, types
class _CM:
    def __init__(self,*a,**kw): pass
    def __enter__(self): return self
    def __exit__(self,*a): return False
    def __getattr__(self,_): return lambda *a,**kw: _CM()
_st=types.ModuleType("streamlit")
_st.cache_data=lambda **kw:(lambda f:f)
_st.session_state={}
_st.button=lambda *a,**kw:False
_st.radio=lambda *a,**kw:"VN30"
_st.checkbox=lambda *a,**kw:True
_st.columns=lambda n,**kw:[_CM() for _ in range(n if isinstance(n,int) else len(n))]
_st.sidebar=_CM()
_st.spinner=_CM
_st.expander=_CM
_st.progress=lambda *a,**kw:_CM()
for _a in ["set_page_config","title","caption","info","warning","error","subheader","markdown","metric","dataframe","line_chart","plotly_chart","selectbox","slider","download_button","stop","rerun","header","write"]:
    setattr(_st,_a,lambda *a,**kw:None)
sys.modules["streamlit"]=_st
sys.modules.setdefault("vnstock3",types.ModuleType("vnstock3"))

import yfinance as yf
import app

SYMS = ["NLG.VN","PDR.VN","DXG.VN","HPG.VN","VHM.VN","SSI.VN","STB.VN","NVL.VN","DIG.VN","KBC.VN","CII.VN","SJS.VN","TDH.VN","HDC.VN","AGG.VN","BCM.VN","VPB.VN","TCB.VN"]

print("\nVCP Reversal debug:")
print(f"{'Sym':<6} {'r1>r2>r3':>10} {'r3<8':>6} {'vol_dry':>8} {'prior_dn':>10} {'near52L':>8} {'result':>8}")
print("-"*65)

for sym in SYMS:
    df = yf.Ticker(sym).history(period="2y")
    if df is None or df.empty or len(df) < 80:
        continue
    if df.index.tz:
        df.index = df.index.tz_localize(None)
    t = sym.replace(".VN","")

    p1,p2,p3 = df.iloc[-60:-40],df.iloc[-40:-20],df.iloc[-20:]
    r1 = app._range_pct(p1)
    r2 = app._range_pct(p2)
    r3 = app._range_pct(p3)
    v1 = float(p1["Volume"].mean())
    v2 = float(p2["Volume"].mean())
    v3 = float(p3["Volume"].mean())
    prior_mid = float(df.iloc[-80:-60]["Close"].mean())
    p1_mid    = float(p1["Close"].mean())
    price     = float(df["Close"].iloc[-1])
    low52     = float(df["Low"].iloc[-252:].min()) if len(df) >= 252 else float(df["Low"].min())

    contracting = r1 > r2 > r3
    tight       = r3 < 8.0
    vol_dry     = v3 < v2 * 0.90 and v2 < v1 * 0.95
    prior_dn    = prior_mid > p1_mid * 1.05
    near_low    = price <= low52 * 1.25
    result      = app._check_vcp_reversal(df)

    print(f"{t:<6} {str(contracting)+'('+str(round(r1,1))+'->'+str(round(r3,1))+')':>12} "
          f"{str(tight):>6} {str(vol_dry):>8} "
          f"{str(prior_dn)+'('+str(round(prior_mid/p1_mid,2))+'x)':>12} "
          f"{str(near_low)+'('+str(round(price/low52,2))+'x)':>10} "
          f"{'SIGNAL' if result else 'none':>8}")

print("\nTriangle Reversal debug:")
for sym in SYMS:
    df = yf.Ticker(sym).history(period="2y")
    if df is None or df.empty or len(df) < 80:
        continue
    if df.index.tz:
        df.index = df.index.tz_localize(None)
    t = sym.replace(".VN","")
    prior_mid   = float(df["Close"].iloc[-80:-60].mean())
    current_mid = float(df["Close"].iloc[-20:].mean())
    prior_dn = prior_mid > current_mid * 1.05
    result = app._check_triangle_reversal(df)
    print(f"  {t:<6} prior_dn={str(prior_dn):>5} ({round(prior_mid/current_mid,2)}x) → {'SIGNAL' if result else 'none'}")
