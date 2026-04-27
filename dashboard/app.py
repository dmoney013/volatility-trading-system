"""
Streamlit Dashboard — Interactive visualization for the GARCH + Transformer
options trading analysis system with Long Straddle backtesting.

Launch: streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import (
    DEFAULT_TICKERS, AFFORDABLE_TICKERS, DEVICE, FEATURE_NAMES,
    INITIAL_CAPITAL, HOLDING_PERIOD_DAYS, ENTRY_VOL_THRESHOLD,
)
from data.fetcher import fetch_all_data
from data.feature_engineer import build_features, normalize_features, build_target
from models.garch_model import GARCHVolatilityModel
from models.transformer_model import TransformerTrainer
from signals.generator import generate_signals, compute_iv_from_options
from backtest.engine import LongStraddleBacktester

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Vol Trading System — GARCH + Transformer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
[data-testid="stSidebar"] { background: rgba(15,15,26,0.95); border-right: 1px solid rgba(100,100,255,0.1); }
.metric-card {
    background: linear-gradient(135deg, rgba(30,30,60,0.8), rgba(20,20,50,0.9));
    border: 1px solid rgba(100,100,255,0.15); border-radius: 12px;
    padding: 20px; margin: 8px 0; backdrop-filter: blur(10px);
}
.signal-buy { background: linear-gradient(135deg, rgba(0,180,80,0.2), rgba(0,120,60,0.3)); border-color: rgba(0,200,80,0.4); }
.signal-sell { background: linear-gradient(135deg, rgba(200,50,50,0.2), rgba(150,30,30,0.3)); border-color: rgba(220,60,60,0.4); }
.signal-neutral { background: linear-gradient(135deg, rgba(100,100,100,0.2), rgba(80,80,80,0.3)); border-color: rgba(150,150,150,0.3); }
.win-card { background: linear-gradient(135deg, rgba(0,200,80,0.15), rgba(0,150,60,0.2)); border-color: rgba(0,200,80,0.3); }
.loss-card { background: linear-gradient(135deg, rgba(220,50,50,0.15), rgba(180,30,30,0.2)); border-color: rgba(220,60,60,0.3); }
h1, h2, h3 { color: #e0e0ff !important; }
.stMetric label { color: #8888cc !important; }
.stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }
.stat-item { flex: 1; min-width: 140px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📊 Vol Trading System")
    st.markdown("*GARCH + Transformer + Backtest*")
    st.markdown("---")

    all_tickers = DEFAULT_TICKERS + AFFORDABLE_TICKERS
    ticker = st.selectbox("Select Ticker", all_tickers + ["Custom..."],
                          help="Budget-friendly tickers (F, SOFI, etc.) work best for $150 straddles")
    if ticker == "Custom...":
        ticker = st.text_input("Enter ticker symbol", "F").upper()

    st.markdown("---")
    st.markdown("### ⚙️ Backtest Settings")
    bt_capital = st.number_input("Starting Capital ($)", value=INITIAL_CAPITAL, min_value=50.0, max_value=10000.0, step=25.0)
    bt_holding = st.slider("Holding Period (days)", 1, 30, HOLDING_PERIOD_DAYS)
    bt_threshold = st.slider("Entry Vol Spread (%)", 1, 15, int(ENTRY_VOL_THRESHOLD * 100)) / 100.0

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis + Backtest", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown("**Data:** 2022 → Present")
    st.markdown("**Strategy:** Long Straddle")
    st.caption("No naked options — defined risk only")

# ─── Main Content ────────────────────────────────────────────────
st.markdown(f"# Volatility Analysis: **{ticker}**")

if run_btn:
    # ═══ DATA FETCH ══════════════════════════════════════════════
    with st.status("Fetching market data...", expanded=True) as status:
        st.write(f"Downloading {ticker}, VIX, Treasury data from 2022...")
        data = fetch_all_data(ticker)
        st.write(f"✅ {len(data['prices'])} trading days loaded")
        status.update(label="Data loaded!", state="complete")

    # ═══ GARCH ═══════════════════════════════════════════════════
    with st.status("Fitting GARCH models...", expanded=True) as status:
        garch = GARCHVolatilityModel()
        diagnostics = garch.fit(data["prices"], verbose=False)
        forecast = garch.forecast(horizon=5)
        cond_vol = garch.get_conditional_volatility()
        st.write(f"✅ Best model: {diagnostics['model_name']}")
        status.update(label="GARCH complete!", state="complete")

    # ═══ TRANSFORMER ═════════════════════════════════════════════
    with st.status("Training Transformer...", expanded=True) as status:
        st.write("Building features...")
        features = build_features(data["prices"], data["vix"], data["treasury"], data["options"])
        normalized = normalize_features(features)
        targets = build_target(data["prices"])

        st.write(f"Training on {len(normalized)} samples, {normalized.shape[1]} features...")
        trainer = TransformerTrainer(n_features=normalized.shape[1])
        t_results = trainer.train(normalized, targets, verbose=False)
        fi = t_results["feature_importance"]
        st.write(f"✅ Val MSE: {t_results['best_val_loss']:.6f}")
        status.update(label="Transformer complete!", state="complete")

    # ═══ BACKTEST ════════════════════════════════════════════════
    with st.status("Running Long Straddle Backtest...", expanded=True) as status:
        bt = LongStraddleBacktester(
            initial_capital=bt_capital,
            holding_period=bt_holding,
            entry_threshold=bt_threshold,
        )
        bt_results = bt.run(data["prices"], cond_vol, data["treasury"], ticker, verbose=False)
        st.write(f"✅ {bt_results['total_trades']} trades executed")
        status.update(label="Backtest complete!", state="complete")

    # ═══ SIGNALS ═════════════════════════════════════════════════
    garch_fv = forecast["Annualized Vol"].iloc[0]
    garch_cv = cond_vol.iloc[-1]
    signal = generate_signals(ticker, garch_fv, garch_cv, data["options"], fi, data["prices"])

    # ─── Signal Banner ───────────────────────────────────────────
    sig_class = "signal-buy" if "BUY" in signal["signal"] else "signal-sell" if "SELL" in signal["signal"] else "signal-neutral"
    st.markdown(f"""
    <div class="metric-card {sig_class}" style="text-align:center; padding:30px;">
        <h2 style="margin:0; font-size:2em;">{signal['signal']}</h2>
        <p style="color:#aaa; margin:5px 0;">Strength: {signal['strength']}/100 | {signal.get('vol_regime','')}</p>
        <p style="color:#ccc; font-size:0.9em;">{signal['rationale']}</p>
        <p style="color:#88ccff; font-size:0.85em;"><b>Strategy:</b> Long Straddle (defined risk, no naked options)</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Key Metrics ─────────────────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        iv_val = f"{signal['iv']*100:.1f}%" if signal['iv'] else "N/A"
        st.metric("Market IV", iv_val)
    with cols[1]:
        st.metric("GARCH RV (1D)", f"{garch_fv*100:.1f}%")
    with cols[2]:
        spread_val = f"{signal['spread']*100:+.1f}pp" if signal['spread'] is not None else "N/A"
        st.metric("IV-RV Spread", spread_val)
    with cols[3]:
        st.metric("Current Vol", f"{garch_cv*100:.1f}%")

    st.markdown("---")

    # ═══ TAB LAYOUT ══════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Backtest Results", "📈 Volatility", "🧠 Feature Importance",
        "📊 GARCH Diagnostics", "📋 Options Chain"
    ])

    # ─── Tab 1: BACKTEST RESULTS (NEW — PRIMARY TAB) ─────────────
    with tab1:
        if bt_results["total_trades"] == 0:
            st.warning("No trades were executed. The entry threshold may be too strict for this ticker, or "
                       "GARCH-forecasted vol never significantly exceeded market IV. Try lowering the "
                       "entry spread threshold or changing the ticker.")
        else:
            # Summary cards
            ret_class = "win-card" if bt_results["total_return_pct"] > 0 else "loss-card"
            st.markdown(f"""
            <div class="metric-card {ret_class}" style="text-align:center; padding:25px;">
                <h2 style="margin:0;">Long Straddle Backtest: {ticker}</h2>
                <p style="font-size:2.5em; margin:10px 0; font-weight:700;">
                    ${bt_results['final_capital']:.2f}
                </p>
                <p style="color:#aaa;">
                    {bt_results['total_return_pct']:+.1f}% return from ${bt_results['initial_capital']:.0f} starting capital
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Stats grid
            mc = st.columns(5)
            with mc[0]:
                st.metric("Total Trades", bt_results["total_trades"])
            with mc[1]:
                st.metric("Win Rate", f"{bt_results['win_rate']:.0f}%")
            with mc[2]:
                st.metric("Profit Factor", f"{bt_results['profit_factor']:.2f}" if bt_results['profit_factor'] < 100 else "∞")
            with mc[3]:
                st.metric("Best Trade", f"${bt_results['best_trade']:+.2f}")
            with mc[4]:
                st.metric("Max Drawdown", f"{bt_results['max_drawdown_pct']:.1f}%")

            mc2 = st.columns(5)
            with mc2[0]:
                st.metric("Winners", bt_results["winners"])
            with mc2[1]:
                st.metric("Losers", bt_results["losers"])
            with mc2[2]:
                st.metric("Avg Win", f"${bt_results['avg_win']:+.2f}")
            with mc2[3]:
                st.metric("Avg Loss", f"${bt_results['avg_loss']:+.2f}")
            with mc2[4]:
                st.metric("Worst Trade", f"${bt_results['worst_trade']:+.2f}")

            st.markdown("---")

            # Equity curve
            st.subheader("Equity Curve")
            equity_df = bt_results["equity_df"]
            if not equity_df.empty:
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=equity_df["date"], y=equity_df["equity"],
                    name="Portfolio Value", fill="tozeroy",
                    line=dict(color="#6C63FF", width=2),
                    fillcolor="rgba(108,99,255,0.1)",
                ))
                fig_eq.add_hline(y=bt_capital, line_dash="dash", line_color="#FFD93D",
                                annotation_text=f"Starting Capital: ${bt_capital:.0f}")

                # Mark trades
                trades_df = bt_results["trades_df"]
                wins = trades_df[trades_df["net_pnl"] > 0]
                losses = trades_df[trades_df["net_pnl"] <= 0]

                if not wins.empty:
                    fig_eq.add_trace(go.Scatter(
                        x=wins["exit_date"], y=[equity_df.loc[equity_df["date"] == d, "equity"].values[0]
                                                 for d in wins["exit_date"] if d in equity_df["date"].values][:len(wins)],
                        mode="markers", name="Winning Trade",
                        marker=dict(color="#00cc66", size=10, symbol="triangle-up"),
                    ))
                if not losses.empty:
                    fig_eq.add_trace(go.Scatter(
                        x=losses["exit_date"], y=[equity_df.loc[equity_df["date"] == d, "equity"].values[0]
                                                   for d in losses["exit_date"] if d in equity_df["date"].values][:len(losses)],
                        mode="markers", name="Losing Trade",
                        marker=dict(color="#ff4444", size=10, symbol="triangle-down"),
                    ))

                fig_eq.update_layout(template="plotly_dark", height=400,
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,30,0.8)",
                                      yaxis_title="Portfolio Value ($)", xaxis_title="",
                                      legend=dict(orientation="h", y=1.1),
                                      margin=dict(l=60, r=20, t=20, b=40))
                st.plotly_chart(fig_eq, use_container_width=True)

            # Trade-level P&L distribution
            st.subheader("Trade P&L Distribution")
            trades_df = bt_results["trades_df"]
            colors = ["#00cc66" if p > 0 else "#ff4444" for p in trades_df["net_pnl"]]
            fig_pnl = go.Figure(go.Bar(
                x=list(range(1, len(trades_df)+1)), y=trades_df["net_pnl"],
                marker_color=colors, name="Net P&L",
            ))
            fig_pnl.add_hline(y=0, line_color="#666")
            fig_pnl.update_layout(template="plotly_dark", height=300,
                                   xaxis_title="Trade #", yaxis_title="Net P&L ($)",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,30,0.8)",
                                   margin=dict(l=60, r=20, t=20, b=40))
            st.plotly_chart(fig_pnl, use_container_width=True)

            # Trade log
            st.subheader("Trade Log")
            st.dataframe(bt.get_trades_df(), use_container_width=True, hide_index=True)

            # Strategy explanation
            st.markdown("""
            <div class="metric-card" style="padding:20px;">
                <h3 style="margin-top:0;">📖 Strategy: GARCH-Informed Long Straddle</h3>
                <p><b>Entry:</b> Buy ATM call + ATM put when GARCH forecasts realized vol will exceed
                the market's current implied volatility (options are underpriced).</p>
                <p><b>Exit:</b> Hold for the specified period, then close at intrinsic value.</p>
                <p><b>Risk:</b> Maximum loss = premium paid. No naked options — fully defined risk.</p>
                <p><b>Edge:</b> GARCH identifies when the market underestimates future volatility.
                The Transformer identifies which conditions are driving volatility.</p>
                <p style="color:#888; font-size:0.85em;">⚠️ Option prices are synthesized via Black-Scholes
                using GARCH conditional volatility. Historical options data is not freely available.
                Actual results may differ due to bid-ask spreads, liquidity, and early exercise.</p>
            </div>
            """, unsafe_allow_html=True)

    # ─── Tab 2: Volatility ───────────────────────────────────────
    with tab2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            row_heights=[0.6, 0.4], subplot_titles=["Price", "Conditional Volatility (Annualized)"])
        fig.add_trace(go.Scatter(x=data["prices"].index, y=data["prices"]["Close"],
                                  name="Close", line=dict(color="#6C63FF", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol.values * 100,
                                  name="GARCH Vol %", fill="tozeroy",
                                  line=dict(color="#FF6B6B", width=1), fillcolor="rgba(255,107,107,0.15)"), row=2, col=1)
        vix_aligned = data["vix"]["Close"].reindex(cond_vol.index, method="ffill")
        fig.add_trace(go.Scatter(x=vix_aligned.index, y=vix_aligned.values,
                                  name="VIX", line=dict(color="#FFD93D", width=1, dash="dot")), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=600,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,30,0.8)",
                          legend=dict(orientation="h", y=1.02), margin=dict(l=50, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Volatility Forecast")
        st.dataframe(forecast.style.format({"Daily Vol": "{:.4f}", "Annualized Vol": "{:.4f}", "Annualized Vol (%)": "{:.2f}%"}),
                      use_container_width=True, hide_index=True)

    # ─── Tab 3: Feature Importance ───────────────────────────────
    with tab3:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Feature Importance Ranking")
            fig_fi = go.Figure(go.Bar(
                y=fi["Feature"], x=fi["Importance"],
                orientation="h", marker=dict(color=fi["Importance"],
                colorscale=[[0,"#1a1a4e"],[0.5,"#6C63FF"],[1,"#FF6B6B"]]),
                error_x=dict(type="data", array=fi["Importance_Std"].values, visible=True, color="#888"),
            ))
            fig_fi.update_layout(template="plotly_dark", height=500, yaxis=dict(autorange="reversed"),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,30,0.8)",
                                  xaxis_title="Importance Score", margin=dict(l=130, r=20, t=10, b=40))
            st.plotly_chart(fig_fi, use_container_width=True)

        with col_b:
            st.subheader("Temporal Importance")
            temp_imp = trainer.get_temporal_importance()
            fig_temp = go.Figure(go.Bar(
                x=list(range(len(temp_imp))), y=temp_imp,
                marker=dict(color=temp_imp, colorscale="Viridis"),
            ))
            fig_temp.update_layout(template="plotly_dark", height=500,
                                    xaxis_title="Days Ago (0=most recent)", yaxis_title="Attention Weight",
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,30,0.8)",
                                    margin=dict(l=50, r=20, t=10, b=40))
            st.plotly_chart(fig_temp, use_container_width=True)

        st.markdown("### 🔑 Top Volatility Drivers")
        for i, row in fi.head(5).iterrows():
            pct = row["Importance"] * 100
            st.markdown(f"""<div class="metric-card" style="padding:12px 20px;">
                <b>#{int(row['Rank'])} {row['Feature']}</b> — {pct:.1f}% importance
            </div>""", unsafe_allow_html=True)

    # ─── Tab 4: GARCH Diagnostics ────────────────────────────────
    with tab4:
        st.subheader(f"Model: {diagnostics['model_name']}")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.metric("AIC", f"{diagnostics['aic']:.2f}")
            st.metric("Log-Likelihood", f"{diagnostics['log_likelihood']:.2f}")
        with dc2:
            st.metric("BIC", f"{diagnostics['bic']:.2f}")
            st.metric("Current Vol", f"{diagnostics['conditional_vol_current']*100:.2f}%")

        st.subheader("Model Parameters")
        params_df = pd.DataFrame({
            "Parameter": list(diagnostics["params"].keys()),
            "Value": [f"{v:.6f}" for v in diagnostics["params"].values()],
            "P-Value": [f"{diagnostics['pvalues'].get(k, 'N/A')}" for k in diagnostics["params"].keys()],
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        if t_results:
            st.subheader("Transformer Training Loss")
            hist = trainer.get_training_history()
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=hist["epoch"], y=hist["train_loss"], name="Train", line=dict(color="#6C63FF")))
            fig_loss.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_loss"], name="Validation", line=dict(color="#FF6B6B")))
            fig_loss.update_layout(template="plotly_dark", height=350, xaxis_title="Epoch", yaxis_title="MSE Loss",
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,30,0.8)")
            st.plotly_chart(fig_loss, use_container_width=True)

    # ─── Tab 5: Options Chain ────────────────────────────────────
    with tab5:
        calls = data["options"].get("calls", pd.DataFrame())
        puts = data["options"].get("puts", pd.DataFrame())

        if not calls.empty:
            st.subheader("IV Smile")
            fig_smile = go.Figure()
            if "impliedVolatility" in calls.columns:
                valid_calls = calls[calls["impliedVolatility"].notna()]
                fig_smile.add_trace(go.Scatter(x=valid_calls["strike"], y=valid_calls["impliedVolatility"] * 100,
                                                mode="lines+markers", name="Call IV", line=dict(color="#6C63FF")))
            if "impliedVolatility" in puts.columns:
                valid_puts = puts[puts["impliedVolatility"].notna()]
                fig_smile.add_trace(go.Scatter(x=valid_puts["strike"], y=valid_puts["impliedVolatility"] * 100,
                                                mode="lines+markers", name="Put IV", line=dict(color="#FF6B6B")))

            spot = data["prices"]["Close"].iloc[-1]
            fig_smile.add_vline(x=spot, line_dash="dash", line_color="#FFD93D", annotation_text=f"Spot: ${spot:.2f}")
            fig_smile.update_layout(template="plotly_dark", height=400, xaxis_title="Strike", yaxis_title="IV (%)",
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,30,0.8)")
            st.plotly_chart(fig_smile, use_container_width=True)

            st.subheader("Options Chain Data")
            oc1, oc2 = st.columns(2)
            with oc1:
                st.markdown("**Calls**")
                display_cols = [c for c in ["strike","lastPrice","volume","openInterest","impliedVolatility"] if c in calls.columns]
                st.dataframe(calls[display_cols].head(20), use_container_width=True, hide_index=True)
            with oc2:
                st.markdown("**Puts**")
                display_cols = [c for c in ["strike","lastPrice","volume","openInterest","impliedVolatility"] if c in puts.columns]
                st.dataframe(puts[display_cols].head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No options chain data available for this ticker.")

else:
    # Landing page with scanner
    st.markdown("""
    <div class="metric-card" style="text-align:center; padding:40px;">
        <h2 style="margin-bottom:10px;">Volatility Trading System</h2>
        <p style="color:#aaa;">GARCH + Transformer | Long Straddle | $150 Budget | Webull-compatible</p>
    </div>
    """, unsafe_allow_html=True)

    # ═══ TOP 8 SCANNER ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🔍 Top 8 Recommendations — Next Trading Day")
    scan_col1, scan_col2 = st.columns([3, 1])
    with scan_col2:
        scan_btn = st.button("🔄 Refresh Scan", use_container_width=True, type="primary")

    if scan_btn or "scan_results" not in st.session_state:
        if scan_btn or "scan_results" not in st.session_state:
            with st.status("Scanning 42 tickers for GARCH signals...", expanded=True) as status:
                st.write("Fetching options chains & running GARCH models...")
                from signals.scanner import scan_for_opportunities
                recs = scan_for_opportunities(budget=bt_capital, top_n=8)
                st.session_state["scan_results"] = recs
                status.update(label=f"Scan complete — {len(recs)} opportunities found!", state="complete")

    recs = st.session_state.get("scan_results", [])
    if recs:
        for i, r in enumerate(recs):
            spread_pct = r['spread'] * 100
            signal_color = "#00cc66" if spread_pct > 10 else "#FFD93D" if spread_pct > 5 else "#ff8844"
            st.markdown(f"""
            <div class="metric-card" style="padding:15px 20px; display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
                <div style="min-width:50px; text-align:center;">
                    <span style="font-size:1.5em; font-weight:700; color:{signal_color};">#{i+1}</span>
                </div>
                <div style="min-width:80px;">
                    <div style="font-size:1.3em; font-weight:700; color:#e0e0ff;">{r['ticker']}</div>
                    <div style="color:#888; font-size:0.85em;">${r['spot']:.2f}</div>
                </div>
                <div style="flex:1; min-width:200px;">
                    <div style="color:#aaa;">
                        <b>${r['strike']} Straddle</b> — {r['expiry']}
                    </div>
                    <div style="color:#888; font-size:0.85em;">
                        {r['contracts']}x @ ${r['call_price']:.2f}C + ${r['put_price']:.2f}P = <b>${r['total_cost']:.2f}</b>
                    </div>
                </div>
                <div style="min-width:140px; text-align:center;">
                    <div style="font-size:0.8em; color:#888;">GARCH Spread</div>
                    <div style="font-size:1.2em; font-weight:700; color:{signal_color};">+{spread_pct:.1f}%</div>
                </div>
                <div style="min-width:120px; text-align:center;">
                    <div style="font-size:0.8em; color:#888;">Liquidity</div>
                    <div style="color:#ccc;">{r['liquidity']:,} vol</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No opportunities found within budget. Click Refresh to scan.")

    st.markdown("---")
    st.caption("Select a ticker in the sidebar and click **Run Analysis + Backtest** for detailed analysis.")

    # ── Live Trading Panel ───────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; margin:30px 0 15px 0;">
        <span style="font-size:1.8em; font-weight:700; color:#ff6b6b;">
            🔴 Live Trading — Webull
        </span>
    </div>
    """, unsafe_allow_html=True)

    from broker.webull_client import get_accounts, activate_token, execute_top_trade

    col_status, col_actions = st.columns([1, 2])

    with col_status:
        st.markdown("#### Connection Status")
        try:
            accts = get_accounts()
            if accts:
                st.success(f"✅ Connected — {len(accts)} account(s)")
                for a in accts:
                    st.markdown(f"**{a.get('account_type', 'Account')}**: `{a.get('account_id', 'N/A')}`")
            else:
                st.warning("⚠️ Not authenticated. Generate a token below.")
        except Exception as e:
            st.error(f"Connection error: {e}")

    with col_actions:
        st.markdown("#### Actions")

        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("🔑 Generate Token", use_container_width=True):
                with st.spinner("Creating token... Check your Webull app!"):
                    success = activate_token()
                    if success:
                        st.success("Token activated! Refresh page to see status.")
                        st.rerun()
                    else:
                        st.error("Token verification timed out. Try again.")

        with act_col2:
            live_mode = st.checkbox("Enable LIVE mode", value=False,
                                     help="⚠️ When enabled, orders will be placed with real money!")

        st.markdown("---")

        trade_btn = st.button(
            "🚀 Execute Top GARCH Signal" if live_mode else "🧪 Dry Run Top Signal",
            use_container_width=True,
            type="primary",
        )

        if trade_btn:
            with st.status(
                "🔴 PLACING LIVE ORDER..." if live_mode else "🧪 Running dry simulation...",
                expanded=True
            ) as status:
                result = execute_top_trade(budget=bt_capital, dry_run=not live_mode)
                if result.get("success"):
                    pick = result.get("pick", {})
                    if live_mode:
                        status.update(label="✅ Order placed!", state="complete")
                        st.balloons()
                    else:
                        status.update(label="✅ Dry run complete", state="complete")

                    if pick:
                        st.markdown(f"""
                        <div class="metric-card signal-buy" style="padding:20px;">
                            <div style="font-size:1.3em; font-weight:700; color:#00ff88;">
                                {'🔴 LIVE ORDER' if live_mode else '🧪 DRY RUN'}: {pick['ticker']} ${pick['strike']} Straddle
                            </div>
                            <div style="margin-top:10px; color:#ccc;">
                                {pick['contracts']}x contracts • Expiry: {pick['expiry']}<br>
                                Call: ${pick['call_price']:.2f} + Put: ${pick['put_price']:.2f} = <b>${pick['total_cost']:.2f}</b><br>
                                GARCH spread: +{pick['spread']*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    status.update(label="❌ No signal or error", state="error")
                    st.error(f"Reason: {result.get('reason', 'unknown')}")

