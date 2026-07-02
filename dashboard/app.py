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
    STRANGLE_OTM_WIDTH, STRANGLE_HOLDING_PERIOD_DAYS, STRANGLE_ENTRY_VOL_THRESHOLD,
)
from data.fetcher import fetch_all_data
from data.feature_engineer import build_features, normalize_features, build_target
from models.garch_model import GARCHVolatilityModel
from models.transformer_model import TransformerTrainer
from signals.generator import generate_signals, compute_iv_from_options
from backtest.engine import LongStraddleBacktester
from backtest.strangle_engine import LongStrangleBacktester

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
    st.markdown("*GARCH (Decision) + Transformer (Interpretability)*")
    st.markdown("---")

    all_tickers = DEFAULT_TICKERS + AFFORDABLE_TICKERS
    ticker = st.selectbox("Select Ticker", all_tickers + ["Custom..."],
                          help="Quick-select from featured tickers, or enter any symbol. "
                               "Full 42-ticker universe is scanned on the landing page.")
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
    st.markdown(f"**Hold:** {HOLDING_PERIOD_DAYS} trading days")
    st.markdown(f"**Entry:** GARCH RV > 30d HV by {ENTRY_VOL_THRESHOLD*100:.0f}%+")
    st.markdown("**Strategy:** Long Straddle")
    st.caption("GARCH drives decisions · Transformer provides feature insight")

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
        status.update(label="Straddle backtest complete!", state="complete")

    # ═══ STRANGLE BACKTEST ═══════════════════════════════════════
    with st.status("Running Long Strangle Backtest...", expanded=True) as status:
        st_bt = LongStrangleBacktester(
            initial_capital=bt_capital,
            holding_period=STRANGLE_HOLDING_PERIOD_DAYS,
            entry_threshold=STRANGLE_ENTRY_VOL_THRESHOLD,
            otm_width=STRANGLE_OTM_WIDTH,
        )
        st_bt_results = st_bt.run(data["prices"], cond_vol, data["treasury"], ticker, verbose=False)
        st.write(f"✅ {st_bt_results['total_trades']} strangle trades executed")
        status.update(label="Strangle backtest complete!", state="complete")

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
    tab1, tab7, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💰 Straddle Backtest", "🔀 Strangle Backtest",
        "📈 Volatility", "🧠 Feature Importance",
        "📊 GARCH Diagnostics", "📋 Options Chain", "📖 Model & Methodology"
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
            st.markdown(f"""
            <div class="metric-card" style="padding:20px;">
                <h3 style="margin-top:0;">📖 Strategy: GARCH-Informed Long Straddle</h3>
                <p><b>Signal (GARCH):</b> GJR-GARCH(1,1,1) with Student-t innovations forecasts
                realized volatility. When GARCH RV exceeds the 30-day rolling historical vol by ≥{ENTRY_VOL_THRESHOLD*100:.0f}%,
                a BUY VOL signal is generated.</p>
                <p><b>Entry:</b> Buy ATM call + ATM put. The scanner compares GARCH forecast RV against
                <em>30-day rolling close-to-close historical volatility</em> (backtested to +1,247% vs +736% for IV proxy).</p>
                <p><b>Exit:</b> Default hold of {HOLDING_PERIOD_DAYS} trading days, then close at intrinsic value.
                In live trading, an auto-close daemon monitors for +12% take-profit or -30% stop-loss
                and exits automatically if either threshold is hit.</p>
                <p><b>Risk:</b> Maximum loss = premium paid. No naked options — fully defined risk.
                Budget capped at ${INITIAL_CAPITAL:.0f} per trade.</p>
                <p><b>Interpretability (Transformer):</b> A Transformer encoder is trained alongside GARCH
                to identify which market features (RSI, VIX, volume, etc.) are driving volatility.
                It does <em>not</em> influence the buy/sell decision — GARCH is the sole decision-maker.
                Feature importance from attention weights and gradient saliency is displayed in the
                "Feature Importance" tab.</p>
                <p style="color:#888; font-size:0.85em;">⚠️ Backtest uses Black-Scholes synthetic pricing
                (historical options data is not freely available). Live trading uses real market prices
                from the options chain. Actual results may differ due to bid-ask spreads and liquidity.</p>
            </div>
            """, unsafe_allow_html=True)

    # ─── Tab 7: STRANGLE BACKTEST ─────────────────────────────────
    with tab7:
        if st_bt_results["total_trades"] == 0:
            st.warning("No strangle trades were executed. The entry threshold may be too strict for this ticker, or "
                       "the strangle premiums exceeded the budget.")
            st.markdown(f"""
            **Why no trades?** The strangle strategy requires:
            - GARCH RV to exceed 30d historical vol by ≥{STRANGLE_ENTRY_VOL_THRESHOLD*100:.0f}%
            - An affordable OTM call + OTM put ({STRANGLE_OTM_WIDTH*100:.0f}% OTM) within the ${bt_capital:.0f} budget

            Try a higher-beta ticker or adjusting the entry threshold.
            """)
        else:
            st_trades_df = st_bt_results.get("trades_df", pd.DataFrame())
            st_equity_df = st_bt_results.get("equity_df", pd.DataFrame())

            # Summary cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st_pnl_color = "#00ff88" if st_bt_results['total_pnl'] >= 0 else "#ff4444"
                st.markdown(f"""
                <div class="metric-card" style="text-align:center; padding:20px; border-color:{st_pnl_color}40;">
                    <div style="color:#888; font-size:0.75em; text-transform:uppercase;">Total P&L</div>
                    <div style="font-size:2em; font-weight:800; color:{st_pnl_color};">${st_bt_results['total_pnl']:+.2f}</div>
                    <div style="color:{st_pnl_color}; font-weight:600;">{st_bt_results['total_return_pct']:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center; padding:20px;">
                    <div style="color:#888; font-size:0.75em; text-transform:uppercase;">Win Rate</div>
                    <div style="font-size:2em; font-weight:800; color:#e0e0ff;">{st_bt_results['win_rate']:.0f}%</div>
                    <div style="color:#888;">{st_bt_results['winners']}W / {st_bt_results['losers']}L</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st_pf = st_bt_results.get('profit_factor', 0)
                st_pf_str = f"{st_pf:.2f}" if st_pf < 100 else "∞"
                st.markdown(f"""
                <div class="metric-card" style="text-align:center; padding:20px;">
                    <div style="color:#888; font-size:0.75em; text-transform:uppercase;">Profit Factor</div>
                    <div style="font-size:2em; font-weight:800; color:#e0e0ff;">{st_pf_str}</div>
                    <div style="color:#888;">Max DD: {st_bt_results.get('max_drawdown_pct', 0):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center; padding:20px;">
                    <div style="color:#888; font-size:0.75em; text-transform:uppercase;">Trades</div>
                    <div style="font-size:2em; font-weight:800; color:#e0e0ff;">{st_bt_results['total_trades']}</div>
                    <div style="color:#888;">{STRANGLE_OTM_WIDTH*100:.0f}% OTM width</div>
                </div>
                """, unsafe_allow_html=True)

            # Equity curve
            if not st_equity_df.empty and "equity" in st_equity_df.columns:
                fig_eq = go.Figure()
                eq_color = "#00ff88" if st_bt_results['total_pnl'] >= 0 else "#ff4444"
                fig_eq.add_trace(go.Scatter(
                    x=st_equity_df["date"], y=st_equity_df["equity"],
                    mode="lines", name="Equity",
                    line=dict(color=eq_color, width=2.5),
                    fill="tozeroy", fillcolor=f"rgba({','.join(str(int(eq_color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))},0.08)",
                ))
                fig_eq.add_hline(y=bt_capital, line_dash="dash", line_color="#666",
                                 annotation_text="Initial Capital")
                fig_eq.update_layout(
                    title="Strangle Equity Curve",
                    template="plotly_dark", height=350,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,30,0.8)",
                    margin=dict(l=60, r=20, t=40, b=30),
                    yaxis_title="Capital ($)", yaxis_tickprefix="$",
                )
                st.plotly_chart(fig_eq, use_container_width=True)

            # P&L Distribution
            if len(st_trades_df) > 0:
                st_colors = ["#00ff88" if x > 0 else "#ff4444" for x in st_trades_df["net_pnl"]]
                fig_pnl = go.Figure(go.Bar(
                    x=list(range(1, len(st_trades_df)+1)), y=st_trades_df["net_pnl"],
                    marker_color=st_colors, name="Net P&L",
                ))
                fig_pnl.add_hline(y=0, line_color="#666")
                fig_pnl.update_layout(template="plotly_dark", height=300,
                                       xaxis_title="Trade #", yaxis_title="Net P&L ($)",
                                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,30,0.8)",
                                       margin=dict(l=60, r=20, t=20, b=40))
                st.plotly_chart(fig_pnl, use_container_width=True)

            # Trade log
            st.subheader("Strangle Trade Log")
            st.dataframe(st_bt.get_trades_df(), use_container_width=True, hide_index=True)

            # Strategy explanation
            st.markdown(f"""
            <div class="metric-card" style="padding:20px;">
                <h3 style="margin-top:0;">📖 Strategy: GARCH-Informed Long Strangle</h3>
                <p><b>Structure:</b> Buy OTM call (strike = spot × {1+STRANGLE_OTM_WIDTH:.2f}) +
                OTM put (strike = spot × {1-STRANGLE_OTM_WIDTH:.2f}). Both legs are
                {STRANGLE_OTM_WIDTH*100:.0f}% out-of-the-money.</p>
                <p><b>Signal:</b> Same GARCH model as the straddle — when GARCH forecast RV exceeds
                30-day historical vol by ≥{STRANGLE_ENTRY_VOL_THRESHOLD*100:.0f}%, a BUY VOL signal fires.</p>
                <p><b>Advantage vs Straddle:</b> Lower premium cost allows more contracts or preserves
                capital. Best suited when expecting very large moves (earnings, catalysts, macro events).</p>
                <p><b>Disadvantage:</b> Needs a bigger move to profit — spot must breach one of the OTM
                strikes. With small moves, both legs expire worthless = full premium loss.</p>
                <p><b>Exit:</b> Hold for {STRANGLE_HOLDING_PERIOD_DAYS} trading days, close at intrinsic value.
                Same auto-close daemon (+12% TP / -30% SL) applies in live trading.</p>
                <p style="color:#888; font-size:0.85em;">⚠️ Backtest uses Black-Scholes synthetic pricing.
                Live trading uses real market prices. Strangles are inherently harder to profit from
                than straddles because both legs start OTM.</p>
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

    # ─── Tab 6: Model & Methodology ──────────────────────────────
    with tab6:
        st.markdown("""
        <div class="metric-card" style="padding:30px; text-align:center; margin-bottom:20px;">
            <h2 style="margin:0;">GARCH Volatility Trading System</h2>
            <p style="color:#aaa; margin:5px 0;">GJR-GARCH(1,1,1) · Student-t Innovations · Long Straddle Strategy</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Overview ──
        st.markdown("### 🏗️ System Architecture")
        st.markdown("""
        This system uses two models with distinct roles:

        | Component | Model | Role |
        |---|---|---|
        | **Decision Engine** | GJR-GARCH(1,1,1) | Forecasts realized volatility and generates buy/sell signals |
        | **Interpretability** | Transformer Encoder | Identifies which features drive volatility (display only) |

        > **The Transformer does NOT influence trading decisions.** It is trained alongside GARCH to
        > provide feature importance rankings via attention weights and gradient saliency. The GARCH
        > model is the sole decision-maker for all trade entries and exits.

        ```
        ┌────────────────────────────────────────────────────────────┐
        │  DATA LAYER — Yahoo Finance API                           │
        │  OHLCV, VIX, 10Y Treasury, Live Options Chains            │
        │  Local CSV cache (12-hour freshness)                       │
        └──────────────┬─────────────────────────────────────────────┘
                       │
        ┌──────────────▼─────────────────────────────────────────────┐
        │  MODEL LAYER                                               │
        │  GARCH(1,1) vs GJR-GARCH(1,1,1) — AIC selection           │
        │  Student-t distribution for fat tails                      │
        │  Transformer — feature importance only (no signal role)    │
        └──────────────┬─────────────────────────────────────────────┘
                       │
        ┌──────────────▼─────────────────────────────────────────────┐
        │  SIGNAL LAYER                                              │
        │  GARCH Forecast RV vs 30d Rolling Historical Vol           │
        │  Signal threshold: ±5pp spread                             │
        │  Entry threshold: ≥3% RV-HV spread for trade execution     │
        │  Scanner: 42 tickers × 3 strikes each                     │
        └──────────────┬─────────────────────────────────────────────┘
                       │
        ┌──────────────▼─────────────────────────────────────────────┐
        │  EXECUTION LAYER                                           │
        │  Webull OpenAPI (HMAC-SHA1 signed)                         │
        │  Two atomic SINGLE-leg orders (call + put)                 │
        │  Auto-close daemon: +12% take-profit, 30s polling          │
        └────────────────────────────────────────────────────────────┘
        ```
        """)

        st.markdown("---")

        # ── GARCH Model ──
        st.markdown("### 📐 The GARCH Model")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("""
            **Symmetric GARCH(1,1):**
            ```
            σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
            ```
            - `ω` — long-run variance constant
            - `α` — ARCH coeff (shock reaction)
            - `β` — GARCH coeff (persistence)
            - Constraint: α + β < 1 (stationarity)
            """)
        with m2:
            st.markdown("""
            **Asymmetric GJR-GARCH(1,1,1):**
            ```
            σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}
            I_{t-1} = 1 if ε_{t-1} < 0, else 0
            ```
            - `γ` — leverage parameter (asymmetry)
            - Captures the **leverage effect**: negative returns
              increase vol more than positive returns
            """)

        st.markdown("""
        Both models are fitted to the full price history. The model with the **lower AIC** is selected.
        GJR-GARCH is selected ~70% of the time, confirming that leverage effects are present
        in most equity returns. Returns are modeled with a **Student-t distribution** to capture
        the fat tails observed in financial markets.
        """)

        st.markdown("""
        | Parameter | Value | Rationale |
        |---|---|---|
        | p (GARCH lag) | 1 | Standard; higher orders rarely improve fit |
        | q (ARCH lag) | 1 | Captures immediate shock reaction |
        | o (Leverage lag) | 1 | One asymmetric term for leverage effect |
        | Distribution | Student-t | Fat tails in equity returns |
        | Return scaling | ×100 | Numerical stability for MLE optimizer |
        | Trading days | 252 | Standard US market annualization |
        """)

        st.markdown("---")

        # ── Transformer ──
        st.markdown("### 🧠 The Transformer Model (Interpretability Only)")
        st.markdown("""
        A PyTorch **Transformer encoder** is trained alongside GARCH to provide
        feature-level interpretability. It does **not** generate signals or influence
        trade decisions.

        | Component | Value |
        |---|---|
        | Architecture | TransformerEncoder → GlobalAvgPool → Linear Head |
        | Lookback window | 30 trading days |
        | Internal dimension | 64 (d_model) |
        | Attention heads | 4 |
        | Encoder layers | 2 |
        | Feedforward dim | 128 |
        | Dropout | 10% |
        | Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
        | Scheduler | Cosine Annealing |
        | Early stopping | Patience = 15 epochs |

        **21 input features** are computed from price data, volume, technicals (RSI, MACD,
        Bollinger, ATR), market regime (VIX), options data (P/C ratio, avg IV), and macro
        (10Y Treasury yield). Feature importance is extracted via:
        1. **Gradient saliency** — |∂output/∂input| averaged across test samples
        2. **Attention weights** — averaged across heads and layers for temporal importance
        """)

        st.markdown("---")

        # ── Scanning Pipeline ──
        st.markdown("### 🔍 Live Scanning Pipeline")
        st.markdown("""
        The scanner runs across a universe of **42 budget-friendly tickers** (stocks priced ~$1–$30)
        to find the best straddle opportunities.

        **For each of the 42 tickers:**
        1. Fetch the **live options chain** from Yahoo Finance
        2. Find the nearest expiration **~14 days out**
        3. Try ATM and **±1 strikes** (3 candidates per ticker)
        4. Compute straddle cost: `(call_price + put_price) × 100 + $1.30 commission`
        5. Filter: must fit within the **$150 budget**
        6. Fit GARCH on full price history → extract **conditional volatility**
        7. Compute 30-day rolling close-to-close historical vol (annualized)
        8. Compute spread: `GARCH_RV − 30d_hist_vol`
           - Uses **30-day rolling historical vol** as the benchmark (backtested to outperform option IV proxy by +69%)
        9. Sort all opportunities by spread (strongest positive signal first)
        10. Return **Top 8** recommendations

        > ⚠️ The sidebar ticker dropdown shows 16 featured tickers for quick individual analysis.
        > The full 42-ticker scan universe is used on this landing page when you click "Refresh Scan".
        """)

        st.markdown("---")

        # ── Signal Logic ──
        st.markdown("### 📡 Signal Generation")
        st.markdown("""
        The system uses **two separate thresholds** for different purposes:

        | Threshold | Value | Purpose |
        |---|---|---|
        | **Signal threshold** | ±5pp spread | Determines signal direction (BUY/SELL/NEUTRAL) in the signal generator |
        | **Entry threshold** | ≥3% RV-HV spread | Required to actually open a backtest or live trade position |

        **Signal direction logic:**
        ```
        Spread = GARCH_RV − 30d_Hist_Vol

        If Spread > +5pp:  BUY VOL  → Long straddle (options underpriced)
        If Spread < −5pp:  SELL VOL → Short straddle (options overpriced)
        Otherwise:         NEUTRAL  → No trade
        ```

        **Signal strength** is computed as `|Spread| / max(Hist_Vol, 0.01)` and scaled 0–100.
        A stronger signal indicates greater relative mispricing.
        """)

        st.markdown("---")

        # ── Backtesting ──
        st.markdown("### 🧪 Backtesting Methodology")
        st.markdown(f"""
        The rigorous backtester uses a **60/20/20 chronological split**:

        ```
        |←── Train (60%) ──→|←── Val (20%) ──→|←── Test (20%) ──→|
           Jan 2022              Mid 2024           Late 2025
           ~648 days             ~216 days           ~217 days
        ```

        - **Training set** — GARCH parameters (ω, α, β, γ, ν) are estimated
        - **Validation set** — Pearson correlation between GARCH vol forecast and 5-day rolling RV
          (target: correlation > 0.5)
        - **Test set** — Completely unseen data used for backtest P&L simulation

        **Default backtest parameters (configurable in sidebar):**

        | Parameter | Value |
        |---|---|
        | Starting capital | ${INITIAL_CAPITAL:.0f} |
        | Holding period | {HOLDING_PERIOD_DAYS} trading days |
        | Entry threshold | GARCH RV > 30d HV by {ENTRY_VOL_THRESHOLD*100:.0f}%+ |
        | Max position size | 90% of capital |
        | Commission | $0.65 per contract per leg |

        **Exit rules:**
        - Hold for the specified period, then exit at intrinsic value
        - In backtesting, the hold strategy outperformed stop-loss strategies

        **Note:** Backtesting uses **Black-Scholes synthetic pricing** because historical
        options data is not freely available. Both the backtester and the live scanner use
        **30-day rolling close-to-close historical vol** as the benchmark. This approach
        outperformed the Garman-Klass OHLC IV proxy in head-to-head backtesting (+1,247% vs +736%).
        """)

        st.markdown("---")

        # ── Live Execution ──
        st.markdown("### 🔴 Live Execution & Risk Management")
        st.markdown("""
        **Execution via Webull OpenAPI:**
        - HMAC-SHA1 signed authentication on every request
        - Token-based 2FA via Webull mobile app
        - Two separate SINGLE-leg LIMIT orders (call + put)
        - GTC time-in-force for buys, DAY for sells

        **Auto-Close Daemon** (`broker/auto_close.py`):
        - Market hours (9:30 AM – 4:00 PM ET): polls every 30 seconds
        - Off-hours: polls every 5 minutes
        - **Take-profit trigger: +12% unrealized return** → auto-sell both legs
        - **Stop-loss trigger: -30% unrealized return** → auto-sell both legs to cap losses
        - All checks logged to `cache/auto_close.log`
        - Closed trades recorded in `cache/closed_trades.json` with exit reason

        **Risk controls:**

        | Control | Value |
        |---|---|
        | Budget per trade | $150 max |
        | Position sizing | 90% of capital max |
        | Concentration | Single best signal executed (max conviction) |
        | Max loss | Premium paid (defined risk, no naked options) |
        | Auto take-profit | +12% unrealized P&L |
        | Auto stop-loss | -30% unrealized P&L |
        | Data freshness | 12-hour cache; live options for scanning |
        """)

else:
    # Landing page with scanner
    st.markdown(f"""
    <div class="metric-card" style="text-align:center; padding:40px;">
        <h2 style="margin-bottom:10px;">Volatility Trading System</h2>
        <p style="color:#aaa;">GARCH (Decision) + Transformer (Interpretability) | Long Straddle |
        ${INITIAL_CAPITAL:.0f} Budget | {HOLDING_PERIOD_DAYS}-Day Hold | +12% TP / -30% SL | Webull</p>
    </div>
    """, unsafe_allow_html=True)

    # ═══ LANDING PAGE TABS ══════════════════════════════════════
    landing_tab1, landing_tab3, landing_tab2 = st.tabs([
        "🔍 Straddle Scanner & Trading", "🔀 Strangle Scanner", "📖 Methodology"
    ])

    with landing_tab1:
        # ═══ TOP 8 SCANNER ══════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 🔍 Top 8 Recommendations — Next Trading Day")
        scan_col1, scan_col2 = st.columns([3, 1])
        with scan_col2:
            scan_btn = st.button("🔄 Refresh Scan", use_container_width=True, type="primary")

        if scan_btn or "scan_results" not in st.session_state:
            if scan_btn or "scan_results" not in st.session_state:
                with st.status("Scanning 42 tickers — GARCH RV vs 30d historical vol...", expanded=True) as status:
                    st.write("Fitting GARCH on each ticker & comparing forecast RV against 30-day rolling historical volatility...")
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
                    <div style="min-width:100px; text-align:center;">
                        <div style="font-size:0.8em; color:#888;">GARCH RV</div>
                        <div style="color:#e0e0ff; font-weight:600;">{r['garch_rv']:.1%}</div>
                    </div>
                    <div style="min-width:100px; text-align:center;">
                        <div style="font-size:0.8em; color:#888;">30d Hist Vol</div>
                        <div style="color:#e0e0ff; font-weight:600;">{r['hist_vol']:.1%}</div>
                    </div>
                    <div style="min-width:140px; text-align:center;">
                        <div style="font-size:0.8em; color:#888;">RV-HV Spread</div>
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

        # ── Live Position Tracker ─────────────────────────────────────
        st.markdown("---")
        st.markdown("""
    <div style="text-align:center; margin:30px 0 15px 0;">
        <span style="font-size:1.8em; font-weight:700;
              background: linear-gradient(90deg, #00ff88, #00ccff);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📈 Live Position Tracker
        </span>
    </div>
    """, unsafe_allow_html=True)

        from broker.position_tracker import fetch_live_positions, get_pnl_history

        refresh_col1, refresh_col2 = st.columns([3, 1])
        with refresh_col2:
            refresh_pos = st.button("🔄 Refresh Positions", use_container_width=True)

        if refresh_pos or "pos_data" not in st.session_state:
            try:
                pos_data = fetch_live_positions()
                if pos_data:
                    st.session_state["pos_data"] = pos_data
            except Exception as e:
                st.error(f"Failed to fetch positions: {e}")

        pos_data = st.session_state.get("pos_data")

        if pos_data and pos_data.get("straddles"):
            # ── Portfolio Summary Metrics ──
            total_pnl = pos_data["total_pnl"]
            total_pnl_pct = pos_data["total_pnl_pct"]
            total_cost = pos_data["total_cost"]
            total_value = pos_data["total_value"]
            pnl_color = "#00ff88" if total_pnl >= 0 else "#ff4444"
            pnl_arrow = "▲" if total_pnl >= 0 else "▼"

            st.markdown(f"""
            <div class="metric-card" style="padding:25px; text-align:center; border-color:{pnl_color}40;">
                <div style="display:flex; justify-content:space-around; flex-wrap:wrap; gap:20px;">
                    <div>
                        <div style="color:#888; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Total P&L</div>
                        <div style="font-size:2.2em; font-weight:800; color:{pnl_color};">
                            {pnl_arrow} ${abs(total_pnl):.2f}
                        </div>
                        <div style="font-size:1.1em; color:{pnl_color}; font-weight:600;">
                            {'+' if total_pnl >= 0 else ''}{total_pnl_pct:.2f}%
                        </div>
                    </div>
                    <div>
                        <div style="color:#888; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Cost Basis</div>
                        <div style="font-size:1.5em; font-weight:600; color:#e0e0ff;">${total_cost:.2f}</div>
                    </div>
                    <div>
                        <div style="color:#888; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Market Value</div>
                        <div style="font-size:1.5em; font-weight:600; color:#e0e0ff;">${total_value:.2f}</div>
                    </div>
                    <div>
                        <div style="color:#888; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Positions</div>
                        <div style="font-size:1.5em; font-weight:600; color:#e0e0ff;">{len(pos_data['straddles'])}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Individual Straddle Cards ──
            for key, s in pos_data["straddles"].items():
                s_pnl_color = "#00ff88" if s["total_pnl"] >= 0 else "#ff4444"
                s_arrow = "▲" if s["total_pnl"] >= 0 else "▼"

                call = s.get("call") or {}
                put = s.get("put") or {}
                call_pnl = call.get("unrealized_pnl", 0)
                put_pnl = put.get("unrealized_pnl", 0)
                call_color = "#00ff88" if call_pnl >= 0 else "#ff4444"
                put_color = "#00ff88" if put_pnl >= 0 else "#ff4444"

                st.markdown(f"""
                <div class="metric-card" style="padding:18px 24px; margin:12px 0; border-color:{s_pnl_color}30;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:15px;">
                        <div>
                            <div style="font-size:1.3em; font-weight:700; color:#e0e0ff;">
                                {s['symbol']} ${s['strike']:.0f} Straddle
                            </div>
                            <div style="color:#888; font-size:0.85em;">Exp: {s['expiry']}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.75em; color:#888; text-transform:uppercase;">Total P&L</div>
                            <div style="font-size:1.6em; font-weight:800; color:{s_pnl_color};">
                                {s_arrow} ${abs(s['total_pnl']):.2f}
                            </div>
                            <div style="font-size:0.9em; color:{s_pnl_color}; font-weight:600;">
                                {'+' if s['pnl_pct'] >= 0 else ''}{s['pnl_pct']:.2f}%
                            </div>
                        </div>
                        <div style="display:flex; gap:20px;">
                            <div style="text-align:center; min-width:100px;">
                                <div style="font-size:0.7em; color:#888; text-transform:uppercase;">📞 Call</div>
                                <div style="color:#e0e0ff; font-weight:600;">
                                    {call.get('quantity', 0)}x @ ${call.get('cost_price', 0):.2f}
                                </div>
                                <div style="font-size:0.85em; color:{call_color};">
                                    Now: ${call.get('last_price', 0):.3f} ({'+' if call_pnl >= 0 else ''}${call_pnl:.2f})
                                </div>
                            </div>
                            <div style="text-align:center; min-width:100px;">
                                <div style="font-size:0.7em; color:#888; text-transform:uppercase;">📉 Put</div>
                                <div style="color:#e0e0ff; font-weight:600;">
                                    {put.get('quantity', 0)}x @ ${put.get('cost_price', 0):.2f}
                                </div>
                                <div style="font-size:0.85em; color:{put_color};">
                                    Now: ${put.get('last_price', 0):.3f} ({'+' if put_pnl >= 0 else ''}${put_pnl:.2f})
                                </div>
                            </div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.7em; color:#888; text-transform:uppercase;">Day P&L</div>
                            <div style="font-size:1em; font-weight:600; color:{'#00ff88' if s['day_pnl'] >= 0 else '#ff4444'};">
                                {'+' if s['day_pnl'] >= 0 else ''}${s['day_pnl']:.2f}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── P&L History Chart ──
            history = get_pnl_history()
            if len(history) >= 2:
                import pandas as pd
                hist_df = pd.DataFrame(history)
                hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
                hist_df = hist_df.sort_values("timestamp")

                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(
                    x=hist_df["timestamp"],
                    y=hist_df["total_pnl"],
                    mode="lines+markers",
                    name="P&L ($)",
                    line=dict(color="#00ff88" if hist_df["total_pnl"].iloc[-1] >= 0 else "#ff4444",
                              width=3),
                    marker=dict(size=6),
                    fill="tozeroy",
                    fillcolor="rgba(0,255,136,0.08)" if hist_df["total_pnl"].iloc[-1] >= 0
                              else "rgba(255,68,68,0.08)",
                ))
                fig_pnl.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                fig_pnl.update_layout(
                    title=dict(text="Position P&L Over Time", font=dict(color="#e0e0ff", size=16)),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(100,100,255,0.08)", color="#888"),
                    yaxis=dict(gridcolor="rgba(100,100,255,0.08)", color="#888",
                               title="Unrealized P&L ($)", tickprefix="$"),
                    height=350,
                    margin=dict(l=50, r=20, t=50, b=30),
                    font=dict(color="#ccc"),
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

                # P&L % chart
                fig_pct = go.Figure()
                fig_pct.add_trace(go.Scatter(
                    x=hist_df["timestamp"],
                    y=hist_df["total_pnl_pct"],
                    mode="lines+markers",
                    name="P&L %",
                    line=dict(color="#00ccff", width=2),
                    marker=dict(size=5),
                ))
                fig_pct.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                fig_pct.update_layout(
                    title=dict(text="Return (%)", font=dict(color="#e0e0ff", size=14)),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(100,100,255,0.08)", color="#888"),
                    yaxis=dict(gridcolor="rgba(100,100,255,0.08)", color="#888",
                               title="Return %", ticksuffix="%"),
                    height=250,
                    margin=dict(l=50, r=20, t=40, b=30),
                    font=dict(color="#ccc"),
                )
                st.plotly_chart(fig_pct, use_container_width=True)
            else:
                st.info("📊 P&L chart will appear after the second refresh — keep clicking **Refresh Positions** to build history.")
        elif pos_data is not None:
            st.info("No open option positions found. Execute a trade to start tracking.")

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

    # ═══ STRANGLE SCANNER TAB ═════════════════════════════════════════
    with landing_tab3:
        st.markdown("---")
        st.markdown(f"## 🔀 Top 8 Strangle Opportunities — {STRANGLE_OTM_WIDTH*100:.0f}% OTM")
        sg_col1, sg_col2 = st.columns([3, 1])
        with sg_col2:
            sg_btn = st.button("🔄 Refresh Strangle Scan", use_container_width=True, type="primary")

        if sg_btn or "strangle_scan_results" not in st.session_state:
            with st.status(f"Scanning 42 tickers for strangles — {STRANGLE_OTM_WIDTH*100:.0f}% OTM...", expanded=True) as status:
                st.write("Fitting GARCH and finding OTM call/put pairs...")
                from signals.strangle_scanner import scan_strangle_opportunities
                sg_recs = scan_strangle_opportunities(budget=bt_capital, top_n=8)
                st.session_state["strangle_scan_results"] = sg_recs
                status.update(label=f"Scan complete — {len(sg_recs)} strangle opportunities!", state="complete")

        sg_recs = st.session_state.get("strangle_scan_results", [])
        if sg_recs:
            for i, r in enumerate(sg_recs):
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
                    <div style="flex:1; min-width:220px;">
                        <div style="color:#aaa;">
                            <b>${r['call_strike']}C / ${r['put_strike']}P Strangle</b> — {r['expiry']}
                        </div>
                        <div style="color:#888; font-size:0.85em;">
                            {r['contracts']}x @ ${r['call_price']:.2f}C + ${r['put_price']:.2f}P = <b>${r['total_cost']:.2f}</b>
                        </div>
                    </div>
                    <div style="min-width:100px; text-align:center;">
                        <div style="font-size:0.8em; color:#888;">GARCH RV</div>
                        <div style="color:#e0e0ff; font-weight:600;">{r['garch_rv']:.1%}</div>
                    </div>
                    <div style="min-width:100px; text-align:center;">
                        <div style="font-size:0.8em; color:#888;">30d Hist Vol</div>
                        <div style="color:#e0e0ff; font-weight:600;">{r['hist_vol']:.1%}</div>
                    </div>
                    <div style="min-width:140px; text-align:center;">
                        <div style="font-size:0.8em; color:#888;">RV-HV Spread</div>
                        <div style="font-size:1.2em; font-weight:700; color:{signal_color};">+{spread_pct:.1f}%</div>
                    </div>
                    <div style="min-width:120px; text-align:center;">
                        <div style="font-size:0.8em; color:#888;">Liquidity</div>
                        <div style="color:#ccc;">{r['liquidity']:,} vol</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strangle opportunities found within budget. Click Refresh to scan.")

        st.markdown("---")
        st.markdown(f"""
        <div class="metric-card" style="padding:16px;">
            <p style="margin:0; color:#aaa; font-size:0.9em;">
                <b>Strangle vs Straddle:</b> Strangles use OTM call + OTM put ({STRANGLE_OTM_WIDTH*100:.0f}% from spot),
                resulting in <b>lower premiums</b> but requiring a <b>bigger move</b> to profit.
                Both strategies use the same GARCH signal (forecast RV vs 30d historical vol).
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ═══ METHODOLOGY TAB ════════════════════════════════════════════
    with landing_tab2:
        st.markdown("""
        <div class="metric-card" style="padding:30px; text-align:center; margin-bottom:24px;">
            <h2 style="margin:0; font-size:1.8em;">How Securities Are Scanned & Selected</h2>
            <p style="color:#aaa; margin:8px 0 0 0; font-size:1em;">
                GARCH Volatility Forecasting · Long Straddle Strategy · Defined Risk
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Section 1: Core Model ──
        st.markdown("### 1 · The Core Volatility Forecasting Model (GARCH)")
        st.markdown("""
        At the heart of the scanner is a **GARCH-family volatility forecasting engine** that
        models daily stock returns to predict future realized volatility.
        """)

        garch_col1, garch_col2 = st.columns(2)
        with garch_col1:
            st.markdown("""
            <div class="metric-card" style="padding:20px;">
                <h4 style="margin-top:0; color:#6C63FF;">Symmetric GARCH(1,1)</h4>
                <code style="color:#e0e0ff; font-size:0.95em;">σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}</code>
                <ul style="color:#ccc; margin-top:12px; font-size:0.9em;">
                    <li><b>ω</b> — long-run variance constant</li>
                    <li><b>α</b> — ARCH coeff (shock reaction)</li>
                    <li><b>β</b> — GARCH coeff (persistence)</li>
                    <li>Stationarity: α + β &lt; 1</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with garch_col2:
            st.markdown("""
            <div class="metric-card" style="padding:20px;">
                <h4 style="margin-top:0; color:#FF6B6B;">Asymmetric GJR-GARCH(1,1,1)</h4>
                <code style="color:#e0e0ff; font-size:0.95em;">σ²_t = ω + (α + γ·I)·ε²_{t-1} + β·σ²_{t-1}</code>
                <ul style="color:#ccc; margin-top:12px; font-size:0.9em;">
                    <li><b>γ</b> — leverage parameter (asymmetry)</li>
                    <li><b>I</b> = 1 if ε_{t-1} &lt; 0, else 0</li>
                    <li>Captures <b>leverage effect</b>: negative returns<br>increase vol more than positive returns</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        Both models are fitted to the stock's full price history. The model with the **lower
        Akaike Information Criterion (AIC)** is automatically selected. GJR-GARCH is chosen
        ~70% of the time, confirming that leverage effects are present in most equity returns.
        Returns are modeled with a **Student-t distribution** to capture the fat tails observed
        in financial markets.
        """)

        st.markdown("""    
        | Parameter | Value | Rationale |
        |---|---|---|
        | p (GARCH lag) | 1 | Standard; higher orders rarely improve fit |
        | q (ARCH lag) | 1 | Captures immediate shock reaction |
        | o (Leverage lag) | 1 | One asymmetric term for leverage effect |
        | Distribution | Student-t | Fat tails in equity returns |
        | Return scaling | ×100 | Numerical stability for MLE optimizer |
        | Trading days | 252 | Standard US market annualization |
        """)

        st.markdown("---")

        # ── Section 2: Signal Methodology ──
        st.markdown("### 2 · Signal Methodology — The 30-Day Realization Benchmark")
        st.markdown("""
        The strategy identifies underpriced options by comparing **forecasted future volatility**
        with recent historical trends:
        """)

        st.markdown("""
        <div class="metric-card" style="padding:20px; text-align:center;">
            <div style="font-size:1.3em; font-weight:600; color:#e0e0ff; margin-bottom:8px;">
                Spread = GARCH Forecast RV − 30-Day Rolling Close-to-Close Historical Volatility
            </div>
            <div style="color:#aaa; font-size:0.9em;">
                A <span style="color:#00cc66; font-weight:600;">positive spread</span> means GARCH
                forecasts volatility expansion above recent history → options are <b>underpriced</b> → BUY VOL
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Why 30-day close-to-close realized vol?** In head-to-head backtesting, this benchmark
        significantly outperformed option IV proxies like the Garman-Klass OHLC estimator:

        | Benchmark | Backtest Return | Period |
        |---|---|---|
        | **30-day close-to-close HV** | **+1,247%** | 43 sequential 5-day trades |
        | Garman-Klass OHLC IV proxy | +736% | Same period |

        The 30-day rolling window provides a stable, noise-resistant measure of the market's
        recent realized volatility that the GARCH forecast can be meaningfully compared against.
        """)

        st.markdown("---")

        # ── Section 3: Universe Selection ──
        st.markdown("### 3 · Universe Selection — 42 Budget-Friendly Tickers")
        st.markdown("""
        The scanner targets a pre-defined universe of **42 highly liquid, high-beta, budget-friendly
        tickers** — stocks priced roughly $1–$30 where ATM straddles fit comfortably within
        tight capital limits.
        """)

        # Show the universe in a styled grid
        from signals.scanner import SCAN_UNIVERSE
        universe_rows = [SCAN_UNIVERSE[i:i+8] for i in range(0, len(SCAN_UNIVERSE), 8)]
        universe_html = '<div class="metric-card" style="padding:20px;"><div style="display:flex; flex-wrap:wrap; gap:8px; justify-content:center;">'
        for sym in SCAN_UNIVERSE:
            universe_html += f'<span style="background:rgba(108,99,255,0.2); border:1px solid rgba(108,99,255,0.3); border-radius:6px; padding:4px 12px; color:#e0e0ff; font-size:0.85em; font-weight:500;">{sym}</span>'
        universe_html += '</div></div>'
        st.markdown(universe_html, unsafe_allow_html=True)

        st.markdown("""
        **Why these tickers?** By prioritizing lower-priced, high-volatility stocks, the system
        ensures that multi-leg option packages (call + put straddles) can fit within tight capital
        limits. These names also tend to have active options markets with reasonable liquidity.
        """)

        st.markdown("---")

        # ── Section 4: Strike & Expiry ──
        st.markdown("### 4 · Option Strike & Expiry Selection")

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("""
            <div class="metric-card" style="padding:20px; text-align:center;">
                <div style="font-size:0.8em; color:#888; text-transform:uppercase; letter-spacing:1px;">Target Expiry</div>
                <div style="font-size:2em; font-weight:700; color:#FFD93D; margin:8px 0;">~14 Days</div>
                <div style="color:#aaa; font-size:0.85em;">Selects the available expiration closest to now + 14 days for high-gamma, short-term exposure</div>
            </div>
            """, unsafe_allow_html=True)
        with exp_col2:
            st.markdown("""
            <div class="metric-card" style="padding:20px; text-align:center;">
                <div style="font-size:0.8em; color:#888; text-transform:uppercase; letter-spacing:1px;">Strike Selection</div>
                <div style="font-size:2em; font-weight:700; color:#6C63FF; margin:8px 0;">ATM ± 1</div>
                <div style="color:#aaa; font-size:0.85em;">Checks ATM and both adjacent strikes (3 candidates per ticker) to find the most cost-efficient entry</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Section 5: Cost & Liquidity Filters ──
        st.markdown("### 5 · Cost, Budget & Liquidity Constraints")
        st.markdown("""
        Before any security is selected, it must pass strict risk and execution filters:
        """)

        st.markdown("""
        <div class="metric-card" style="padding:20px;">
            <div style="display:flex; flex-wrap:wrap; gap:24px; justify-content:space-around;">
                <div style="text-align:center; min-width:160px;">
                    <div style="font-size:0.75em; color:#888; text-transform:uppercase; letter-spacing:1px;">Budget Limit</div>
                    <div style="font-size:1.8em; font-weight:700; color:#FF6B6B;">$150</div>
                    <div style="color:#aaa; font-size:0.8em;">Hard cap per trade</div>
                </div>
                <div style="text-align:center; min-width:160px;">
                    <div style="font-size:0.75em; color:#888; text-transform:uppercase; letter-spacing:1px;">Transaction Fee</div>
                    <div style="font-size:1.8em; font-weight:700; color:#FFD93D;">$1.30</div>
                    <div style="color:#aaa; font-size:0.8em;">Per straddle (Webull)</div>
                </div>
                <div style="text-align:center; min-width:160px;">
                    <div style="font-size:0.75em; color:#888; text-transform:uppercase; letter-spacing:1px;">Min Price Filter</div>
                    <div style="font-size:1.8em; font-weight:700; color:#00cc66;">$0.03</div>
                    <div style="color:#aaa; font-size:0.8em;">Per leg (excludes pennies)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Straddle cost formula:**
        ```
        Straddle Cost = (Call Price + Put Price) × 100 + $1.30 fee
        ```

        **Liquidity filter:** The scanner tracks current trading volume for both the call and
        put legs. Contracts with zero or negligible volume are excluded to ensure high fill
        rates and narrow bid-ask spreads.

        **Pricing source:** Uses actual `lastPrice` from the live options chain (or bid-ask
        midpoint if the market is closed). This is *not* synthetic pricing — the scanner works
        with real market data.
        """)

        st.markdown("---")

        # ── Section 6: Ranking & Selection ──
        st.markdown("### 6 · Ranking & Final Selection")

        st.markdown("""
        <div class="metric-card" style="padding:24px;">
            <h4 style="margin-top:0; color:#e0e0ff;">Selection Pipeline (per scan)</h4>
            <ol style="color:#ccc; line-height:2em; font-size:0.95em;">
                <li>Fetch <b>live options chain</b> from Yahoo Finance for each of 42 tickers</li>
                <li>Find the nearest expiration <b>~14 days out</b></li>
                <li>Try ATM and <b>±1 strikes</b> (3 candidates per ticker)</li>
                <li>Compute straddle cost — must fit within the <b>$150 budget</b></li>
                <li>Fit <b>GARCH</b> on full price history → extract conditional volatility forecast</li>
                <li>Compute <b>30-day rolling close-to-close historical vol</b> (annualized)</li>
                <li>Compute spread: <code style="color:#6C63FF;">GARCH_RV − 30d_hist_vol</code></li>
                <li>Filter: only <b>positive spread</b> candidates (GARCH predicts vol expansion)</li>
                <li>Sort all candidates by spread — <b>strongest signal first</b></li>
                <li>Return <b>Top 8</b> recommendations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Section 7: Role of the Transformer ──
        st.markdown("### 7 · The Transformer Model — Interpretability Only")
        st.markdown("""
        A PyTorch **Transformer encoder** is trained alongside GARCH to provide
        feature-level interpretability. It does **not** generate signals or influence
        trade decisions.

        | Component | Role |
        |---|---|
        | **GARCH** | **Decision engine** — forecasts vol and generates all buy/sell signals |
        | **Transformer** | **Interpretability** — identifies which of 21 features drive volatility |

        The 21 input features span price returns, realized volatility (5/10/21-day), volume,
        technicals (RSI, MACD, Bollinger, ATR), market regime (VIX), options data (P/C ratio,
        avg IV), and macro (10Y Treasury yield). Feature importance is extracted via:

        1. **Gradient saliency** — |∂output/∂input| averaged across test samples
        2. **Attention weights** — averaged across heads and layers for temporal importance

        > ⚠️ The Transformer is displayed in the *Feature Importance* tab after running an
        > analysis. It helps you understand *why* volatility is elevated, but it never
        > overrides the GARCH signal.
        """)

        st.markdown("---")

        # ── Section 8: Risk Management ──
        st.markdown("### 8 · Risk Management & Execution")

        risk_col1, risk_col2 = st.columns(2)
        with risk_col1:
            st.markdown("""
            <div class="metric-card" style="padding:20px;">
                <h4 style="margin-top:0; color:#00cc66;">Defined Risk</h4>
                <ul style="color:#ccc; font-size:0.9em; line-height:1.8em;">
                    <li><b>Max loss = premium paid</b> (no naked options)</li>
                    <li>Budget capped at <b>$150</b> per trade</li>
                    <li>Max position size: <b>90%</b> of capital</li>
                    <li>Single best signal executed (max conviction)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with risk_col2:
            st.markdown("""
            <div class="metric-card" style="padding:20px;">
                <h4 style="margin-top:0; color:#FF6B6B;">Auto-Close Daemon</h4>
                <ul style="color:#ccc; font-size:0.9em; line-height:1.8em;">
                    <li><b>+12%</b> take-profit → auto-sell both legs</li>
                    <li><b>-30%</b> stop-loss → auto-sell to cap losses</li>
                    <li>Market hours: polls every <b>30 seconds</b></li>
                    <li>Off-hours: polls every <b>5 minutes</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Disclaimer ──
        st.markdown("""
        <div class="metric-card" style="padding:20px; border-color:rgba(255,200,50,0.3);">
            <p style="color:#FFD93D; font-weight:600; margin-top:0;">⚠️ Disclaimer</p>
            <p style="color:#aaa; font-size:0.85em; margin-bottom:0;">
                This is a research and educational tool. It does not constitute financial advice.
                Backtested results use Black-Scholes synthetic pricing (historical options data is not
                freely available). Live scanning uses real market prices from the options chain, but
                actual results may differ due to bid-ask spreads, slippage, and liquidity conditions.
                Past performance does not guarantee future results.
            </p>
        </div>
        """, unsafe_allow_html=True)

