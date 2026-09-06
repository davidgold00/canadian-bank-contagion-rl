"""Source-verified explanatory content for the static research site."""
from __future__ import annotations
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import json
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.dashboard.reporting import classifier_dataset, return_reconciliation, sha256
from src.dashboard.investment_signals import compute_market_positioning
from src.portfolio.allocation_policy import BANKS, FINANCIAL_EXPOSURE_ASSETS

ETF_COVERAGE = """<p><strong>ETF coverage limit.</strong> Zero centrality and node stress for XFN and XIU mean “not represented in the six-bank graph,” not zero economic contagion risk. They receive zero explicit node/centrality penalty and no direct bank-block covariance uplift, but retain ordinary historical-return, covariance and CVaR risk. The final positive-semidefinite projection can affect the full covariance matrix.</p><p>The modeled financial-exposure proxy sums direct Big Six weights plus XFN and excludes financial holdings inside XIU. XFN is financial-sector ETF exposure; XIU is broad-market ETF exposure with unmeasured embedded bank/financial holdings. Direct single-name caps do not constrain aggregate issuer exposure through ETFs. A reliable dated look-through estimate is unavailable: the repository contains an ETF template, not current verified holdings. No look-through penalty is added to optimization or backtests.</p>"""

GRAPH_METHOD = """<p><strong>Three graph definitions.</strong> Risk uses 63 trading days of bank returns, edges at absolute correlation ≥0.35, and average weighted connectivity across five possible peers; node size represents that computed measure. Scenarios use 126 days, positive correlations only, zero self-links and row normalization. The optimizer uses its 126-return lookback, edges at absolute correlation ≥0.35, weighted eigenvector centrality divided by its maximum (weighted degree fallback if the eigenvector is ambiguous), and absolute average pairwise correlation. These are separate configured views for co-movement description, directional scenario accounting and portfolio penalties, not a single calibrated transmission model. Statistical co-movement does not identify causal contagion.</p>"""

SCENARIO_METHOD = """<p><strong>Propagation rule.</strong> Let A be the bank positive-correlation matrix with diagonal zero, divided by each source row’s sum (a zero row uses denominator 1). At severity q, s₀ = clip(q × preset, 0, 100). For k = 0,…,4: sₖ₊₁ = clip(0.70 sₖ + 0.45 Aᵀsₖ, 0, 100). Thus rows distribute outgoing stress and columns collect incoming stress. Persistence is 0.70 and spillover 0.45; the combined coefficients can amplify stress. These are five abstract steps, without a calibrated calendar interpretation.</p><p>Preset magnitudes and thresholds are assumptions. Scenario “severe” starts at 70; the composite Risk page’s severe band starts at 90. These are distinct scales and configured thresholds. A finite five-step run does not establish convergence; capped paths may flatten because of saturation. Equal-weight stress scores and shares are not monetary loss forecasts and do not use portfolio holdings. Scenario selection does not rerun the optimizer.</p>"""

METRIC_METHOD = """<p><strong>Return estimate.</strong> The current optimizer uses daily simple adjusted-price returns over 126 observations. The annual input is 0.35 × mean(last 63 returns) × 252 + 0.65 × mean(last min(252, available) returns) × 252. Because only 126 returns enter this solve, the second window is 126, not a full year. Each estimate is clipped to [−25%, +35%]; repeated 0.350000 values are the upper clip, not independent forecasts. Cash expected return is zero. Portfolio expected return is the weighted sum, a model estimate with no promised-return interpretation.</p><p><strong>CVaR.</strong> Current-target CVaR is the negative mean of historical daily portfolio returns at or below their 5th percentile (95% confidence), using current fixed weights on the 126-return window. It is a daily loss fraction, not an annual number or the realized rebalanced strategy. Performance CVaR uses realized simulated daily returns over the stated run period with the same quantile/tail rule. The sign is positive for average tail losses and may be negative if even the selected tail gains.</p><p><strong>Contribution formulas.</strong> The chart is a true historical-tail attribution: cᵢ = −wᵢ × mean(rᵢ | portfolio return ≤ 5th percentile), divided by Σc when positive. If the total is nonpositive, the implementation returns raw signed contributions instead of shares. Annual volatility contribution is wᵢ(Σw)ᵢ / √(wᵀΣw); it sums to volatility, not 100%. Expected-return contribution is wᵢμᵢ. Contagion contribution is wᵢ(0.65 centralityᵢ + 0.35 node-stressᵢ/100).</p><p><strong>Performance conventions.</strong> Cumulative return divides ending NAV by gross initial capital inferred from first NAV minus first daily P&amp;L. The old post-cost-first-NAV convention is retained in the reconciliation table. Other metrics are unchanged: volatility is population daily standard deviation × √252; Sharpe uses zero risk-free rate and population standard deviation; Sortino uses zero target and population standard deviation of negative returns only (not root-mean-square shortfall across all days). Both ratios multiply by √252. CAGR, where used, remains based on first post-cost NAV and 252 observations per year. Maximum drawdown compares with the running NAV peak. Days below starting value compares with first post-cost NAV and is distinct from duration below a prior peak. Feature rolling volatility uses sample standard deviation, unlike the performance summary.</p><p><strong>Turnover.</strong> Daily paper turnover is total absolute risky-asset purchase and sale notional / pre-trade NAV, without a half factor or separate cash leg. The displayed average includes initial purchases and zero-trade days. The optimizer’s turnover objective is instead Σ(w−w_previous)². Explicit target governance does not mean lower observed trading turnover.</p>"""

PORTFOLIO_METHOD = """<p><strong>Different mandates.</strong> The current CVaR snapshot and historical CVaR paper fund both use the Big Six, XFN, XIU and cash, but are distinct configurations. Snapshot: 126 returns; risk-aversion 7, CVaR penalty 9, contagion penalty 0.9, volatility penalty 1, turnover penalty 0.15 and an equal-all-asset prior because no previous holdings are passed. Its limits are 20% per direct bank, 70% modeled financial proxy, 5–60% cash. It is not the paper fund’s next scheduled order.</p><p>The paper CVaR run solves every 10 observations using actual prior weights, risk-aversion 6, CVaR penalty 8, contagion penalty 0.8, volatility penalty 1, turnover penalty 0.20, graph strength 0.40 and 126 returns. Regime limits tighten direct-bank, proxy and cash bounds at 30, 60 and 80; caller caps combine with these limits. It suppresses risky-asset trades below 1% of pre-trade NAV. Small trades, transaction costs and between-rebalance price drift can leave holdings different from target limits; bounds are not continuously enforced on holdings.</p><p>The historical PPO paper runs share the nine-asset universe, deterministic inference, 22% direct-bank cap and 80% financial-proxy cap. They assess targets daily with a 1% per-asset trade threshold. Inputs are 21 days of asset returns, the first 40 feature columns and current weights; actions are softmax-transformed, then constrained. This differs from the CVaR return/covariance/graph inputs and penalized objective. The standalone heuristic excludes XIU and is neither PPO nor the simulator’s fallback.</p><p><strong>PPO evidence boundary.</strong> The saved policy artifact can be identified by hash, but a dated training-dataset manifest and a verified held-out evaluation boundary are unavailable. Training source calls the full dataset and environment; configuration train fractions are not consumed by that training function. These paper evaluations are historical simulations, not established out-of-sample profitability tests. Nonlinear adaptation is architectural capability; aggregate exposure diagnostics do not prove every holding is static or that training failed. A matched static allocation comparison is absent, so incremental value over a comparable simple allocation remains unestablished.</p>"""

EXECUTION_METHOD = """<p><strong>Data and execution.</strong> The downloader requests yfinance Close with auto_adjust=True (split/dividend adjustments supplied by the vendor); dividends are not separately credited. Canadian bank/ETF prices are treated as CAD. USD oil, VIX and CAD/USD are contextual inputs, not tradable USD holdings; no FX conversion is applied to portfolio NAV. Processed rows require at least six numeric market observations. Macro features are aligned to market dates and forward-filled, so a feature date can be later than a macro observation.</p><p>Paper simulators forward-fill prices and features without backward-fill, skip dates with incomplete risky-asset prices, and supply neutral/zero defaults downstream where specified. CVaR return gaps are filled with zero after price forward-fill. Cash earns zero. Both paper engines charge 5 basis points per absolute risky-asset trade notional, reserve costs and may sell to repair negative cash; no tax, market impact or liquidity/slippage model is added. Prior holdings are marked at today’s close before new weights trade at that same close; new targets use that close’s features. This idealized same-close execution is not a verified executable signal delay.</p><p>The separate PPO training environment applies newly selected weights to the current indexed return and uses current-index features; paper-engine ordering does not establish that training timing is free of look-ahead. The inference return-window alignment also differs by one observation from the training environment. These newly identified issues are documented for a separate correction, with training and historical results preserved here.</p><p><strong>Transaction explanations.</strong> Original reason strings are contextual templates selected by direction, stress and centrality thresholds, not solver causal traces or PPO explanations. Public activity tables describe observed direction, target and pre-trade weight; no claim is made that an indicator caused a trade.</p>"""


def optimizer_explanation(result, constraints, score, strength):
    d=result.diagnostics
    pressure=.45*np.clip(score/100,0,1)+.30*np.clip(d['graph_density'],0,1)+.25*np.clip(d['average_correlation'],0,1)
    uplift=strength*pressure
    exposure=result.weights.reindex(FINANCIAL_EXPOSURE_ASSETS,fill_value=0).sum()
    cash=result.weights.get('cash',0)
    return (f"<p><strong>Signal-dependent input:</strong> pressure = 0.45 × score/100 + 0.30 × graph density + 0.25 × average absolute correlation. At this snapshot, strength {strength:.2f} gives a {uplift:.2%} common multiplier increase on covariance entries in the direct Big Six block, including variances; this is not a {uplift:.0%} standard-deviation increase. Each bank pair then receives the additional product aᵢaⱼ where aᵢ = 1 + {strength:.2f}(0.65 centralityᵢ + 0.35 stressᵢ/100). A positive-semidefinite projection follows.</p>"
            f"<p><strong>Observed solution:</strong> {exposure:.2%} modeled financial-proxy exposure and {cash:.2%} cash; financial-cap slack {(constraints.max_bank_exposure-exposure)*100:.2f} percentage points. "
            + (f"Cash is at the {constraints.min_cash_weight:.0%} floor. " if abs(cash-constraints.min_cash_weight)<1e-6 else "Cash differs from the configured floor. ")
            + "The objective jointly uses return, tail loss, covariance, contagion and turnover. A counterfactual solve would be needed to attribute these weights to the score alone or establish the unconstrained cash allocation.</p>")


def policy_bands(features, macro):
    # Discover transitions from the implemented lookup, rather than duplicate boundaries.
    groups=[]
    for score in range(101):
        p=compute_market_positioning(features,macro,float(score))
        key=(p['total_bank_budget'],p['cash_guidance'].split('.')[0])
        if not groups or groups[-1]['key']!=key: groups.append({'start':score,'end':score,'key':key})
        else: groups[-1]['end']=score
    return pd.DataFrame([{'Score condition':f"{g['start']} ≤ score < {groups[i+1]['start']}" if i+1<len(groups) else f"{g['start']} ≤ score ≤ 100",
                          'Editorial bank-risk budget':g['key'][0], 'Editorial cash guidance':g['key'][1]+' cash',
                          'Implementation':'Guidance only; no hysteresis or automatic policy-compliant allocation'} for i,g in enumerate(groups)])


def validation_disclosure(features, metrics, table):
    ds,cols,split,cut=classifier_dataset(features)
    train,test=ds.iloc[:split],ds.iloc[split:]
    rows=[]
    for name,part in [('Train',train),('Test',test)]:
        rows.append({'Partition':name,'Dates':f'{part.index[0].date()}–{part.index[-1].date()}', 'Rows':len(part),
                     'Positive labels':int(part.target.sum()), 'Prevalence':f'{part.target.mean():.2%}'})
    rf=metrics.set_index('Model').loc['Random Forest']
    missing=int(ds.index.isin(features.index[-5:]).sum())
    crossed=int((features.index.get_indexer(train.index)+5 >= features.index.get_loc(test.index[0])).sum())
    return (f"<p><strong>Scope:</strong> AUC {metrics.iloc[0]['AUC']:.3f} evaluates supervised future-stress ranking only. It does not validate CVaR allocation quality, PPO profitability, causal contagion or rebalance recommendations. No mapping to portfolio confidence is documented.</p>"
            f"<p>Target: the composite score five dataset observations ahead is at or above the full-sample future-score 80th percentile ({cut:.6f}/100); this is a five-observation endpoint, not any stress event within five calendar days. All {len(cols)} numeric columns except the current composite are retained; infinities become missing and rows with missing feature values are dropped. First floor(70% × N) rows train, the remainder test, without shuffling.</p>"
            +table(pd.DataFrame(rows),label='Supervised classifier split details')
            +f"<p>Logistic regression: training-only StandardScaler, balanced class weights, max_iter 1500. Random Forest: 180 trees, depth 5, minimum leaf 10, balanced class weights, seed 42; no feature scaling. Both predict at probability ≥0.50, a fixed threshold with no documented tuning procedure. Two configured models are fitted at export and displayed in test-AUC order; this is not a nested model-selection evaluation. These results are distinct from artifacts/supervised training outputs.</p>"
            +f"<p><strong>Verified validation limitations, unchanged:</strong> the label quantile uses the full sample, including the test period. No purge/embargo is applied; {crossed} training labels reach into the test period. The last {missing} included rows have unavailable future scores converted to negative labels by the existing comparison. These source paths preclude claiming that leakage safeguards passed; effects on reported statistics have not been isolated. Proposed separate corrections are train-only label calibration, horizon-aware split purging and removal of unavailable future labels, followed by new separately labeled evaluations.</p>"
            +f"<p>Random Forest precision {rf.Precision:.3f}, recall {rf.Recall:.3f}: approximately {(1-rf.Precision)*100:.1f}% of positive alerts are false alerts at this threshold. That is not the false-positive rate among negative cases. Confidence intervals, repeated temporal evaluation and a persistence/simple baseline are unavailable. Those tests are needed to establish robust incremental predictive value.</p>")


def tradeoff_text(cvar, rl):
    gap=rl['ending_value']-cvar['ending_value']
    wealth=gap/rl['ending_value']
    dd=abs(rl['max_drawdown'])-abs(cvar['max_drawdown'])
    relative=dd/abs(rl['max_drawdown']) if rl['max_drawdown'] else 0
    return (f"Common-period CVaR ends at ${cvar['ending_value']:,.2f}; PPO ends at ${rl['ending_value']:,.2f}. "
            f"CVaR minus PPO ending wealth is ${-gap:+,.2f} ({-wealth:+.2%} relative to PPO). "
            f"CVaR’s maximum-drawdown magnitude is {dd*100:.2f} percentage points smaller ({relative:.2%} relative reduction). "
            f"Cumulative return differs by {(rl['cumulative_return']-cvar['cumulative_return'])*100:.2f} percentage points using gross initial capital. "
            f"Sharpe: CVaR {cvar['sharpe_ratio']:.2f}, PPO {rl['sharpe_ratio']:.2f}; Sortino: CVaR {cvar['sortino_ratio']:.2f}, PPO {rl['sortino_ratio']:.2f}. "
            "These relative wealth and drawdown differences are not return percentage points. The choice depends on the mandate’s return participation and drawdown priorities, not a universal ranking.")


def run_disclosure(cvar, rl, extended_ledger, extended_weights, extended_source, target, heuristic, date, table):
    path=Path('artifacts/rl/ppo_model.zip'); artifact=sha256(path) if path.exists() else 'unavailable'
    run_rows=[]; rec_rows=[]; records={}
    for name,ledger,weights,source in [('CVaR common period',cvar.ledger,cvar.weights,cvar.policy_source),('PPO common period',rl.ledger,rl.weights,rl.policy_source),('PPO extended period',extended_ledger,extended_weights,extended_source)]:
        r=return_reconciliation(ledger)
        run_rows.append({'Run':name, 'Source':source,'Evaluation dates':f'{ledger.index[0].date()}–{ledger.index[-1].date()}',
                         'Observations':len(ledger),'Latest cash':f"{weights.iloc[-1]['cash']:.2%}",
                         'Last traded':str(ledger.index[ledger.number_of_trades>0][-1].date()) if (ledger.number_of_trades>0).any() else 'None'})
        rec_rows.append({'Run':name,'Initial capital':f"${r['initial_capital']:,.2f}",'First post-cost NAV':f"${r['first_nav']:,.2f}",
                         'Initial costs':f"${r['initial_cost']:,.2f}",'Ending NAV':f"${r['ending_value']:,.2f}",
                         'Return / initial capital':f"{r['gross_initial_return']:.4%}",'Return / first NAV (old)':f"{r['post_cost_nav_return']:.4%}"})
        records[name]={'policy_source':source,'start':str(ledger.index[0].date()),'end':str(ledger.index[-1].date()),
                       'assets':list(weights.columns),'latest_weights':weights.iloc[-1].to_dict(), 'return_reconciliation':r,
                       'policy_notes':list(ledger.policy_note.unique()) if 'policy_note' in ledger else [],
                       'daily_values':{str(k.date()):float(v) for k,v in ledger.portfolio_value.items()}}
    manifest={'as_of':date,'ppo_sha256':artifact,'runs':records,'current_cvar_target':target.weights.to_dict(),
              'snapshot_status':target.diagnostics['status'],'standalone_heuristic':heuristic.to_dict()}
    download='data:application/json;charset=utf-8,'+quote(json.dumps(manifest),safe='')
    last_solve=pd.Timestamp(cvar.ledger.iloc[-1]['allocation_observation_date']).date()
    return ("<h3>Portfolio and run identities</h3>"+table(pd.DataFrame(run_rows),label='Portfolio run identities')
            +f"<p>Current CVaR snapshot target: cvar_snapshot / optimize_cvar_portfolio, as of {date}; solver status {escape(str(target.diagnostics['status']))}. Actual CVaR paper holdings: common-period simulator, as of {cvar.ledger.index[-1].date()}, last scheduled solve {last_solve}. The snapshot is a separate configuration, not the paper run’s live target. PPO common and extended runs start at the 72% and 55% positions of the available price history respectively; the longer run is never joined into the headline comparison. All three paper universes: {', '.join(cvar.weights.columns)}.</p>"
            +f"<p>Historical PPO policy: artifacts/rl/ppo_model.zip, SHA-256 <code style='overflow-wrap:anywhere'>{artifact}</code>. Common-period runtime notes: {escape('; '.join(rl.ledger.policy_note.unique())) if 'policy_note' in rl.ledger else 'unavailable'}. Saved training dates/universe manifest: unavailable; the listed universe is the inference universe verified at runtime. The current heuristic is source allocation_chart, as of {date}; no PPO inference is attempted for it. Independent bank scores are from compute_bank_signals as of {date}; editorial ranges are from compute_market_positioning on that date.</p>"
            +"<p>Exact portfolio vectors must sum to 100% within 0.0001 percentage points. One-decimal display rounding allows at most 0.05 percentage points per asset (0.45 for nine assets); values are not rescaled to hide residuals.</p>"
            +"<h3>Capital and return reconciliation</h3>"+table(pd.DataFrame(rec_rows),label='Capital and return denominators')
            +"<p>The headline cumulative return now includes initial costs by using original capital. Ending NAV, daily returns, trades, Sharpe, Sortino and drawdowns are unchanged. Compounding ledger daily returns reconciles to the original-capital return.</p>"
            +f"<a class='button secondary' download='northern-signal-run-evidence.json' href='{download}'>Download precise run evidence</a>")


def provenance_disclosure(root, features, prices, macro, build_time, table):
    rows=[]
    for rel,usage in [('data/raw/market_prices.csv','Candidate market source cache; no retrieval-status manifest'),('data/raw/boc_yields.csv','Selected macro source cache'),('data/processed/prices.csv','Active portfolio and scenario price input'),('data/processed/model_dataset.csv','Active score, classifier and portfolio features'),('data/sample/market_prices.csv','Unused synthetic sample when processed/raw files exist'),('data/sample/macro.csv','Unused synthetic fallback when raw macro exists'),('data/templates/etf_holdings_template.csv','Unused template; no reliable current ETF look-through'),('data/templates/housing_stress_template.csv','Unused manual template'),('data/templates/cds_template.csv','Unused manual template')]:
        p=root/rel
        if not p.exists(): continue
        d=pd.read_csv(p); col='date' if 'date' in d else d.columns[0]
        dates=pd.to_datetime(d[col],errors='coerce').dropna()
        rows.append({'File':rel,'Use in this snapshot':usage,'Last dated row':str(dates.max().date()) if len(dates) else 'Unavailable',
                     'File modified (UTC)':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(timespec='seconds')})
    market_dates=[]
    for col in prices:
        if prices[col].notna().any():
            market_dates.append({'Market series':col,'Last observed price in selected input':str(prices[col].dropna().index[-1].date())})
    macro_dates=[]
    for col in ['policy_rate','ca_2y','ca_5y','ca_10y']:
        if col in macro and macro[col].notna().any():
            macro_dates.append({'Macro series':col,'Last source observation':str(macro[col].dropna().index[-1].date()),'Value (%)':f'{macro[col].dropna().iloc[-1]:.3f}'})
    docs=[]
    for rel in ['reports/methodology.md','reports/model_card.md','reports/limitations.md','configs/config.yaml','configs/data_sources.yaml','configs/graph_config.yaml','configs/rl_config.yaml','data/README.md','docs/interview-audit.md']:
        p=root/rel
        docs.append(f"<li><a href='https://github.com/davidgold00/canadian-bank-contagion-rl/blob/simplified/{rel}'>{rel}</a></li>" if p.exists() else f'<li>{rel}: unavailable</li>')
    return (f"<p>Site build time: {escape(build_time)}. Feature processing file time is listed below; it is not a source observation timestamp. Public refresh is static, not a live feed.</p>"
            +table(pd.DataFrame(rows),label='Active inputs and unused files')+table(pd.DataFrame(macro_dates),label='Source-specific macro observation dates')+table(pd.DataFrame(market_dates),label='Source-specific market observation dates')
            +"<p>These pages read local caches. Downloaders are configured for Yahoo Finance and Bank of Canada Valet, but can write synthetic fallback data to the same raw paths without recording status. No retrieval manifest certifies live-derived versus synthetic lineage for the existing market cache; file presence alone is insufficient. The source selection and dates above are verified, while original download provenance remains unverified. Source observations and forward-filled feature dates must not be conflated.</p>"
            +"<h3>Available documentation</h3><ul>"+''.join(docs)+"</ul><p>Legacy reports are introductory and may describe capabilities not active in this export; the source-verified disclosures here describe the displayed runs. Training data manifest, classifier uncertainty report and dated current ETF holdings: unavailable.</p>")
