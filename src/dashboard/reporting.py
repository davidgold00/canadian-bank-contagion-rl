"""Reporting-only diagnostics. Never used by optimization, scoring or simulation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.portfolio.allocation_policy import BANKS, FINANCIAL_EXPOSURE_ASSETS
from src.portfolio.performance_metrics import performance_summary

ACTION_TOLERANCE = 0.001  # 0.10 percentage points; distinct from trade threshold
WEIGHT_TOLERANCE = 1e-6


def portfolio_action(delta: float, tolerance: float = ACTION_TOLERANCE) -> str:
    return 'BUY' if delta > tolerance else 'SELL' if delta < -tolerance else 'HOLD'


def portfolio_comparison(current: pd.Series, target: pd.Series) -> pd.DataFrame:
    assets = list(dict.fromkeys([*BANKS, 'XFN.TO', 'XIU.TO', *current.index, *target.index, 'cash']))
    for vector in (current, target):
        if not np.isfinite(vector).all() or abs(float(vector.sum()) - 1) > WEIGHT_TOLERANCE:
            raise ValueError('Portfolio weights must be finite and sum to 100%.')
    table = pd.DataFrame({'Bank': assets, 'Current Weight': current.reindex(assets, fill_value=0).values,
                          'Target Weight': target.reindex(assets, fill_value=0).values})
    table['Delta'] = table['Target Weight'] - table['Current Weight']
    table['Action'] = table['Delta'].map(portfolio_action)
    table['Reason'] = table.apply(lambda row: (
        'Cash balance change from the same portfolio vectors; no separate cash security trade.' if row['Bank'] == 'cash'
        else 'No material target change; no sale of a zero holding.' if row['Action'] == 'HOLD'
        else 'Current CVaR snapshot target minus actual CVaR paper holdings. Portfolio construction action; independent bank signals are separate.'
    ), axis=1)
    return table.assign(materiality=table['Delta'].abs()).sort_values(
        ['materiality', 'Bank'], ascending=[False, True], kind='stable').drop(columns='materiality').reset_index(drop=True)


def rebalance_display(table: pd.DataFrame) -> pd.DataFrame:
    out = table.rename(columns={'Bank': 'Asset'}).copy()
    for col in ['Current Weight', 'Target Weight']:
        out[col] = out[col].map(lambda x: f'{x:.1%}')
    out['Delta'] = out['Delta'].map(lambda x: f'{100*x:+.2f} pp')
    return out


def return_reconciliation(ledger: pd.DataFrame) -> dict[str, float]:
    first = ledger.iloc[0]
    initial = float(first['portfolio_value'] - first['daily_pnl'])
    end = float(ledger.iloc[-1]['portfolio_value'])
    if initial <= 0:
        raise ValueError('Cannot establish positive initial capital from ledger.')
    return {'initial_capital': initial, 'first_nav': float(first['portfolio_value']),
            'initial_cost': float(first['transaction_costs']), 'ending_value': end,
            'gross_initial_return': end / initial - 1,
            'post_cost_nav_return': end / float(first['portfolio_value']) - 1,
            'compounded_daily_return': float((1 + ledger['daily_return']).prod() - 1)}


def reported_summary(ledger: pd.DataFrame) -> dict:
    """Fix the display denominator only; retain the original ledger and other metrics."""
    out = performance_summary(ledger['portfolio_value'], ledger['daily_return'],
                              ledger['turnover'], ledger['transaction_costs'])
    out['cumulative_return'] = return_reconciliation(ledger)['gross_initial_return']
    return out


def constraint_status(weights: pd.Series, constraints) -> pd.DataFrame:
    rows = []
    def upper(label, value, limit):
        slack = limit - value
        rows.append({'Constraint': label, 'Actual': f'{value:.2%}', 'Limit': f'≤ {limit:.2%}',
                     'Slack': f'{slack*100:.2f} pp',
                     'Status': 'Breach' if slack < -WEIGHT_TOLERANCE else 'Active' if slack <= WEIGHT_TOLERANCE else 'Slack'})
    upper('Direct Big Six single-name maximum', float(weights.reindex(BANKS, fill_value=0).max()), constraints.max_single_name_weight)
    upper('Modeled financial-exposure proxy (Big Six + XFN)', float(weights.reindex(FINANCIAL_EXPOSURE_ASSETS, fill_value=0).sum()), constraints.max_bank_exposure)
    upper('Cash ceiling', float(weights.get('cash', 0)), constraints.max_cash_weight)
    cash = float(weights.get('cash', 0)); slack = cash - constraints.min_cash_weight
    rows.append({'Constraint':'Cash floor', 'Actual':f'{cash:.2%}', 'Limit':f'≥ {constraints.min_cash_weight:.2%}',
                 'Slack':f'{slack*100:.2f} pp', 'Status':'Breach' if slack < -WEIGHT_TOLERANCE else 'Active' if slack <= WEIGHT_TOLERANCE else 'Slack'})
    rows.extend([{'Constraint':'Fully invested', 'Actual':f'{weights.sum():.6%}', 'Limit':'100%', 'Slack':'—',
                  'Status':'Pass' if abs(weights.sum()-1) <= WEIGHT_TOLERANCE else 'Breach'},
                 {'Constraint':'Long-only', 'Actual':f'Minimum weight {weights.min():.2%}', 'Limit':'≥ 0%', 'Slack':'—',
                  'Status':'Pass' if weights.min() >= -WEIGHT_TOLERANCE else 'Breach'}])
    return pd.DataFrame(rows)


def exposure_diagnostics(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows=[]
    for name, ledger in runs.items():
        for key, label in [('bank_exposure','Modeled financial proxy'), ('cash_weight','Cash')]:
            x=ledger[key]
            rows.append({'Run':name, 'Measure':label, 'Minimum':f'{x.min():.2%}', 'Maximum':f'{x.max():.2%}',
                         'Mean':f'{x.mean():.2%}', 'Daily level std. dev.':f'{x.std(ddof=0)*100:.3f} pp',
                         'Average daily turnover':f'{ledger.turnover.mean():.2%}'})
    return pd.DataFrame(rows)


def scenario_leaders(values: pd.Series) -> dict:
    """Display ties at one decimal; report exact ties separately, without forcing ranks."""
    maximum = float(values.max())
    exact = values.index[np.isclose(values, maximum, rtol=0, atol=1e-9)].tolist()
    displayed = values.index[values.map(lambda x: f'{x:.1f}') == f'{maximum:.1f}'].tolist()
    return {'exact':exact, 'displayed':displayed, 'saturated':values.index[values >= 100-1e-9].tolist()}


def classifier_dataset(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int, float]:
    """Reproduce existing validation intake exactly, including its documented limitations."""
    future = features['contagion_risk_score'].shift(-5)
    threshold = float(future.quantile(.80))
    y = (future >= threshold).astype(int)
    cols = [c for c in features if c != 'contagion_risk_score' and pd.api.types.is_numeric_dtype(features[c])]
    dataset = pd.concat([features[cols].replace([np.inf,-np.inf],np.nan),y.rename('target')],axis=1).dropna()
    return dataset, cols, int(len(dataset)*.70), threshold


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
