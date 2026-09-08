from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


TRADABLES = ['RY.TO', 'TD.TO', 'BMO.TO', 'BNS.TO', 'CM.TO', 'NA.TO', 'XFN.TO', 'XIU.TO']
ASSETS = TRADABLES + ['cash']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()[:16]


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json(path, value, immutable=False):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + '\n'
    if immutable and path.exists():
        if path.read_text() != payload:
            raise ValueError(f'Immutable artifact already exists: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(payload)
    os.replace(temp, path)


def validate_panel(df, required, *, positive=False, minimum_rows=1):
    import numpy as np
    import pandas as pd
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.isna().any():
        raise ValueError('A valid dated index is required.')
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('Dates must be unique and chronological.')
    if len(df) < minimum_rows:
        raise ValueError('Insufficient observations.')
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required series: {sorted(missing)}')
    values = df[required].apply(pd.to_numeric, errors='coerce')
    if (df[required].notna() & values.isna()).any().any():
        raise ValueError('Non-numeric source values')
    if np.isinf(values.to_numpy()).any():
        raise ValueError('Infinite source values.')
    if (values.notna().sum() < minimum_rows).any():
        raise ValueError('Required series has insufficient observed values.')
    if positive and (values <= 0).any().any():
        raise ValueError('Tradable adjusted prices must be positive.')
    last = {c: str(values[c].dropna().index[-1].date()) for c in required}
    return {'row_count': len(df), 'first_date': str(df.index[0].date()),
            'last_observation': last, 'missing_counts': values.isna().sum().astype(int).to_dict(),
            'incomplete_dates': [str(x.date()) for x in values.index[values.isna().any(axis=1)]]}


def record_download(path, frame, provider, series, required, positive=False, adjustments=None):
    quality = validate_panel(frame, required, positive=positive, minimum_rows=126)
    manifest = {'status': 'downloaded_verified', 'provider': provider,
                'requested_series': series, 'retrieved_at': utcnow(),
                'sha256': digest(path), 'quality': quality, 'adjustments': adjustments,
                'historical_release_timestamps': 'Unavailable; observation dates are not release timestamps.'}
    write_json(str(path) + '.manifest.json', manifest)
    return manifest


def verify_download(path):
    path = Path(path)
    manifest = json.loads(Path(str(path) + '.manifest.json').read_text())
    if manifest.get('status') != 'downloaded_verified' or manifest.get('sha256') != digest(path):
        raise ValueError(f'Unverified or modified input: {path}')
    return manifest


def prepare_dataset(market_path, macro_path, destination):
    """Strict production intake. No synthetic fallback or tradable price filling."""
    import pandas as pd
    from src.features.market_features import make_market_features
    from src.features.macro_features import make_macro_features
    from src.features.stress_features import make_contagion_risk_score
    market_manifest, macro_manifest = verify_download(market_path), verify_download(macro_path)
    prices = pd.read_csv(market_path, index_col=0, parse_dates=True)
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    quality = validate_panel(prices, TRADABLES + ['^VIX'], positive=True, minimum_rows=252)
    validate_panel(macro, ['policy_rate', 'ca_2y', 'ca_5y', 'ca_10y'], minimum_rows=252)
    # A complete close across the actual tradable universe is required.
    prices = prices.loc[prices[TRADABLES + ['^VIX']].notna().all(axis=1)].copy()
    if len(prices) < 252:
        raise ValueError('Insufficient complete market sessions.')
    latest=prices.index[-1]
    for column in TRADABLES+['^VIX']:
        if (latest-prices[column].dropna().index[-1]).days>7:raise ValueError(f'Stale market series: {column}')
    for column in ['policy_rate','ca_2y','ca_5y','ca_10y']:
        if (latest-macro[column].dropna().index[-1]).days>7:raise ValueError(f'Stale macro series: {column}')
    macro['slope_10y_2y'] = macro.ca_10y - macro.ca_2y
    macro['curvature'] = 2 * macro.ca_5y - macro.ca_2y - macro.ca_10y
    # No historical release-time archive exists: use an explicit conservative
    # one-market-session delay after observation date, rather than inventing one.
    aligned = make_macro_features(macro).reindex(prices.index, method='ffill').shift(1)
    observed_date = pd.Series(macro.index, index=macro.index).reindex(prices.index, method='ffill').shift(1)
    age = pd.Series(prices.index, index=prices.index).sub(observed_date).dt.days
    if age.iloc[-1] > 7 or pd.isna(age.iloc[-1]):
        raise ValueError('Macro observations are stale by more than seven calendar days.')
    mf = make_market_features(prices)
    features = mf.join(aligned, how='left')
    features['contagion_risk_score'] = make_contagion_risk_score(features)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in [('prices', prices), ('features', features), ('macro', macro)]:
        frame.to_csv(destination / f'{name}.csv', index_label='date')
    source = {'status': 'downloaded_verified', 'market': market_manifest, 'macro': macro_manifest,
              'quality': quality, 'macro_age_days_latest': int(age.iloc[-1]),
              'macro_alignment': 'Forward aligned with one market-session delay; actual release timestamps unavailable.',
              'tradable_price_fill': 'None; incomplete sessions excluded.',
              'feature_date': str(features.index[-1].date()),
              'hashes': {name: digest(destination / f'{name}.csv') for name in ['prices', 'features', 'macro']}}
    write_json(destination / 'provenance.json', source)
    return prices, features, source
