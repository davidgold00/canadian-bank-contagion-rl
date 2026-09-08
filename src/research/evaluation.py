"""Chronological classifier evaluation. Never called by the HTML renderer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                            recall_score, confusion_matrix, accuracy_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .provenance import object_id, write_json

METHOD = 'classifier-chronological-v2'


def labeled_frame(features, horizon=5):
    cols = [c for c in features if c != 'contagion_risk_score' and pd.api.types.is_numeric_dtype(features[c])]
    frame = features[cols].replace([np.inf, -np.inf], np.nan).copy()
    frame['current_score'] = features.contagion_risk_score
    frame['future_score'] = features.contagion_risk_score.shift(-horizon)
    frame['outcome_date'] = pd.Series(features.index, index=features.index).shift(-horizon)
    return frame.dropna(), cols


def freeze_protocol(frame, feature_cols, input_hash):
    n = len(frame)
    if n < 150:
        raise ValueError('At least 150 complete labeled observations are required.')
    dates = frame.index
    return {'method': METHOD, 'input_hash': input_hash, 'horizon_observations': 5,
            'features': feature_cols, 'train_start': str(dates[0].date()),
            'validation_start': str(dates[int(n * .70)].date()),
            'test_start': str(dates[int(n * .85)].date()), 'test_end': str(dates[-1].date()),
            'portfolio_evaluation_end': str(frame.outcome_date.max().date()),
            'selection': 'validation average precision; fixed 0.50 decision threshold',
            'seed': 42, 'label_quantile': .80}


def partition(frame, protocol):
    v, t, end = [pd.Timestamp(protocol[k]) for k in ['validation_start', 'test_start', 'test_end']]
    train = frame[(frame.index >= pd.Timestamp(protocol['train_start'])) & (frame.index < v) & (frame.outcome_date < v)]
    val = frame[(frame.index >= v) & (frame.index < t) & (frame.outcome_date < t)]
    test = frame[(frame.index >= t) & (frame.index <= end)]
    if min(map(len, [train, val, test])) < 10:
        raise ValueError('Insufficient purged partition coverage.')
    threshold = float(train.future_score.quantile(protocol['label_quantile']))
    return train, val, test, threshold


def metrics(y, score, predictions):
    return {'auroc': float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
            'average_precision': float(average_precision_score(y, score)) if np.any(y) else None,
            'precision': float(precision_score(y, predictions, zero_division=0)),
            'recall': float(recall_score(y, predictions, zero_division=0)),
            'accuracy': float(accuracy_score(y, predictions)),
            'confusion_matrix_tn_fp_fn_tp': confusion_matrix(y, predictions, labels=[0, 1]).ravel().tolist(),
            'prevalence': float(np.mean(y)), 'observations': len(y)}


def models():
    return {'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight='balanced')),
            'Random forest': RandomForestClassifier(n_estimators=180, max_depth=5, min_samples_leaf=10,
                                                   random_state=42, class_weight='balanced')}


def evaluate_classifier(features, input_hash, output, protocol_path=None):
    frame, cols = labeled_frame(features)
    output = Path(output)
    protocol_path = Path(protocol_path or output.with_name('classifier-protocol.json'))
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol['input_hash'] != input_hash or protocol['features'] != cols:
            raise ValueError('Changed inputs need a new evaluation version and protocol path.')
    else:
        protocol = freeze_protocol(frame, cols, input_hash)
        write_json(protocol_path, protocol, immutable=True)
    if output.exists():
        saved = json.loads(output.read_text())
        if saved['protocol'] != protocol:
            raise ValueError('Evaluation already exists with a different protocol.')
        return saved
    train, val, test, threshold = partition(frame, protocol)
    fitted, validation = {}, {}
    for name, model in models().items():
        model.fit(train[cols], (train.future_score >= threshold).astype(int))
        prob = model.predict_proba(val[cols])[:, list(model.classes_).index(1)]
        validation[name] = metrics((val.future_score >= threshold).astype(int), prob, prob >= .5)
        fitted[name] = model
    selected = max(validation, key=lambda n: validation[n]['average_precision'] if validation[n]['average_precision'] is not None else -1)
    # Selection is complete before any final test prediction is inspected.
    final = {}
    predictions = {}
    y = (test.future_score >= threshold).astype(int)
    for name, model in fitted.items():
        prob = model.predict_proba(test[cols])[:, list(model.classes_).index(1)]
        final[name] = metrics(y, prob, prob >= .5)
        predictions[name] = prob.tolist()
    final['Current-score persistence'] = metrics(y, test.current_score, test.current_score >= threshold)
    prior = float((train.future_score >= threshold).mean())
    final['Training class prior'] = metrics(y, np.full(len(y), prior), np.full(len(y), prior >= .5))
    # Expanding development folds do not touch the final test partition.
    dev = frame[(frame.index < pd.Timestamp(protocol['test_start'])) & (frame.outcome_date < pd.Timestamp(protocol['test_start']))]
    folds = []
    for frac in [.55, .70, .85]:
        boundary = dev.index[int(len(dev) * frac)]
        stop = dev.index[min(int(len(dev) * (frac + .15)), len(dev) - 1)]
        a = dev[(dev.index < boundary) & (dev.outcome_date < boundary)]
        b = dev[(dev.index >= boundary) & (dev.index <= stop) & (dev.outcome_date <= stop)]
        q = float(a.future_score.quantile(.8))
        scores = {}
        for name, model in models().items():
            model.fit(a[cols], (a.future_score >= q).astype(int))
            prob = model.predict_proba(b[cols])[:, list(model.classes_).index(1)]
            scores[name] = metrics((b.future_score >= q).astype(int), prob, prob >= .5)
        folds.append({'validation_start': str(boundary.date()), 'validation_end': str(stop.date()), 'metrics': scores})
    result = {'id': METHOD + '-' + object_id(protocol), 'protocol': protocol, 'threshold': threshold,
              'selected_model': selected, 'validation': validation, 'test': final,
              'development_folds': folds, 'test_prediction_dates': [str(d.date()) for d in test.index],
              'test_outcome_dates': [str(d.date()) for d in test.outcome_date],
              'predictions': predictions, 'test_labels': y.tolist(),
              'partitions': {name: {'start': str(f.index[0].date()), 'end': str(f.index[-1].date()), 'count': len(f),
                                    'last_outcome': str(f.outcome_date.max().date())} for name, f in [('train', train), ('validation', val), ('test', test)]},
              'excluded_unavailable_or_incomplete': len(features) - len(frame),
              'purged_boundary_rows': len(frame) - len(train) - len(val) - len(test),
              'scope': 'Ranking the composite score five observations ahead; not bank failure, portfolio approval or profitability.'}
    write_json(output, result, immutable=True)
    return result
