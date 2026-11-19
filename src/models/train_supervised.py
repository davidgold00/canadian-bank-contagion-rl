from pathlib import Path
import pandas as pd, joblib
from sklearn.metrics import roc_auc_score
from src.data.build_dataset import build_dataset
from src.features.target_generation import make_stress_event_labels
from .baselines import make_logistic, make_random_forest

def train(output_dir='artifacts/supervised'):
    df=build_dataset(); y=make_stress_event_labels(df); X=df.select_dtypes('number').replace([float('inf'),-float('inf')],0).fillna(0)
    data=X.join(y).dropna(); X=data[X.columns]; y=data[y.name].astype(int)
    cut=int(len(data)*.7); Xtr,Xte,ytr,yte=X.iloc[:cut],X.iloc[cut:],y.iloc[:cut],y.iloc[cut:]
    models={'logistic':make_logistic(),'random_forest':make_random_forest()}; Path(output_dir).mkdir(parents=True,exist_ok=True)
    metrics={}
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; metrics[name]={'auc':float(roc_auc_score(yte,p)) if yte.nunique()>1 else None}; joblib.dump(m, f'{output_dir}/{name}.joblib')
    pd.DataFrame(metrics).T.to_csv(f'{output_dir}/metrics.csv'); return metrics
if __name__=='__main__': print(train())
