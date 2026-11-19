from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_logistic(): return Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=1000))])
def make_random_forest(): return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
