from pathlib import Path
import pandas as pd

def load_housing_data(path='data/templates/housing_stress_template.csv'):
    df=pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()
    return df.resample('B').ffill()
