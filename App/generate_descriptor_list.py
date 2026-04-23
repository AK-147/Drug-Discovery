import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

df = pd.read_csv('Data/aromatase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')

X = df.drop('pIC50', axis=1)
Y = df['pIC50']

selection = VarianceThreshold(threshold=0.1)
X_sel = selection.fit_transform(X)

selected_cols = X.columns[selection.get_support()]
print(f"Selected {len(selected_cols)} features (no index column)")

pd.DataFrame(columns=selected_cols).to_csv('descriptor_list.csv', index=False)
print("Saved descriptor_list.csv")

model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_sel, Y)
print(f"Trained model — R²: {model.score(X_sel, Y):.4f}")

pickle.dump(model, open('aromatase_model.pkl', 'wb'))
print("Saved aromatase_model.pkl")
