# Predicción del Total de Delitos en México con XGBoost y Streamlit

Aplicación web en **Streamlit** para ejecutar inferencia con un modelo **XGBoost** ya entrenado (`XGBRegressor`, 500 árboles, 208 características) a partir de variables numéricas y categóricas de delitos en México.

## Supuestos documentados

1. **Reutilización sin reentrenamiento:** El modelo cargado desde `artifacts/` se utiliza únicamente en modo inferencia (`predict`).
2. **Consistencia de columnas (208 features):** En el volcado del modelo no vienen incrustados los nombres de las 208 columnas ni el mapeo exacto de categorías. Si exportas tu preprocesador (`artifacts/preprocessor.joblib`), nombres de columnas (`artifacts/feature_columns.json`) y categorías exactas (`artifacts/categories.json`), la app los cargará automáticamente. Si no están presentes, aplica un alineamiento determinista hacia las 208 columnas esperadas.
3. **Rangos numéricos:** `Año` en `[2015, 2030]` y `Clave_Ent` en `[1, 32]`.

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Coloca tu modelo entrenado en artifacts/xgb_crime_model.ubj (o .json / .joblib)
streamlit run app.py
