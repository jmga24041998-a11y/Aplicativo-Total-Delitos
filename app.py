from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.predictor import CrimePredictor
from src.preprocessing import (
    NUMERIC_RANGES,
    ArtifactNotFoundError,
    CrimePreprocessor,
    InputPayload,
    PreprocessingError,
    load_categories,
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Candidatos de archivos serializados del modelo entrenado (sin reentrenar)
MODEL_CANDIDATES = [
    ARTIFACTS_DIR / "xgb_crime_model.ubj",
    ARTIFACTS_DIR / "xgb_crime_model.json",
    ARTIFACTS_DIR / "xgb_crime_model.joblib",
    ARTIFACTS_DIR / "model.ubj",
    ARTIFACTS_DIR / "model.json",
    ARTIFACTS_DIR / "model.joblib",
]
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
CATEGORIES_PATH = ARTIFACTS_DIR / "categories.json"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_columns.json"


def resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return MODEL_CANDIDATES[0]


@st.cache_resource(show_spinner="Cargando modelo XGBoost entrenado...")
def get_predictor_and_preprocessor(model_file: str) -> tuple[CrimePredictor, CrimePreprocessor]:
    predictor = CrimePredictor(Path(model_file))
    preprocessor = CrimePreprocessor(
        preprocessor_path=PREPROCESSOR_PATH if PREPROCESSOR_PATH.exists() else None,
        feature_names_path=FEATURE_NAMES_PATH if FEATURE_NAMES_PATH.exists() else None,
        expected_feature_count=predictor.expected_features,
    )
    return predictor, preprocessor


def main() -> None:
    st.set_page_config(
        page_title="Predicción de Delitos en México (XGBoost)",
        page_icon="📊",
        layout="centered",
    )

    st.title("Predicción del Total de Delitos en México")
    st.caption(
        "Aplicación de inferencia con modelo XGBoost preentrenado (500 árboles, 208 características)."
    )

    model_path = resolve_model_path()

    try:
        categories = load_categories(CATEGORIES_PATH)
    except PreprocessingError as exc:
        st.error(f"Error de configuración de categorías: {exc}")
        st.stop()

    try:
        predictor, preprocessor = get_predictor_and_preprocessor(str(model_path))
        model_ready = True
    except ArtifactNotFoundError as exc:
        model_ready = False
        st.error(
            f"**Modelo no encontrado:** {exc}\n\n"
            "Sube tu archivo serializado (`xgb_crime_model.ubj`, `xgb_crime_model.json` o `xgb_crime_model.joblib`) "
            "a la carpeta `artifacts/` del repositorio."
        )
    except PreprocessingError as exc:
        model_ready = False
        st.error(f"**Error al inicializar el modelo o preprocesador:** {exc}")

    min_year, max_year = NUMERIC_RANGES["Año"]
    min_ent, max_ent = NUMERIC_RANGES["Clave_Ent"]

    with st.form("crime_prediction_form"):
        st.subheader("Variables numéricas")
        col_num1, col_num2 = st.columns(2)
        with col_num1:
            anio = st.number_input(
                "Año",
                min_value=min_year,
                max_value=max_year,
                value=2024,
                step=1,
                help=f"Rango válido: {min_year} a {max_year}.",
            )
        with col_num2:
            clave_ent = st.number_input(
                "Clave_Ent",
                min_value=min_ent,
                max_value=max_ent,
                value=1,
                step=1,
                help=f"Clave de entidad federativa ({min_ent} a {max_ent}).",
            )

        st.subheader("Variables categóricas")
        col_cat1, col_cat2 = st.columns(2)

        with col_cat1:
            entidad = st.selectbox("Entidad", options=categories["Entidad"])
            bien_juridico = st.selectbox(
                "Bien jurídico afectado", options=categories["Bien jurídico afectado"]
            )
            tipo_de_delito = st.selectbox("Tipo de delito", options=categories["Tipo de delito"])
            subtipo_de_delito = st.selectbox(
                "Subtipo de delito", options=categories["Subtipo de delito"]
            )

        with col_cat2:
            modalidad = st.selectbox("Modalidad", options=categories["Modalidad"])
            mes = st.selectbox("Mes", options=categories["Mes"])
            sexo = st.selectbox("Sexo", options=categories["Sexo"])
            rango_edad = st.selectbox("Rango de edad", options=categories["Rango de edad"])

        submitted = st.form_submit_button("Predecir total de delitos", type="primary")

    if submitted:
        if not model_ready:
            st.warning(
                "No es posible ejecutar la predicción porque el archivo del modelo no está disponible o falló su carga."
            )
            return

        payload = InputPayload(
            anio=int(anio),
            clave_ent=int(clave_ent),
            entidad=entidad,
            bien_juridico=bien_juridico,
            tipo_de_delito=tipo_de_delito,
            subtipo_de_delito=subtipo_de_delito,
            modalidad=modalidad,
            mes=mes,
            sexo=sexo,
            rango_edad=rango_edad,
        )

        try:
            features_df = preprocessor.transform(payload, categories)
            estimated_crimes = predictor.predict(features_df)

            st.success("Predicción generada correctamente.")
            st.metric(
                label="Total de delitos estimado",
                value=f"{estimated_crimes:,.2f}",
            )
            with st.expander("Ver resumen de entrada y vector preprocesado"):
                st.json(payload.to_raw_dict())
                st.caption(f"Dimensiones enviadas a XGBoost: {features_df.shape[0]} fila × {features_df.shape[1]} columnas.")
        except PreprocessingError as exc:
            st.error(f"Error de validación o preprocesamiento: {exc}")
        except Exception as exc:
            st.error(f"Ocurrió un error inesperado durante la predicción: {exc}")


if __name__ == "__main__":
    main()
