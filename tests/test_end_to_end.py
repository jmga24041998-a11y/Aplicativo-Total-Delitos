from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from src.predictor import CrimePredictor
from src.preprocessing import (
    ArtifactNotFoundError,
    CrimePreprocessor,
    InputPayload,
    PreprocessingError,
    load_categories,
)


def _build_dummy_xgb_model(model_path: Path, n_features: int = 208) -> None:
    """Crea un modelo XGBRegressor válido con n_features sin reentrenar en app.py."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(25, n_features)).astype(np.float32)
    y = (np.abs(X[:, 0]) * 120.0 + 15.0).astype(np.float32)
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    model.save_model(str(model_path))


def test_end_to_end_prediction(tmp_path: Path) -> None:
    model_file = tmp_path / "xgb_crime_model.ubj"
    _build_dummy_xgb_model(model_file, n_features=208)

    categories = load_categories(None)
    predictor = CrimePredictor(model_file)
    preprocessor = CrimePreprocessor(expected_feature_count=predictor.expected_features)

    payload = InputPayload(
        anio=2023,
        clave_ent=14,
        entidad="Jalisco",
        bien_juridico="El patrimonio",
        tipo_de_delito="Robo",
        subtipo_de_delito="Robo a transeúnte en vía pública",
        modalidad="Con violencia",
        mes="Octubre",
        sexo="Mujer",
        rango_edad="Adulto (18 y más)",
    )

    X_ready = preprocessor.transform(payload, categories)
    assert X_ready.shape == (1, 208)

    prediction = predictor.predict(X_ready)
    assert isinstance(prediction, float)
    assert prediction >= 0.0


def test_missing_model_raises_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        CrimePredictor(tmp_path / "no_existe.ubj")


def test_out_of_range_and_unknown_category_raise_error(tmp_path: Path) -> None:
    categories = load_categories(None)
    preprocessor = CrimePreprocessor(expected_feature_count=208)

    bad_year = InputPayload(
        anio=1900,
        clave_ent=1,
        entidad="Aguascalientes",
        bien_juridico="El patrimonio",
        tipo_de_delito="Robo",
        subtipo_de_delito="Robo a casa habitación",
        modalidad="Sin violencia",
        mes="Enero",
        sexo="Hombre",
        rango_edad="Adulto (18 y más)",
    )
    with pytest.raises(PreprocessingError, match="fuera de rango"):
        preprocessor.transform(bad_year, categories)

    bad_cat = InputPayload(
        anio=2023,
        clave_ent=1,
        entidad="Entidad Inexistente",
        bien_juridico="El patrimonio",
        tipo_de_delito="Robo",
        subtipo_de_delito="Robo a casa habitación",
        modalidad="Sin violencia",
        mes="Enero",
        sexo="Hombre",
        rango_edad="Adulto (18 y más)",
    )
    with pytest.raises(PreprocessingError, match="Categoría desconocida"):
        preprocessor.transform(bad_cat, categories)
