from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import ArtifactNotFoundError, PreprocessingError

try:
    import joblib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Falta 'joblib'. Verifica que 'requirements.txt' esté en la raíz del repositorio y reinicia la app."
    ) from exc

try:
    import xgboost as xgb
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Falta 'xgboost'. Verifica que 'requirements.txt' esté en la raíz del repositorio y reinicia la app."
    ) from exc


class CrimePredictor:
    """Envoltorio de inferencia para un modelo XGBoost ya entrenado (sin reentrenamiento)."""

    SUPPORTED_EXTENSIONS = {".ubj", ".json", ".model", ".joblib", ".pkl"}

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model: Any = self._load_model(model_path)

    def _load_model(self, path: Path) -> Any:
        if not path.exists() or not path.is_file():
            raise ArtifactNotFoundError(
                f"No se encontró el archivo del modelo entrenado en '{path}'. "
                "Verifica la ruta o coloca el modelo serializado en la carpeta 'artifacts/'."
            )

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise PreprocessingError(
                f"Formato de modelo no soportado ('{suffix}'). "
                f"Formatos admitidos: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}."
            )

        try:
            if suffix in {".ubj", ".json", ".model"}:
                regressor = xgb.XGBRegressor()
                regressor.load_model(str(path))
                return regressor
            return joblib.load(path)
        except Exception as exc:
            raise PreprocessingError(
                f"Error al cargar el modelo XGBoost desde '{path}': {exc}"
            ) from exc

    @property
    def expected_features(self) -> int:
        if hasattr(self.model, "n_features_in_") and self.model.n_features_in_ is not None:
            return int(self.model.n_features_in_)
        booster = self.model.get_booster() if hasattr(self.model, "get_booster") else None
        if booster is not None and booster.num_features() > 0:
            return int(booster.num_features())
        return 208

    def predict(self, features_df: pd.DataFrame) -> float:
        if features_df.empty:
            raise PreprocessingError("El conjunto de características para predecir está vacío.")

        expected = self.expected_features
        if features_df.shape[1] != expected:
            raise PreprocessingError(
                f"Número de columnas incompatible: se recibieron {features_df.shape[1]}, "
                f"pero el modelo espera {expected}."
            )

        try:
            raw_pred = self.model.predict(features_df)
        except Exception as exc:
            raise PreprocessingError(
                f"Error durante la ejecución de predicción con XGBoost: {exc}"
            ) from exc

        pred_value = float(np.asarray(raw_pred).reshape(-1)[0])
        if np.isnan(pred_value) or np.isinf(pred_value):
            raise PreprocessingError("El modelo devolvió un valor no numérico o infinito.")

        return max(0.0, pred_value)
            raise PreprocessingError("El modelo devolvió un valor no numérico o infinito.")

        return max(0.0, pred_value)
