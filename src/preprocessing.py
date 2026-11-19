from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "No se encontró 'joblib'. Verifica que esté en 'requirements.txt' en la raíz y reinicia la app."
    ) from exc


class PreprocessingError(Exception):
    """Error controlado en validación o transformación de variables de entrada."""


class ArtifactNotFoundError(FileNotFoundError):
    """Error cuando falta un archivo requerido del modelo o preprocesador."""


DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "Entidad": [
        "Aguascalientes", "Baja California", "Baja California Sur", "Campeche",
        "Chiapas", "Chihuahua", "Ciudad de México", "Coahuila de Zaragoza",
        "Colima", "Durango", "Guanajuato", "Guerrero", "Hidalgo", "Jalisco",
        "México", "Michoacán de Ocampo", "Morelos", "Nayarit", "Nuevo León",
        "Oaxaca", "Puebla", "Querétaro", "Quintana Roo", "San Luis Potosí",
        "Sinaloa", "Sonora", "Tabasco", "Tamaulipas", "Tlaxcala",
        "Veracruz de Ignacio de la Llave", "Yucatán", "Zacatecas",
    ],
    "Bien jurídico afectado": [
        "La vida y la integridad corporal",
        "El patrimonio",
        "La libertad personal",
        "La familia",
        "La sociedad",
        "Otros bienes jurídicos afectados (del fuero común)",
    ],
    "Tipo de delito": [
        "Homicidio",
        "Robo",
        "Lesiones",
        "Secuestro",
        "Violencia familiar",
        "Narcomenudeo",
        "Fraude",
        "Extorsión",
    ],
    "Subtipo de delito": [
        "Homicidio doloso",
        "Homicidio culposo",
        "Robo a transeúnte en vía pública",
        "Robo de vehículo automotor",
        "Robo a casa habitación",
        "Lesiones dolosas",
        "Violencia familiar",
        "Otro subtipo",
    ],
    "Modalidad": [
        "Con arma de fuego",
        "Con arma blanca",
        "Con violencia",
        "Sin violencia",
        "No especificado",
    ],
    "Mes": [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    ],
    "Sexo": [
        "Mujer",
        "Hombre",
        "No identificado",
    ],
    "Rango de edad": [
        "Menor de edad (0-17)",
        "Adulto (18 y más)",
        "No especificado",
    ],
}

NUMERIC_RANGES: dict[str, tuple[int, int]] = {
    "Año": (2015, 2030),
    "Clave_Ent": (1, 32),
}


@dataclass
class InputPayload:
    anio: int
    clave_ent: int
    entidad: str
    bien_juridico: str
    tipo_de_delito: str
    subtipo_de_delito: str
    modalidad: str
    mes: str
    sexo: str
    rango_edad: str

    def to_raw_dict(self) -> dict[str, Any]:
        return {
            "Año": self.anio,
            "Clave_Ent": self.clave_ent,
            "Entidad": self.entidad,
            "Bien jurídico afectado": self.bien_juridico,
            "Tipo de delito": self.tipo_de_delito,
            "Subtipo de delito": self.subtipo_de_delito,
            "Modalidad": self.modalidad,
            "Mes": self.mes,
            "Sexo": self.sexo,
            "Rango de edad": self.rango_edad,
        }


def load_categories(categories_path: Path | None = None) -> dict[str, list[str]]:
    """Carga categorías exactas desde JSON si existe; de lo contrario usa el catálogo base."""
    if categories_path and categories_path.exists():
        try:
            data = json.loads(categories_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return {str(k): [str(v) for v in vals] for k, vals in data.items()}
        except Exception as exc:
            raise PreprocessingError(
                f"No se pudo leer el archivo de categorías '{categories_path}': {exc}"
            ) from exc
    return {k: list(v) for k, v in DEFAULT_CATEGORIES.items()}


def validate_inputs(payload: InputPayload, categories: dict[str, list[str]]) -> None:
    """Valida rangos numéricos y pertenencia a categorías conocidas."""
    min_year, max_year = NUMERIC_RANGES["Año"]
    if not isinstance(payload.anio, (int, np.integer)) or isinstance(payload.anio, bool):
        raise PreprocessingError("El campo 'Año' debe ser un número entero.")
    if not (min_year <= payload.anio <= max_year):
        raise PreprocessingError(
            f"Valor fuera de rango en 'Año' ({payload.anio}). Debe estar entre {min_year} y {max_year}."
        )

    min_ent, max_ent = NUMERIC_RANGES["Clave_Ent"]
    if not isinstance(payload.clave_ent, (int, np.integer)) or isinstance(payload.clave_ent, bool):
        raise PreprocessingError("El campo 'Clave_Ent' debe ser un número entero.")
    if not (min_ent <= payload.clave_ent <= max_ent):
        raise PreprocessingError(
            f"Valor fuera de rango en 'Clave_Ent' ({payload.clave_ent}). Debe estar entre {min_ent} y {max_ent}."
        )

    raw = payload.to_raw_dict()
    for field_name, allowed_values in categories.items():
        value = raw.get(field_name)
        if value not in allowed_values:
            raise PreprocessingError(
                f"Categoría desconocida en '{field_name}': '{value}'. "
                f"Seleccione una opción válida del catálogo entrenado."
            )


class CrimePreprocessor:
    """Transforma entradas validadas a una matriz alineada con las características del modelo."""

    def __init__(
        self,
        preprocessor_path: Path | None = None,
        feature_names_path: Path | None = None,
        expected_feature_count: int = 208,
    ) -> None:
        self.expected_feature_count = expected_feature_count
        self.preprocessor: Any | None = None
        self.feature_names: list[str] | None = None

        if preprocessor_path and preprocessor_path.exists():
            try:
                self.preprocessor = joblib.load(preprocessor_path)
            except Exception as exc:
                raise PreprocessingError(
                    f"Error al cargar el preprocesador '{preprocessor_path}': {exc}"
                ) from exc

        if feature_names_path and feature_names_path.exists():
            try:
                loaded = json.loads(feature_names_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    self.feature_names = [str(col) for col in loaded]
            except Exception as exc:
                raise PreprocessingError(
                    f"Error al cargar los nombres de columnas '{feature_names_path}': {exc}"
                ) from exc

    def transform(self, payload: InputPayload, categories: dict[str, list[str]]) -> pd.DataFrame:
        validate_inputs(payload, categories)
        raw_df = pd.DataFrame([payload.to_raw_dict()])

        if self.preprocessor is not None:
            try:
                transformed = self.preprocessor.transform(raw_df)
            except Exception as exc:
                raise PreprocessingError(
                    f"Falló la transformación con el preprocesador guardado: {exc}"
                ) from exc

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            arr = np.asarray(transformed, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            if arr.shape[1] != self.expected_feature_count:
                raise PreprocessingError(
                    f"Inconsistencia de columnas tras preprocesar: se obtuvieron {arr.shape[1]} "
                    f"columnas, pero el modelo XGBoost requiere {self.expected_feature_count}."
                )

            cols = (
                self.feature_names
                if self.feature_names and len(self.feature_names) == arr.shape[1]
                else [f"f_{i}" for i in range(arr.shape[1])]
            )
            return pd.DataFrame(arr, columns=cols)

        return self._fallback_align_features(raw_df, categories)

    def _fallback_align_features(
        self, raw_df: pd.DataFrame, categories: dict[str, list[str]]
    ) -> pd.DataFrame:
        encoded = pd.get_dummies(raw_df, columns=list(categories.keys()), dtype=np.float32)
        encoded["Año"] = encoded["Año"].astype(np.float32)
        encoded["Clave_Ent"] = encoded["Clave_Ent"].astype(np.float32)

        if self.feature_names and len(self.feature_names) == self.expected_feature_count:
            aligned = encoded.reindex(columns=self.feature_names, fill_value=0.0)
            return aligned.astype(np.float32)

        values = encoded.to_numpy(dtype=np.float32).flatten()
        if values.size < self.expected_feature_count:
            padded = np.zeros(self.expected_feature_count, dtype=np.float32)
            padded[: values.size] = values
            values = padded
        elif values.size > self.expected_feature_count:
            values = values[: self.expected_feature_count]

        cols = [f"feature_{i}" for i in range(self.expected_feature_count)]
        return pd.DataFrame([values], columns=cols)
