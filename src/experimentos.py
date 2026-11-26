"""Evalúa todas las estrategias con validación cruzada y escribe reports/experimentos.csv (R3).

    uv run python -m src.experimentos
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.datos import INFORMES, PARTICIONES_CV, cargar_entrenamiento, limpiar_objetivo
from src.estrategias import ESTRATEGIAS, TIPOS
from src.validacion import validacion_cruzada

COLUMNAS_MAE = [f"mae_p{p}" for p in PARTICIONES_CV]


def comprobar_estrategias() -> None:
    """Cada estrategia tiene un tipo válido y su referencia es una estrategia anterior de la lista."""
    anteriores: set[str] = set()
    for estrategia in ESTRATEGIAS:
        if estrategia.tipo not in TIPOS:
            raise ValueError(f"{estrategia.nombre}: tipo {estrategia.tipo!r} no es uno de {TIPOS}")
        if estrategia.referencia and estrategia.referencia not in anteriores:
            raise ValueError(f"{estrategia.nombre}: su referencia {estrategia.referencia!r} "
                             "no es una estrategia anterior de ESTRATEGIAS")
        anteriores.add(estrategia.nombre)


def comparar_con_referencia(resultados: pd.DataFrame) -> pd.DataFrame:
    """Añade la comparación de cada estrategia con su referencia, partición a partición.

    - `mejora_media`: MAE de la referencia − MAE de la estrategia, de media (positivo = mejora).
    - `particiones_mejor`: en cuántas de las 5 particiones el MAE es menor que el de la referencia.
    """
    resultados = resultados.copy()
    por_nombre = resultados.set_index("estrategia")[COLUMNAS_MAE]
    mejora, particiones = [], []
    for _, fila in resultados.iterrows():
        if fila["referencia"]:
            diferencia = (por_nombre.loc[fila["referencia"]].to_numpy(dtype=float)
                          - fila[COLUMNAS_MAE].to_numpy(dtype=float))
            mejora.append(float(diferencia.mean()))
            particiones.append(int((diferencia > 0).sum()))
        else:
            mejora.append(np.nan)
            particiones.append(pd.NA)
    resultados["mejora_media"] = mejora
    resultados["particiones_mejor"] = pd.array(particiones, dtype="Int64")
    return resultados


def euros(valor: float) -> str:
    """Un importe en euros, sin decimales y con punto de miles: 74.980."""
    return f"{valor:,.0f}".replace(",", ".")


def linea(fila: dict) -> str:
    """El MAE de una estrategia en cada partición, su media y su desviación, en una línea."""
    particiones = "  ".join(f"{euros(fila[c]):>7}" for c in COLUMNAS_MAE)
    return f"{fila['estrategia']:28s} {particiones}  │ {euros(fila['mae_medio']):>7} ± {euros(fila['mae_sd']):>6}"


def cabecera() -> str:
    """Los títulos de las columnas que escribe `linea`."""
    particiones = "  ".join(f"{'p' + p:>7}" for p in PARTICIONES_CV)
    return f"{'MAE (€) por partición':28s} {particiones}  │ {'media':>7} ± {'sd':>6}"


def evaluar_estrategias(datos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por estrategia: MAE de cada partición, media, desviación y comparación con su referencia (R3)."""
    comprobar_estrategias()
    print(cabecera())
    filas = []
    for estrategia in ESTRATEGIAS:
        errores = validacion_cruzada(estrategia.crear(), datos)
        fila = {"estrategia": estrategia.nombre, "tipo": estrategia.tipo,
                "descripcion": estrategia.descripcion, "referencia": estrategia.referencia}
        fila |= dict(zip(COLUMNAS_MAE, errores))
        fila |= {"mae_medio": float(np.mean(errores)), "mae_sd": float(np.std(errores, ddof=1))}
        filas.append(fila)
        print(linea(fila))
    return comparar_con_referencia(pd.DataFrame(filas))


def main() -> None:
    resultados = evaluar_estrategias(limpiar_objetivo(cargar_entrenamiento()))
    INFORMES.mkdir(exist_ok=True)
    resultados.to_csv(INFORMES / "experimentos.csv", index=False)
    print(resultados[["estrategia", "tipo", "referencia", "mejora_media", "particiones_mejor"]]
          .to_string(index=False))
    print(f"Escrito {INFORMES / 'experimentos.csv'}")


if __name__ == "__main__":
    main()
