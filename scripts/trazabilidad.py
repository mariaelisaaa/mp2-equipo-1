"""Comprueba la trazabilidad entre SPECS.md, el código y los tests.

    uv run python scripts/trazabilidad.py

1. Criterios: cada "CA<número>" de la sección "Criterios de aceptación" de SPECS.md tiene al menos
   un test con @pytest.mark.criterio("CA<número>").
2. Estrategias: cada fila de SPECS.md §4 (E0, L1, T2…) está en ESTRATEGIAS de src/estrategias.py,
   y cada estrategia del código está en SPECS.md §4. El nombre en el código es el Id en minúsculas
   seguido de una palabra (`l1_superficie_corregida`); E0 es `baseline` y E1 es `modelo_minimo`.
   También avisa si reports/experimentos.csv no tiene todas las estrategias del código.

Sale con código 1 si falta algún test, si un test cita un criterio que no existe o si el código tiene
una estrategia que no está en SPECS.md. Una fila de SPECS.md todavía sin implementar es solo un aviso:
primero se escribe la hipótesis y después se implementa.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
PROFESORADO = {f"CM{i}" for i in range(1, 8)}
NOMBRES_FIJOS = {"E0": "baseline", "E1": "modelo_minimo"}  # el punto de partida de la plantilla


def seccion_de_specs(titulo: str) -> str:
    """Texto de una sección '## N. <titulo>' de SPECS.md, sin las indicaciones de la plantilla."""
    texto = (RAIZ / "SPECS.md").read_text(encoding="utf-8")
    texto = re.sub(r"<!--.*?-->", "", texto, flags=re.S)        # ignorar las indicaciones de la plantilla
    m = re.search(rf"^##\s*\d*\.?\s*{titulo}\s*$(.*?)(?=^##\s|\Z)", texto, flags=re.M | re.S | re.I)
    return m.group(1) if m else ""


def criterios_en_specs() -> set[str]:
    return set(re.findall(r"\bCA\d+\b", seccion_de_specs("Criterios de aceptación")))


def estrategias_en_specs() -> list[str]:
    """Ids de las filas de las tablas de SPECS.md §4 (la primera celda: E0, L1, T2, N1, M3…)."""
    ids = re.findall(r"^\|\s*([A-Z]{1,2}\d+)\s*\|", seccion_de_specs("Estrategias y experimentos"), flags=re.M)
    return list(dict.fromkeys(ids))


def corresponde(id_specs: str, nombre: str) -> bool:
    """El nombre del código corresponde al Id de SPECS.md: E0 → baseline, L1 → l1_…"""
    if id_specs in NOMBRES_FIJOS:
        return nombre == NOMBRES_FIJOS[id_specs]
    return nombre == id_specs.lower() or nombre.startswith(id_specs.lower() + "_")


def criterios_en_tests() -> dict[str, list[str]]:
    usados: dict[str, list[str]] = {}
    for f in (RAIZ / "tests").rglob("test_*.py"):
        for cid in re.findall(r"criterio\(\s*[\"']((?:CA|CM)\d+)[\"']\s*\)", f.read_text(encoding="utf-8")):
            usados.setdefault(cid, []).append(str(f.relative_to(RAIZ)))
    return usados


def comprobar_estrategias() -> int:
    """Compara SPECS.md §4 con ESTRATEGIAS y con experimentos.csv. Devuelve el número de errores."""
    try:
        from src.estrategias import ESTRATEGIAS
    except Exception as error:  # noqa: BLE001 - se informa y se sigue
        print(f"✘ No se puede importar src/estrategias.py: {error}")
        return 1
    en_codigo = [e.nombre for e in ESTRATEGIAS]
    en_specs = estrategias_en_specs()
    errores = 0
    print("\nEstrategias (SPECS.md §4 → src/estrategias.py):")
    for id_specs in en_specs:
        nombres = [n for n in en_codigo if corresponde(id_specs, n)]
        if nombres:
            print(f"✔ {id_specs}: {', '.join(nombres)}")
        else:
            print(f"· {id_specs}: en SPECS.md, todavía sin implementar")
    for nombre in en_codigo:
        if not any(corresponde(i, nombre) for i in en_specs):
            print(f"✘ {nombre}: está en ESTRATEGIAS pero no en SPECS.md §4")
            errores += 1
    csv = RAIZ / "reports" / "experimentos.csv"
    if not csv.exists():
        print("· Todavía no hay reports/experimentos.csv: ejecutad `uv run python -m src.experimentos`")
    else:
        en_csv = set(re.findall(r"^([^,\n]+),", csv.read_text(encoding="utf-8"), flags=re.M)[1:])
        faltan = [n for n in en_codigo if n not in en_csv]
        if faltan:
            print(f"· experimentos.csv no tiene {', '.join(faltan)}: ejecutad `uv run python -m src.experimentos`")
    return errores


def main() -> int:
    errores_estrategias = comprobar_estrategias()
    print("\nCriterios de aceptación (SPECS.md §7 → tests):")
    definidos = criterios_en_specs()
    usados = criterios_en_tests()
    orden = lambda c: (c[:2], int(c[2:]))
    if not definidos:
        print("✘ SPECS.md no tiene criterios de aceptación (CA1, CA2…) en la sección 'Criterios de aceptación'.")
        return 1
    sin_test = sorted(definidos - set(usados), key=orden)
    desconocidos = sorted(set(usados) - definidos - PROFESORADO, key=orden)
    for c in sorted(definidos, key=orden):
        print(f"{'✔' if c in usados else '✘'} {c}: {', '.join(sorted(set(usados.get(c, [])))) or 'SIN TEST'}")
    for c in desconocidos:
        print(f"✘ {c}: lo cita {', '.join(sorted(set(usados[c])))} pero no existe en SPECS.md")
    if sin_test or desconocidos or errores_estrategias:
        print(f"\n{len(sin_test)} criterio(s) sin test, {len(desconocidos)} referencia(s) desconocida(s) "
              f"y {errores_estrategias} estrategia(s) del código que no están en SPECS.md.")
        return 1
    print(f"\nTodos los criterios ({len(definidos)}) tienen al menos un test y todas las estrategias "
          "del código están en SPECS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
