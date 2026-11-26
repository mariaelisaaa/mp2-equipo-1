# SPECS · Tasación de viviendas para Banc Avellaner

> **Qué construimos.** Lo escribe el equipo; el agente lo lee pero no lo edita. Las indicaciones entre `<!-- -->` son de la plantilla: borradlas cuando escribáis cada sección. Lo marcado como **ejemplo resuelto** ya está implementado y testeado: no lo borréis, ampliadlo.

## 1. Objetivo

## 2. Usuarios y preguntas

## 3. Datos de entrada y problemas

Lo encontramos con `notebooks/exploracion.ipynb`. Entrenamiento: 6.237 filas. Test: 4.002.

| Id | Columna | Qué pasa (con cifras) | Por qué importa | Fase |
| --- | --- | --- | --- | --- |
| P1 | `estado_operacion` | 620 filas (10 %) son anuncios retirados, que no se vendieron. El test solo tiene ventas | Su precio es lo que se pedía, no lo que se pagó | 1 · Limpieza |
| P2 | `planta` | 231 1a y 139 de 1a planta deben ser el mismo dato | Es importante porque es el mismo nivel de planta y se deberian juntar esos valores. | Limpieza |
| P3 | `barri` | 4 viviendas estan en zonas sin operaciones en el entrenamiento. | Importa porque no podemos clasificar la vivienda segun zona. Podriamos estimarla por cercania a una conocida segun longitud y latitud | Limpieza |
| P4 | `ascensor` | Ascensor tiene 13 formas de definir Si (1) o No (0). \['1' 'SI' 'Si' 'Sí' 'True' 'si', 's'\] deberian ser 1. \['0' 'False' 'No' 'no', 'n'\] deberian ser 0 | Importa porque la pregunta si tiene o no ascensor tiene una respuesta binaria que permite filtrar. | Limpieza |
| P5 | `precio` | 22 viviendas estan por debajo de 1.000 euros que deberian ser multiplicadas por mil. | Importa porque muestra un valor de precio erroneo por un defasaje en la unidad. | Limpieza |
| P6 | `parking` | Parking tiene "Sí" o "Inlcuido" que deberia ser "Si". Y tiene "No" "" que es "No". Y una tercera opcion se mantiene que es opcional. | Importa para poder agrupar los datos de forma correcta. | Limpieza |

## 4. Estrategias y experimentos

**Regla de decisión** (escrita antes de ejecutar nada):

Cómo leemos los resultados: `src.experimentos` da el MAE de cada partición (`mae_p1` … `mae_p5`). Comparamos cada estrategia con su referencia partición a partición, porque hay particiones más difíciles que otras para todas las estrategias a la vez.

### Punto de partida (ejemplo resuelto)

| Id | Tipo | Qué cambia | Referencia | Hipótesis | MAE medio ± sd | Mejora media · particiones | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | baseline | €/m² mediano de la zona × superficie construida | — | — |  | — | La tasación que hay que batir |
| E1 | minimo | Regresión lineal con `superficie_construida`, `habitaciones`, `banos`, `distancia_centro_km` y `poblacion_zona`, sin transformar | — | *Vuestra hipótesis: ¿mejor o peor que E0? ¿Por qué?* |  | — |  |
|  |  |  |  |  |  |  |  |

### Fase 1 · Limpieza

| Id | Tipo | Qué cambia | Referencia | Hipótesis | MAE medio ± sd | Mejora media · particiones | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | limpieza | Modelo mínimo con la superficie corregida | E1 | Mejora: las superficies de 1.500 m² inflan la tasación |  |  |  |
| L2 | limpieza | plantas repetidas, unifica los valoresa numericos | L1 | permite definir y unificar mismas plantas |  |  |  |
| L3 | limpieza | ascensor | L2 | Permite definir Si tiene ascensor o No |  |  |  |

### Fase 2 · Transformaciones

| Id | Tipo | Qué cambia | Referencia | Hipótesis | MAE medio ± sd | Mejora media · particiones | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | transformacion | Añade la zona como categoría | L3 | Mejora mucho: en Pedralbes el m² vale el triple que en Nou Barris |  |  |  |
| T2 | transformacion | unifica plantas | T1 | bajos, principal y primeros, bajos. segundos tercero y cuarto, medio. de quinto para arriba, alto. |  |  |  |
| T3 | transformacion | ascensor | T2 | \['1' 'SI' 'Si' 'Sí' 'True' 'si', 's'\] deberian ser 1. \['0' 'False' 'No' 'no', 'n'\] deberian ser 0 |  |  |  |

### Fase 3 · Variables nuevas

| Id | Tipo | Qué cambia | Referencia | Hipótesis | MAE medio ± sd | Mejora media · particiones | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N1 | caracteristica | Añade la relación baños / habitaciones | Última aceptada | *Vuestra hipótesis* |  |  |  |
|  |  |  |  |  |  |  |  |

### Fase 4 · Modelo

| Id | Tipo | Qué cambia | Referencia | Hipótesis | MAE medio ± sd | Mejora media · particiones | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | modelo | Ridge con λ = 0,01 | Última aceptada | Igual que la regresión lineal: λ muy pequeño |  |  |  |
|  |  |  |  |  |  |  |  |

## 5. Salidas

| Fichero | Qué contiene | Lo escribe |
| --- | --- | --- |
| `reports/experimentos.csv` | Una fila por estrategia con su MAE por partición | `src.experimentos` |
| `models/modelo.joblib` | El modelo final entrenado con las particiones 1-5 | `src.entrenar` |
| `reports/metricas.json` | MAE y RMSE del baseline y del modelo en `validacion`, con IC 95 % | `src.entrenar` |
| `reports/predicciones_validacion.csv` | Precio real, predicción del modelo y del baseline de cada vivienda de `validacion` | `src.entrenar` |
| `entregables/envio_kaggle.csv` | Predicción de cada vivienda del test | `src.predecir` |

## 6. Requisitos

**Ejemplo resuelto: datos y validación**

- R1. `cargar_entrenamiento()` devuelve las 6.237 filas de `train_set.csv` con la columna `particion` de `particiones.csv`; `cargar_test()` devuelve las 4.002 del test. No se limpia nada al cargar. `limpiar_objetivo()` todavía no quita ninguna fila.
- R2. Baseline: la tasación de una vivienda es el €/m² mediano de las operaciones de entrenamiento de su zona (el barrio en Barcelona; el municipio fuera de Barcelona o si falta el barrio) × su `superficie_construida`. Si la zona no tiene operaciones, el €/m² mediano global; si la vivienda no tiene superficie (vacía o 0), el precio mediano de su zona.
- R3. Validación cruzada: cada estrategia se entrena cinco veces, cada vez con cuatro de las particiones 1-5, y se mide el MAE en la que queda fuera. Las filas de `validacion` no se usan.
- R4. El error en `validacion` se da con un intervalo de confianza del 95 % por bootstrap percentil de las filas (2.000 remuestras, semilla 2026).
- R5. Modelo mínimo: regresión lineal con `superficie_construida`, `habitaciones`, `banos`, `distancia_centro_km` y `poblacion_zona`, sin transformar. Es la referencia de las estrategias.
- R6. El modelo final se entrena con las particiones 1-5 y se mide una vez en `validacion`, junto al baseline entrenado con las mismas filas.
- R7. El envío a Kaggle son las predicciones del modelo guardado para las 4.002 viviendas del test.
- R8. limpiar_objetivo() elimina únicamente las filas cuya estado_operacion sea retirada. Conserva las filas vendida y todas las columnas, incluida particion. Esta decisión depende exclusivamente de la fila.
- R9. superficie_construida se corrige dentro del Pipeline de L1: si vale 0 o supera 500 m² y existe una superficie_util positiva, se sustituye por superficie_util; en los demás casos conserva su valor original.
- R10. Una estrategia de limpieza se acepta si mejora la referencia en al menos 4 de las 5 particiones y baja el MAE medio; si no cumple ambos requisitos, se descarta y la última aceptada sigue siendo E1. **Cómo decidimos**

**Fase 1 · Limpieza**

**Fase 2 · Transformaciones**

**Fase 3 · Variables nuevas**

**Fase 4 · Modelo y explicación**

**Cuadro de mando**

## 7. Criterios de aceptación

**Ejemplo resuelto** (tests en `tests/acceptance/test_ejemplo_resuelto.py`)

- CA1 (R1). Hay 6.237 filas de entrenamiento; cada una está en `1`-`5` o en `validacion`, que tiene entre el 18 % y el 22 % de las filas, y ningún inmueble (`ref_catastral`) aparece en dos particiones.
- CA2 (R2). Con tres ventas de ejemplo (barrio A: 100 m² por 300.000 € y 50 m² por 200.000 €; Badalona: 80 m² por 160.000 €), el baseline tasa un piso de 60 m² en A en 210.000 €, uno de Badalona sin superficie en 160.000 €, uno de 100 m² en un barrio sin ventas en 300.000 € (€/m² global) y uno de A sin superficie en 250.000 €.
- CA3 (R3). La validación cruzada devuelve cinco errores, y cambiar el precio de las filas de `validacion` no los cambia.
- CA4 (R4). Con precios reales 100, 200, 300 y 400 y predicciones 110, 190, 330 y 400, el MAE es 12,5, su intervalo lo contiene y repetir el cálculo da el mismo intervalo.

**Vuestros criterios**

## 8. Fuera de alcance

- Modelos no lineales (árboles, *ensembles*, redes, k vecinos): llegan en las sesiones 14 y 15.
- Usar el test o sus precios públicos para entrenar o para elegir el modelo.

## 9. Preguntas abiertas

## 10. Decisiones