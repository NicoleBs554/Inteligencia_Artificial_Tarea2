# Tarea 2: Regresión Ridge y Lasso

Proyecto de regresión sobre la resistencia a la compresión del concreto. El cuaderno realiza el análisis exploratorio, construye un caso de sobreajuste con regresión lineal, implementa Ridge desde su solución cerrada y compara Ridge con Lasso.

## Dataset

Se utiliza el dataset **Concrete Compressive Strength**, disponible en el [repositorio UCI (dataset 165)](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength). La copia utilizada está en [`src/tarea2/concrete_data.csv`](src/tarea2/concrete_data.csv).

- **Observaciones:** 1.030 filas originales; se eliminan 25 filas duplicadas y quedan 1.005.
- **Predictores:** ocho variables cuantitativas sobre la mezcla y su edad.
- **Objetivo continuo:** resistencia a la compresión del concreto, en MPa.
- **Exploración y limpieza:** se revisan tipos, valores faltantes, duplicados, estadísticas descriptivas, correlaciones y distribución del objetivo. No se encontraron valores faltantes; las filas duplicadas se eliminaron.

El cuaderno identifica el dataset como procedente de Kaggle/UCI, pero el repositorio no registra una URL específica de Kaggle ni deja constancia de qué dataset se usó en la Tarea 1. Por ello, esa comparación debe confirmarse antes de entregar.

## Qué incluye

El trabajo está desarrollado en [`src/tarea2/Tarea2_IA.ipynb`](src/tarea2/Tarea2_IA.ipynb) y la implementación propia de Ridge está en [`src/tarea2/my_ridge.py`](src/tarea2/my_ridge.py).

1. **Sobreajuste de regresión lineal:** expande los ocho predictores a 44 características con `PolynomialFeatures` de grado 2 y entrena `LinearRegression` con solo 80 ejemplos; los 925 restantes forman el conjunto de prueba.
2. **Implementación de `MyRidge`:** usa únicamente NumPy. En `fit` centra las características y el objetivo con las medias del entrenamiento, resuelve el sistema con `np.linalg.solve` y recupera el intercepto sin penalizarlo. Incluye `fit`, `predict`, `score` (R²) y `rmse`, y los atributos `coef_`, `intercept_`, `X_mean_` y `y_mean_`.
3. **Selección de regularización:** estandariza las características, selecciona `alpha` mediante validación cruzada de cinco particiones sobre una grilla logarítmica y grafica los errores de entrenamiento y validación frente a `alpha`.
4. **Validación contra scikit-learn:** compara coeficientes, intercepto, R² y RMSE de `MyRidge` y `Ridge` tanto con características estandarizadas como sin estandarizar, mediante `np.allclose`.
5. **Ridge frente a Lasso:** selecciona un `alpha` distinto para cada modelo mediante validación cruzada, presenta los coeficientes, grafica sus caminos de regularización, identifica los coeficientes nulos de Lasso y explica la diferencia geométrica entre las penalizaciones L2 y L1.

## Resultados registrados

Los siguientes valores corresponden a las salidas guardadas en el cuaderno:

| Modelo/medida | Resultado |
|---|---:|
| `LinearRegression`, RMSE de entrenamiento | 4.721 MPa |
| `LinearRegression`, RMSE de prueba | 19.122 MPa |
| `MyRidge`, `alpha` elegido | 1.2068 |
| `MyRidge`, RMSE de prueba | 10.898 MPa |
| `Ridge` de scikit-learn, `alpha` elegido | 1.2068 |
| `Lasso` de scikit-learn, `alpha` elegido | 0.1326 |
| `Lasso`, coeficientes exactamente iguales a cero | 25 de 44 |
| `Ridge` de scikit-learn, coeficientes exactamente iguales a cero | 0 de 44 |
| `Ridge` de scikit-learn, RMSE de prueba | 10.898 MPa |
| `Lasso` de scikit-learn, RMSE de prueba | 10.836 MPa |

El cuaderno imprime los nombres de las características que Lasso elimina y conserva. La regularización L1 de Lasso puede llevar coeficientes exactamente a cero; la penalización L2 de Ridge normalmente los reduce sin anularlos.

## Revisión de requisitos y aspectos por confirmar

- **Cumplidos según los archivos y resultados revisados:** tamaño y tipo del dataset; ocho predictores cuantitativos; limpieza y exploración; demostración de sobreajuste; API y fórmula requeridas para `MyRidge`; selección de `alpha`; gráficas; comparación con `Ridge` de scikit-learn; y comparación Ridge/Lasso.
- **No verificable desde este repositorio:** que el dataset sea distinto al de la Tarea 1. Confirmarlo con el material de esa tarea.
- **Mejora metodológica recomendada:** el `StandardScaler` se ajusta solo con el conjunto de entrenamiento general, pero antes de dividirlo en folds. Para evitar que cada fold de validación influya en el escalado, conviene ajustar el scaler dentro de cada fold.
- **Ajuste de presentación pendiente:** el RMSE de prueba de `MyRidge` aparece tanto en la sección del punto 3 como en la tabla resumen final. La consigna pide reportarlo una sola vez al final.
- **Advertencia de ejecución:** las salidas guardadas muestran `ConvergenceWarning` de Lasso para algunos valores de `alpha`, aunque se configuró `max_iter=100000`. Conviene volver a ejecutar el cuaderno y comprobar la convergencia de la búsqueda y de los caminos de coeficientes.

## Requisitos

El proyecto declara Python 3.14 o superior y sus dependencias en [`pyproject.toml`](pyproject.toml). Con `uv` instalado, en PowerShell:

```powershell
uv sync
Set-Location src\tarea2
uv run --project ..\.. jupyter lab Tarea2_IA.ipynb
```

Ejecuta las celdas del cuaderno en orden. La ruta `concrete_data.csv` se resuelve desde `src\tarea2`, por lo que se recomienda iniciarlo desde ese directorio.

## Estructura

```text
src/
└── tarea2/
    ├── Tarea2_IA.ipynb
    ├── concrete_data.csv
    └── my_ridge.py
```
