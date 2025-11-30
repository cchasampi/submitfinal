Workflow de Modelado con LightGBM y Optimización Bayesiana

Este proyecto contiene un notebook en R ("cintia950_WorkFlow_01_submitfinal.ipynb") cuyo objetivo es desarrollar un modelo de clasificación binaria usando LightGBM, optimizado mediante búsqueda bayesiana de hiperparámetros.


 1. Objetivos del script

1. Cargar y preparar el dataset para un problema de clasificación binaria.
2. Definir una estrategia de particionado en training y validation.
3. Configurar los parámetros de LightGBM a través de una lista global `PARAM`.
4. Implementar una función de evaluación que:

   - entrena un modelo LightGBM con un conjunto de hiperparámetros,
   -evalúa su desempeño en el conjunto de validación usando Accuracy o AUC,
    -devuelve esa métrica como objetivo para la optimización.
5. Utilizar mlrMBO (u otra librería similar) para realizar la optimización bayesiana de los hiperparámetros.
6. Entrenar un modelo final con los mejores hiperparámetros encontrados.



 2. Requerimientos de software

El notebook está desarrollado en R y utiliza los siguientes paquetes principales:

 `lightgbm` – entrenamiento del modelo de gradient boosting.
 `data.table` – manipulación eficiente de datos.
 `mlrMBO` / `smoof` – optimización bayesiana de hiperparámetros.
 `ggplot2` (opcional) – para algunos gráficos exploratorios.
 `Matrix` / `magrittr` / otros (según se cargue en el notebook).

> Nota: LightGBM requiere instalación previa de la librería nativa. Dependiendo del entorno, puede ser necesario compilarla o usar una versión ya preparada.


 3. Descripción general del workflow

El notebook sigue, a grandes rasgos, las siguientes etapas:

1. Carga de datos

    Lectura del dataset original .
    Conversión a `data.table` para facilitar el manejo.
    Selección de la variable objetivo y de las variables predictoras.

2. Definición de parámetros globales (`PARAM`)

    `PARAM$semilla_primigenia` → semilla para reproducibilidad.
    `PARAM$trainingstrategy` → define qué períodos o subconjuntos se usan para training y cuáles para validation (variable `foto_mes`).
    `PARAM$lgbm$param_fijos` → lista de parámetros fijos de LightGBM:

      `boosting`, `objective`, `metric`, `max_depth`, `min_data_in_leaf`, etc.
    `PARAM$lgbm$nrounds` → cantidad de iteraciones (árboles) a entrenar.

3. Armado de los datasets de LightGBM

    Creación de `dtrain` con los registros de entrenamiento.
    Creación de `dvalidate` con los registros de validación.
    Conversión a `lgb.Dataset` usando las variables seleccionadas (`campos_buenos`).

4. Función de evaluación para la optimización: `EstimarGanancia_AUC_lightgbm` ( o Accuracy: EstimarGanancia_ACCU2_lightgbm)
   Esta función:

    Recibe un vector/lista de hiperparámetros `x` propuesto por la optimización.
    Combina `x` con los parámetros fijos de `PARAM$lgbm$param_fijos` usando `modifyList`.
    Ajusta internamente hiperparámetros derivados como:

      `min_data_in_leaf` (a partir de `leaf_size`),
      `num_leaves` (a partir de `coverage` y el tamaño del dataset de entrenamiento).
    Entrena un modelo LightGBM con `dtrain` (y `dvalidate` como validación).
    Predice las probabilidades sobre el conjunto de validación.
    Convierte probabilidades a clase usando un umbral (0.5).
    Calcula el Accuracy en validación:
     [
     Accuracy = \frac{\text{nº de predicciones correctas}}{\text{nº total de casos de validación}}
     ]
    Devuelve el Accuracy como valor objetivo para la búsqueda bayesiana.
    Además, guarda información extra (como `best_iter`) en los atributos del valor devuelto.

5. Optimización bayesiana de hiperparámetros

    Se define un espacio de búsqueda (por ejemplo):

      `learning_rate`
      `feature_fraction`
      `coverage`
      `leaf_size`
    Se arma una función objetivo `smoof` que envuelve a `EstimarGanancia_AUC_lightgbm`.
    Se ejecuta `mbo()` (mlrMBO) para:

      probar configuraciones iniciales,
      modelar la superficie de respuesta,
      ir proponiendo nuevas combinaciones de hiperparámetros en base al Accuracy obtenido,
      encontrar la combinación que maximiza el Accuracy en el conjunto de validación.

6. Entrenamiento del modelo final

    Una vez obtenidos los mejores hiperparámetros, el script:

      los combina con los parámetros fijos,
      entrena un modelo final con LightGBM (posiblemente usando más datos y/o más iteraciones),
      genera las predicciones finales (por ejemplo, para un conjunto de test o para envío a una competencia / evaluación externa).


 4. Funciones clave

 4.1. `EstimarGanancia_AUC_lightgbm` (versión Accuracy)

 Función central para la optimización.
 Entrena un modelo LightGBM con los hiperparámetros propuestos.
 Evalúa con Accuracy sobre la partición de validación.
 Es utilizada internamente por `mbo()` como función objetivo.

 4.2. Estructura `PARAM`

 Permite centralizar todas las configuraciones:

   semillas, folds, particiones temporales,
   parámetros fijos del modelo,
   cantidad de iteraciones (`nrounds`),
   etc.
 Facilita cambiar estrategias de entrenamiento/validación sin modificar muchas líneas de código.


 5. Cómo ejecutar el notebook

1. Abrir el archivo `cintia950_WorkFlow_01_submitfinal.ipynb` en el entorno que soporte R en notebooks (por ejemplo, RKernel en Jupyter o RStudio con soporte a `.ipynb`).
2. Verificar que todas las librerías necesarias estén instaladas:

   ```r
   install.packages(c("data.table", "ggplot2", "mlrMBO", "smoof"))
    LightGBM puede requerir instalación específica:
    remotes::install_github("microsoft/LightGBM", subdir = "R-package")
   ```
3. Ejecutar las celdas en orden:

    Carga de datos y definición de parámetros.
    Armado de `dtrain` y `dvalidate`.
    Definición de la función `EstimarGanancia_AUC_lightgbm`.
    Configuración del problema de optimización y llamada a `mbo()`.
    Entrenamiento del modelo final con los mejores hiperparámetros.
4. Revisar las métricas impresas en consola (mensajes de `Accuracy` por iteración) y los resultados finales.



 6. Resultados esperados

 Un conjunto de hiperparámetros óptimos (o cercanos al óptimo) encontrados mediante optimización bayesiana.
 Un modelo LightGBM entrenado con dichos valores.
 Un valor de Accuracy en el conjunto de validación que sirve como indicador de desempeño.
 (Opcional) Predicciones sobre un conjunto de test para evaluación externa / competencia.


