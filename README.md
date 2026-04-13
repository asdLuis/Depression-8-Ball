# Modelo de Depresión Estudiantil

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo capaz de predecir si un estudiante presenta depresión a partir de factores demográficos, académicos y de estilo de vida. La meta es identificar, de forma rápida y precisa, aquellos perfiles estudiantiles con alta probabilidad de presentar depresión.

## Descripción del Dataset

Para este proyecto se utiliza el dataset  **Student Depression Dataset**, descargado de [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset). Este conjunto de datos contiene **27,901 instancias** y **18 columnas** en total, donde cada fila representa un estudiante individual.

Las variables incluyen información demográfica, académica y de hábitos de vida como edad, género, ciudad, promedio académico (CGPA), horas de sueño, hábitos alimenticios, presión académica y satisfacción con los estudios. La variable objetivo (`Depression`) es de tipo binaria (0 = sin depresión, 1 = con depresión).

### Tabla de Variables

| Variable | Tipo | Descripción |
|---|---|---|
| id | Identificador | Identificador único del estudiante |
| Gender | Categórico | Género del estudiante (Male/Female) |
| Age | Numérico | Edad del estudiante |
| City | Categórico | Ciudad de residencia |
| Profession | Categórico | Ocupación del estudiante |
| Academic Pressure | Numérico | Nivel de presión académica (escala 0-5) |
| Work Pressure | Numérico | Nivel de presión laboral (escala 0-5) |
| CGPA | Numérico | Promedio académico |
| Study Satisfaction | Numérico | Nivel de satisfacción con los estudios (escala 0-5) |
| Job Satisfaction | Numérico | Nivel de satisfacción laboral (escala 0-5) |
| Sleep Duration | Categórico | Duración promedio de sueño diario |
| Dietary Habits | Categórico | Hábitos alimenticios |
| Degree | Categórico | Nivel académico actual |
| Have you ever had suicidal thoughts? | Binario | Historial de pensamientos suicidas |
| Work/Study Hours | Numérico | Horas de trabajo o estudio por día |
| Financial Stress | Numérico | Nivel de estrés financiero |
| Family History of Mental Illness | Binario | Antecedentes familiares de enfermedad mental |
| Depression | **Target** | Variable objetivo binaria (0 = No, 1 = Sí) |

---

## Limpieza

### Eliminación de Columnas No Relevantes

Se eliminaron las siguientes columnas antes de cualquier otro procesamiento:

- **`id`**: Es un identificador único sin valor predictivo, solo causaria que nuestro modelo intente encontrar una relación con nuestras demás variables.
- **`Work Pressure`** y **`Job Satisfaction`**: Al revisar el dataset, se encontró que el **100% de los valores de estas columnas son 0**, lo que indica que los estudiantes de este dataset no tienen actividad laboral registrada. Columnas sin varianza no aportan información al modelo.

### Validación de Columnas Categóricas

Se detectó que las columnas `City`, `Degree` y `Profession` contenían **entradas inválidas o erróneas**. Valores como nombres de personas, grados académicos en el campo de ciudad  o entradas sin sentido. Para limpiar estas columnas se aplicó un **filtro de frecuencia mínima de 10 apariciones**: cualquier valor que aparezca menos de 10 veces se considera un dato atípico o erróneo y se reemplaza con `NaN`. 

### Tratamiento de Valores Faltantes

Después de la validación categórica, el dataset presentó **60 valores faltantes** distribuidos en las columnas `City` (26), `Profession` (31) y `Financial Stress` (3).

Se decidió **eliminar las filas con valores faltantes** (`dropna`) en lugar de imputar un valor (como la media o moda) ya que las 60 instancias afectadas representan apenas el **0.21%** del total de **27,901 registros**, por lo que su eliminación no altera significativamente la distribución del dataset.

### Eliminación de Duplicados

Se verificó la existencia de filas duplicadas y no se encontraron duplicados en el dataset, por lo que este paso no tuvo impacto en el número de instancias.

### División del Dataset

Con los datos ya limpios, se realizó la división en subconjuntos de entrenamiento y prueba:

- **`X_train`** (80%): utilizado para entrenar el modelo
- **`X_test`** (20%): utilizado para evaluar el desempeño del modelo en datos no vistos

Se utilizó el parámetro `stratify=y` para garantizar que la proporción de instancias con y sin depresión sea la misma en ambos subconjuntos. Esto es para evitar que una clase este subrepresentada en la división de prueba.

### Preprocesamiento de Features

Se utilizó un `ColumnTransformer` de scikit-learn que aplica transformaciones distintas según el tipo de variable:

#### Variables Numéricas — `StandardScaler`

Se normalizaron las columnas `Age`, `Academic Pressure`, `CGPA`, `Study Satisfaction`, `Work/Study Hours` y `Financial Stress` utilizando **StandardScaler**, que transforma cada valor para que la distribución resultante tenga **media 0 y desviación estándar 1**: `z = (valor - media) / desviación_estándar`.

#### Variables Categóricas — `OneHotEncoder`

Se aplicó **OneHotEncoder** a las columnas `Gender`, `City`, `Profession`, `Sleep Duration`, `Dietary Habits`, `Degree` y `Family History of Mental Illness`, convirtiendo cada categoría en una columna binaria independiente.

### Referencias

Yang, T., He, Y., Wu, L., Ren, L., Lin, J., Wang, C., Wu, S., & Liu, X. (2023). The relationships between anxiety and suicidal ideation and between depression and suicidal ideation among Chinese college students: A network analysis. Heliyon, 9(10), e20938. https://doi.org/10.1016/j.heliyon.2023.e20938