# mlcore — Handoff técnico para packaging, migración a Polars y preparación de release

**Responsable del handoff científico:** David Medina-Ortiz
**Responsable de hardening técnico:** Diego
**Estado de entrada:** arquitectura científica congelada después de testing, demos, CLI, cleanup y documentación
**Alcance de este handoff:** packaging/distribución, migración tabular a Polars, clean-install hardening y preparación de release
**Fuera de alcance:** rediseño científico, nuevos algoritmos, nuevas métricas, cambios de particionado, nuevas estrategias de tuning o cambios en la semántica experimental

---

## 1. Objetivo

`mlcore` es una librería agnóstica al dominio para **aprendizaje supervisado clásico**, diseñada para ejecutar workflows reproducibles de clasificación y regresión sobre representaciones numéricas ya preparadas.

El desarrollo científico y arquitectónico principal está cerrado. El trabajo de este handoff consiste en llevar la librería desde un estado funcional y científicamente congelado a un estado **package-ready**, con especial énfasis en dos tareas.

1. Migrar la capa tabular interna desde **pandas hacia Polars** sin alterar los contratos científicos.
2. Preparar y validar el paquete para distribución limpia mediante wheel/sdist, extras opcionales, entornos limpios, CI de packaging y eventual publicación.

La prioridad absoluta es **preservar comportamiento**. La migración a Polars y el hardening de packaging no deben modificar resultados científicos, memberships de particiones, métricas, predicciones, fingerprints sin versionado, orden de clases, protected-test semantics ni estructura de los workflows.

---

## 2. Regla principal del handoff

> **Cambiar infraestructura, no cambiar ciencia.**

Diego puede modificar libremente implementación tabular, dependencias, packaging, CI, adapters de I/O y utilidades relacionadas con distribución, siempre que los contratos públicos y científicos indicados en este documento permanezcan estables.

Si durante el trabajo aparece un problema que exige modificar una decisión científica o una API pública importante, **no resolverlo unilateralmente**. Documentar el problema y devolverlo para revisión.

---

## 3. Qué se considera congelado

Los siguientes elementos deben tratarse como contratos de la librería.

### 3.1 Alcance científico

Soportado

- clasificación binaria
- clasificación multiclase
- regresión single-target
- features numéricas preparadas
- estimadores clásicos compatibles con scikit-learn
- proveedores sklearn, XGBoost y LightGBM
- optimización Grid, Random, Halving y Optuna
- benchmarking multi-representación, multi-partición, multi-modelo y multi-seed
- persistencia reproducible
- workflows Python, YAML/JSON y CLI

Fuera de alcance

- deep learning
- representation learning
- generación de embeddings/descriptores
- procesamiento de secuencias o SMILES
- PU learning
- multilabel
- multi-output
- AutoML general
- reducción de redundancia

### 3.2 BioSieve

BioSieve es el motor canónico de generación de particiones cuando los datos no vienen previamente particionados.

`mlcore` no debe implementar nuevamente

- KFold propio
- StratifiedKFold propio
- GroupKFold propio
- reducción de redundancia
- MMseqs2
- clustering de secuencias
- homology reduction
- selection of representatives

Flujo esperado

```text
prepared data
    │
    ├── existing PartitionPlan
    │
    └── no partitions
             │
             ▼
          BioSieve
             │
             ▼
       PartitionPlan
             │
             ▼
           mlcore
```

### 3.3 Preprocessing

El preprocessing debe seguir siendo **leakage-safe**.

```text
training fold
    ↓
fit imputer
    ↓
fit scaler
    ↓
fit estimator

held-out fold
    ↓
transform only
    ↓
predict
```

Nunca ajustar imputación o scaling usando el dataset completo antes de crear folds.

### 3.4 Métricas y prediction semantics

Se deben preservar

- `PredictionResult`
- orden de clases
- `positive_class`
- diferencia entre probabilities y decision scores
- OOF predictions
- compatibilidad task/métrica
- handling explícito de folds donde una métrica no está definida

No inferir silenciosamente que `proba[:, 1]` es siempre la clase positiva.

### 3.5 Tuning

Se debe preservar

- typed search spaces
- `Categorical`
- `Integer`
- `Float`
- `LogFloat`
- Grid
- Random
- Halving Grid
- Halving Random
- Optuna
- multi-metric scoring
- `refit_metric`
- `refit=False`
- sample weights
- failure tracking
- seeded reproducibility

Protected-test semantics

```text
train + validation
       ↓
hyperparameter selection
       ↓
refit
       ↓
FINAL TEST
```

No utilizar el mismo ordinary CV para seleccionar hiperparámetros y reportar performance final unbiased.

### 3.6 Benchmarking

Se debe preservar la semántica

```text
representations
× partitions
× algorithms
× seeds
× modes
```

Cuando un benchmark contiene `untuned` y `tuned` con `train/validation/test`, los modos comparados deben reportarse sobre **el mismo protected test**.

También deben preservarse

- `run_id`
- `configuration_id`
- fold metrics
- aggregate metrics
- sample-level predictions
- optimization history
- runtime
- isolated failures

### 3.7 Persistence

Artifact schema actual

```text
artifact/
├── manifest.json
├── model.joblib
├── environment.json
├── feature_schema.json
├── provenance.json
├── parameters.json
├── metrics.json
├── training_config.json
├── partition_plan.json      # optional
└── checksums.sha256
```

Mantener

- artifact schema version
- checksums antes de deserializar
- feature schema validation
- dataset fingerprint
- partition fingerprint
- class ordering
- positive class
- package/environment metadata

No cambiar silenciosamente el schema de artifacts existentes.

### 3.8 Public API

Mantener disponibles los entry points públicos

```python
mlcore.train(...)
mlcore.validate(...)
mlcore.evaluate(...)
mlcore.tune(...)
mlcore.optimize(...)
mlcore.benchmark(...)
mlcore.predict(...)
```

También mantener los contratos públicos principales

```text
DatasetBundle
FeatureSchema
PartitionPlan
BioSievePartitionConfig
PreprocessingConfig
TuningConfig
SearchSpace
BenchmarkConfig
PredictionResult
EvaluationResult
ValidationResult
OptimizationResult
BenchmarkResult
MODEL_REGISTRY
```

### 3.9 Config y CLI

Mantener config schema `1.0` salvo acuerdo explícito para versionarlo.

Mantener workflows

```text
train
evaluate
validate
tune
benchmark
predict
```

Mantener CLI principal

```text
mlcore run
mlcore train
mlcore evaluate
mlcore validate
mlcore tune
mlcore optimize
mlcore benchmark
mlcore predict
mlcore models
mlcore artifact
mlcore config
mlcore doctor
```

Mantener exit codes

```text
0  success
2  config / CLI contract error
3  workflow or domain failure
4  unexpected internal failure
```

---

## 4. Arquitectura que Diego debe preservar

```text
Prepared numerical data
          │
          ▼
     DatasetBundle
          │
          ├──── existing PartitionPlan
          │
          └──── BioSieve → PartitionPlan
          │
          ▼
  Leakage-safe Pipeline
          │
          ▼
     AlgorithmSpec
          │
          ▼
   EstimatorFactory
          │
    ┌─────┼───────────────┐
    ▼     ▼               ▼
Validate Tune          Final train
    │     │               │
    └─────┼───────────────┘
          ▼
 Prediction / Evaluation
          │
      ┌───┴────────┐
      ▼            ▼
 Benchmark      Persistence
```

No reintroducir

- `Trainer` legacy
- `BackendBase`
- runners
- optimizers legacy
- execution paths alternativos

---

# PARTE A — MIGRACIÓN A POLARS

## 5. Objetivo de la migración

El objetivo no es simplemente reemplazar `pd.DataFrame` por `pl.DataFrame` línea por línea.

La migración debe conseguir que

- pandas deje de ser una dependencia obligatoria de `mlcore`
- Polars sea la implementación tabular canónica
- el core siga aceptando `numpy.ndarray`
- las operaciones tabulares sean más consistentes con BioSieve
- los outputs tabulares de benchmarking/tuning/persistence puedan expresarse con Polars
- scikit-learn/XGBoost/LightGBM reciban inputs estables y predecibles
- fingerprints y schemas no cambien accidentalmente por diferencias entre backends tabulares

---

## 6. Principio de frontera recomendado

### 6.1 Dentro de mlcore

Usar Polars para

- lectura/escritura tabular
- selección de columnas
- result tables
- benchmark tables
- tuning history tables
- persistence tables
- config-driven CSV/TSV loading
- partition interchange tables

### 6.2 Frontera de estimadores

Usar **NumPy** como frontera estable hacia estimadores.

```text
Polars / DatasetBundle
        │
        ▼
FeatureSchema + validation
        │
        ▼
NumPy arrays
        │
        ▼
sklearn / XGBoost / LightGBM
```

No depender de que todas las versiones de todos los proveedores soporten directamente Polars.

Preservar nombres y orden de features mediante `FeatureSchema`, no delegándolos al estimator backend.

---

## 7. Contrato público durante la migración

Target recomendado después de la migración

- `numpy.ndarray` sigue soportado
- `polars.DataFrame` pasa a ser el DataFrame principal
- pandas deja de ser dependencia core
- se puede mantener compatibilidad opcional con pandas si resulta barata, pero no debe condicionar el diseño

### Cambio público permitido

Los métodos tabulares existentes pueden pasar de devolver pandas a devolver Polars, siempre que

- el nombre del método se conserve
- las columnas se conserven
- la semántica de filas se conserve
- el orden determinista se conserve
- los tipos lógicos se documenten

Ejemplos

```python
BenchmarkResult.runs_frame()
BenchmarkResult.metrics_frame()
BenchmarkResult.predictions_frame()
OptimizationResult.history_frame()
```

Pueden retornar `pl.DataFrame` después de la migración.

Este cambio debe documentarse claramente como parte del release.

---

## 8. Inventario actual de pandas

En la versión entregada, los principales puntos de uso están concentrados en aproximadamente estos módulos.

| Área | Módulo | Prioridad | Comentario |
|---|---|---:|---|
| Benchmark | `mlcore/benchmark/results.py` | Alta | Genera múltiples DataFrames públicos |
| Config | `mlcore/config/builders.py` | Alta | Lectura de CSV/TSV y construcción de DatasetBundle |
| Config | `mlcore/config/runner.py` | Alta | Export de outputs/predictions |
| Datasets | `mlcore/datasets/validation.py` | Crítica | Validación de inputs y missing values |
| Datasets | `mlcore/datasets/schemas.py` | Crítica | FeatureSchema y DatasetBundle |
| Datasets | `mlcore/datasets/loaders.py` | Alta | PartitionPlan desde CSV/TSV/DataFrame |
| Datasets | `mlcore/datasets/_fingerprint.py` | Crítica | Fingerprints científicos |
| BioSieve | `mlcore/datasets/biosieve.py` | Alta | Adapter; BioSieve ya trabaja naturalmente con Polars |
| Preprocessing | `mlcore/preprocessing/validation.py` | Alta | Conversión/validación antes del Pipeline |
| Tuning | `mlcore/tuning/results.py` | Media | History table |
| Persistence | `mlcore/persistence/artifacts.py` | Alta | Benchmark artifact tables |
| Persistence | `mlcore/persistence/load.py` | Alta | Lectura de tables persistidas |
| Persistence | `mlcore/persistence/save.py` | Alta | Escritura de tables persistidas |
| Environment | `mlcore/persistence/environment.py` | Baja | Metadata de paquetes instalados |

Antes de comenzar, repetir el inventario con el estado exacto del repo

```bash
grep -RIn --include='*.py' \
  -E 'import pandas|from pandas|pd\.|DataFrame|Series' \
  mlcore
```

---

## 9. Orden recomendado de migración

No migrar los archivos de forma aleatoria.

### Etapa P1 — Capa de compatibilidad tabular

Crear un pequeño módulo interno con funciones canónicas, por ejemplo

```text
mlcore/utils/tabular.py
```

Responsabilidades posibles

```python
to_numpy(...)
to_polars(...)
is_dataframe_like(...)
feature_names(...)
canonical_dtype(...)
missing_mask(...)
```

Evitar dispersar checks de pandas/Polars en 20 módulos.

No convertir este módulo en una abstracción enorme. Debe ser una frontera pequeña.

### Etapa P2 — DatasetBundle y FeatureSchema

Migrar primero

```text
mlcore/datasets/schemas.py
mlcore/datasets/validation.py
mlcore/datasets/_fingerprint.py
```

Esta es la parte más delicada.

Gates obligatorios

- NumPy sigue funcionando
- Polars funciona
- sample order idéntico
- feature order idéntico
- sample IDs idénticos
- targets idénticos
- groups idénticos
- sample weights idénticos
- classification target detection idéntica
- missing/infinite validation idéntica

### Etapa P3 — BioSieve

Migrar/simplificar

```text
mlcore/datasets/biosieve.py
```

BioSieve ya utiliza Polars. El objetivo debe ser **eliminar conversiones innecesarias**, no crear un adapter nuevo más complejo.

Mantener exactamente los IDs recibidos/devueltos.

### Etapa P4 — Loaders y config

Migrar

```text
mlcore/datasets/loaders.py
mlcore/config/builders.py
mlcore/config/runner.py
```

Validar

- CSV
- TSV
- target columns
- sample_id
- groups
- sample_weight
- explicit feature list
- relative paths desde config

### Etapa P5 — Results

Migrar

```text
mlcore/benchmark/results.py
mlcore/tuning/results.py
```

Mantener nombres y orden de columnas.

Actualizar notebooks y docs solamente donde la API tabular cambie de pandas a Polars.

### Etapa P6 — Persistence

Migrar

```text
mlcore/persistence/artifacts.py
mlcore/persistence/load.py
mlcore/persistence/save.py
```

No modificar artifact schema silenciosamente.

Si el formato persistido cambia de CSV a Parquet, eso se considera un cambio de artifact schema y debe versionarse. Para este handoff se recomienda **mantener CSV** y cambiar solamente el engine utilizado para leer/escribir.

### Etapa P7 — Eliminar pandas del core

Al final debe pasar

```bash
grep -RIn --include='*.py' \
  -E 'import pandas|from pandas|pd\.' \
  mlcore
```

Target ideal

```text
0 runtime pandas imports
```

Si se decide mantener compatibilidad opcional con pandas, concentrarla en **un único adapter opcional** y no volver a introducir pandas como dependencia obligatoria.

---

## 10. NaN versus null en Polars

Este punto requiere tests específicos.

En Polars

- `NaN` es un valor floating-point
- `null` representa missingness a nivel de columna

En NumPy/scikit-learn, gran parte del preprocessing actual interpreta `np.nan` como missing.

Por tanto, definir una política explícita.

### Política recomendada

Antes de la frontera NumPy/estimator

```text
Polars null
      ↓
canonical missing representation
      ↓
np.nan where appropriate
      ↓
SimpleImputer inside the fold
```

No imputar durante la conversión.

La conversión solamente normaliza representación de missingness.

Tests necesarios

- float NaN
- Polars null
- columna con NaN + null
- target null debe fallar
- feature null puede llegar al Pipeline si imputation está habilitada
- `Inf/-Inf` deben seguir rechazándose según el contrato actual

---

## 11. Dtypes y FeatureSchema

Pandas y Polars nombran/representan tipos de forma diferente.

Ejemplo conceptual

```text
pandas float64
Polars Float64
```

No permitir que esa diferencia cambie fingerprints o schema compatibility sin intención.

### Recomendación

Introducir **canonical logical dtypes** internos.

Ejemplo

```text
float32
float64
int32
int64
uint*
bool
string
```

`FeatureSchema` debe almacenar la forma canónica, no `str(dtype)` específico del backend tabular.

Si esto modifica fingerprints actuales, proceder como se indica en la siguiente sección.

---

## 12. Fingerprints — zona crítica

Los fingerprints son parte de provenance científica.

**No cambiar fingerprints accidentalmente durante la migración.**

Primero crear fixtures con datasets conocidos y registrar sus fingerprints actuales.

Ejemplo

```python
assert dataset.fingerprint == "<known hash>"
assert partition_plan.fingerprint == "<known hash>"
```

Después migrar a Polars.

### Dos resultados posibles

#### Opción A — Los fingerprints permanecen idénticos

Preferida.

No requiere cambios de schema/version.

#### Opción B — El nuevo canonical dtype/data serialization exige nuevos fingerprints

No reemplazar silenciosamente el algoritmo.

Versionar el fingerprint contract, por ejemplo

```text
fingerprint_version = 2
```

Preservar compatibilidad de lectura con artifacts anteriores cuando sea razonable.

Documentar la migración.

---

## 13. Row order

Polars no debe introducir cambios de orden implícitos.

Contratos que deben permanecer

- `DatasetBundle` preserva orden de entrada
- `subset()` devuelve muestras en el orden del dataset original
- OOF predictions vuelven al orden del dataset original
- benchmark predictions conservan `sample_id`
- partition membership no implica un nuevo training order

Agregar regression tests explícitos con tablas reordenadas.

---

## 14. Result tables

Cuando los métodos públicos cambien a Polars, verificar exactamente los schemas.

### BenchmarkResult

Mantener como mínimo

```text
runs_frame()
aggregate_metrics_frame()
fold_metrics_frame()
metrics_frame()
predictions_frame()
failures_frame()
optimization_history_frame()
```

### OptimizationResult

Mantener

```text
history_frame()
```

La migración no debe renombrar columnas ni cambiar significado de scores.

Agregar tests del tipo

```python
assert result.metrics_frame().columns == EXPECTED_COLUMNS
```

con Polars.

---

## 15. Notebooks y demos después de Polars

Los notebooks son ejemplos ejecutables, pero no deben condicionar la arquitectura.

Después de la migración

- actualizar imports pandas → polars donde corresponda
- mantener matplotlib para visualización
- convertir a NumPy únicamente cuando una función gráfica lo requiera
- no meter seaborn como dependencia core
- no meter visualización dentro de `mlcore`

Los notebooks deben seguir representando workflows reales y no convertirse en adapters de compatibilidad pandas.

---

# PARTE B — PACKAGING Y DISTRIBUCIÓN

## 16. Estado actual de packaging

Actualmente el proyecto utiliza

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

La distribución se llama

```text
mlcore
```

El paquete importable también se llama

```text
mlcore
```

Python soportado actualmente

```text
3.11
3.12
3.13
```

Entry point CLI

```toml
[project.scripts]
mlcore = "mlcore.cli.main:main"
```

Se recomienda **mantener Hatchling**. Cambiar build backend no aporta valor salvo que aparezca un problema concreto.

---

## 17. Dependencias target después de Polars

Core propuesto

```toml
dependencies = [
    "numpy>=2.0,<3.0",
    "polars>=1.0,<2.0",
    "scikit-learn>=1.8,<2.0",
    "scipy>=1.10,<2.0",
    "joblib>=1.3,<2.0",
    "rich>=13.0,<16.0",
    "pyyaml>=6.0,<7.0",
]
```

Pandas debe salir del core.

Mantener extras separados

```text
biosieve
xgboost
lightgbm
optuna
all
```

Si se decide mantener input compatibility con pandas, usar un extra explícito

```toml
pandas = ["pandas>=2.0,<3.0"]
```

solo si realmente se implementa y testea.

No añadir pandas “por si acaso”.

---

## 18. Extras y optional imports

Core install debe funcionar sin

- BioSieve
- XGBoost
- LightGBM
- Optuna

Y estos providers deben registrarse solo cuando están disponibles.

Gates

```bash
python -c "import mlcore; print(mlcore.__version__)"
mlcore --help
mlcore doctor --json
```

con instalación core-only.

Luego repetir con extras individuales.

---

## 19. Release/dev dependencies

No mezclar herramientas de release con runtime.

Se recomienda agregar un extra separado, por ejemplo

```toml
release = [
    "build>=1,<2",
    "twine>=5,<7",
    "check-wheel-contents>=0.6,<1",
]
```

No es obligatorio usar exactamente estas versiones; ajustarlas a la versión actual disponible durante el hardening.

Ruff/linting no forma parte de los gates científicos actuales. Si se desea utilizar durante release hardening, mantenerlo separado y no convertirlo en condición para rediseñar código estable.

---

## 20. Build artifacts

Build esperado

```bash
rm -rf dist build *.egg-info
python -m build
```

Debe producir

```text
dist/mlcore-<version>.tar.gz
dist/mlcore-<version>-py3-none-any.whl
```

Verificar

```bash
python -m twine check dist/*
check-wheel-contents dist/*.whl
```

Inspeccionar contenido

```bash
unzip -l dist/*.whl
```

No deben incluirse

- tests
- caches
- `.git`
- `.vscode`
- notebooks salvo decisión explícita
- temporary artifacts
- development files sin función runtime

Sí deben incluirse

- `mlcore/**`
- package metadata
- license metadata
- dependency metadata
- CLI entrypoint metadata

---

## 21. Clean install — core only

Crear un environment completamente limpio.

Ejemplo

```bash
python3.11 -m venv /tmp/mlcore-core-311
source /tmp/mlcore-core-311/bin/activate
python -m pip install --upgrade pip
python -m pip install dist/mlcore-*.whl
```

Gates

```bash
python - <<'PY'
import mlcore
print(mlcore.__version__)
print(mlcore.MODEL_REGISTRY.summary())
PY

mlcore --help
mlcore doctor
mlcore models list --provider sklearn
```

Después ejecutar un smoke mínimo de clasificación y regresión usando únicamente sklearn.

El core install **no debe importar XGBoost, LightGBM, Optuna o BioSieve**.

---

## 22. Clean install — extras individuales

Validar al menos

```text
[biosieve]
[xgboost]
[lightgbm]
[optuna]
[all]
```

Idealmente sobre wheel, no solo editable install.

Ejemplo conceptual

```bash
python -m pip install './dist/mlcore-0.1.0-py3-none-any.whl[biosieve]'
```

Verificar para cada provider

- optional import
- registry population
- minimal training/splitting/tuning workflow
- `mlcore doctor`

---

## 23. Python matrix

Mínimo obligatorio

```text
Python 3.11
Python 3.12
Python 3.13
```

Core tests en los tres.

Full extras pueden ejecutarse al menos en

```text
3.11
3.13
```

si el coste de CI resulta alto.

No reducir `requires-python` sin discusión.

---

## 24. Operating systems

Target mínimo recomendado para packaging

- Ubuntu latest
- Windows latest
- macOS latest

Core package smoke en los tres.

Full scientific test suite puede permanecer principalmente en Linux si los optional providers vuelven la matriz demasiado pesada.

Lo importante es detectar problemas de

- paths
- encoding
- subprocess
- CLI entry points
- temporary directories
- wheel install

---

## 25. CI mínima recomendada

### Workflow A — Core CI

Eventos

```text
push
pull_request
```

Matrix

```text
Python 3.11
Python 3.12
Python 3.13
```

Pasos

```bash
pip install -e '.[tests]'
pytest -q
python -m compileall -q mlcore tests
```

Después de la migración a Polars, asegurarse de que este entorno no instale pandas salvo dependencia transitiva inevitable.

### Workflow B — Package build

```bash
python -m build
python -m twine check dist/*
check-wheel-contents dist/*.whl
```

Después crear un nuevo venv e instalar **el wheel**, no el source tree.

Smoke

```bash
python -c 'import mlcore; print(mlcore.__version__)'
mlcore --help
mlcore doctor --json
```

### Workflow C — Optional providers

Al menos Linux.

Probar

```text
biosieve
xgboost
lightgbm
optuna
all
```

### Workflow D — Notebook smoke

Los 13 notebooks pueden mantenerse como integration gates si el tiempo total es razonable.

Usar modo reducido

```text
MLCORE_DEMO_TEST=1
MPLBACKEND=Agg
```

No exigir notebooks sin outputs/execution counts como gate funcional.

---

## 26. Tests de documentación

No reintroducir una suite que haga fallar `pytest` por cuestiones editoriales como

- existencia de `docs/index.md`
- links Markdown
- execution counts de notebooks
- outputs guardados en notebooks

La documentación puede validarse durante release mediante herramientas específicas si se desea, pero **no es parte del contrato científico de pytest**.

---

## 27. Versioning

Versión actual

```text
0.1.0
```

No cambiar automáticamente el `Development Status` de Alpha a Beta/Stable.

La decisión sobre versión de release y classifier final corresponde al owner.

Si la migración Polars cambia públicamente los return types de DataFrame, documentarlo claramente en release notes.

No utilizar un bump de versión para ocultar cambios silenciosos de fingerprints/artifact schema.

---

## 28. TestPyPI antes de PyPI

Antes de cualquier publicación final, recomendado

1. construir wheel + sdist
2. `twine check`
3. publicar en TestPyPI
4. crear environment limpio
5. instalar desde TestPyPI
6. probar CLI
7. ejecutar un classification smoke
8. ejecutar un regression smoke
9. ejecutar al menos un optional-provider smoke

Publicar en PyPI únicamente con autorización explícita del owner.

Si se utiliza GitHub Actions, preferir PyPI Trusted Publishing en vez de guardar tokens de larga duración.

---

# PARTE C — VALIDACIÓN DE LA MIGRACIÓN

## 29. Baseline antes de tocar código

Antes de iniciar la migración

```bash
pytest -q
python -m compileall -q mlcore tests
mlcore --help
mlcore doctor --json
```

Registrar

- número de tests
- warnings existentes
- Python version
- sklearn version
- polars version una vez instalada
- optional providers disponibles

No hardcodear el número de tests como contrato permanente. El contrato es que **todos los tests vigentes pasen**.

---

## 30. Suite específica Polars requerida

Agregar tests específicos para Polars.

### DatasetBundle

- NumPy input
- Polars DataFrame input
- feature names
- sample IDs
- string labels
- integer labels
- NaN
- null
- Inf rejection
- sample weights
- groups

### FeatureSchema

- schema from Polars
- feature reordering rejected
- missing feature rejected
- extra feature rejected
- logical dtype normalization

### Fingerprints

- fixed dataset fixture
- fixed partition fixture
- row reorder behavior according to existing contract
- DataFrame backend does not silently alter hash semantics

### BioSieve

- Polars path end-to-end
- no pandas intermediate required
- memberships unchanged
- metadata/provenance unchanged

### BenchmarkResult

- result methods return Polars
- expected columns
- deterministic row ordering
- `run_id` traceability

### Persistence

- table save/load round trip
- no changed artifact semantics
- legacy artifact load where supported

### Config

- CSV/TSV loaded through Polars
- same DatasetBundle semantics

---

## 31. Regression matrix científica post-Polars

No hace falta crear otra arquitectura de tests. Reutilizar los torture tests existentes.

Debe seguir pasando al menos

```text
binary classification
multiclass classification
regression
high-dimensional p >> n
imbalanced data
NaN/imputation
sample weights
groups
external partitions
BioSieve partitions
Grid
Random
Halving
Optuna
multi-representation benchmark
persistence
Python/YAML/CLI parity
```

Y los 13 notebooks ejecutables.

---

## 32. Equivalencia numérica

Para un conjunto pequeño de fixtures deterministas, comparar pre/post Polars.

Mantener tolerancias razonables

```python
np.testing.assert_allclose(...)
```

Comparar

- predictions
- probabilities
- metric values
- OOF sample ordering
- best params para seeded deterministic searches cuando aplique
- benchmark row counts
- partition membership

No exigir bitwise equality cuando el algoritmo no la garantiza.

---

## 33. Performance smoke

Polars no debe empeorar catastróficamente memoria/tiempo de I/O.

Crear al menos un benchmark simple, por ejemplo

```text
50,000 rows
256 numerical features
```

Medir

- CSV load
- DatasetBundle construction
- result table creation
- benchmark export

No establecer thresholds universales dependientes del hardware. Guardar resultados y revisar regresiones obvias.

---

# PARTE D — COSAS QUE DIEGO NO DEBE CAMBIAR

## 34. No modificar sin revisión

- catálogo científico de algoritmos
- métrica definitions
- positive-class semantics
- OOF behavior
- BioSieve partition semantics
- sample membership
- protected-test logic
- tuning objective semantics
- benchmark comparison semantics
- preprocessing leakage guarantees
- artifact scientific provenance
- config schema semantics
- CLI exit codes
- public workflow names

---

## 35. No agregar durante este handoff

- nuevos modelos
- nuevos optimizers
- AutoML
- SHAP
- plotting dentro del core
- cloud deployment
- server/API web
- database backend
- remote artifact store
- experiment tracker
- Dask/Ray
- GPU abstraction
- new sequence-specific code

Este handoff es para **hardening y distribución**, no para expandir scope.

---

# PARTE E — CRITERIOS DE ACEPTACIÓN

## 36. Polars migration — Definition of Done

Se considera terminada cuando

- Polars es dependencia core
- pandas ya no es dependencia core
- no hay pandas runtime imports distribuidos por `mlcore`
- NumPy sigue soportado
- DatasetBundle soporta Polars
- FeatureSchema preserva semántica
- fingerprints están preservados o formalmente versionados
- NaN/null están cubiertos por tests
- BioSieve usa el camino Polars sin conversiones redundantes
- result tables usan Polars
- persistence tables usan Polars
- config CSV/TSV loading usa Polars
- docs/notebooks relevantes están actualizados
- suite completa pasa
- 13 notebooks pasan en modo test

---

## 37. Packaging — Definition of Done

Se considera terminado cuando

- `python -m build` genera wheel + sdist
- `twine check` pasa
- wheel contents son correctos
- core wheel instala en environment limpio
- `mlcore --help` funciona desde wheel instalado
- `mlcore doctor` funciona
- classification smoke core-only pasa
- regression smoke core-only pasa
- Python 3.11 pasa
- Python 3.12 pasa
- Python 3.13 pasa
- extras individuales instalan
- `[all]` instala
- BioSieve integration smoke pasa
- XGBoost smoke pasa
- LightGBM smoke pasa
- Optuna smoke pasa
- CLI entrypoint funciona fuera del repo
- artifacts pueden save/load en clean install
- sdist puede reconstruir wheel

---

## 38. Release candidate — Definition of Done

Antes de declarar release candidate

```text
[ ] scientific test suite green
[ ] Polars-specific suite green
[ ] notebook suite green
[ ] core clean-install green
[ ] all-extras clean-install green
[ ] wheel green
[ ] sdist green
[ ] CLI smoke green
[ ] TestPyPI smoke green
[ ] no accidental pandas core dependency
[ ] no cache/editor/dev files in distribution
[ ] release notes prepared
[ ] known limitations documented
```

---

# PARTE F — ENTREGABLE QUE DIEGO DEBE DEVOLVER

## 39. Código

Un branch/PR con

- migración a Polars
- packaging hardening
- CI necesaria
- test updates estrictamente relacionados
- docs/notebooks actualizados únicamente por cambios Polars/packaging

Evitar mezclar refactors estéticos grandes no relacionados.

---

## 40. `POLARS_MIGRATION_REPORT.md`

Debe indicar

1. módulos migrados
2. estrategia de canonical dtype
3. tratamiento de NaN/null
4. cambios en inputs públicos
5. cambios en outputs tabulares
6. fingerprints pre/post
7. BioSieve integration
8. persistence compatibility
9. performance smoke
10. tests agregados
11. cualquier incompatibilidad conocida

---

## 41. `PACKAGING_REPORT.md`

Debe indicar

1. build backend final
2. dependencies core
3. optional extras
4. Python matrix probada
5. OS probados
6. wheel/sdist checks
7. clean-install results
8. CLI smoke results
9. TestPyPI status
10. CI workflows agregados
11. problemas conocidos

---

## 42. Release artifacts

Entregar, al menos localmente

```text
dist/mlcore-<version>.tar.gz
dist/mlcore-<version>-py3-none-any.whl
```

más output de

```bash
python -m twine check dist/*
check-wheel-contents dist/*.whl
```

No publicar a PyPI final sin autorización.

---

## 43. Validation report

Entregar una tabla simple.

| Gate | Environment | Resultado |
|---|---|---|
| pytest | Python 3.11 | PASS/FAIL |
| pytest | Python 3.12 | PASS/FAIL |
| pytest | Python 3.13 | PASS/FAIL |
| notebook demos | test mode | PASS/FAIL |
| core wheel install | clean env | PASS/FAIL |
| biosieve extra | clean env | PASS/FAIL |
| xgboost extra | clean env | PASS/FAIL |
| lightgbm extra | clean env | PASS/FAIL |
| optuna extra | clean env | PASS/FAIL |
| all extras | clean env | PASS/FAIL |
| TestPyPI install | clean env | PASS/FAIL |

---

# PARTE G — SECUENCIA RECOMENDADA PARA DIEGO

## 44. Orden de trabajo

### Paso 1 — congelar baseline

```bash
git status
pytest -q
python -m compileall -q mlcore tests
mlcore doctor --json
```

Crear un commit limpio antes de migrar.

### Paso 2 — agregar Polars

Agregar dependency, todavía sin retirar pandas.

Crear los adapters tabulares mínimos.

### Paso 3 — migrar DatasetBundle/FeatureSchema/fingerprints

No seguir hasta que estos tests estén verdes.

### Paso 4 — migrar BioSieve/loaders/config

Comprobar membership equality.

### Paso 5 — migrar result tables

Actualizar tests/notebooks.

### Paso 6 — migrar persistence

Comprobar round trips.

### Paso 7 — retirar pandas del core

Repetir grep.

### Paso 8 — full scientific regression

```bash
pytest -q
```

más notebooks.

### Paso 9 — packaging hardening

Build wheel/sdist y clean installs.

### Paso 10 — CI

Automatizar los gates que realmente aportan valor.

### Paso 11 — TestPyPI

Solo después de todos los gates anteriores.

### Paso 12 — handback

Entregar PR/branch + dos reports + validation matrix + artifacts.

---

# 45. Checklist corto para comenzar

```text
[ ] Estoy trabajando sobre el baseline congelado
[ ] La suite actual pasa antes de tocar Polars
[ ] Entiendo que BioSieve genera particiones
[ ] No voy a modificar contratos científicos
[ ] Tengo fixtures para fingerprints
[ ] Tengo tests para NaN/null
[ ] Mantendré NumPy como frontera hacia estimadores
[ ] Migraré tabular I/O/results a Polars
[ ] pandas saldrá de dependencies core
[ ] Probaré wheels en environments limpios
[ ] No publicaré en PyPI final sin autorización
```

---

# 46. Resultado esperado del handoff

Al finalizar, `mlcore` debería poder describirse de esta manera

> **A reproducible classical supervised machine-learning library using Polars for its tabular layer, BioSieve for partition generation, NumPy as a stable estimator boundary, and clean optional integrations for XGBoost, LightGBM and Optuna, distributed as a validated Python package with reproducible artifacts and a polished CLI.**

La arquitectura científica ya existe y está congelada. El objetivo de este handoff es **hacerla distribuible, mantenible y eficiente sin cambiar lo que científicamente significa cada workflow**.

---

## 47. Contacto / decisión

Cualquier cambio que afecte

- scientific semantics
- fingerprints
- artifact schema
- public API names
- config schema
- BioSieve semantics
- protected-test behavior

se devuelve para revisión antes de integrarlo.

Cambios puramente relacionados con

- Polars implementation
- packaging
- wheel/sdist
- dependency metadata
- CI
- clean-install compatibility
- release tooling

están dentro del scope de Diego siempre que los gates anteriores permanezcan verdes.
