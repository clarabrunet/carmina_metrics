## Estructura

- **`models/`**: Carpeta donde copiar los output de cada modelo en formato JSON.
- **`results/`**: Carpeta donde aparecerán los informes finales.
- **`run_evaluation.py`**: Script de ejecución
- **`scripts/`**: Código interno del programa

## Pasos para ejecutar

1.  Copiar archivos `.json` dentro de la carpeta **`models/`**.
2.  Ejecuta el script principal:
    ```
    python run_evaluation.py
    ```
3.  I ja ho tindríem!

## ¿Qué significan las métricas?

El programa evalúa el rendimiento en tres niveles:

### 1. Identity 

Identificar etiquetas

### 2. Category

Clasificar categoría de las etiquetas

### 3. Combined

Identificación + Clasificación
