# Procesamiento de Imágenes en CPU y GPU
**Curso:** Computación Paralela y Distribuida  
**Proyecto:** 2 — Procesamiento de Imágenes  
**Integrantes:**
- Integrante 1 — Escala de grises y desenfoque (secuencial + MPI)
- Integrante 2 — Detección de bordes (secuencial + MPI)
- Integrante 3 — Filtros en GPU con CuPy (CUDA)
---
 
## Estructura del proyecto
 
```
mi_proyecto_paralela/
├── imagenes_prueba/
│   ├── color_1920p.jpg        
│   ├── color_2400.jpg
|   ├── gris_1920p.jpg
|   └── gris_2400p.jpg  
├── resultados/             # las imágenes procesadas se guardan aquí
├── filtros/
│   ├── __init__.py
│   ├── escala_grises.py    # secuencial + MPI
│   ├── desenfoque.py       # secuencial + MPI con halos
│   ├── deteccion_bordes.py # secuencial + MPI
│   └── filtros_cuda.py     # GPU con CuPy
├── main.py                 # script maestro unificado
├── evaluar_rendimiento.py  # orquestador de pruebas comparativas
├── requirements.txt
├── Dockerfile
└── README.md
```
 
---
 
## Requisitos previos
 
- Python 3.11+
- OpenMPI instalado en el sistema (`sudo apt install libopenmpi-dev openmpi-bin`)
- GPU NVIDIA con CUDA 12.x (solo para la versión GPU)
- Docker (opcional, para ejecución en contenedor)
---
 
## Instalación local
 
```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd mi_proyecto_paralela
 
# 2. Instalar dependencias Python
pip install -r requirements.txt
 
# 3. (Opcional) Instalar CuPy si se tiene GPU con CUDA 12.x
pip install cupy-cuda12x==13.1.0
```
 
---
 
## Ejecución local
 
### Versión secuencial
 
```bash
# Escala de grises
python filtros/escala_grises.py --seq imagenes_prueba/color_4k.jpg
```
 
> Los resultados se guardan en `resultados/`.
 
### Versión MPI (paralela en CPU)
 
```bash
# Escala de grises con 4 procesos (si hay suficientes núcleos)
mpirun -np 4 python filtros/escala_grises.py --mpi imagenes_prueba/color_4k.jpg
 
# Dentro de Docker, o en máquina con pocos núcleos
mpiexec --allow-run-as-root --oversubscribe -n 4 python filtros/escala_grises.py --mpi imagenes_prueba/color_4k.jpg
```
 
> Puedes variar `-np` con los valores 1, 2, 4, 8 para las pruebas de speedup.
 
### Versión GPU (CUDA)
 
```bash
python filtros/filtros_cuda.py imagenes_prueba/color_4k.jpg
```
 
### Script maestro (todos los filtros de una vez)
 
```bash
# Secuencial
python main.py --seq imagenes_prueba/color_4k.jpg
 
# MPI con 4 procesos
mpirun -np 4 python main.py --mpi imagenes_prueba/color_4k.jpg
 
# GPU
python main.py --cuda imagenes_prueba/color_4k.jpg
```
 
### Evaluación de Rendimiento
 
```bash
# Corre todas las pruebas y genera las tablas y gráficos comparativos
python evaluar_rendimiento.py
```
 
---
 
## Ejecución con Docker
 
### Construir la imagen
 
```bash
docker build -t procesamiento-imagenes .
```
 
### Ejecutar en Docker
 
```bash
# Secuencial (montando la carpeta de imágenes y resultados)
docker run --rm \
  -v $(pwd)/imagenes_prueba:/app/imagenes_prueba \
  -v $(pwd)/resultados:/app/resultados \
  procesamiento-imagenes \
  python filtros/escala_grises.py --seq imagenes_prueba/color_4k.jpg
 
# MPI con 4 procesos dentro del contenedor
docker run --rm \
  -v $(pwd)/imagenes_prueba:/app/imagenes_prueba \
  -v $(pwd)/resultados:/app/resultados \
  procesamiento-imagenes \
  mpirun -np 4 python filtros/desenfoque.py --mpi imagenes_prueba/color_4k.jpg
```
 
### Descargar desde Docker Hub
 
```bash
# Reemplaza <usuario> con el usuario de Docker Hub del equipo
docker pull <usuario>/procesamiento-imagenes:latest
 
docker run --rm \
  -v $(pwd)/imagenes_prueba:/app/imagenes_prueba \
  -v $(pwd)/resultados:/app/resultados \
  <usuario>/procesamiento-imagenes:latest \
  python main.py --seq imagenes_prueba/color_4k.jpg
```
 
### Publicar en Docker Hub
 
```bash
docker tag procesamiento-imagenes <usuario>/procesamiento-imagenes:latest
docker push <usuario>/procesamiento-imagenes:latest
```
 
---
 
## Resultados esperados
 
Los archivos de salida se guardan en `resultados/`:
 
| Filtro | Archivo de salida |
|---|---|
| Escala de grises | `resultado_gris.jpg` |
| Desenfoque | `resultado_blur.jpg` |
| Detección de bordes | `resultado_bordes.jpg` |
 
---
 
## Notas
 
- La versión MPI **no** debe ejecutarse con `mpirun` en modo secuencial (`--seq`); hacerlo dejará los procesos secundarios inactivos.
- CuPy requiere una GPU NVIDIA compatible. Sin GPU, usar únicamente `--seq` o `--mpi`.
- Las pruebas de rendimiento del informe se realizaron con `-np 1, 2, 4, 8` sobre `color_4k.jpg`.
