# ─────────────────────────────────────────────────────────────────
# Imagen base: Ubuntu 22.04 con Python 3.11 y OpenMPI preinstalado
# ─────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

# Evita prompts interactivos durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

# ── 1. Dependencias del sistema ──────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    libopenmpi-dev \
    openmpi-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Hace que "python" y "pip" apunten a la versión 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3     1

# ── 2. Directorio de trabajo ─────────────────────────────────────
WORKDIR /app

# ── 3. Dependencias Python ───────────────────────────────────────
# Se copian primero para aprovechar la caché de Docker:
# si el requirements no cambia, esta capa no se reconstruye.
COPY requirements.txt .

# Se instala todo excepto cupy (requiere GPU en el host para compilar)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    Pillow==10.3.0 \
    mpi4py==3.1.6 \
    scipy==1.13.0 \
    matplotlib==3.8.4

# ── 4. Código fuente ─────────────────────────────────────────────
COPY . .

# Crea carpetas necesarias si no existen
RUN mkdir -p resultados imagenes_prueba

# ── 5. Configuración de MPI ──────────────────────────────────────
# Permite ejecutar MPI como root dentro del contenedor (necesario en Docker)
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# ── 6. Comando por defecto ───────────────────────────────────────
# Ejecuta el script maestro en modo secuencial como prueba de humo.
# El usuario puede sobreescribir este comando al hacer docker run.
CMD ["python", "main.py", "--help"]
