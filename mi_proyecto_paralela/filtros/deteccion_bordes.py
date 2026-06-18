import numpy as np
import time
import sys
from mpi4py import MPI
from PIL import Image


# ==========================================
# RGB -> GRIS
# ==========================================
def a_gris(img):
    if len(img.shape) == 2:
        return img

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)


# ==========================================
# SOBEL OPTIMIZADO
# ==========================================
def aplicar_sobel(imagen):

    imagen = imagen.astype(np.int16)  # 🔥 FIX CLAVE

    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    Ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    padded = np.pad(imagen, 1, mode="edge")

    gx = (
        -padded[:-2, :-2] + padded[:-2, 2:] +
        -2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:] +
        -padded[2:, :-2] + padded[2:, 2:]
    )

    gy = (
        -padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:] +
         padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
    )

    mag = np.sqrt(gx ** 2 + gy ** 2)

    return np.clip(mag, 0, 255).astype(np.uint8)


# ==========================================
# SECUENCIAL
# ==========================================
def bordes_seq(imagen):
    start = time.perf_counter()

    gris = a_gris(imagen)
    res = aplicar_sobel(gris)

    end = time.perf_counter()

    return res, end - start


# ==========================================
# MPI
# ==========================================
def bordes_mpi(imagen):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        gris = a_gris(imagen)
        bloques = np.array_split(gris, size, axis=0)
    else:
        bloques = None

    start = time.perf_counter()

    bloque = comm.scatter(bloques, root=0)

    h, w = bloque.shape

    arriba = None
    abajo = None

    # recibir halos
    if rank > 0:
        arriba = np.empty((1, w), dtype=np.uint8)
        comm.Recv(arriba, source=rank - 1)

    if rank < size - 1:
        abajo = np.empty((1, w), dtype=np.uint8)
        comm.Recv(abajo, source=rank + 1)

    # enviar halos
    if rank > 0:
        comm.Send(bloque[0], dest=rank - 1)

    if rank < size - 1:
        comm.Send(bloque[-1], dest=rank + 1)

    # construir bloque extendido
    ext = bloque

    if arriba is not None:
        ext = np.vstack([arriba, ext])

    if abajo is not None:
        ext = np.vstack([ext, abajo])

    local = aplicar_sobel(ext)

    # quitar halos del resultado
    if arriba is not None:
        local = local[1:]

    if abajo is not None:
        local = local[:-1]

    out = comm.gather(local, root=0)

    end = time.perf_counter()

    if rank == 0:
        final = np.vstack(out)
        return final, end - start

    return None, end - start


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    modo = sys.argv[1] if len(sys.argv) > 1 else "--seq"
    ruta = sys.argv[2] if len(sys.argv) > 2 else None

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    imagen = None

    if rank == 0:
        if ruta:
            print(f"[Raíz] Cargando imagen: {ruta}")
            imagen = np.array(Image.open(ruta).convert("RGB"))
        else:
            imagen = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)

    if modo == "--seq" and rank == 0:

        print("=== BORDES SECUENCIAL ===")

        res, t = bordes_seq(imagen)

        print(f"Tiempo secuencial: {t:.6f} s")

        if ruta:
            Image.fromarray(res).save("resultados/bordes_seq.jpg")

    elif modo == "--mpi":

        if rank == 0:
            print(f"=== BORDES MPI con {comm.Get_size()} procesos ===")

        res, t = bordes_mpi(imagen)

        if rank == 0:

            print(f"Tiempo MPI: {t:.6f} s")

            if ruta:
                Image.fromarray(res).save("resultados/bordes_mpi.jpg")
