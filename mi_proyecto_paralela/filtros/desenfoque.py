import numpy as np
import time
import sys
from mpi4py import MPI

def aplicar_desenfoque_local(bloque_padded, start_row, end_row):
    """Aplica el promedio de convolución 3x3 leyendo las dimensiones internas del bloque."""
    h, w, c = bloque_padded.shape
    resultado = np.zeros((end_row - start_row, w, c), dtype=np.uint8)

    for i in range(start_row, end_row):
        for j in range(w):
            r_min, r_max = max(0, i - 1), min(h, i + 2)
            c_min, c_max = max(0, j - 1), min(w, j + 2)

            vecindad = bloque_padded[r_min:r_max, c_min:c_max, :]
            resultado[i - start_row, j, :] = np.mean(vecindad, axis=(0, 1))

    return resultado

def aplicar_desenfoque_secuencial(imagen):
    """Versión secuencial adaptada al tamaño real de la imagen entrante."""
    start_time = time.perf_counter()
    h, w, c = imagen.shape
    resultado = aplicar_desenfoque_local(imagen, 0, h)
    end_time = time.perf_counter()
    return resultado, end_time - start_time

def aplicar_desenfoque_mpi(imagen_rgb):
    """Versión paralela distribuida con cálculo dinámico de halos perimetrales."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        alto_total = imagen_rgb.shape[0]
        bloques_base = np.array_split(imagen_rgb, size, axis=0)
        paquetes_a_enviar = []
        inicio_fila = 0

        for i in range(size):
            filas_bloque = bloques_base[i].shape[0]
            fin_fila = inicio_fila + filas_bloque

            pad_arriba = 1 if inicio_fila > 0 else 0
            pad_abajo = 1 if fin_fila < alto_total else 0

            bloque_con_halo = imagen_rgb[inicio_fila - pad_arriba : fin_fila + pad_abajo, :, :]
            idx_inicio_real = pad_arriba
            idx_fin_real = pad_arriba + filas_bloque

            paquetes_a_enviar.append((bloque_con_halo, idx_inicio_real, idx_fin_real))
            inicio_fila = fin_fila
    else:
        paquetes_a_enviar = None

    comm.Barrier()
    start_time = time.perf_counter()
    mi_paquete = comm.scatter(paquetes_a_enviar, root=0)
    mi_bloque_padded, start_r, end_r = mi_paquete

    mi_bloque_desenfocado = aplicar_desenfoque_local(mi_bloque_padded, start_r, end_r)
    bloques_finales = comm.gather(mi_bloque_desenfocado, root=0)

    end_time = time.perf_counter()
    duracion = end_time - start_time

    if rank == 0:
        imagen_desenfocada_final = np.vstack(bloques_finales)
        return imagen_desenfocada_final, duracion
    return None, duracion

if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "--seq"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    imagen_prueba = None
    if rank == 0:
        if len(sys.argv) > 2:
            from PIL import Image
            ruta_img = sys.argv[2]
            print(f"[Raíz] Cargando imagen real desde: {ruta_img}")
            imagen_prueba = np.array(Image.open(ruta_img).convert("RGB"))
        else:
            print("[Raíz] Sin imagen externa. Generando matriz sintética predeterminada de 400x600...")
            imagen_prueba = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)

    if modo == "--seq" and rank == 0:
        print("=== INICIANDO DESENFOQUE SECUENCIAL ===")
        print(f"[Raíz] Dimensiones dinámicas detectadas: {imagen_prueba.shape}")
        resultado, t_seq = aplicar_desenfoque_secuencial(imagen_prueba)
        print("¡Filtro Secuencial completado con éxito!")
        print(f"Tiempo total de ejecución Secuencial: {t_seq:.6f} segundos.")

        if len(sys.argv) > 2:
            from PIL import Image
            Image.fromarray(resultado).save("resultados/resultado_blur.jpg")
            print("[Raíz] Copia guardada con éxito en 'resultados/resultado_blur.jpg'")

    elif modo == "--mpi":
        if rank == 0:
            print(f"=== INICIANDO DESENFOQUE DISTRIBUIDO CON {comm.Get_size()} PROCESOS ===")
            print(f"[Raíz] Dimensiones dinámicas detectadas: {imagen_prueba.shape}")

        resultado, t_mpi = aplicar_desenfoque_mpi(imagen_prueba)

        if rank == 0:
            print("\n[Raíz] ¡Filtro de Desenfoque completado con éxito!")
            print(f"Tiempo total de ejecución MPI: {t_mpi:.6f} segundos.")

            if len(sys.argv) > 2 and resultado is not None:
                from PIL import Image
                Image.fromarray(resultado).save("resultados/resultado_blur.jpg")
                print("[Raíz] Copia distribuida guardada con éxito en 'resultados/resultado_blur.jpg'")
