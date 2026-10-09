
import cv2
import numpy as np
import time
import os

# ============================================================
# 1. PARÁMETROS

lk_params = dict( winSize=(15, 15),  maxLevel=2, criteria=( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03))
feature_params = dict( maxCorners=20, qualityLevel=0.3, minDistance=10, blockSize=7)

trajectory_len = 20
detect_interval = 1

trajectories = []
frame_idx = 0

# ============================================================
# 2. ABRIR VIDEO

video_path = "trackingJenifer-2.mp4"
os.makedirs("opticalflow", exist_ok=True)
output_path = os.path.join("opticalflow", "resultado_optical_flow.mp4")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video")

fps_video = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ok, frame = cap.read()

if not ok:
    cap.release()
    raise RuntimeError("No se pudo leer el primer fotograma")

prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ============================================================
# 3. VIDEO DE SALIDA

out = cv2.VideoWriter( output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_video if fps_video > 0 else 30, (width, height))
if not out.isOpened():
    cap.release()
    raise RuntimeError("No se pudo crear el video de salida")

# ============================================================
# 4. MÉTRICAS

puntos_detectados_total = 0
puntos_seguidos_total = 0
puntos_perdidos_total = 0
desplazamientos = []
tiempos_frame = []
puntos_evaluados_total = 0
frames_procesados = 0
desplazamientos_x = []
desplazamientos_y = []

# ============================================================
# 5. VENTANA
cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)


while True:
    start = time.perf_counter()
    if frame_idx > 0:
        suc, frame = cap.read()

        if not suc:
            break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    # --------------------------------------------------------
    # A. ACTUALIZAR TRAYECTORIAS EXISTENTES
    if len(trajectories) > 0:

        p0 = np.float32([
            trajectory[-1] for trajectory in trajectories
        ]).reshape(-1, 1, 2)

        puntos_anteriores = len(p0)
        puntos_evaluados_total += puntos_anteriores

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray, p0, None, **lk_params)

        if p1 is None or st is None:
            puntos_perdidos_total += puntos_anteriores
            trajectories = []

        else:

            # Comprobación 
            p0r, st_back, err_back = cv2.calcOpticalFlowPyrLK( frame_gray,  prev_gray, p1, None, **lk_params)
            if p0r is None or st_back is None:
                puntos_perdidos_total += puntos_anteriores
                trajectories = []

            else:
                d = np.abs(p0 - p0r).reshape(-1, 2).max(axis=1)
                good = (   (d < 1.0) &  (st.ravel() == 1) & (st_back.ravel() == 1))
                puntos_seguidos = int(np.count_nonzero(good))
                puntos_perdidos = puntos_anteriores - puntos_seguidos

                puntos_seguidos_total += puntos_seguidos
                puntos_perdidos_total += puntos_perdidos

                new_trajectories = []

                posiciones_nuevas = p1.reshape(-1, 2)
                posiciones_anteriores = p0.reshape(-1, 2)

                
                # Desplazamiento horizontal y vertical por punto
                vectores = posiciones_nuevas - posiciones_anteriores
                vectores_validos = vectores[good]
                if len(vectores_validos) > 0:
                    u = vectores_validos[:, 0]  # Horizontal
                    v = vectores_validos[:, 1]  # Vertical
                    desplazamientos_x.extend(u.tolist())
                    desplazamientos_y.extend(v.tolist())
                    movimientos = np.linalg.norm(vectores_validos, axis=1)
                    desplazamientos.extend(movimientos.tolist())

                # Actualizar trayectorias válidas.
                for trajectory, (x, y), good_flag in zip( trajectories, posiciones_nuevas, good):

                    if not good_flag:
                        continue

                    trajectory.append((x, y))

                    if len(trajectory) > trajectory_len:
                        del trajectory[0]

                    new_trajectories.append(trajectory)

                    # Punto actual en rojo.
                    cv2.circle(  img, (int(x), int(y)), 2,  (0, 0, 255), -1)

                trajectories = new_trajectories

                # Dibujar trayectorias en .
                for trajectory in trajectories:
                    if len(trajectory) >= 2:
                        cv2.polylines( img, [np.int32(trajectory)], False, (0, 255, 0), 2 )

    # --------------------------------------------------------
    # B. DETECTAR NUEVOS PUNTOS

    if frame_idx % detect_interval == 0:
        mask = np.full_like(frame_gray, 255)
        for trajectory in trajectories:
            x, y = np.int32(trajectory[-1])
            cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack( frame_gray, mask=mask, **feature_params)

        if p is not None:
            nuevos_puntos = np.float32(p).reshape(-1, 2)
            puntos_detectados_total += len(nuevos_puntos)
            for x, y in nuevos_puntos:
                trajectories.append([(x, y)])

    # --------------------------------------------------------
    # C. AP METRICA

    elapsed = time.perf_counter() - start
    tiempos_frame.append(elapsed)
    frames_procesados += 1
    fps_actual = 1.0 / elapsed if elapsed > 0 else 0

    # Información mostrada sobre el video.
    cv2.putText( img, f"FPS: {fps_actual:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText( img, f"Track count: {len(trajectories)}",  (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),2)

    # Guardar fotograma .
    out.write(img)
    # Mostrar 
    cv2.imshow("Optical Flow", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # iteración 
    prev_gray = frame_gray.copy()
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# ============================================================
# 8.  RESULTADOS

desplazamiento_promedio = ( np.mean(desplazamientos)
    if desplazamientos else 0
)


# Desplazamiento horizontal promedio
mov_horizontal_medio = (
    np.mean(desplazamientos_x) if desplazamientos_x else 0
)

# Desplazamiento vertical promedio
mov_vertical_medio = (
    np.mean(desplazamientos_y) if desplazamientos_y else 0
)


# ============================================================
# 9. IMPRIMIR 

print("\n" + "=" * 52)
print("          RESULTADOS DEL FLUJO ÓPTICO")
print("=" * 52)

print("\n--- VIDEO ---")
print(f"Archivo analizado: {video_path}")
print(f"Resolución: {width} x {height}")
print(f"FPS originales: {fps_video:.2f}")
print(f"Frames totales del video: {frames_video}")
print(f"Frames procesados: {frames_procesados}")

print("\n--- PUNTOS Y TRAYECTORIAS ---")
print(f"Puntos nuevos detectados: {puntos_detectados_total}")
print(f"Puntos evaluados : {puntos_evaluados_total}")
print(f"Puntos seguidos : {puntos_seguidos_total}")
print(f"Puntos perdidos : {puntos_perdidos_total}")
print(f"Trayectorias activas: {len(trajectories)}")

print("\n--- MOVIMIENTO  ---")
print(f"Magnitud promedio: {desplazamiento_promedio:.3f} px/frame")
print(f"Horizontal promedio: {mov_horizontal_medio:.3f} px/frame")
print(f"Vertical promedio: {mov_vertical_medio:.3f} px/frame")

print("\n--- ARCHIVO DE SALIDA ---")
print(f"Video guardado: {os.path.abspath(output_path)}")

print("=" * 52)
