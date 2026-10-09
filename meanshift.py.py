import cv2
import numpy as np
import os
import time

video_nombre = "trackingJenifer-2.mp4"
# Buscar  video
video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), video_nombre)

if not os.path.exists(video_path):
    print("No se encontró el video:")
    print(video_path)
    exit()

print("Video encontrado:")
print(video_path)


# ============================================================
# 2. ABRIR VIDEO
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Error al abrir el video.")
    exit()

fps_video = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nInformación del video:")
print(f"Resolución: {width} x {height}")
print(f"FPS: {fps_video:.2f}")
print(f"Frames: {total_frames}")



ok, frame = video.read()
if not ok:
    print("No se pudo leer el primer frame.")
    video.release()
    exit()


scale = 0.6
new_width = int(width * scale)
new_height = int(height * scale)
frame = cv2.resize( frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

print(f"Resolución procesada: {new_width} x {new_height}")
print(f"Escala aplicada: {scale * 100:.0f}%")
# ============================================================
# 4. SELECCIONAR EL OBJETO

print("\nSelecciona el objeto que quieres rastrear.")
print("Presiona ENTER cuando termines.")
print("Presiona ESC para cancelar.")

cv2.namedWindow("Seleccionar objeto", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Seleccionar objeto", 800, 500)

bbox = cv2.selectROI("Seleccionar objeto",frame, False, False)
cv2.destroyWindow("Seleccionar objeto")

if bbox == (0, 0, 0, 0):
    print("No se seleccionó ningún objeto.")
    video.release()
    exit()
x, y, w, h = bbox
print(f"\nROI seleccionada: x={x}, y={y}, w={w}, h={h}")

# ============================================================
# 5.  HISTOGRAMA DEL OBJETO

#  ROI
roi = frame[y:y+h, x:x+w]
#  ROI de BGR a HSV
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#  máscara  eliminar píxeles oscuros
mask = cv2.inRange(hsv_roi,np.array((0., 90., 50.)), np.array((180., 255., 255.)))

# Histograma  Hue
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

# Normalizar h
cv2.normalize( roi_hist, roi_hist,0,255,cv2.NORM_MINMAX)


# ============================================================
# 6. CRITERIO  TERMINACIÓN  MEANSHIFT
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,20,1)

nombre_video = os.path.splitext(video_nombre)[0]
# Carpeta  guardar  
carpeta_salida = os.path.join(os.path.dirname(video_path),"meanshift")
#   ruta  video  salida
output_path = os.path.join( carpeta_salida, nombre_video + "_MeanShift.mp4")

codec = cv2.VideoWriter_fourcc(*"mp4v")
os.makedirs(carpeta_salida, exist_ok=True)
out = cv2.VideoWriter( output_path, codec, fps_video,(new_width, new_height))

if not out.isOpened():
    print("Error al crear el video de salida.")
    video.release()
    exit()

# ============================================================
# 8. MÉTRICAS

frame_count = 0

total_tracking_time = 0
total_elapsed_start = time.time()

previous_center = None
total_displacement = 0
displacement_count = 0

# ============================================================
# 9. TRACKING MEANSHIFT
cv2.namedWindow("MeanShift Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MeanShift Tracking", 800, 500)
while True:

    ok, frame = video.read()
    if not ok:
        break
    frame_count += 1

    frame = cv2.resize(  frame,  (new_width, new_height), interpolation=cv2.INTER_AREA)
    # --------------------------------------------------------
    # Convertir frame a HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # --------------------------------------------------------
    # Backprojection
    dst = cv2.calcBackProject( [hsv], [0], roi_hist, [0, 180], 1)
    # --------------------------------------------------------
    # MEANSHIFT
    timer = cv2.getTickCount()
    iterations, track_window = cv2.meanShift( dst, (x, y, w, h), term_crit)

    tracking_time = ( cv2.getTickCount() - timer) / cv2.getTickFrequency()
    total_tracking_time += tracking_time

    # Actualizar posición
    x, y, w, h = track_window

    # ========================================================
    # CENTRO DEL BOUNDING BOX
    center_x = x + w / 2
    center_y = y + h / 2
    current_center = ( center_x, center_y)

    # ========================================================
    # DESPLAZAMIENTO

    if previous_center is not None:
        displacement = (
            (current_center[0] - previous_center[0]) ** 2+(current_center[1] - previous_center[1]) ** 2
        ) ** 0.5

        total_displacement += displacement
        displacement_count += 1
    previous_center = current_center


    # ========================================================
    # DIBUJO BOUNDING BOX
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText( frame, "Tracking - MeanShift", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ========================================================
    # FPS  TRACKER
    if tracking_time > 0:
        current_fps = 1 / tracking_time
    else:
        current_fps = 0
    cv2.putText( frame, f"FPS tracker: {current_fps:.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
#guardar video
    out.write(frame)
    # MOSTRAR VIDEO
    cv2.namedWindow("MeanShift Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow( "MeanShift Tracking", 800, 500)
    cv2.imshow( "MeanShift Tracking", frame)

    # ESC PARA SALIR
    if cv2.waitKey(1) & 0xFF == 27:
        break
# ============================================================
# 10. MÉTRICAS 

total_elapsed_time = time.time() - total_elapsed_start
if frame_count > 0:
    average_fps = ( frame_count / total_elapsed_time)
    average_frame_time = ( total_elapsed_time / frame_count)
else:
    average_fps = 0
    average_frame_time = 0
if total_tracking_time > 0:
    tracker_fps = ( frame_count / total_tracking_time )
else:
    tracker_fps = 0
if displacement_count > 0:
    average_displacement = (total_displacement / displacement_count)
else:
    average_displacement = 0

# ============================================================
# 11. MOSTRAR RESULTADOS
# ============================================================

print("\n========================================")
print("RESULTADOS MEANSHIFT")
print("========================================")
print(f"Frames procesados: {frame_count}")
print(f"FPS promedio general: " f"{average_fps:.2f} fps")
print( f"FPS del tracker: " f"{tracker_fps:.2f} fps")
print( f"Tiempo total de procesamiento: " f"{total_elapsed_time:.4f} segundos")
print( f"Tiempo promedio por frame: " f"{average_frame_time * 1000:.4f} ms")
print(f"Desplazamiento promedio del bounding box: " f"{average_displacement:.2f} px")
print("\nVideo guardado en:")
print(output_path)

video.release()
out.release()
cv2.destroyAllWindows()
