
import cv2
import numpy as np
import time

# ============================================================
# 1. PARÁMETROS
# ============================================================

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

feature_params = dict(
    maxCorners=20,
    qualityLevel=0.3,
    minDistance=10,
    blockSize=7
)

trajectory_len = 20
detect_interval = 1
trajectories = []
frame_idx = 0

# ============================================================
# 2. ABRIR VIDEO
cap = cv2.VideoCapture("slow_traffic_small.mp4")
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video")

fps_video = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ok, frame = cap.read()
if not ok:
    cap.release()
    raise RuntimeError("No se pudo leer el primer fotograma")

prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Crear video de salida
out = cv2.VideoWriter("resultado_optical_flow.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
 fps_video if fps_video > 0 else 30,(width, height))

while True:

    start = time.time()
    if frame_idx > 0:
        suc, frame = cap.read()
        if not suc:
            break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    # --------------------------------------------------------
    # 4. ACTUALIZAR TRAYECTORIAS

    if len(trajectories) > 0:

        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK( prev_gray, frame_gray, p0, None,**lk_params)

        if p1 is not None and st is not None:

            p0r, st_back, err_back = cv2.calcOpticalFlowPyrLK( frame_gray, prev_gray,  p1, None,**lk_params)
            if p0r is not None and st_back is not None:

                d = np.abs(p0 - p0r).reshape(-1, 2).max(axis=1)
                good = ((d < 1.0) &(st.ravel() == 1) &(st_back.ravel() == 1))
                new_trajectories = []

                for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2),good):

                    if not good_flag:
                        continue
                    trajectory.append((x, y))
                    if len(trajectory) > trajectory_len:
                        del trajectory[0]

                    new_trajectories.append(trajectory)

                    cv2.circle( img, (int(x), int(y)), 2,(0, 0, 255),-1)

                trajectories = new_trajectories

                # Dibujar  trayectorias
                if trajectories:
                    cv2.polylines(
                        img,
                        [ np.int32(trajectory)
                            for trajectory in trajectories
                            if len(trajectory) >= 2
                        ],
                        False,(0, 255, 0),2
                    )

    # --------------------------------------------------------
    # 5. DETECTAR NUEVOS PUNTOS
    if frame_idx % detect_interval == 0:

        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        for trajectory in trajectories:
            x, y = np.int32(trajectory[-1])
            cv2.circle(mask, (x, y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    # --------------------------------------------------------
    # 6. FPS Y VISUALIZACIÓN

    elapsed = time.time() - start
    fps = 1.0 / elapsed if elapsed > 0 else 0

    cv2.putText( img, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0), 2)
    cv2.putText(img,f"Track count: {len(trajectories)}",(20, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(img)
    cv2.imshow("Optical Flow", img)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    prev_gray = frame_gray.copy()
    frame_idx += 1


cap.release()
out.release()
cv2.destroyAllWindows()

print("Video guardado : resultado_optical_flow.mp4")