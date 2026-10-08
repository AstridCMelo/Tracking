import cv2 
import numpy as np
import time

video_path = ['trackingAstrid-2.mp4', 'slow_traffic_Small.mp4', "trackingAstrid-1.mp4"]

for video_name in video_path:
    for opcion in range(4): 

        #Metodo de tracking
        match opcion:
            case 0:
                tracker_name = 'Boosting'
                tracker = cv2.legacy.TrackerBoosting_create() 
            case 1:
                tracker_name = 'MIL'
                tracker = cv2.TrackerMIL_create() 
            case 2:
                tracker_name = 'KCF'
                tracker = cv2.TrackerKCF_create() 
            case 3:
                tracker_name = 'TLD'
                tracker = cv2.legacy.TrackerTLD_create() 

        print(f"Tracking con {tracker_name}")

        #Video a probar
        video = cv2.VideoCapture(video_name)

        #Para guardar el video con las mismas propiedades que el video actual
        fps_video = video.get(cv2.CAP_PROP_FPS) 
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(f'{tracker_name}_{video_name}', codec, fps_video,(width, height))

        ok,frame = video.read()

        #Resize video
        cv2.namedWindow("Seleccionar objeto", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Seleccionar objeto", 640, 360)

        if not ok :
            print ("Error al leer video ")
            exit()

        bbox = (0, 0, 0, 0)

        while(bbox == (0, 0, 0, 0)):
            bbox = cv2.selectROI ("Seleccionar objeto", frame , False)

        tracker.init (frame, bbox)

        start_time = time.time()
        frame_count = 0

        while True :
            ok , frame = video.read()
            if not ok :
                break

            frame_count += 1

            #frame = cv2.resize(frame, (640, 360))
            #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            timer = cv2.getTickCount()

            ok,bbox = tracker.update(frame)

            #fps por frame
            fps = cv2.getTickFrequency() / (cv2.getTickCount()-timer)
            cv2.putText (frame," FPS : " + str(int(fps)),
            (10 ,70), cv2 . FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0),2)


            if ok :
                x,y,w,h = [int (v) for v in bbox ]
                cv2.rectangle (frame, (x,y), (x+w, y+h) , (255, 0, 0), 2)
            else :
                cv2.putText(frame, "Tracking fallido " ,(10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0 ,0 ,255), 2)

            out.write(frame)

            cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)  
            cv2.resizeWindow("Tracking", 640, 360)
            cv2.imshow ("Tracking" , frame )

            #Terminar programa con esc
            if cv2.waitKey(1) & 0xFF == 27:
                    break

        #Para medir el rendimiento en velocidad promedio
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print(f"FPS: {fps}")

        
        video.release ()
        out.release()
        cv2.destroyAllWindows()
