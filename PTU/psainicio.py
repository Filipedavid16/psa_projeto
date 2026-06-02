import cv2

# Esta função será substituída pelo cammexerptu_waveshare.py
seguir_face = None

# ALTERA AQUI
CAMERA_ID = 0


def run():

    global seguir_face

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"Erro: não consegui abrir a camera {CAMERA_ID}")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    print("Camera ligada.")
    print("Q = sair")

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Erro ao ler frame.")
            break

        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        bbox = None

        if len(faces) > 0:

            # usa a maior face encontrada
            x, y, fw, fh = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            x1 = x
            y1 = y
            x2 = x + fw
            y2 = y + fh

            bbox = (x1, y1, x2, y2)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                "FACE",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # enviar coordenadas para o PTU
        if seguir_face is not None:
            seguir_face(bbox, w, h)

        # cruz central
        cv2.line(
            frame,
            (w // 2, 0),
            (w // 2, h),
            (255, 255, 255),
            1
        )

        cv2.line(
            frame,
            (0, h // 2),
            (w, h // 2),
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "Waveshare Face Tracking",
            frame
        )

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()