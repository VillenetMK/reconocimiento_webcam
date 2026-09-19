"""Deteccion local de objetos y rostros con webcam. Ejecutar PY1.PY --help."""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WINDOW = "Objetos y rostros | Q o ESC: salir"
MODEL = ROOT / "models" / "yolo11n.pt"

# Traducciones de las 80 clases COCO; el orden es el del modelo YOLO11n.
LABELS = (
    "persona|bicicleta|auto|moto|avion|autobus|tren|camion|bote|semaforo|"
    "hidrante|senal de pare|parquimetro|banco|pajaro|gato|perro|caballo|oveja|vaca|"
    "elefante|oso|cebra|jirafa|mochila|paraguas|bolso|corbata|maleta|frisbee|"
    "esquis|tabla de snowboard|pelota|cometa|bate de beisbol|guante de beisbol|"
    "patineta|tabla de surf|raqueta|botella|copa|taza|tenedor|cuchillo|cuchara|"
    "tazon|platano|manzana|sandwich|naranja|brocoli|zanahoria|hot dog|pizza|dona|"
    "pastel|silla|sofa|maceta|cama|mesa|inodoro|televisor|laptop|raton|"
    "control remoto|teclado|celular|microondas|horno|tostadora|lavadero|"
    "refrigerador|libro|reloj|florero|tijeras|oso de peluche|secadora|cepillo de dientes"
).split("|")


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    label: str
    kind: str = "objeto"


def clipped_box(values, width, height):
    """Devuelve una caja valida dentro de la imagen, o None si no tiene area."""
    values = tuple(values)
    if len(values) != 4 or not all(math.isfinite(float(v)) for v in values):
        return None
    x1, y1, x2, y2 = map(round, values)
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class Detector:
    def __init__(self, confidence=0.45, size=416, faces_only=False, model_path=MODEL):
        import cv2

        self.confidence, self.size = confidence, size
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_detector.empty():
            raise RuntimeError("No se encontro el detector de rostros de OpenCV.")
        self.object_model = None
        if not faces_only:
            from ultralytics import YOLO

            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print("Preparando YOLO11n para detectar objetos...", flush=True)
            if not model_path.is_file():
                print("Primera ejecucion: descargando el modelo. Necesitas internet.", flush=True)
            try:
                self.object_model = YOLO(str(model_path))
            except Exception as error:
                raise RuntimeError(
                    "No se pudo cargar/descargar YOLO11n. Revisa tu conexion y "
                    "vuelve a intentarlo, o inicia con --solo-rostros. "
                    f"Detalle: {error}"
                ) from error

    def detect(self, frame):
        import cv2

        height, width = frame.shape[:2]
        detections = []
        # Los detectores reciben el fotograma original, antes de dibujar marcas.
        if self.object_model is not None:
            result = self.object_model.predict(
                source=frame, conf=self.confidence, imgsz=self.size,
                device="cpu", verbose=False, save=False
            )[0]
            if result.boxes is not None:
                for row in result.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, score, class_id = row[:6]
                    if not math.isfinite(float(score)) or score < self.confidence:
                        continue
                    box = clipped_box((x1, y1, x2, y2), width, height)
                    index = int(class_id)
                    label = LABELS[index] if 0 <= index < len(LABELS) else "objeto"
                    if box is not None:
                        detections.append(Detection(box, label))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
        )
        for x, y, width, height in faces:
            detections.append(Detection((int(x), int(y), int(x + width), int(y + height)), "ROSTRO", "rostro"))
        return detections


def put_label(frame, text, x, y):
    import cv2

    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    (width, height), baseline = cv2.getTextSize(text, font, scale, 1)
    x = max(4, min(int(x), frame.shape[1] - width - 5))
    y = max(height + 5, min(int(y), frame.shape[0] - baseline - 5))
    cv2.rectangle(frame, (x - 3, y - height - 4),
                  (x + width + 3, y + baseline + 3), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def render(frame, detections):
    """Vista de alto contraste. Objetos: rectangulos. Rostros: elipses."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for item in detections:
        x1, y1, x2, y2 = item.box
        for shade, thickness in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            if item.kind == "rostro":
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
                cv2.ellipse(view, center, axes, 0, 0, 360, shade, thickness)
            else:
                cv2.rectangle(view, (x1, y1), (x2, y2), shade, thickness)
        put_label(view, item.label, x1, y1 - 8)
    faces = sum(d.kind == "rostro" for d in detections)
    objects = len(detections) - faces
    put_label(view, f"Objetos: {objects} | Rostros: {faces} | Q / ESC: salir", 8, view.shape[0] - 12)
    return view


def open_camera(index):
    import cv2

    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        camera = cv2.VideoCapture(index, backend)
        keep = False
        try:
            if not camera.isOpened():
                continue
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Algunos dispositivos necesitan varios fotogramas de calentamiento.
            for _ in range(10):
                ok, frame = camera.read()
                if ok and frame is not None and frame.size:
                    keep = True
                    return camera, frame
        except cv2.error:
            # Un backend de Windows puede fallar aunque el otro funcione.
            continue
        finally:
            if not keep:
                camera.release()
    raise RuntimeError(
        f"No se pudo leer la camara {index}. Cierra Zoom/Teams/Camara, revisa "
        "el permiso de camara para aplicaciones de escritorio o prueba --camara 1."
    )


def camera_loop(detector, index=0, mirror=True):
    import cv2

    if sys.platform.startswith("linux") and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
        raise RuntimeError("La webcam necesita una sesion de escritorio. Para probar sin pantalla usa --autoprueba.")
    camera, frame = open_camera(index)
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 720)
        print("Camara abierta. Pulsa Q o ESC en la ventana para salir.", flush=True)
        while True:
            height, width = frame.shape[:2]
            if width > 640:
                frame = cv2.resize(frame, (640, max(1, round(height * 640 / width))))
            if mirror:
                frame = cv2.flip(frame, 1)
            view = render(frame, detector.detect(frame))
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            ok, frame = camera.read()
            if not ok or frame is None or not frame.size:
                raise RuntimeError("La webcam dejo de enviar imagen. Revisa la conexion.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def confidence_value(value):
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 1:
        raise argparse.ArgumentTypeError("Usa un numero mayor que 0 y menor o igual a 1.")
    return number


def camera_index(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("El indice de camara debe ser 0 o mayor.")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Detector local de objetos y rostros con webcam.")
    parser.add_argument("--camara", type=camera_index, default=0, help="Indice de la webcam (0)")
    parser.add_argument("--confianza", type=confidence_value, default=0.45, help="Umbral de objetos, entre 0 y 1 (0.45)")
    parser.add_argument("--tamano", type=int, choices=(320, 416, 640), default=416, help="320: rapidez; 640: detalle (416)")
    parser.add_argument("--solo-rostros", action="store_true", help="Detectar caras sin cargar/descargar YOLO")
    parser.add_argument("--sin-espejo", action="store_true", help="Mostrar la orientacion original de la webcam")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnostico", action="store_true", help="Comprobar librerias sin abrir la camara")
    mode.add_argument("--autoprueba", action="store_true", help="Ejecutar los detectores con una imagen artificial sin abrir ventanas")
    return parser.parse_args(argv)


def diagnostics():
    import cv2
    import numpy
    import torch
    import ultralytics

    Detector(faces_only=True)
    print(f"Python: {sys.version.split()[0]} | {sys.executable}")
    for name, module in (("OpenCV", cv2), ("NumPy", numpy), ("PyTorch", torch), ("Ultralytics", ultralytics)):
        print(f"{name}: {module.__version__}")
    print("Detector de rostros: disponible")
    print(f"Modelo de objetos: {MODEL if MODEL.exists() else 'se descargara al iniciar'}")
    print("Diagnostico completado. La webcam se comprueba al iniciar la vista.")


def main(argv=None):
    args = arguments(argv)
    try:
        if args.diagnostico:
            diagnostics()
            return 0
        detector = Detector(args.confianza, args.tamano, args.solo_rostros)
        if args.autoprueba:
            import numpy as np

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)
            view = render(frame, detections)
            if view.shape != frame.shape:
                raise RuntimeError("La imagen resultante tiene dimensiones incorrectas.")
            print(f"Autoprueba correcta: {len(detections)} detecciones en imagen artificial.")
            print("Esto verifica la ejecucion; la precision y la webcam requieren una prueba real.")
        else:
            camera_loop(detector, args.camara, not args.sin_espejo)
        return 0
    except KeyboardInterrupt:
        print("\nPrograma cerrado.")
        return 130
    except ImportError as error:
        print(f"Falta una dependencia: {error}. Ejecuta INICIAR.bat para instalarla.", file=sys.stderr)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
