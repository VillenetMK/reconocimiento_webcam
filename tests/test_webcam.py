import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import webcam_app as app


class ArgumentsTests(unittest.TestCase):
    def test_rejects_invalid_confidence(self):
        for value in ("nan", "inf", "-1", "0", "1.1"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    app.arguments(["--confianza", value])
                self.assertEqual(error.exception.code, 2)

    def test_camera_and_resolution_validation(self):
        for args in (["--camara", "-1"], ["--tamano", "17"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    app.arguments(args)
        parsed = app.arguments(["--camara", "2", "--tamano", "320", "--solo-rostros"])
        self.assertEqual((parsed.camara, parsed.tamano, parsed.solo_rostros), (2, 320, True))


class DetectionTests(unittest.TestCase):
    def test_model_class_mapping(self):
        self.assertEqual(len(app.LABELS), 80)
        for number, name in ((0, "persona"), (15, "gato"), (39, "botella"),
                             (56, "silla"), (67, "celular"), (79, "cepillo de dientes")):
            self.assertEqual(app.LABELS[number], name)

    def test_outside_and_invalid_boxes(self):
        self.assertEqual(app.clipped_box((-10, -9, 700, 500), 640, 480), (0, 0, 639, 479))
        self.assertIsNone(app.clipped_box((20, 20, 10, 10), 640, 480))
        self.assertIsNone(app.clipped_box((0, 0, float("nan"), 20), 640, 480))

    def test_objects_and_faces_share_clean_input(self):
        frame = np.full((200, 300, 3), 70, dtype=np.uint8)
        before = frame.copy()
        boxes = MagicMock()
        boxes.data.cpu.return_value.numpy.return_value = np.array([
            [30, 40, 100, 120, 0.9, 67],   # celular
            [10, 20, 50, 60, 0.1, 39],    # descartar baja puntuacion
            [200, 100, 600, 500, 0.8, 0], # recortar a los limites
        ])
        detector = app.Detector(faces_only=True)
        detector.object_model = MagicMock()
        detector.object_model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        detector.face_detector = MagicMock()
        detector.face_detector.detectMultiScale.return_value = [(40, 60, 70, 80)]
        result = detector.detect(frame)
        self.assertEqual([item.label for item in result], ["celular", "persona", "ROSTRO"])
        self.assertEqual(result[1].box, (200, 100, 299, 199))
        self.assertEqual(result[2].box, (40, 60, 110, 140))
        np.testing.assert_array_equal(frame, before)
        self.assertIs(detector.object_model.predict.call_args.kwargs["source"], frame)

    def test_real_face_detector_handles_empty_image(self):
        detector = app.Detector(faces_only=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(detector.detect(frame), [])

    def test_render_does_not_change_source(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [app.Detection((0, 0, 60, 60), "celular"),
                      app.Detection((550, 400, 639, 479), "ROSTRO", "rostro")]
        view = app.render(frame, detections)
        self.assertEqual(view.shape, frame.shape)
        self.assertEqual(view.dtype, frame.dtype)
        self.assertFalse(np.any(frame))
        self.assertTrue(np.any(view))


class CameraTests(unittest.TestCase):
    def test_unavailable_camera_is_released(self):
        cameras = []

        def create(*args):
            camera = MagicMock()
            camera.isOpened.return_value = False
            cameras.append(camera)
            return camera

        with patch.object(cv2, "VideoCapture", side_effect=create):
            with self.assertRaisesRegex(RuntimeError, "No se pudo leer"):
                app.open_camera(0)
        self.assertGreater(len(cameras), 0)
        for camera in cameras:
            camera.release.assert_called_once()

    def test_warmup_allows_delayed_first_frame(self):
        camera = MagicMock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera.read.side_effect = [(False, None), (True, frame)]
        with patch.object(cv2, "VideoCapture", return_value=camera):
            actual_camera, actual_frame = app.open_camera(0)
        self.assertIs(actual_camera, camera)
        self.assertIs(actual_frame, frame)
        camera.release.assert_not_called()
        camera.release()

    def run_loop(self, failure=None):
        camera = MagicMock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = MagicMock()
        detector.detect.return_value = []
        detector.detect.side_effect = failure
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"DISPLAY": ":test"}))
            stack.enter_context(patch.object(app, "open_camera", return_value=(camera, frame)))
            for method in ("namedWindow", "resizeWindow", "imshow"):
                stack.enter_context(patch.object(cv2, method))
            stack.enter_context(patch.object(cv2, "waitKey", return_value=ord("q")))
            destroy = stack.enter_context(patch.object(cv2, "destroyAllWindows"))
            if failure:
                with self.assertRaisesRegex(RuntimeError, "fallo de inferencia"):
                    app.camera_loop(detector)
            else:
                app.camera_loop(detector)
            camera.release.assert_called_once()
            destroy.assert_called_once()

    def test_quit_releases_camera(self):
        self.run_loop()

    def test_inference_error_releases_camera(self):
        self.run_loop(RuntimeError("fallo de inferencia"))


if __name__ == "__main__":
    unittest.main()
