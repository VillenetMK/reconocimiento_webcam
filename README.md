# Reconocimiento por webcam

Abre la cámara y muestra etiquetas en español para objetos como celulares,
botellas, sillas y personas. Las caras aparecen como **ROSTRO**, dentro de una
elipse. Usa YOLO11n para objetos y el detector frontal de OpenCV para rostros.

La vista usa alto contraste y formas diferentes para distinguir los resultados.
La aplicación procesa los fotogramas localmente y no guarda fotos ni video.

## Iniciar en Windows

1. Descarga este repositorio con **Code → Download ZIP** y **extrae todo el ZIP**,
   o clónalo con Git.
2. Abre la carpeta extraída y haz doble clic en **INICIAR.bat**.
3. Espera la instalación inicial y la descarga del modelo. Se abrirá la webcam.

Necesitas Python **3.10–3.14 de 64 bits**, una webcam y acceso a internet durante
la instalación y la primera descarga del modelo. El iniciador busca Python
automáticamente, incluidas las instalaciones `pythoncore-3.14-64`; evita depender
del acceso `python` de Microsoft Store.

El iniciador crea `.venv_vision` e instala las dependencias allí. Las siguientes
ejecuciones reutilizan el entorno y el modelo `models/yolo11n.pt`. No necesitas
activar un entorno desde PowerShell. La instalación puede descargar cientos de
MB por PyTorch; una vez preparada, la detección se ejecuta en tu PC.

**Q**, **Esc** o cerrar la ventana libera la cámara y termina la aplicación.

## Opciones

Desde PowerShell, dentro de la carpeta del repositorio:

```powershell
.\INICIAR.bat --camara 1
.\INICIAR.bat --tamano 320
.\INICIAR.bat --confianza 0.60
.\INICIAR.bat --solo-rostros
.\INICIAR.bat --diagnostico
```

| Opción | Uso |
| --- | --- |
| `--camara 1` | Elegir otra webcam; el valor inicial es `0`. |
| `--tamano 320` | Reducir el trabajo del detector de objetos. También admite `416` (inicial) y `640`. |
| `--confianza 0.60` | Exigir una puntuación mayor al detector de objetos; el valor inicial es `0.45`. |
| `--solo-rostros` | Detectar caras sin cargar ni descargar el modelo de objetos. |
| `--sin-espejo` | Mostrar la orientación original de la webcam. |
| `--diagnostico` | Comprobar las dependencias sin abrir la webcam. |
| `--autoprueba` | Ejecutar los detectores con una imagen artificial sin abrir ventanas. |

Si el entorno ya existe, también puedes ejecutar directamente:

```powershell
.\.venv_vision\Scripts\python.exe .\PY1.PY
```

En VS Code abre **la carpeta completa del repositorio** y usa el intérprete
`.venv_vision\Scripts\python.exe`. El archivo que ejecutas es `PY1.PY` de esta
carpeta; evita editar otra copia con el mismo nombre en Descargas.

## Linux y macOS

Con Python y una sesión de escritorio:

```bash
bash iniciar.sh
```

En Debian/Ubuntu, si `venv` o las bibliotecas gráficas no están disponibles,
instala `python3-venv`, `libgl1` y `libglib2.0-0` mediante el gestor de paquetes.
Concede acceso a la cámara cuando lo solicite tu sistema. La validación
automatizada cubre Windows y Linux; macOS necesita una prueba con hardware real.

## Solución de problemas

- **No abre la webcam:** cierra otras aplicaciones que la utilicen, revisa los
  permisos de cámara para aplicaciones de escritorio y prueba `--camara 1`.
- **La descarga del modelo falla:** comprueba la conexión y vuelve a iniciar.
  `--solo-rostros` permite probar las caras sin el modelo de objetos, una vez
  instaladas las dependencias.
- **La imagen va lenta:** prueba `--tamano 320` o `--solo-rostros`.
- **Cambiaste la ubicación del proyecto y el entorno falla:** elimina únicamente
  la carpeta generada `.venv_vision` y vuelve a abrir `INICIAR.bat`.
- **Falla la instalación:** conserva el mensaje de error de la consola. El
  iniciador deja la ventana abierta para que puedas leerlo y reintentar.
- **`Unknown word` en VS Code:** es el corrector ortográfico, no un error de Python.

## Alcance

El modelo identifica **80 categorías COCO**, no cualquier objeto posible. El
detector de caras funciona mejor con rostros de frente y buena iluminación.
Puede producir omisiones y falsas detecciones. Las personas cuentan también
como una categoría de objetos.

**Detectar un rostro no identifica a una persona por su nombre.** Esta versión
no incluye registro de identidades, reconocimiento facial por nombre ni prueba
de vida. El umbral de objetos es una puntuación del modelo, no una garantía de
acierto.

## Desarrollo y comprobaciones

```powershell
.\.venv_vision\Scripts\python.exe -m unittest discover -s tests -v
.\.venv_vision\Scripts\python.exe .\PY1.PY --autoprueba
```

Las pruebas verifican el manejo de resultados, las etiquetas, los argumentos,
la representación y la liberación de la webcam al salir o ante un fallo.
GitHub Actions comprueba dependencias y ejecución sin cámara en Windows y Linux.
La prueba con webcam y la precisión visual se deben comprobar en el equipo del
usuario; la prueba artificial no mide precisión.

Archivos principales: `webcam_app.py` contiene los detectores y la ventana;
`PY1.PY` es el punto de entrada; `scripts/bootstrap.py` prepara el entorno;
`requirements.txt` define las dependencias. Los modelos, entornos y posibles
capturas locales están excluidos de Git.

Referencias: [YOLO11](https://docs.ultralytics.com/models/yolo11/),
[clases COCO](https://docs.ultralytics.com/datasets/detect/coco/),
[OpenCV](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html).
