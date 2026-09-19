@echo off
setlocal
cd /d "%~dp0"
title Reconocimiento de objetos y rostros

if exist ".venv_vision\Scripts\python.exe" (
    ".venv_vision\Scripts\python.exe" "scripts\bootstrap.py" %*
    goto finish
)

rem Busca primero instalaciones reales; evita el alias de Microsoft Store.
for %%V in (3.14 3.13 3.12 3.11 3.10) do (
    if exist "%LOCALAPPDATA%\Python\pythoncore-%%V-64\python.exe" (
        set "WEBCAM_PYTHON=%LOCALAPPDATA%\Python\pythoncore-%%V-64\python.exe"
        goto found
    )
)
for %%V in (314 313 312 311 310) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
        set "WEBCAM_PYTHON=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
        goto found
    )
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 "scripts\bootstrap.py" %*
    goto finish
)
python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>nul
if not errorlevel 1 (
    python "scripts\bootstrap.py" %*
    goto finish
)
echo No se encontro Python. Instala Python 3.10-3.14 de 64 bits desde python.org.
if not "%WEBCAM_NO_PAUSE%"=="1" pause
exit /b 1

:found
"%WEBCAM_PYTHON%" "scripts\bootstrap.py" %*

:finish
set "WEBCAM_EXIT=%ERRORLEVEL%"
if not "%WEBCAM_NO_PAUSE%"=="1" pause
exit /b %WEBCAM_EXIT%
