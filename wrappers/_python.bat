@echo off
setlocal
set "ROOT=%~dp0.."
if exist "%ROOT%\.cache\tools\ffmpeg" (
    set "PATH=%ROOT%\.cache\tools\ffmpeg;%PATH%"
)
set "PY=%ROOT%\.venv\Scripts\python.exe"
if exist "%PY%" goto run
set "PY=py"
:run
"%PY%" %*
exit /b %ERRORLEVEL%
