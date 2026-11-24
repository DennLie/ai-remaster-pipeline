@echo off
setlocal
if exist "%~dp0..\venv\Scripts\activate.bat" (
    call "%~dp0..\venv\Scripts\activate.bat"
    python "%~dp0..\scripts\qc_color_consistency.py" %*
    exit /b %ERRORLEVEL%
)
if exist "%~dp0..\.venv\Scripts\activate.bat" (
    call "%~dp0..\.venv\Scripts\activate.bat"
    python "%~dp0..\scripts\qc_color_consistency.py" %*
    exit /b %ERRORLEVEL%
)
call "%~dp0_python.bat" "%~dp0..\scripts\qc_color_consistency.py" %*
exit /b %ERRORLEVEL%
