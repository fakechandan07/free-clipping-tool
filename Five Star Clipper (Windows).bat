@echo off
rem Double-click me (Windows) to open Five Star Clipper. The first time, it sets everything up.
cd /d "%~dp0tool"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tool\windows\open-clipper.ps1"
if errorlevel 1 pause
