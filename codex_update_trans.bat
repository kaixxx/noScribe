@echo off
chcp 65001 >nul
echo Running codex via bash -ilc...
echo.

wsl bash -ilc "codex exec --sandbox workspace-write 'The '\''trans'\'' subdirectory contains translation files for this app. The english version is your reference. Open it and check for each entry if it is also contained in the other language files. Update them if needed.'"

echo.
echo ---- Command finished ----
pause
