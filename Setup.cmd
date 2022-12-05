@ECHO OFF

ECHO Initializing...
ECHO.

git submodule update --remote --progress --init --recursive --force

ECHO.
ECHO Finished
PAUSE