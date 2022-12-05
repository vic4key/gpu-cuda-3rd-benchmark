@ECHO OFF

ECHO Initializing...
ECHO.

git submodule update --remote --progress --init --recursive --force
rem git submodule update --progress --init --recursive --force
rem git submodule update --init --recursive
rem git submodule foreach --recursive git checkout master
rem git submodule foreach --recursive git pull

ECHO.
ECHO Finished
PAUSE