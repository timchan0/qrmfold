@ECHO OFF

pushd %~dp0

set SOURCEDIR=source
set BUILDDIR=build
set APIDOCDIR=%SOURCEDIR%\api
set PACKAGEDIR=..\qrmfold

if "%1"=="" goto help

python -m sphinx.ext.apidoc --force --separate --module-first -o "%APIDOCDIR%" "%PACKAGEDIR%" "_*"
python -m sphinx -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
python -m sphinx -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
