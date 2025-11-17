@echo off
REM Algen-PP: Script de Automação para Windows
REM Detecta Python, instala dependências e executa o projeto

setlocal enabledelayedexpansion

echo ========================================
echo Algen-PP: Script de Automação
echo ========================================
echo.

REM 1. Detectar Python
echo [1/6] Detectando Python...

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    where python3 >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo ERRO: Python não encontrado!
        echo Por favor, instale Python 3.8 ou superior.
        exit /b 1
    ) else (
        set PYTHON_CMD=python3
    )
) else (
    set PYTHON_CMD=python
)

REM Verificar versão do Python
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo  Python encontrado: %PYTHON_VERSION% (%PYTHON_CMD%)
echo.

REM 2. Criar ambiente virtual se não existir
echo [2/6] Verificando ambiente virtual...

if not exist "venv" (
    echo   Criando ambiente virtual...
    %PYTHON_CMD% -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ERRO: Falha ao criar ambiente virtual
        exit /b 1
    )
    echo  Ambiente virtual criado
) else (
    echo  Ambiente virtual já existe
)

REM 3. Ativar ambiente virtual
echo [3/6] Ativando ambiente virtual...

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo  Ambiente virtual ativado
) else (
    echo ERRO: Não foi possível ativar o ambiente virtual
    exit /b 1
)

REM 4. Verificar pip
echo [4/6] Verificando pip...

%PYTHON_CMD% -m pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   pip não encontrado. Instalando...
    %PYTHON_CMD% -m ensurepip --upgrade
    if %ERRORLEVEL% NEQ 0 (
        echo ERRO: Falha ao instalar pip
        exit /b 1
    )
    echo  pip instalado
) else (
    for /f "tokens=2" %%i in ('%PYTHON_CMD% -m pip --version 2^>^&1') do set PIP_VERSION=%%i
    echo  pip encontrado: !PIP_VERSION!
)

REM Atualizar pip
echo   Atualizando pip...
%PYTHON_CMD% -m pip install --upgrade pip --quiet
if %ERRORLEVEL% NEQ 0 (
    echo AVISO: Falha ao atualizar pip, continuando...
) else (
    echo  pip atualizado
)
echo.

REM 5. Instalar dependências
echo [5/6] Instalando dependências...

if not exist "requirements.txt" (
    echo ERRO: requirements.txt não encontrado!
    exit /b 1
)

echo   Instalando pacotes de requirements.txt...
%PYTHON_CMD% -m pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Falha ao instalar dependências
    exit /b 1
)
echo  Dependências instaladas
echo.

REM 6. Verificar estrutura do projeto
echo [6/6] Verificando estrutura do projeto...

if not exist "src" (
    echo ERRO: Diretório 'src' não encontrado!
    exit /b 1
)

if not exist "src\main.py" (
    echo ERRO: src\main.py não encontrado!
    exit /b 1
)

echo  Estrutura do projeto verificada
echo.

REM 7. Executar o projeto
echo ========================================
echo Executando Algen-PP...
echo ========================================
echo.

REM Mudar para o diretório src/ pois o código usa caminhos relativos (..\images, ..\outputs)
cd src

REM Executar com -u para output não-bufferizado (ver output em tempo real)
%PYTHON_CMD% -u main.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Execução concluída com sucesso!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Execução falhou com código: %ERRORLEVEL%
    echo ========================================
    exit /b %ERRORLEVEL%
)

endlocal

