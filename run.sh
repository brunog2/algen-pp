#!/bin/bash

# Algen-PP: Script de Automação
# Detecta Python, instala dependências e executa o projeto

set -e  # Para na primeira erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Algen-PP: Script de Automação${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 1. Detectar Python
echo -e "${YELLOW}[1/6] Detectando Python...${NC}"

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}ERRO: Python não encontrado!${NC}"
    echo -e "${RED}Por favor, instale Python 3.8 ou superior.${NC}"
    exit 1
fi

# Verificar versão do Python
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}ERRO: Python 3.8 ou superior é necessário!${NC}"
    echo -e "${RED}Versão encontrada: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN} Python encontrado: $PYTHON_VERSION ($PYTHON_CMD)${NC}\n"

# 2. Criar ambiente virtual se não existir
echo -e "${YELLOW}[2/6] Verificando ambiente virtual...${NC}"

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}  Criando ambiente virtual...${NC}"
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN} Ambiente virtual criado${NC}"
else
    echo -e "${GREEN} Ambiente virtual já existe${NC}"
fi

# 3. Ativar ambiente virtual
echo -e "${YELLOW}[3/6] Ativando ambiente virtual...${NC}"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo -e "${GREEN} Ambiente virtual ativado${NC}"
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
    echo -e "${GREEN} Ambiente virtual ativado (Windows)${NC}"
else
    echo -e "${RED}ERRO: Não foi possível ativar o ambiente virtual${NC}"
    exit 1
fi

# Verificar se pip está instalado
echo -e "${YELLOW}[4/6] Verificando pip...${NC}"

# Usar python -m pip que é mais confiável
# Primeiro tenta garantir que pip está instalado
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo -e "${YELLOW}  pip não encontrado. Instalando...${NC}"
    $PYTHON_CMD -m ensurepip --upgrade
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERRO: Falha ao instalar pip${NC}"
        exit 1
    fi
    echo -e "${GREEN} pip instalado${NC}"
else
    PIP_VERSION=$($PYTHON_CMD -m pip --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN} pip encontrado: $PIP_VERSION${NC}"
fi

# Atualizar pip usando python -m pip
echo -e "${YELLOW}  Atualizando pip...${NC}"
$PYTHON_CMD -m pip install --upgrade pip --quiet
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}AVISO: Falha ao atualizar pip, continuando...${NC}"
else
    echo -e "${GREEN} pip atualizado${NC}"
fi
echo ""

# 4. Instalar dependências
echo -e "${YELLOW}[5/6] Instalando dependências...${NC}"

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}ERRO: requirements.txt não encontrado!${NC}"
    exit 1
fi

echo -e "${YELLOW}  Instalando pacotes de requirements.txt...${NC}"
$PYTHON_CMD -m pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo -e "${RED}ERRO: Falha ao instalar dependências${NC}"
    exit 1
fi
echo -e "${GREEN} Dependências instaladas${NC}\n"

# 5. Verificar se o diretório src existe
echo -e "${YELLOW}[6/6] Verificando estrutura do projeto...${NC}"

if [ ! -d "src" ]; then
    echo -e "${RED}ERRO: Diretório 'src' não encontrado!${NC}"
    exit 1
fi

if [ ! -f "src/main.py" ]; then
    echo -e "${RED}ERRO: src/main.py não encontrado!${NC}"
    exit 1
fi

echo -e "${GREEN} Estrutura do projeto verificada${NC}\n"

# 6. Executar o projeto
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Executando Algen-PP...${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Mudar para o diretório do projeto (garantir que estamos na raiz)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Mudar para o diretório src/ pois o código usa caminhos relativos (../images, ../outputs)
cd src

# Executar o main.py com output não-bufferizado (-u) para ver output em tempo real
$PYTHON_CMD -u main.py

# Verificar código de saída
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Execução concluída com sucesso!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}Execução falhou com código: $EXIT_CODE${NC}"
    echo -e "${RED}========================================${NC}"
    exit $EXIT_CODE
fi

