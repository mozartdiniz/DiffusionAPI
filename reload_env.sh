#!/bin/bash

set -e  # Para o script se algo falhar

VENV_DIR="venv"

echo "🔁 Limpando ambiente anterior (se existir)..."
rm -rf $VENV_DIR

echo "🐍 Criando novo ambiente virtual em '$VENV_DIR'..."
python3 -m venv $VENV_DIR

echo "✅ Ambiente criado. Ativando..."
source $VENV_DIR/bin/activate

echo "📦 Instalando dependências do requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Ambiente configurado com sucesso!"
echo "🧪 Para ativar o ambiente manualmente depois, use:"
echo "   source $VENV_DIR/bin/activate"