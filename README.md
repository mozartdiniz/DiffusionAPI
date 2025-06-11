# DiffusionAPI

A FastAPI-based API for Stable Diffusion image generation.

## English

### Prerequisites
- Python 3.10 or higher
- Git
- pip (Python package manager)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DiffusionAPI.git
cd DiffusionAPI
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the Stable Diffusion model:
```bash
python3 download_model.py
```

5. Run the API server:
```bash
uvicorn diffusionapi.main:app --host 0.0.0.0 --port 7860
```

The API will be available at `http://localhost:7860`

### API Endpoints

- `POST /generate`: Generate an image from a text prompt
  - Request body: `{"prompt": "your text prompt here"}`
  - Returns: Generated image as PNG

### Environment Variables
- No environment variables are required for basic usage

## Português

### Pré-requisitos
- Python 3.10 ou superior
- Git
- pip (gerenciador de pacotes Python)

### Instruções de Configuração

1. Clone o repositório:
```bash
git clone https://github.com/yourusername/DiffusionAPI.git
cd DiffusionAPI
```

2. Crie e ative um ambiente virtual:
```bash
# No macOS/Linux
python3 -m venv venv
source venv/bin/activate

# No Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Instale os pacotes necessários:
```bash
pip install -r requirements.txt
```

4. Baixe o modelo Stable Diffusion:
```bash
python3 download_model.py
```

5. Execute o servidor da API:
```bash
uvicorn diffusionapi.main:app --host 0.0.0.0 --port 7860
```

A API estará disponível em `http://localhost:7860`

### Endpoints da API

- `POST /generate`: Gera uma imagem a partir de um prompt de texto
  - Corpo da requisição: `{"prompt": "seu prompt de texto aqui"}`
  - Retorna: Imagem gerada em formato PNG

### Variáveis de Ambiente
- Nenhuma variável de ambiente é necessária para uso básico

## Troubleshooting / Solução de Problemas

### Common Issues / Problemas Comuns

1. **CUDA/MPS not available / CUDA/MPS não disponível**
   - The model will run on CPU by default if no GPU is available
   - O modelo rodará na CPU por padrão se nenhuma GPU estiver disponível

2. **Model download fails / Falha no download do modelo**
   - Check your internet connection
   - Verify you have enough disk space
   - Verifique sua conexão com a internet
   - Verifique se você tem espaço suficiente em disco

3. **Port 7860 already in use / Porta 7860 já em uso**
   - Change the port number in the uvicorn command
   - Altere o número da porta no comando uvicorn

## License / Licença

This project is licensed under the MIT License - see the LICENSE file for details.
Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.