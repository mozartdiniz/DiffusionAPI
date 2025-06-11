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

4. Create a `.env` file in the project root:

```bash
# Create .env file
cat > .env << EOL
# Path to the directory where models are stored
MODELS_DIR=/path/to/your/models/directory

# API settings (optional)
API_HOST=0.0.0.0
API_PORT=7860
EOL
```

Then edit the `.env` file to set your actual model directory path.

5. Download the Stable Diffusion model:

```bash
# Download default model (stable-diffusion-v1-5)
python3 download_model.py

# Or download a specific model
python3 download_model.py --model your-model-name
```

6. Run the API server:

```bash
uvicorn diffusionapi.main:app --host 0.0.0.0 --port 7860
```

The API will be available at `http://localhost:7860`

### API Endpoints

- `POST /sdapi/v1/txt2img`: Generate an image from a text prompt
  - Request body:
    ```json
    {
      "prompt": "your text prompt here",
      "model": "your-model-name", // optional, defaults to "stable-diffusion-v1-5"
      "steps": 30, // optional, defaults to 30
      "cfg_scale": 7.5, // optional, defaults to 7.5
      "width": 512, // optional, defaults to 512
      "height": 512 // optional, defaults to 512
    }
    ```
  - Returns: Generated image as base64-encoded PNG

### Environment Variables

The following environment variables can be set in the `.env` file:

- `MODELS_DIR`: Path to the directory where models are stored (default: "stable_diffusion/models")
- `API_HOST`: Host address for the API server (default: "0.0.0.0")
- `API_PORT`: Port number for the API server (default: 7860)

### Available Models

You can use any model that is compatible with the Stable Diffusion pipeline. Some popular options include:

- `stable-diffusion-v1-5` (default)
- `stable-diffusion-v2-1`
- `stable-diffusion-xl-base-1.0`
- `runwayml/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-2-1`

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

4. Crie um arquivo `.env` na raiz do projeto:

```bash
# Criar arquivo .env
cat > .env << EOL
# Caminho para o diretório onde os modelos são armazenados
MODELS_DIR=/caminho/para/seu/diretorio/de/modelos

# Configurações da API (opcional)
API_HOST=0.0.0.0
API_PORT=7860
EOL
```

Depois edite o arquivo `.env` para definir o caminho real do seu diretório de modelos.

5. Baixe o modelo Stable Diffusion:

```bash
# Baixar modelo padrão (stable-diffusion-v1-5)
python3 download_model.py

# Ou baixar um modelo específico
python3 download_model.py --model seu-nome-de-modelo
```

6. Execute o servidor da API:

```bash
uvicorn diffusionapi.main:app --host 0.0.0.0 --port 7860
```

A API estará disponível em `http://localhost:7860`

### Endpoints da API

- `POST /sdapi/v1/txt2img`: Gera uma imagem a partir de um prompt de texto
  - Corpo da requisição:
    ```json
    {
      "prompt": "seu prompt de texto aqui",
      "model": "seu-nome-de-modelo", // opcional, padrão: "stable-diffusion-v1-5"
      "steps": 30, // opcional, padrão: 30
      "cfg_scale": 7.5, // opcional, padrão: 7.5
      "width": 512, // opcional, padrão: 512
      "height": 512 // opcional, padrão: 512
    }
    ```
  - Retorna: Imagem gerada em formato PNG codificado em base64

### Variáveis de Ambiente

As seguintes variáveis de ambiente podem ser definidas no arquivo `.env`:

- `MODELS_DIR`: Caminho para o diretório onde os modelos são armazenados (padrão: "stable_diffusion/models")
- `API_HOST`: Endereço do host para o servidor da API (padrão: "0.0.0.0")
- `API_PORT`: Número da porta para o servidor da API (padrão: 7860)

### Modelos Disponíveis

Você pode usar qualquer modelo compatível com o pipeline Stable Diffusion. Algumas opções populares incluem:

- `stable-diffusion-v1-5` (padrão)
- `stable-diffusion-v2-1`
- `stable-diffusion-xl-base-1.0`
- `runwayml/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-2-1`

## Troubleshooting / Solução de Problemas

### Common Issues / Problemas Comuns

1. **CUDA/MPS not available / CUDA/MPS não disponível**

   - The model will run on CPU by default if no GPU is available
   - O modelo rodará na CPU por padrão se nenhuma GPU estiver disponível

2. **Model download fails / Falha no download do modelo**

   - Check your internet connection
   - Verify you have enough disk space
   - Check if MODELS_DIR is set correctly in your .env file
   - Verify the model name is correct
   - Verifique sua conexão com a internet
   - Verifique se você tem espaço suficiente em disco
   - Verifique se MODELS_DIR está configurado corretamente no seu arquivo .env
   - Verifique se o nome do modelo está correto

3. **Port 7860 already in use / Porta 7860 já em uso**
   - Change the port number in the .env file or uvicorn command
   - Altere o número da porta no arquivo .env ou no comando uvicorn

## License / Licença

This project is licensed under the MIT License - see the LICENSE file for details.
Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.
