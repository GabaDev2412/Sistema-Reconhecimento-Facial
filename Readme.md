# Sistema de Reconhecimento Facial

Este projeto utiliza técnicas de reconhecimento facial para identificar pessoas a partir de uma webcam. Ele carrega um conjunto de imagens conhecidas e compara rostos detectados em tempo real para identificar indivíduos.

## Funcionalidades

- **Carregamento de Imagens**: Carrega imagens de um diretório especificado e extrai as codificações faciais para reconhecimento.
- **Detecção de Rostos**: Utiliza a webcam para capturar imagens e detectar rostos em tempo real.
- **Reconhecimento Facial**: Compara rostos detectados com as imagens conhecidas para identificar indivíduos.
- **Exibição de Resultados**: Exibe a imagem capturada com o nome da pessoa identificada ou uma mensagem indicando que a pessoa não é reconhecida.

## Dependências

- `opencv-python`: Biblioteca para processamento de imagens e vídeo.
- `face_recognition`: Biblioteca para reconhecimento facial.
- `cvzone`: Biblioteca para detecção de rostos com uma interface simplificada.
- `os`: Biblioteca padrão para manipulação de caminhos e arquivos.

## Instalação

Para instalar as dependências necessárias, você pode usar o `pip`:

```bash
pip install opencv-python face_recognition cvzone
```

## Uso

1. **Prepare o Dataset:** Coloque as imagens de rosto em um diretório específico. As imagens devem estar no formato `.jpg` ou `.png`. O nome do arquivo será usado como identificação da pessoa.

2. **Configure o Caminho do Dataset:** Modifique o caminho `dataset_folder` no código para apontar para o diretório onde suas imagens estão localizadas.

```python
dataset_folder = "Seu/Diretório/Aqui"
```

3. **Execute o Código:** Execute o script para iniciar a detecção e reconhecimento facial. A janela da webcam será aberta e exibirá os resultados em tempo real.

```bash
python seu_script.py
```

4. **Interaja com a Aplicação:** A aplicação exibirá o nome da pessoa reconhecida ou uma mensagem indicando que a pessoa não é conhecida. Pressione `Esc` para sair da aplicação.

## Exemplos de uso

- **Imagem Reconhecida:** Se um rosto for reconhecido, o nome da pessoa será exibido sobre a imagem capturada pela webcam.
- **Imagem Não Reconhecida:** Se um rosto não for reconhecido, será exibida uma mensagem indicando que a pessoa não é conhecida.