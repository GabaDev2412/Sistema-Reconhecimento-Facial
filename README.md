# Sistema de Reconhecimento Facial

Este projeto implementa um sistema de reconhecimento facial utilizando as bibliotecas DeepFace e OpenCV em Python. O sistema é capaz de detectar faces, registrar novos usuários e realizar o reconhecimento facial tanto em tempo real através da webcam quanto a partir de imagens estáticas.

## Sumário

- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Funcionalidades](#funcionalidades)
- [Como Usar](#como-usar)
- [Funcionamento Técnico](#funcionamento-técnico)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Algoritmos Implementados](#algoritmos-implementados)
- [Solução de Problemas](#solução-de-problemas)

## Requisitos

Para executar este sistema, você precisa de:

- Python 3.7+
- Webcam (para captura em tempo real)
- Bibliotecas listadas no arquivo `requirements.txt`

## Instalação

1. Clone este repositório:

```bash
git clone https://github.com/seu-usuario/sistema-reconhecimento-facial.git
cd sistema-reconhecimento-facial
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o script principal:

```bash
python main.py
```

## Funcionalidades

O sistema oferece as seguintes funcionalidades:

1. **Reconhecimento Facial em Tempo Real**:

   - Detecta e identifica rostos em um fluxo de vídeo através da webcam
   - Exibe identificação e nível de confiança para cada rosto
   - Permite registrar novas faces a partir da tela de reconhecimento

2. **Reconhecimento a partir de Imagem**:

   - Processa uma imagem estática para identificar faces
   - Marca as faces identificadas com nome e percentual de confiança

3. **Cadastro de Novas Faces**:

   - Sistema de captura em movimento circular para registrar diferentes ângulos do rosto
   - Armazena múltiplas imagens da mesma pessoa para melhorar a precisão do reconhecimento
   - Gera e armazena encodings faciais para acelerar o reconhecimento futuro

4. **Menu Principal**:
   - Interface de linha de comando com opções claras para escolher a função desejada

## Como Usar

### Menu Principal

Ao iniciar o programa, você verá o menu principal com as seguintes opções:

```
===== SISTEMA DE RECONHECIMENTO FACIAL =====
1. Reconhecimento em tempo real
2. Reconhecer a partir de imagem de teste
3. Registrar nova face
4. Sair
```

### Registrando uma Nova Face

1. Selecione a opção 3 no menu principal (ou pressione 'r' durante o reconhecimento em tempo real)
2. Digite o nome da pessoa a ser registrada
3. Siga as instruções na tela:
   - Posicione seu rosto no centro da tela e pressione ESPAÇO para iniciar
   - Mova seu rosto em um movimento circular AMPLO e LENTO
   - Incline a cabeça em diferentes ângulos durante o movimento
   - Uma barra de progresso mostrará o avanço da captura
   - ESC para cancelar a captura

O sistema capturará automaticamente diferentes ângulos do seu rosto à medida que você se move, armazenando imagens e encodings para uso posterior.

### Reconhecimento em Tempo Real

1. Selecione a opção 1 no menu principal
2. O sistema iniciará a webcam e começará a detectar e reconhecer rostos
3. Pressione 'r' para registrar um novo rosto (útil para rostos não reconhecidos)
4. Pressione 'q' para sair e voltar ao menu principal

### Reconhecimento a partir de Imagem

1. Selecione a opção 2 no menu principal
2. O sistema processará a imagem definida em `test_image_path` (por padrão "test_image.jpg")
3. As faces reconhecidas serão marcadas com nomes e níveis de confiança
4. Se não houver faces registradas ou imagem de teste disponível, o sistema oferecerá opções para capturar ou registrar novas faces

## Funcionamento Técnico

### Detecção Facial

O sistema utiliza o detector da biblioteca OpenCV via DeepFace para identificar faces em imagens:

- Processa cada quadro/imagem para detectar áreas faciais
- Aplica melhorias para aumentar a qualidade da detecção (equalização de histograma, filtro bilateral)
- Extrai a região facial para processamento adicional

### Encoding Facial

Para cada face detectada:

- Gera um vetor de características (embedding) usando o modelo VGG-Face
- Estes embeddings são representações numéricas das características faciais (512 dimensões)
- Os embeddings são armazenados para comparação futura

### Comparação e Reconhecimento

Para reconhecer uma face:

1. O sistema extrai o embedding da face atual
2. Calcula a distância de cosseno entre este embedding e os embeddings armazenados
3. Aplica um threshold para determinar se há uma correspondência
4. Exibe o nome da pessoa e o nível de confiança para correspondências encontradas

### Cadastro com Movimento Circular

O sistema implementa um método avançado de captura facial:

- Orienta o usuário a mover o rosto em padrão circular
- Monitora a diversidade entre frames capturados
- Avalia a qualidade de cada imagem (contraste, brilho)
- Captura automaticamente frames quando detecta posições adequadas
- Mantém múltiplos ângulos para melhorar o reconhecimento

## Estrutura do Projeto

O sistema cria e utiliza os seguintes diretórios:

- **known_faces/**: Armazena imagens de faces registradas, organizadas em subpastas por pessoa
- **face_encodings/**: Armazena os embeddings faciais extraídos de cada pessoa registrada

Arquivos principais:

- **main.py**: Script principal que contém toda a lógica do sistema
- **test_image.jpg**: Imagem para teste de reconhecimento (criada durante o uso)
- **requirements.txt**: Lista de dependências do projeto

## Algoritmos Implementados

### Cálculo de Distância de Cosseno

Utilizado para comparar a similaridade entre vetores de características faciais:

```python
def calculate_cosine_distance(vector_a, vector_b):
    # Normalizar os vetores
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # Calcular similaridade de cosseno
    dot_product = np.dot(vector_a, vector_b)
    similarity = dot_product / (norm_a * norm_b)

    # Converter similaridade para distância
    distance = 1.0 - similarity
    return max(0.0, min(distance, 1.0))
```

### Avaliação de Qualidade de Imagem

Avalia a qualidade de uma imagem facial baseada em contraste e brilho:

```python
def evaluate_face_quality(face_img):
    # Converter para escala de cinza
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    # Calcular contraste usando desvio padrão
    std_dev = np.std(gray)

    # Calcular brilho médio
    mean_brightness = np.mean(gray)

    # Verificar faixa de brilho adequada
    brightness_score = 1.0 - 2.0 * abs(mean_brightness - 128) / 255

    # Combinar métricas
    quality_score = 0.7 * (std_dev / 128) + 0.3 * brightness_score
    return min(1.0, max(0.0, quality_score))
```

### Cálculo de Diversidade entre Frames

Calcula o quão diferente é um novo embedding facial dos já capturados:

```python
def calculate_frame_diversity(new_embedding, existing_embeddings, min_distance=0.15):
    if not existing_embeddings:
        return 1.0  # Se é o primeiro, é totalmente diverso

    # Calcular distâncias em relação aos embeddings existentes
    distances = []
    for existing in existing_embeddings:
        dist = calculate_cosine_distance(new_embedding, existing)
        distances.append(dist)

    # Média das distâncias
    avg_distance = np.mean(distances)

    # Normalizar para pontuação de diversidade
    if avg_distance < min_distance:
        return 0.0  # Muito similar aos existentes
    elif avg_distance > 0.6:
        return.0  # Muito diferente (possivelmente outra pessoa)
    else:
        return min(1.0, (avg_distance - min_distance) / (0.5 - min_distance))
```

## Solução de Problemas

**Webcam não disponível:**

- Verifique se outros programas estão usando a webcam
- Aguarde alguns segundos e tente novamente
- O sistema oferece alternativa para selecionar uma imagem do computador

**Baixa qualidade de reconhecimento:**

- Registre novamente a face usando o método de movimento circular
- Certifique-se de capturar diferentes ângulos durante o registro
- Verifique se há boa iluminação, preferencialmente frontal e uniforme

**Faces não detectadas:**

- Verifique a iluminação do ambiente
- Posicione o rosto mais próximo e centralizado na imagem
- Evite oclusões como óculos escuros, máscaras ou chapéus

**Falsos positivos:**

- O sistema utiliza thresholds de confiança, que podem ser ajustados no código:
  - Diminua o valor de `threshold` na função `process_recognition_result` para tornar o reconhecimento mais rigoroso
  - Aumente o valor de `smoothed_confidence` na função `realtime_recognition` para exigir maior confiança
