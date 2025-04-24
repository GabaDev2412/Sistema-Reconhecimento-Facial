import cv2
from deepface import DeepFace
import os
import glob
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
import threading
import pickle
from datetime import datetime

# Definir caminhos
known_faces_dir = "known_faces"  # Diretório com subpastas para cada pessoa
test_image_path = "test_image.jpg"  # Imagem para teste
encodings_dir = "face_encodings"  # Diretório para armazenar os encodings

# Criar diretórios necessários se não existirem
for directory in [known_faces_dir, encodings_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Diretório {directory} criado.")

# Variável global para controle de acesso à webcam
webcam_in_use = False
last_webcam_release_time = 0

# Configurações para captura em movimento circular - VALORES AJUSTADOS
NUM_FRAMES_TO_CAPTURE = 8  # Reduzido para facilitar a conclusão
MIN_FACE_SIZE = 80  # Reduzido para aceitar rostos um pouco menores
MIN_FACE_QUALITY_THRESHOLD = 0.3  # Reduzido para aceitar imagens de menor qualidade
MIN_FACE_CONFIDENCE = 0.7  # Reduzido para aumentar aceitação de detecções
MIN_DIVERSITY_THRESHOLD = 0.15  # Novo: threshold mínimo para diversidade entre frames

# Função utilitária para calcular a distância de cosseno entre dois vetores
def calculate_cosine_distance(vector_a, vector_b):
    """
    Calcula a distância de cosseno entre dois vetores.
    
    A distância é calculada como: 1 - (dot_product(a, b) / (norm(a) * norm(b)))
    Valores próximos a 0 indicam alta similaridade, próximos a 1 indicam baixa similaridade.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Os vetores devem ter o mesmo tamanho")
        
    # Converter para arrays numpy se já não forem
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    
    # Normalizar os vetores
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    # Evitar divisão por zero
    if norm_a == 0 or norm_b == 0:
        return 1.0  # Máxima distância se algum vetor for zero
    
    # Calcular similaridade de cosseno (produto escalar dos vetores normalizados)
    dot_product = np.dot(vector_a, vector_b)
    similarity = dot_product / (norm_a * norm_b)
    
    # Converter similaridade para distância
    distance = 1.0 - similarity
    
    # Garantir que a distância esteja entre 0 e 1
    return max(0.0, min(distance, 1.0))

# Função para avaliar a qualidade da imagem
def evaluate_face_quality(face_img):
    """Avalia a qualidade da imagem facial baseada em claridade e contraste"""
    # Converter para escala de cinza se não estiver
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
        
    # Calcular contraste usando desvio padrão
    std_dev = np.std(gray)
    
    # Calcular brilho médio
    mean_brightness = np.mean(gray)
    
    # Verificar se o brilho está em uma faixa adequada (nem muito escuro nem muito claro)
    brightness_score = 1.0 - 2.0 * abs(mean_brightness - 128) / 255
    
    # Combinar métricas (ajustar pesos conforme necessário)
    quality_score = 0.7 * (std_dev / 128) + 0.3 * brightness_score
    
    return min(1.0, max(0.0, quality_score))

# Função para calcular o quão diferente é um frame de outros já capturados
def calculate_frame_diversity(new_embedding, existing_embeddings, min_distance=0.15):
    """Calcula o quão diferente é um embedding facial de outros já coletados"""
    if not existing_embeddings:
        return 1.0  # Se é o primeiro, é totalmente diverso
    
    distances = []
    for existing in existing_embeddings:
        dist = calculate_cosine_distance(new_embedding, existing)
        distances.append(dist)
    
    avg_distance = np.mean(distances)
    
    # Retorna uma pontuação de diversidade normalizada
    # Se avg_distance < min_distance, a diversidade é baixa
    # Se avg_distance > 0.5, a diversidade é muito alta (possivelmente pessoa diferente)
    if avg_distance < min_distance:
        return 0.0  # Muito similar aos existentes
    elif avg_distance > 0.6:  # Aumentado para permitir maior variabilidade
        return 0.0  # Muito diferente (possivelmente outra pessoa)
    else:
        # Normalizar entre 0 e 1 na faixa útil
        return min(1.0, (avg_distance - min_distance) / (0.5 - min_distance))

def check_webcam_available():
    """Verifica se uma webcam está disponível"""
    global webcam_in_use, last_webcam_release_time
    
    # Se a webcam estiver em uso, espere
    if webcam_in_use:
        print("Webcam já está em uso. Aguarde...")
        return False
    
    # Verificar se passou tempo suficiente desde o último fechamento
    current_time = time.time()
    if current_time - last_webcam_release_time < 2.0:  # 2 segundos de espera
        wait_time = 2.0 - (current_time - last_webcam_release_time)
        print(f"Aguardando {wait_time:.1f} segundos para inicializar a webcam...")
        time.sleep(wait_time)
    
    # Tentar abrir a webcam
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.release()
        return True
    return False

def capture_from_webcam():
    """Função para capturar imagem da webcam"""
    global webcam_in_use, last_webcam_release_time
    
    if not check_webcam_available():
        print("Webcam não disponível. Verifique a conexão ou tente novamente em alguns segundos.")
        user_input = input("Deseja selecionar uma imagem do computador? (s/n): ").lower()
        if user_input == 's':
            return select_image_file()
        return None
    
    webcam_in_use = True
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        webcam_in_use = False
        print("Não foi possível abrir a webcam. Verifique se outro programa está usando-a.")
        user_input = input("Deseja selecionar uma imagem do computador? (s/n): ").lower()
        if user_input == 's':
            return select_image_file()
        return None
    
    print("Pressione 'c' para capturar ou 'q' para sair")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cv2.putText(frame, "Pressione 'c' para capturar ou 'q' para sair", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Webcam", frame)
            
            key = cv2.waitKey(1)
            if key == ord('c'):  # Capturar imagem
                cap.release()
                last_webcam_release_time = time.time()
                cv2.destroyAllWindows()
                webcam_in_use = False
                return frame
            elif key == ord('q'):  # Sair
                break
    finally:
        cap.release()
        last_webcam_release_time = time.time()
        cv2.destroyAllWindows()
        webcam_in_use = False
        
    return None

def select_image_file():
    """Seleciona uma imagem do computador usando um seletor de arquivos"""
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal
    
    file_path = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=(
            ("Imagens", "*.jpg *.jpeg *.png *.bmp"),
            ("Todos os arquivos", "*.*")
        )
    )
    
    if not file_path:
        print("Nenhum arquivo selecionado.")
        return None
    
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Não foi possível carregar a imagem: {file_path}")
            return None
        return img
    except Exception as e:
        print(f"Erro ao carregar a imagem: {e}")
        return None

def extract_face_from_frame(frame):
    """Extrai a face de um frame e retorna a imagem da face"""
    try:
        # Extrair faces com a API atual do DeepFace
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="opencv",
            enforce_detection=False
        )
        
        if faces and len(faces) > 0:
            face_obj = faces[0]
            facial_area = face_obj['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            # Recortar a face do frame
            face_img = frame[y:y+h, x:x+w].copy()
            return face_img, facial_area, True
        else:
            return None, None, False
    except Exception as e:
        print(f"Erro ao extrair face: {e}")
        return None, None, False

def register_face_continuous_motion(initial_frame=None):
    """Versão melhorada para registrar face com movimento circular contínuo"""
    print("Registrando nova face com movimento contínuo...")
    
    # Pedir nome da pessoa
    name = input("Digite o nome da pessoa: ").strip()
    if not name:
        print("Nome inválido. Registro cancelado.")
        return False
    
    # Criar pasta para a pessoa se não existir
    person_dir = os.path.join(known_faces_dir, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    # Lista para armazenar os encodings faciais e frames capturados
    face_encodings = []
    captured_images = []
    
    # Verificar disponibilidade da webcam
    if not check_webcam_available():
        print("Webcam não disponível para captura contínua.")
        return False
    
    # Iniciar a webcam
    webcam_in_use = True
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        webcam_in_use = False
        print("Não foi possível abrir a webcam para captura contínua.")
        return False
    
    print("\nInstruções para captura facial contínua (MELHORADA):")
    print("1. Posicione seu rosto centralizado na webcam")
    print("2. Quando estiver pronto, pressione ESPAÇO para iniciar")
    print("3. Mova seu rosto em um movimento circular AMPLO e LENTO")
    print("4. INCLINE a cabeça em diferentes ângulos durante o movimento")
    print("5. Uma barra de progresso mostrará o avanço da captura")
    print("6. Pressione ESC a qualquer momento para cancelar")
    
    # Variáveis para controle da captura
    capturing_active = False
    frames_captured = 0
    last_capture_time = 0
    capture_interval = 0.3  # Reduzido para capturar frames com mais frequência
    progress = 0
    start_time = None
    last_pos = None
    movement_score = 0  # Para medir quantidade de movimento entre frames
    face_candidates = []  # Lista para pré-selecionar potenciais capturas
    auto_capture_mode = False  # Ativa captura automática após timeout parcial
    
    # Usar o último frame se fornecido
    if initial_frame is not None:
        last_frame = initial_frame.copy()
    else:
        last_frame = None
    
    # Guias visuais para o movimento circular
    guide_points = []
    guide_radius = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Fazer uma cópia do frame para trabalhar
            display_frame = frame.copy()
            height, width = display_frame.shape[:2]
            
            # Configurar pontos de guia para o movimento circular se ainda não feito
            if not guide_points:
                center_x, center_y = width // 2, height // 2
                guide_radius = min(width, height) // 3  # Círculo maior para movimento mais amplo
                num_points = 8
                for i in range(num_points):
                    angle = 2 * np.pi * i / num_points
                    x = int(center_x + guide_radius * np.cos(angle))
                    y = int(center_y + guide_radius * np.sin(angle))
                    guide_points.append((x, y))
            
            # Se ainda não estiver em modo de captura ativa
            if not capturing_active:
                cv2.putText(display_frame, "Posicione seu rosto e pressione ESPACO para iniciar", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "ESC para cancelar", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mostrar guia circular para o movimento
                center_x, center_y = width // 2, height // 2
                cv2.circle(display_frame, (center_x, center_y), guide_radius, (0, 255, 255), 2)
                
                # Mostrar retângulo guia para posicionamento inicial
                rect_size = min(width, height) // 3  # Retângulo maior
                cv2.rectangle(display_frame, 
                             (center_x - rect_size//2, center_y - rect_size//2),
                             (center_x + rect_size//2, center_y + rect_size//2),
                             (255, 255, 0), 2)
            else:
                # Modo de captura ativa
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = max(0, 25 - elapsed)  # Aumentado para 25 segundos
                
                # Ativar modo de captura automática após 10 segundos se poucos frames capturados
                if elapsed > 10 and frames_captured < NUM_FRAMES_TO_CAPTURE * 0.4 and not auto_capture_mode:
                    auto_capture_mode = True
                    print("Ativando modo de captura automática! Continue movendo o rosto...")
                
                # Calcular e mostrar progresso
                progress = min(1.0, frames_captured / NUM_FRAMES_TO_CAPTURE)
                progress_width = int(width * progress)
                cv2.rectangle(display_frame, (0, height-30), (progress_width, height-10), (0, 255, 0), -1)
                cv2.rectangle(display_frame, (0, height-30), (width, height-10), (255, 255, 255), 2)
                
                # Instruções durante captura
                cv2.putText(display_frame, f"Capturando: {frames_captured}/{NUM_FRAMES_TO_CAPTURE} frames", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Mova seu rosto em CÍRCULO AMPLO. Tempo: {remaining:.1f}s", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if auto_capture_mode:
                    cv2.putText(display_frame, "MODO AUTOMÁTICO ATIVADO", 
                              (width//2-150, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                # Desenhar círculo guia
                center_x, center_y = width // 2, height // 2
                cv2.circle(display_frame, (center_x, center_y), guide_radius, (0, 255, 255), 2)
                
                # Desenhar pontos guia para o movimento circular
                for point_idx, (x, y) in enumerate(guide_points):
                    # Destacar o ponto atual sugerido para o movimento
                    highlight = (point_idx == int((elapsed * 2) % len(guide_points)))
                    color = (0, 0, 255) if highlight else (0, 165, 255)
                    size = 10 if highlight else 5
                    cv2.circle(display_frame, (x, y), size, color, -1)
                
                # Tentar capturar em intervalos regulares ou quando há movimento suficiente
                if ((current_time - last_capture_time >= capture_interval and frames_captured < NUM_FRAMES_TO_CAPTURE) or
                    auto_capture_mode and current_time - last_capture_time >= 0.7):  # Mais frequente no modo automático
                    
                    # Salvar o frame temporariamente para processamento
                    temp_img_path = f"temp_capture_{frames_captured}.jpg"
                    cv2.imwrite(temp_img_path, frame)
                    
                    try:
                        # Detectar faces no frame
                        face_objs = DeepFace.extract_faces(
                            img_path=temp_img_path,
                            detector_backend="opencv",
                            enforce_detection=False
                        )
                        
                        if face_objs and len(face_objs) > 0:
                            face_obj = face_objs[0]
                            facial_area = face_obj['facial_area']
                            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                            
                            # Verificar tamanho do rosto
                            size_ok = w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE
                            
                            if size_ok:
                                # Recortar a face do frame
                                face_img = frame[y:y+h, x:x+w].copy()
                                
                                # Verificar qualidade da imagem
                                quality = evaluate_face_quality(face_img)
                                
                                # Extrair embedding facial
                                if quality >= MIN_FACE_QUALITY_THRESHOLD or auto_capture_mode:
                                    try:
                                        embedding_objs = DeepFace.represent(
                                            img_path=temp_img_path,
                                            model_name="VGG-Face",
                                            enforce_detection=False
                                        )
                                        
                                        if embedding_objs and len(embedding_objs) > 0:
                                            new_embedding = embedding_objs[0]['embedding']
                                            
                                            # Registrar a posição atual do rosto
                                            current_pos = (x + w//2, y + h//2)
                                            
                                            # Calcular movimento se houver posição anterior
                                            if last_pos:
                                                movement = np.sqrt((current_pos[0] - last_pos[0])**2 + 
                                                                 (current_pos[1] - last_pos[1])**2)
                                                movement_score = min(1.0, movement / 30.0)  # normalizar para [0,1]
                                            else:
                                                movement_score = 0
                                            
                                            last_pos = current_pos
                                            
                                            # Verificar diversidade do frame em relação aos já capturados
                                            diversity = calculate_frame_diversity(new_embedding, face_encodings)
                                            
                                            # Adicionar como candidato para captura com suas métricas
                                            face_candidates.append({
                                                'embedding': new_embedding,
                                                'frame': frame.copy(),
                                                'face_area': (x, y, w, h),
                                                'quality': quality,
                                                'diversity': diversity,
                                                'movement': movement_score,
                                                'timestamp': current_time
                                            })
                                            
                                            # Limitar a lista de candidatos
                                            if len(face_candidates) > 10:
                                                face_candidates = face_candidates[-10:]
                                            
                                            # Se for diverso o suficiente, ou auto_capture_mode, armazenar
                                            should_capture = (diversity > MIN_DIVERSITY_THRESHOLD or 
                                                           auto_capture_mode or 
                                                           frames_captured < 2 or  # Primeiro frame sempre aceito
                                                           movement_score > 0.5)  # Captura se houver movimento significativo
                                                           
                                            if should_capture:
                                                # Verificar se já não há um frame muito similar
                                                if frames_captured >= 2:
                                                    # Se este frame é muito similar ao último, pular
                                                    last_embedding = face_encodings[-1]
                                                    if calculate_cosine_distance(new_embedding, last_embedding) < 0.1:
                                                        should_capture = False
                                            
                                            if should_capture:
                                                face_encodings.append(new_embedding)
                                                
                                                # Criar nome de arquivo com timestamp e número da captura
                                                timestamp = int(time.time())
                                                angle_info = f"frame_{frames_captured}_{timestamp}"
                                                img_path = os.path.join(person_dir, f"{name}_{angle_info}.jpg")
                                                
                                                # Salvar imagem e informação
                                                cv2.imwrite(img_path, frame)
                                                captured_images.append(img_path)
                                                frames_captured += 1
                                                last_capture_time = current_time
                                                
                                                # Feedback visual da captura
                                                cv2.putText(display_frame, "Capturado!", 
                                                          (width//2-60, height//2), 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                                                
                                                # Destaque para a face detectada
                                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                                                cv2.putText(display_frame, f"Q:{quality:.2f} D:{diversity:.2f}", 
                                                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    except Exception as e:
                                        print(f"Erro ao extrair embedding: {e}")
                            
                        # Limpar arquivo temporário
                        if os.path.exists(temp_img_path):
                            os.remove(temp_img_path)
                            
                    except Exception as e:
                        print(f"Erro durante tentativa de captura: {e}")
                        if os.path.exists(temp_img_path):
                            os.remove(temp_img_path)
                
                # Verificar se terminamos a captura
                if frames_captured >= NUM_FRAMES_TO_CAPTURE:
                    break
                
                # Se tempo está acabando e temos candidatos mas poucos frames capturados, forçar captura dos melhores
                if elapsed > 18 and frames_captured < NUM_FRAMES_TO_CAPTURE * 0.5 and len(face_candidates) > 0:
                    # Ordenar candidatos por qualidade + diversidade
                    sorted_candidates = sorted(face_candidates, 
                                             key=lambda x: x['quality'] + x['diversity'] + x['movement'],
                                             reverse=True)
                    
                    # Selecionar os melhores candidatos e adicionar
                    for candidate in sorted_candidates[:min(4, len(sorted_candidates))]:
                        if frames_captured >= NUM_FRAMES_TO_CAPTURE:
                            break
                            
                        # Verificar se este embedding é diverso o suficiente comparado aos já aceitos
                        if len(face_encodings) > 0:
                            similar_found = False
                            for existing_embedding in face_encodings:
                                if calculate_cosine_distance(candidate['embedding'], existing_embedding) < 0.12:
                                    similar_found = True
                                    break
                            if similar_found:
                                continue
                        
                        # Adicionar este candidato às capturas
                        face_encodings.append(candidate['embedding'])
                        timestamp = int(time.time())
                        angle_info = f"frame_{frames_captured}_{timestamp}"
                        img_path = os.path.join(person_dir, f"{name}_{angle_info}.jpg")
                        
                        cv2.imwrite(img_path, candidate['frame'])
                        captured_images.append(img_path)
                        frames_captured += 1
                        
                        x, y, w, h = candidate['face_area']
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 165, 255), 3)
                        cv2.putText(display_frame, "Captura Automática!", 
                                  (width//2-120, height//2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                    
                    # Limpar candidatos após processamento
                    face_candidates = []
                
                # Se o tempo máximo foi excedido
                if elapsed > 30:  # Aumentado para 30 segundos
                    print("Tempo máximo excedido. Finalizando com frames capturados.")
                    break
            
            # Mostrar o frame na tela
            cv2.imshow("Captura Facial", display_frame)
            
            # Processar teclas
            key = cv2.waitKey(1)
            if key == 27:  # ESC para cancelar
                print("Captura cancelada pelo usuário.")
                cap.release()
                cv2.destroyAllWindows()
                webcam_in_use = False
                last_webcam_release_time = time.time()
                return False
            elif key == 32 and not capturing_active:  # ESPAÇO para iniciar
                capturing_active = True
                start_time = time.time()
                print("Iniciando captura contínua. Mova seu rosto em círculo lentamente...")
    
    finally:
        # Liberar a webcam
        cap.release()
        last_webcam_release_time = time.time()
        cv2.destroyAllWindows()
        webcam_in_use = False
    
    # Verificar se capturamos frames suficientes
    if frames_captured < NUM_FRAMES_TO_CAPTURE * 0.5:  # Reduzido para 50% do desejado
        print(f"Poucos frames capturados: {frames_captured}/{NUM_FRAMES_TO_CAPTURE}.")
        user_input = input("Deseja continuar mesmo assim? (s/n): ").lower()
        if user_input != 's':
            return False
    
    # Salvar os encodings faciais
    if len(face_encodings) > 0:
        encodings_path = os.path.join(encodings_dir, f"{name}_encodings.pkl")
        with open(encodings_path, 'wb') as f:
            pickle.dump(face_encodings, f)
        print(f"Encodings faciais salvos em {encodings_path}")
    
    print(f"Face registrada com sucesso para {name} com {len(captured_images)} imagens!")
    
    # Atualizar banco de dados de representações
    try:
        print("Atualizando banco de dados de faces...")
        representations_path = os.path.join(known_faces_dir, "representations_vgg_face.pkl")
        if os.path.exists(representations_path):
            os.remove(representations_path)
        DeepFace.build_model("VGG-Face")
    except Exception as e:
        print(f"Aviso: Erro ao atualizar banco de dados: {e}")
    
    return True

def register_new_face(frame=None):
    """Função para registrar uma nova face - agora usamos o método de captura contínua"""
    return register_face_continuous_motion(frame)

def compare_with_stored_encodings(frame):
    """Compara um frame com os encodings armazenados para identificação mais precisa"""
    if not os.path.exists(encodings_dir) or not os.listdir(encodings_dir):
        return None, 0.0
    
    # Salvar frame temporariamente
    temp_frame_path = "temp_comparison_frame.jpg"
    cv2.imwrite(temp_frame_path, frame)
    
    try:
        # Extrair embedding do frame atual
        embedding_objs = DeepFace.represent(
            img_path=temp_frame_path,
            model_name="VGG-Face",
            enforce_detection=False
        )
        
        if not embedding_objs or len(embedding_objs) == 0:
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
            return None, 0.0
        
        current_embedding = embedding_objs[0]['embedding']
        
        # Comparar com todos os encodings armazenados
        best_match = None
        best_confidence = 0.0
        
        for encoding_file in os.listdir(encodings_dir):
            if encoding_file.endswith("_encodings.pkl"):
                person_name = encoding_file.split("_encodings.pkl")[0]
                
                # Carregar encodings desta pessoa
                with open(os.path.join(encodings_dir, encoding_file), 'rb') as f:
                    person_encodings = pickle.load(f)
                
                # Comparar com cada encoding desta pessoa
                for idx, stored_encoding in enumerate(person_encodings):
                    try:
                        # Calcular similaridade de cosseno (1 - distância) usando nossa própria função
                        distance = calculate_cosine_distance(current_embedding, stored_encoding)
                        similarity = 1 - distance
                        
                        if similarity > best_confidence:
                            best_confidence = similarity
                            best_match = person_name
                    except Exception as e:
                        print(f"Erro ao comparar com encoding {idx} de {person_name}: {e}")
        
        # Remover arquivo temporário
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
        
        return best_match, best_confidence
        
    except Exception as e:
        print(f"Erro ao comparar com encodings armazenados: {e}")
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
        return None, 0.0

def improve_face_detection(img):
    """Melhora a detecção facial com processamento de imagem"""
    # Equalização de histograma para melhorar contraste
    if len(img.shape) == 3:  # Colorida
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:  # Escala de cinza
        enhanced = cv2.equalizeHist(img)
    
    # Aplicar filtro bilateral para reduzir ruído mantendo bordas
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return enhanced

def realtime_recognition():
    """Função para reconhecimento facial em tempo real com suporte a múltiplos encodings"""
    global webcam_in_use, last_webcam_release_time
    
    print("Iniciando reconhecimento facial em tempo real...")
    print("Pressione 'q' para sair, 'r' para registrar nova face")
    
    if not check_webcam_available():
        print("Webcam não disponível. Voltando ao menu principal.")
        return
    
    webcam_in_use = True
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        webcam_in_use = False
        print("Não foi possível abrir a webcam. Voltando ao menu principal.")
        return
    
    # Parâmetros aprimorados para reconhecimento
    last_recognition_time = 0
    recognition_interval = 0.5  # Reduzido para 0.5 segundos para aumentar responsividade
    current_faces = []
    frame_count = 0
    face_confidence_history = {}  # Para suavizar resultados
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Melhorar qualidade da imagem para detecção
            if frame_count % 10 == 0:  # Aplicar a cada 10 frames para não sobrecarregar
                enhanced_frame = improve_face_detection(frame)
            else:
                enhanced_frame = frame
            
            # Exibir instruções na tela
            cv2.putText(frame, "Pressione 'q' para sair, 'r' para registrar nova face", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Processar reconhecimento a cada intervalo
            if current_time - last_recognition_time > recognition_interval:
                try:
                    # Salvar frame temporário para análise
                    temp_frame_path = "temp_realtime_frame.jpg"
                    cv2.imwrite(temp_frame_path, enhanced_frame)
                    
                    # Detectar faces
                    faces = DeepFace.extract_faces(
                        img_path=temp_frame_path,
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                    
                    # Limpar lista de faces atuais
                    current_faces = []
                    
                    # Se encontrou faces, tentar reconhecer
                    if faces and len(faces) > 0:
                        for face_idx, face in enumerate(faces):
                            region = face['facial_area']
                            x, y, w, h = region['x'], region['y'], region['w'], region['h']
                            
                            # Desenhar retângulo para face detectada
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            
                            # Adicionar à lista de faces com label padrão
                            face_info = {
                                'region': (x, y, w, h),
                                'label': "Analisando...",
                                'color': (255, 165, 0),  # Laranja enquanto processa
                                'id': face_idx  # ID único para esta face
                            }
                            current_faces.append(face_info)
                            
                            # Verificar usando encodings armazenados para maior precisão
                            has_stored_encodings = os.path.exists(encodings_dir) and len(os.listdir(encodings_dir)) > 0
                            
                            if has_stored_encodings:
                                try:
                                    # Recortar a face do frame
                                    face_img = enhanced_frame[y:y+h, x:x+w].copy()
                                    if face_img.size > 0:
                                        # Comparar com encodings armazenados
                                        best_match, confidence = compare_with_stored_encodings(face_img)
                                        
                                        # Atualizar histórico de confiança para esta face
                                        face_id = f"face_{face_idx}"
                                        if face_id not in face_confidence_history:
                                            face_confidence_history[face_id] = []
                                        
                                        # Manter histórico limitado
                                        face_confidence_history[face_id].append((best_match, confidence))
                                        if len(face_confidence_history[face_id]) > 5:
                                            face_confidence_history[face_id].pop(0)
                                        
                                        # Calcular média de confiança recente
                                        recent_matches = [match for match, conf in face_confidence_history[face_id]]
                                        if recent_matches and len(set(recent_matches)) == 1 and None not in recent_matches:
                                            # Se todos os resultados recentes são consistentes
                                            smoothed_match = recent_matches[0]
                                            smoothed_confidence = np.mean([conf for _, conf in face_confidence_history[face_id]])
                                            
                                            if smoothed_confidence > 0.6:  # Threshold ajustado
                                                face_info['label'] = f"{smoothed_match} ({smoothed_confidence:.2%})"
                                                face_info['color'] = (0, 255, 0)  # Verde para faces reconhecidas
                                            else:
                                                # Fallback para reconhecimento tradicional
                                                results = DeepFace.find(
                                                    img_path=temp_frame_path,
                                                    db_path=known_faces_dir,
                                                    detector_backend="opencv",
                                                    enforce_detection=False,
                                                    model_name="VGG-Face",
                                                    distance_metric="cosine"
                                                )
                                                
                                                if results and len(results) > face_idx and not results[face_idx].empty:
                                                    process_recognition_result(results[face_idx], face_info)
                                                else:
                                                    face_info['label'] = "Desconhecido"
                                                    face_info['color'] = (0, 0, 255)
                                        else:
                                            # Resultados inconsistentes, usar último resultado
                                            if best_match and confidence > 0.65:  # Threshold ligeiramente maior
                                                face_info['label'] = f"{best_match} ({confidence:.2%})"
                                                face_info['color'] = (0, 255, 0)
                                            else:
                                                # Fallback para reconhecimento tradicional
                                                results = DeepFace.find(
                                                    img_path=temp_frame_path,
                                                    db_path=known_faces_dir,
                                                    detector_backend="opencv",
                                                    enforce_detection=False,
                                                    model_name="VGG-Face",
                                                    distance_metric="cosine"
                                                )
                                                
                                                if results and len(results) > face_idx and not results[face_idx].empty:
                                                    process_recognition_result(results[face_idx], face_info)
                                                else:
                                                    face_info['label'] = "Desconhecido"
                                                    face_info['color'] = (0, 0, 255)
                                except Exception as e:
                                    print(f"Erro no reconhecimento detalhado: {e}")
                                    face_info['label'] = "Erro"
                                    face_info['color'] = (0, 0, 255)
                            else:
                                # Método padrão se não houver encodings armazenados
                                has_faces_db = False
                                for person_dir in glob.glob(os.path.join(known_faces_dir, "*")):
                                    if os.path.isdir(person_dir) and len(os.listdir(person_dir)) > 0:
                                        has_faces_db = True
                                        break
                                
                                if has_faces_db:
                                    try:
                                        # Verificar similares no banco
                                        results = DeepFace.find(
                                            img_path=temp_frame_path,
                                            db_path=known_faces_dir,
                                            detector_backend="opencv",
                                            enforce_detection=False,
                                            model_name="VGG-Face",
                                            distance_metric="cosine"
                                        )
                                        
                                        if results and len(results) > face_idx and not results[face_idx].empty:
                                            process_recognition_result(results[face_idx], face_info)
                                        else:
                                            face_info['label'] = "Desconhecido"
                                            face_info['color'] = (0, 0, 255)
                                    except Exception as e:
                                        print(f"Erro no reconhecimento: {e}")
                                        face_info['label'] = "Erro"
                                        face_info['color'] = (0, 0, 255)
                                else:
                                    face_info['label'] = "Banco Vazio"
                                    face_info['color'] = (0, 165, 255)  # Laranja para banco vazio
                    
                    # Atualizar timestamp do último reconhecimento
                    last_recognition_time = current_time
                    
                    # Remover arquivo temporário
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
                        
                except Exception as e:
                    print(f"Erro durante processamento: {e}")
            
            # Desenhar retângulos e labels para todas as faces detectadas
            for face in current_faces:
                x, y, w, h = face['region']
                cv2.rectangle(frame, (x, y), (x + w, y + h), face['color'], 2)
                cv2.putText(frame, face['label'], (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, face['color'], 2)
                           
                # Se a face for desconhecida, sugerir registro
                if face['label'] == "Desconhecido":
                    cv2.putText(frame, "Pressione 'r' para registrar", (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Exibir o frame
            cv2.imshow("Reconhecimento Facial em Tempo Real", frame)
            
            # Capturar tecla
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Guardar o último frame
                last_frame = frame.copy()
                
                # Liberar a webcam para o registro
                cap.release()
                webcam_in_use = False
                last_webcam_release_time = time.time()
                cv2.destroyAllWindows()
                
                # Perguntar se deseja usar o frame atual
                use_current = input("Deseja usar o frame atual para registro? (s/n): ").lower()
                if use_current == 's':
                    # Usar o método de captura contínua
                    if register_face_continuous_motion(last_frame):
                        print("Face registrada com sucesso!")
                    else:
                        print("Falha ao registrar face.")
                else:
                    if register_face_continuous_motion():
                        print("Face registrada com sucesso!")
                    else:
                        print("Falha ao registrar face.")
                
                # Reiniciar webcam para continuar reconhecimento
                time.sleep(1.0)  # Pequena pausa antes de reabrir
                webcam_in_use = True
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Não foi possível reabrir a webcam. Voltando ao menu principal.")
                    webcam_in_use = False
                    return
                
                print("Continuando reconhecimento...")
                
                # Resetar variáveis
                current_faces = []
                last_recognition_time = 0
                face_confidence_history = {}
    
    finally:
        # Liberar recursos
        cap.release()
        last_webcam_release_time = time.time()
        cv2.destroyAllWindows()
        webcam_in_use = False

def process_recognition_result(match_df, face_info):
    """Função auxiliar para processar resultados do DeepFace.find()"""
    top_match = match_df.iloc[0]
    
    # Buscar a coluna de distância
    distance_col = None
    for col in top_match.index:
        if col.endswith('_cosine'):
            distance_col = col
            break
    
    identity = os.path.basename(os.path.dirname(top_match['identity']))
    
    # Verificar limite de confiança (threshold)
    threshold = 0.35  # Threshold menor = mais rigoroso
    if distance_col and float(top_match[distance_col]) < threshold:  
        confidence = 1 - float(top_match[distance_col])
        face_info['label'] = f"{identity} ({confidence:.2%})"
        face_info['color'] = (0, 255, 0)  # Verde para faces reconhecidas
    else:
        face_info['label'] = "Desconhecido"
        face_info['color'] = (0, 0, 255)  # Vermelho para faces desconhecidas

def main_menu():
    """Função para exibir menu principal"""
    print("\n===== SISTEMA DE RECONHECIMENTO FACIAL =====")
    print("1. Reconhecimento em tempo real")
    print("2. Reconhecer a partir de imagem de teste")
    print("3. Registrar nova face")
    print("4. Sair")
    
    choice = input("Escolha uma opção (1-4): ")
    
    if choice == '1':
        realtime_recognition()
        return True
    elif choice == '2':
        return False  # Continuar com o fluxo original do programa
    elif choice == '3':
        register_new_face()
        return True
    elif choice == '4':
        print("Encerrando programa.")
        exit()
    else:
        print("Opção inválida.")
        return True

# Verificar se existem faces registradas
has_registered_faces = False
for person_dir in glob.glob(os.path.join(known_faces_dir, "*")):
    if os.path.isdir(person_dir) and len(os.listdir(person_dir)) > 0:
        has_registered_faces = True
        break

# Exibir menu principal
if main_menu():
    exit()

# Se não houver faces registradas ou se o arquivo de teste não existir, registrar nova face
if not has_registered_faces or not os.path.exists(test_image_path):
    print("Não há faces registradas ou imagem de teste não encontrada.")
    user_input = input("Deseja registrar uma nova face? (s/n): ").lower()
    
    if user_input == 's':
        if register_new_face():
            # Atualizar o banco de dados de representações faciais
            DeepFace.build_model("VGG-Face")
            # Opção para usar a última imagem capturada como teste
            use_last = input("Deseja usar a última imagem capturada para teste? (s/n): ").lower()
            if use_last == 's':
                # Encontra a imagem mais recente
                newest_img = max(
                    [f for f in glob.glob(os.path.join(known_faces_dir, "**/*.jpg"), recursive=True)],
                    key=os.path.getctime
                )
                # Copia a imagem para o arquivo de teste
                img = cv2.imread(newest_img)
                cv2.imwrite(test_image_path, img)
                print(f"Imagem salva como {test_image_path} para teste.")
            else:
                # Capturar nova imagem para teste
                print("Capturando imagem para teste...")
                test_frame = capture_from_webcam()
                if test_frame is not None:
                    cv2.imwrite(test_image_path, test_frame)
                    print(f"Imagem de teste salva como {test_image_path}")
                else:
                    print("Operação cancelada.")
                    exit()
        else:
            print("Falha ao registrar face. Encerrando programa.")
            exit()
    else:
        print("Registro não realizado. Encerrando programa.")
        exit()

# Verificar novamente se a imagem de teste existe
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Imagem {test_image_path} não encontrada. Adicione uma imagem válida.")

# Encontrar correspondências no banco de dados
try:
    print("Processando reconhecimento facial...")
    results = DeepFace.find(
        img_path=test_image_path,
        db_path=known_faces_dir,
        detector_backend="opencv",
        enforce_detection=True,
        model_name="VGG-Face",  # Modelo para reconhecimento
        distance_metric="cosine"
    )

    # Carregar a imagem de teste
    img = cv2.imread(test_image_path)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem {test_image_path}.")

    # Verificar se há resultados
    if not results or len(results) == 0:
        print("Nenhum resultado encontrado.")
        user_input = input("Deseja registrar uma nova face? (s/n): ").lower()
        if user_input == 's':
            register_new_face()
        exit()

    # Processar cada rosto detectado
    for i, df in enumerate(results):
        if not df.empty:
            # Extrair informações do rosto mais próximo
            top_match = df.iloc[0]
            identity = os.path.basename(os.path.dirname(top_match['identity']))  # Extrai o nome da subpasta
            
            # Verificar as colunas disponíveis no DataFrame
            distance_col = None
            for col in top_match.index:
                if col.endswith('_cosine'):
                    distance_col = col
                    break
            
            if distance_col:
                confidence = 1 - float(top_match[distance_col])  # Converte distância em confiança
                confidence_text = f"{identity} ({confidence:.2%})"
            else:
                confidence_text = identity
                
            x, y, w, h = int(top_match['source_x']), int(top_match['source_y']), \
                        int(top_match['source_w']), int(top_match['source_h'])
        else:
            # Caso não haja correspondência, marcar como desconhecido
            identity = "Desconhecido"
            confidence_text = identity
            # Usar DeepFace para detectar rostos (fallback)
            try:
                face_data = DeepFace.extract_faces(
                    img_path=test_image_path,
                    detector_backend="opencv",
                    enforce_detection=True
                )
                if face_data and len(face_data) > 0:
                    region = face_data[0]['facial_area']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                else:
                    print("Nenhum rosto detectado na imagem de teste.")
                    continue
            except Exception as e:
                print(f"Erro ao detectar rosto: {e}")
                continue

        # Desenhar retângulo e nome
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Exibir a imagem
    cv2.imshow("Reconhecimento Facial", img)
    print("Pressione qualquer tecla para encerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Erro durante o reconhecimento facial: {e}")
    
    # Para debug - mostrar mais informações sobre o erro
    import traceback
    traceback.print_exc()
    
    # Perguntar se deseja registrar nova face em caso de erro
    user_input = input("Deseja registrar uma nova face? (s/n): ").lower()
    if user_input == 's':
        register_new_face()