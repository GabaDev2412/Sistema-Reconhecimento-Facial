import cv2
import face_recognition
from cvzone.FaceDetectionModule import FaceDetector
import os


# Função para extrair o nome do arquivo sem a extensão
def get_name_from_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0]


# Função para carregar todas as imagens de uma pasta e suas codificações faciais
def load_images_from_folder(folder):
    known_encodings = []
    known_names = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Filtrar arquivos de imagem
            filepath = os.path.join(folder, filename)
            image = face_recognition.load_image_file(filepath)

            try:
                encoding = face_recognition.face_encodings(image)[0]  # Extrair codificação facial
                known_encodings.append(encoding)
                known_names.append(get_name_from_filename(filename))  # Adicionar nome correspondente
            except IndexError:
                print(f"Nao foi possível detectar rosto em {filename}")

    return known_encodings, known_names


# Caminho para a pasta que contém o dataset de imagens
dataset_folder = "D:/PROJETOS_PYTHON/recFacial1.0/dataset_images"

# Carregar as imagens e codificações faciais
known_encodings, known_names = load_images_from_folder(dataset_folder)

# Iniciar a webcam
video = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    ret, img = video.read()

    if not ret:
        print("Erro ao acessar a webcam")
        break

    # Detectar rostos na imagem da webcam
    img, bboxes = detector.findFaces(img, draw=True)

    if bboxes:
        # Pegar a área do rosto na imagem capturada
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        # Comparar cada rosto detectado com as imagens conhecidas do dataset
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            # Achar a melhor correspondência
            best_match_index = None
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()

            if matches and matches[best_match_index]:
                name = known_names[best_match_index]
                cv2.putText(img, f"Conheco essa pessoa: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)
                print(f"Conheço essa pessoa: {name}")
            else:
                cv2.putText(img, "Não conheco essa pessoa", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print("Não conheço essa pessoa")

    # Mostrar a imagem com a detecção de rosto
    cv2.imshow('Resultado', img)

    # Sair do loop se a tecla 'Esc' for pressionada
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos
video.release()
cv2.destroyAllWindows()
