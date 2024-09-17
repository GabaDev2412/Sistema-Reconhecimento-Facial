# import cv2
# import face_recognition
# from cvzone.FaceDetectionModule import FaceDetector
# import os
#
#
# # Função para extrair o nome do arquivo sem a extensão
# def get_name_from_filename(filename):
#     return os.path.splitext(os.path.basename(filename))[0]
#
#
# # Função para carregar todas as imagens de uma pasta e suas codificações faciais
# def load_images_from_folder(folder):
#     known_encodings = []
#     known_names = []
#
#     for filename in os.listdir(folder):
#         if filename.endswith(".jpg") or filename.endswith(".png"):  # Filtrar arquivos de imagem
#             filepath = os.path.join(folder, filename)
#             image = face_recognition.load_image_file(filepath)
#
#             try:
#                 encoding = face_recognition.face_encodings(image)[0]  # Extrair codificação facial
#                 known_encodings.append(encoding)
#                 known_names.append(get_name_from_filename(filename))  # Adicionar nome correspondente
#             except IndexError:
#                 print(f"Nao foi possível detectar rosto em {filename}")
#
#     return known_encodings, known_names
#
#
# # Caminho para a pasta que contém o dataset de imagens
# dataset_folder = "D:/PROJETOS_PYTHON/recFacial1.0/dataset_images"
#
# # Carregar as imagens e codificações faciais
# known_encodings, known_names = load_images_from_folder(dataset_folder)
#
# # Iniciar a webcam
# video = cv2.VideoCapture(0)
# detector = FaceDetector()
#
# while True:
#     ret, img = video.read()
#
#     if not ret:
#         print("Erro ao acessar a webcam")
#         break
#
#     # Detectar rostos na imagem da webcam
#     img, bboxes = detector.findFaces(img, draw=True)
#
#     if bboxes:
#         # Pegar a área do rosto na imagem capturada
#         face_locations = face_recognition.face_locations(img)
#         face_encodings = face_recognition.face_encodings(img, face_locations)
#
#         # Comparar cada rosto detectado com as imagens conhecidas do dataset
#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(known_encodings, face_encoding)
#             face_distances = face_recognition.face_distance(known_encodings, face_encoding)
#
#             # Achar a melhor correspondência
#             best_match_index = None
#             if len(face_distances) > 0:
#                 best_match_index = face_distances.argmin()
#
#             if matches and matches[best_match_index]:
#                 name = known_names[best_match_index]
#                 cv2.putText(img, f"Conheco essa pessoa: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
#                             2)
#                 print(f"Conheço essa pessoa: {name}")
#             else:
#                 cv2.putText(img, "Não conheco essa pessoa", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                 print("Não conheço essa pessoa")
#
#     # Mostrar a imagem com a detecção de rosto
#     cv2.imshow('Resultado', img)
#
#     # Sair do loop se a tecla 'Esc' for pressionada
#     if cv2.waitKey(1) == 27:
#         break
#
# # Liberar recursos
# video.release()
# cv2.destroyAllWindows()

import cv2
import face_recognition
from cvzone.FaceDetectionModule import FaceDetector
import os
import tkinter as tk
from tkinter import messagebox, Text
from PIL import Image, ImageTk


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
                print(f"Não foi possível detectar rosto em {filename}")

    return known_encodings, known_names


# Classe principal que gerencia a interface gráfica e o reconhecimento facial
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento Facial")

        # Definir o tamanho da janela
        self.root.geometry("1000x600")

        # Frame para o vídeo (metade esquerda)
        self.video_frame = tk.Label(self.root)
        self.video_frame.place(x=0, y=0, width=500, height=600)

        # Caixa de texto para exibir as atualizações do reconhecimento (metade direita)
        self.text_box = Text(self.root, wrap="word", state="disabled")
        self.text_box.place(x=500, y=0, width=500, height=600)

        # Criar um botão para iniciar o reconhecimento facial
        self.start_button = tk.Button(self.root, text="Iniciar Reconhecimento", command=self.start_face_recognition)
        self.start_button.pack(pady=20)

        # Variável para controlar o loop de vídeo
        self.video_running = False
        self.video = None
        self.detector = FaceDetector()

        # Carregar as imagens e codificações faciais
        self.dataset_folder = "D:/PROJETOS_PYTHON/recFacial1.0/dataset_images"
        self.known_encodings, self.known_names = load_images_from_folder(self.dataset_folder)

    # Função chamada quando o botão é clicado
    def start_face_recognition(self):
        self.video_running = True
        self.start_button.pack_forget()  # Esconder o botão enquanto o vídeo estiver ativo
        self.video = cv2.VideoCapture(0)
        self.update_video()

    # Função para atualizar o frame de vídeo
    def update_video(self):
        if self.video_running:
            ret, img = self.video.read()
            if ret:
                # Detectar rostos na imagem da webcam
                img, bboxes = self.detector.findFaces(img, draw=True)

                if bboxes:
                    # Pegar a área do rosto na imagem capturada
                    face_locations = face_recognition.face_locations(img)
                    face_encodings = face_recognition.face_encodings(img, face_locations)

                    # Comparar cada rosto detectado com as imagens conhecidas do dataset
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)

                        # Achar a melhor correspondência
                        best_match_index = None
                        if len(face_distances) > 0:
                            best_match_index = face_distances.argmin()

                        if matches and matches[best_match_index]:
                            name = self.known_names[best_match_index]
                            self.update_text(f"Conheço essa pessoa: {name}")
                            cv2.putText(img, f"Conheco essa pessoa: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 2)
                        else:
                            self.update_text("Não conheço essa pessoa")
                            cv2.putText(img, "Não conheco essa pessoa", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), 2)

                # Converter imagem para o formato compatível com Tkinter
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img_pil)

                # Atualizar o frame do vídeo na metade esquerda
                self.video_frame.imgtk = img_tk
                self.video_frame.configure(image=img_tk)

            # Continuar chamando esta função para atualizar o vídeo
            self.root.after(10, self.update_video)

    # Função para parar o vídeo e mostrar o botão novamente
    def stop_video(self, event=None):
        self.video_running = False
        if self.video is not None:
            self.video.release()
            self.video = None
        self.video_frame.config(image="")  # Limpar o frame de vídeo
        self.start_button.pack(pady=20)  # Mostrar o botão de iniciar

    # Função para atualizar a caixa de texto com as mensagens de reconhecimento
    def update_text(self, message):
        self.text_box.config(state="normal")
        self.text_box.insert(tk.END, message + "\n")
        self.text_box.config(state="disabled")
        self.text_box.yview(tk.END)  # Rolar automaticamente para o fim do texto


# Função principal
def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)

    # Atalho para fechar o reconhecimento facial ao pressionar "Esc"
    root.bind("<Escape>", app.stop_video)

    root.mainloop()


if __name__ == "__main__":
    main()
