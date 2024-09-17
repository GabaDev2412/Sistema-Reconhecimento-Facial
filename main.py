import cv2
import face_recognition
from cvzone.FaceDetectionModule import FaceDetector
import os
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk


# Função para carregar imagens e codificações de uma pasta
def load_images_from_folder(folder):
    known_encodings, known_names = [], []

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".png")):  # Filtrar apenas arquivos de imagem
            filepath = os.path.join(folder, filename)
            image = face_recognition.load_image_file(filepath)

            try:
                encoding = face_recognition.face_encodings(image)[0]
                known_encodings.append(encoding)
                known_names.append(os.path.splitext(filename)[0])  # Nome sem extensão
            except IndexError:
                print(f"Falha ao detectar rosto em {filename}")

    return known_encodings, known_names


# Classe principal para o app de reconhecimento facial
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento Facial")
        self.root.geometry("1000x600")

        # Frame de vídeo e caixa de texto
        self.video_frame = tk.Label(self.root)
        self.video_frame.place(x=10, y=50, width=480, height=500)
        self.text_box = tk.Text(self.root, wrap="word", state="disabled")
        self.text_box.place(x=500, y=50, width=480, height=500)

        # Botão de iniciar reconhecimento
        self.start_button = tk.Button(self.root, text="Iniciar Reconhecimento", command=self.start_face_recognition)
        self.start_button.pack(pady=20)

        # Botão de cadastro oculto inicialmente
        self.register_button = tk.Button(self.root, text="Cadastrar Novo Rosto", command=self.register_new_face)
        self.hide_register_button()

        self.video_running = False
        self.video = None
        self.detector = FaceDetector()

        # Carregar dataset de imagens
        self.dataset_folder = "C:/Users/renat/OneDrive/Documentos/REPOSITORIOS/dataset_images"
        self.known_encodings, self.known_names = load_images_from_folder(self.dataset_folder)

    def start_face_recognition(self):
        self.video_running = True
        self.start_button.pack_forget()
        self.video = cv2.VideoCapture(0)
        self.update_video()

    def update_video(self):
        if self.video_running:
            ret, img = self.video.read()
            if ret:
                img, bboxes = self.detector.findFaces(img, draw=True)

                if bboxes:
                    face_encodings = face_recognition.face_encodings(img, face_recognition.face_locations(img))

                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)

                        if matches and matches[face_distances.argmin()]:
                            name = self.known_names[face_distances.argmin()]
                            self.update_text(f"Conheço essa pessoa: {name}")
                            cv2.putText(img, f"Conheco essa pessoa: {name}", (100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 2)
                        else:
                            self.update_text("Não conheço essa pessoa")
                            cv2.putText(img, "Não conheco essa pessoa", (100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), 2)
                            self.show_register_button()

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tk = ImageTk.PhotoImage(image=Image.fromarray(img))
                self.video_frame.imgtk = img_tk
                self.video_frame.configure(image=img_tk)

            self.root.after(10, self.update_video)

    def stop_video(self, event=None):
        self.video_running = False
        if self.video:
            self.video.release()
        self.video_frame.config(image="")
        self.start_button.pack(pady=20)

    def show_register_button(self):
        self.register_button.place(x=400, y=10)

    def hide_register_button(self):
        self.register_button.place_forget()

    def update_text(self, message):
        self.text_box.config(state="normal")
        self.text_box.insert(tk.END, message + "\n")
        self.text_box.config(state="disabled")
        self.text_box.yview(tk.END)

    def register_new_face(self):
        self.video_running = False  # Pausar o reconhecimento facial
        ret, img = self.video.read()

        if ret:
            name = simpledialog.askstring("Cadastrar Novo Rosto", "Digite o nome do funcionário:")
            if name:
                filepath = os.path.join(self.dataset_folder, f"{name}.jpg")
                cv2.imwrite(filepath, img)

                new_encoding = face_recognition.face_encodings(img)[0]
                self.known_encodings.append(new_encoding)
                self.known_names.append(name)

                messagebox.showinfo("Cadastro Concluído", f"Rosto de {name} cadastrado com sucesso!")
                self.hide_register_button()
                self.video_running = True
            else:
                messagebox.showwarning("Cadastro Cancelado", "O cadastro foi cancelado.")
                self.video_running = True


# Função principal
def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.bind("<Escape>", app.stop_video)
    root.mainloop()


if __name__ == "__main__":
    main()
