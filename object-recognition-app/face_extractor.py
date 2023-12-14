import cv2
import os
from PIL import Image
# Ścieżka do katalogu wejściowego zawierającego foldery z obrazami
input_base_directory = ""

# Ścieżka do katalogu wyjściowego, gdzie zostaną zapisane same twarze
output_directory = ""

# Utwórz katalog wyjściowy, jeśli nie istnieje
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Załaduj wstępnie nauczony model detekcji twarzy z OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Funkcja do wyciągania twarzy z obrazu
def extract_faces(image_path, output_path):

    if not os.path.isfile(image_path):
        print(f"Nie można odnaleźć pliku: {image_path}")
        return

    # Wczytaj obraz
    img = cv2.imread(image_path)

    # Sprawdź, czy udało się wczytać obraz
    if img is None:
        im = Image.open(image_path)
        rgb_im = im.convert('RGB')
        rgb_im.save(image_path)
        img = cv2.imread(image_path)

    
    # Konwertuj obraz na odcienie szarości
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Wykryj twarze na obrazie
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Wyciągnij i zapisz każdą twarz
    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y+h, x:x+w]
        original_filename, file_extension = os.path.splitext(os.path.basename(image_path))
        face_filename = f"{output_path}/{original_filename}_face{file_extension}"
        cv2.imwrite(face_filename, face)
        print(f"Twarz wyciągnięta i zapisana jako {face_filename}")

extract_faces(input_base_directory, output_directory)