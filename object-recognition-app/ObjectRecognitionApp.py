import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keras.models import load_model
from keras.preprocessing import image
from matplotlib import pyplot as plt
import numpy as np
import os

class ImageRecognitionApp(tk.Tk):
    def __init__(self, model_path='saved_models\data_best_model'):
        super().__init__()
        self.title("Object recognition app")
        self.geometry("1200x800")
        self.resizable(False, False)

        self.model = load_model(model_path)
        self.result_map = {
            0: 'konrad',
            1: 'subject01',
            2: 'subject02',
            3: 'subject03',
            4: 'subject04',
            5: 'subject05',
            6: 'subject06',
            7: 'subject07',
            8: 'subject08',
            9: 'subject09',
            10: 'subject10',
            11: 'subject11',
            12: 'subject12',
            13: 'subject13',
            14: 'subject14',
            15: 'subject15'
        }

        self.create_widgets()
        self.create_plot()

    def create_widgets(self):
        self.open_button = tk.Button(self, text="Select Image", command=self.open_file_dialog)
        self.open_button.pack(pady=10)

        white_image = ImageTk.PhotoImage(Image.new("RGB", (300, 400), "white"))

        # Set white background for image_label
        self.image_label = tk.Label(self, image=white_image)
        self.image_label.pack(pady=10, fill='both', side='left', padx=10)

        self.filename_label = tk.Label(self, text="", font=("Helvetica", 12))
        self.filename_label.pack(pady=5)

        self.result_label = tk.Label(self, text="", font=("Helvetica", 16, "bold"))
        self.result_label.pack(pady=10)

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side='right', fill='both', expand='yes', pady=10)

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(title="Choose file", filetypes=[("Graphical files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.display_image(file_path)
            self.predict_object(file_path)

    def display_image(self, file_path):
        image_display = Image.open(file_path)
        target_size = (300, 400)
        
        resized_image = image_display.resize(target_size, Image.LANCZOS)

        photo = ImageTk.PhotoImage(resized_image)

        self.image_label.config(image=photo)
        self.image_label.image = photo
        filename = os.path.basename(file_path)
        self.filename_label.config(text=f"File: {filename}")

    def predict_object(self, file_path):
        self.ax.clear()  # Clear previous plot

        test_image = image.load_img(file_path, target_size=(64, 64), color_mode='grayscale')
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        predictions = self.model.predict(test_image, verbose=0)
        result_index = np.argmax(predictions)
        result_text = f"Recognized object: {self.result_map[result_index]}"
        self.result_label.config(text=result_text)

        labels = list(self.result_map.values())
        values = predictions.flatten() * 100

        self.ax.bar(labels, values, color='skyblue')
        self.ax.set_xlabel('Categories')
        self.ax.set_ylabel('Probability (%)')
        self.ax.set_title('Probability for each class')
        self.ax.tick_params(axis='x', rotation=45)
        self.canvas.draw()  # Update the canvas

if __name__ == "__main__":
    app = ImageRecognitionApp()
    app.mainloop()
