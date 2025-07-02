import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import timm
from torchvision import transforms
from torch import nn
from torch.cuda.amp import autocast
import os

dataset_classes = ['Ahaetulla_nasuta', 'Amphiesma_stolatum', 'Boiga_ochracea', 'Boiga_trigonata', 'Bungarus_caeruleus', 'Bungarus_fasciatus', 'Calliophis_bivirgatus', 'Coelongnathus_radiatus', 'Craspedocephalus_albolabris', 'Daboia_russelii', 'Dendrelaphis_tristis', 'Eryx_johnii', 'Gloydius_himalayanus', 'Indotyphlops_braminus', 'Lycodon_aulicus', 'Naja_kaouthia', 'Naja_naja', 'Oligodon_arnensis', 'Ophiophagus_hannah', 'Oreocryptophis_porphyraceus', 'Ovophis_monticola', 'Pytas_mucosa', 'Python_molurus', 'Rhabdophis_subminiatus', 'Sibynophis_subpunctatus', 'Xenochrophis_piscator']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #you can use cpu or gpu


inference_transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_model():
    model = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True)
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(model.num_features),
        nn.Dropout(0.3),
        nn.Linear(model.num_features, len(dataset_classes))
    )
    model.load_state_dict(torch.load("best_efficientv2m.pth", map_location=device))
    model.eval()
    model.to(device)
    return model

model = load_model()


class SnakeClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🐍 Snake Species Classifier")
        self.root.geometry("720x600")
        self.root.configure(bg="#1e1e1e")

        self.style = ttk.Style()
        self.style.configure("TButton", font=("Segoe UI", 12), padding=10)
        self.style.configure("TLabel", font=("Segoe UI", 12), background="#1e1e1e", foreground="white")

        
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(pady=10)

        
        self.button = ttk.Button(self.root, text="🖼️ Select Image", command=self.load_image)
        self.button.pack(pady=10)

        
        self.result_label = ttk.Label(self.root, text="Prediction will appear here.", font=("Segoe UI", 14, "bold"))
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        try:
            image = Image.open(file_path).convert("RGB")
            image.thumbnail((400, 400))
            tk_img = ImageTk.PhotoImage(image)
            self.image_label.configure(image=tk_img)
            self.image_label.image = tk_img

            self.predict(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_tensor = inference_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                top3_probs, top3_indices = torch.topk(probs, 3)

        prediction_text = "\n".join([
            f"{dataset_classes[top3_indices[0][i]]}: {top3_probs[0][i].item()*100:.2f}%"
            for i in range(3)
        ])

        self.result_label.config(text=prediction_text)


# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = SnakeClassifierGUI(root)
    root.mainloop()
