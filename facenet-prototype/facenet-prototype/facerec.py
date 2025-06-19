import pickle
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os
import torch.nn.functional as F

# Load models
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Get face embedding
def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            return resnet(face.unsqueeze(0))
    return None

# Cosine similarity
def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1, emb2).item()

# Load or compute known face embeddings
if os.path.exists("known_faces.pkl"):
    with open("known_faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}
    for file in os.listdir("known_faces"):
        if file.lower().endswith((".jpg", ".jpeg", ".png")) and not file.startswith('.'):
            name = file.split('.')[0]
            emb = get_embedding(os.path.join("known_faces", file))
            if emb is not None:
                known_faces[name] = emb
    with open("known_faces.pkl", "wb") as f:
        pickle.dump(known_faces, f)

# Compare unknowns
for file in os.listdir("unknown_faces"):
    if file.lower().endswith((".jpg", ".jpeg", ".png")) and not file.startswith('.'):
        emb = get_embedding(os.path.join("unknown_faces", file))
        if emb is None:
            print(f"{file}: No face found")
            continue
        matched = False
        for name, known_emb in known_faces.items():
            sim = cosine_similarity(emb, known_emb)
            if sim > 0.9:
                known_img = Image.open(os.path.join("known_faces", f"{name}.jpg")).convert('RGB')
                unknown_img = Image.open(os.path.join("unknown_faces", file)).convert('RGB')

                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(known_img)
                axs[0].set_title(f"Known: {name}")
                axs[1].imshow(unknown_img)
                axs[1].set_title(f"Matched: {file}")
                for ax in axs:
                    ax.axis('off')
                plt.show()

                print(f"{file}: Recognized as {name} ({sim:.2f})")
                matched = True
                break
        if not matched:
            print(f"{file}: Unknown face")