import pickle
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os
import torch.nn.functional as F

def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png")) and not filename.startswith('.')

def get_embedding(img_path, mtcnn, resnet):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            return resnet(face.unsqueeze(0))
    return None

def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1, emb2).item()

def load_known_faces(mtcnn, resnet):
    known_faces = {}
    for file in os.listdir("known_faces"):
        if is_image_file(file):
            name = os.path.splitext(file)[0]
            emb = get_embedding(os.path.join("known_faces", file), mtcnn, resnet)
            if emb is not None:
                known_faces[name] = emb
    return known_faces

def match_unknown_faces(known_faces, mtcnn, resnet, threshold=0.9):
    for file in os.listdir("unknown_faces"):
        if not is_image_file(file):
            continue

        emb = get_embedding(os.path.join("unknown_faces", file), mtcnn, resnet)
        if emb is None:
            print(f"{file}: No face found")
            continue

        matched_name = None
        matched_sim = None
        for name, known_emb in known_faces.items():
            sim = cosine_similarity(emb, known_emb)
            if sim > threshold:
                matched_name = name
                matched_sim = sim
                break
        if matched_name is not None:
            # Find the actual known image file for this name (any extension)
            known_img_path = next(
                (os.path.join("known_faces", f) for f in os.listdir("known_faces")
                 if os.path.splitext(f)[0] == matched_name and is_image_file(f)),
                None
            )
            if known_img_path:
                known_img = Image.open(known_img_path).convert('RGB')
                unknown_img = Image.open(os.path.join("unknown_faces", file)).convert('RGB')

                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(known_img)
                axs[0].set_title(f"Known: {matched_name}")
                axs[1].imshow(unknown_img)
                axs[1].set_title(f"Matched: {file}")
                for ax in axs:
                    ax.axis('off')
                plt.show()

            print(f"{file}: Recognized as {matched_name} ({matched_sim:.2f})")
        else:
            print(f"{file}: Unknown face")

def main():
    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    if os.path.exists("known_faces.pkl"):
        with open("known_faces.pkl", "rb") as f:
            known_faces = pickle.load(f)
    else:
        known_faces = load_known_faces(mtcnn, resnet)
        with open("known_faces.pkl", "wb") as f:
            pickle.dump(known_faces, f)

    match_unknown_faces(known_faces, mtcnn, resnet)

if __name__ == "__main__":
    main()