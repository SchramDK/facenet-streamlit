import os
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import base64
from io import BytesIO
import streamlit as st

UPLOAD_DIR = "uploads"

def get_image_files():
    if os.path.exists(UPLOAD_DIR):
        return [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return []

def upload_path(filename):
    return os.path.join(UPLOAD_DIR, filename)

@st.cache_resource
def load_models():
    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

@st.cache_resource(show_spinner="Detecting faces...")
def detect_faces_grouped(image_files):
    mtcnn, resnet = load_models()
    faces_by_group = []
    embeddings = []

    for file in image_files:
        path = upload_path(file)
        image = Image.open(path).convert('RGB')
        boxes, _ = mtcnn.detect(image)
        if not boxes:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_crop = image.crop((x1, y1, x2, y2))

            if face_crop.width < 20 or face_crop.height < 20:
                continue

            face_tensor = mtcnn(face_crop)
            if face_tensor is None or not isinstance(face_tensor, torch.Tensor):
                continue
            with torch.no_grad():
                emb = resnet(face_tensor.unsqueeze(0))

            matched = False
            for i, group_emb_list in enumerate(embeddings):
                group_mean_emb = torch.stack(group_emb_list).mean(dim=0, keepdim=True)
                sim = F.cosine_similarity(emb, group_mean_emb).item()
                if sim > 0.8:
                    faces_by_group[i].append((face_crop, file))
                    group_emb_list.append(emb.squeeze(0))
                    matched = True
                    break
            if not matched:
                embeddings.append([emb.squeeze(0)])
                faces_by_group.append([(face_crop, file)])

    return faces_by_group

@st.cache_resource
def load_faces():
    image_files = get_image_files()
    faces_by_group = detect_faces_grouped(image_files)
    return faces_by_group

def image_to_base64(image):
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def resize_image(image, max_size=(800, 800)):
    image.thumbnail(max_size)
    return image

def render_person_box(b64, display_name, photo_count, person_link):
    st.markdown(f"""
    <a href="{person_link}" style="text-decoration: none;" target="_self">
        <div class="person-box">
            <img src="data:image/jpeg;base64,{b64}" alt="Thumbnail of {display_name}" />
            <strong>{display_name}</strong>
            <span style="color: gray;">Photos: {photo_count}</span>
        </div>
    </a>
    """, unsafe_allow_html=True)

def render_profile_header(b64, selected_person, photo_count):
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px; width: 100%;">
        <img src="data:image/jpeg;base64,{b64}" alt="Profile picture of Person {selected_person}" style="border-radius: 50%; width: 80px; height: 80px; object-fit: cover;" />
        <div>
            <h2 style="margin: 0;">Person {selected_person}</h2>
            <p style="margin: 0; color: gray;">{photo_count} photo(s)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="FaceNet Viewer", layout="wide")

css = """
<style>
.person-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 175px;
    border: 1px solid #ddd;
    padding: 10px;
    box-sizing: border-box;
    border-radius: 8px;
    margin-bottom: 15px;
    text-align: center;
}
.person-box img {
    width: 80px;
    height: 80px;
    object-fit: cover;
    border-radius: 50%;
    margin-bottom: 10px;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

if "menu" not in st.session_state:
    st.session_state.menu = "All Files"
if "person" not in st.session_state:
    st.session_state.person = None

query_params = st.query_params
query_menu = query_params.get("menu")
query_person = query_params.get("person")

if query_menu:
    st.session_state.menu = query_menu
if query_person and query_person.isdigit():
    st.session_state.person = int(query_person)
else:
    st.session_state.person = None

options = ["All Files", "People"]
with st.sidebar.expander("Navigation", expanded=True):
    st.markdown(
        "<style>.streamlit-expanderHeader {pointer-events: none;}</style>",
        unsafe_allow_html=True,
    )
    menu_selection = st.radio("Select View", options=options, index=options.index(st.session_state.menu), key="menu_radio")
    if menu_selection != st.session_state.menu:
        st.query_params.clear()
        st.query_params["menu"] = menu_selection
        st.rerun()

menu = st.session_state.menu
selected_person = st.session_state.person

if menu == "People":
    st.title("🧑 Detected People in Uploaded Images")

    image_files = get_image_files()
    if not image_files:
        st.warning("No images uploaded. Please upload files in 'All Files' tab.")
        if "faces_by_group" in st.session_state:
            del st.session_state.faces_by_group
        st.stop()

    faces_by_group = load_faces()

    if selected_person is not None and 1 <= selected_person <= len(faces_by_group):
        idx = selected_person - 1
        group_faces = faces_by_group[idx]

        profile_face = group_faces[0][0]
        b64 = image_to_base64(profile_face)

        render_profile_header(b64, selected_person, len(group_faces))

        if st.button("⬅ Back to overview"):
            st.query_params.clear()
            st.query_params["menu"] = "People"
            st.rerun()

        unique_files = list(dict.fromkeys(src_file for (_, src_file) in group_faces))
        st.markdown("### Photo Gallery")
        gallery_cols = st.columns(4)
        for i, file in enumerate(unique_files):
            full_path = upload_path(file)
            image = Image.open(full_path)
            image = resize_image(image)  # Begræns billedstørrelse
            gallery_cols[i % 4].image(image, caption=file, use_container_width=True)

    else:
        st.markdown("## People Overview")

        cols = st.columns(5)
        for idx, group_faces in enumerate(faces_by_group):
            profile_face = group_faces[0][0]
            photo_count = len(group_faces)
            display_name = f"Person {idx+1}"

            b64 = image_to_base64(profile_face)

            with cols[idx % 5]:
                person_link = f"/?menu=People&person={idx + 1}"
                render_person_box(b64, display_name, photo_count, person_link)

if menu == "All Files":
    st.session_state.person = None
    st.query_params.clear()
    st.query_params["menu"] = "All Files"

    st.title("📂 All Files")

    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
    if uploaded_files:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        for file in uploaded_files:
            file_path = upload_path(file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"{len(uploaded_files)} file(s) uploaded.")

    st.markdown("### Uploaded Files")
    if os.path.exists(UPLOAD_DIR):
        file_list = os.listdir(UPLOAD_DIR)
        for file in file_list:
            file_path = upload_path(file)
            col1, col2 = st.columns([3, 1])
            with col1:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    st.image(file_path, caption=file, use_container_width=True)
                else:
                    st.markdown(f"📄 {file}")
            with col2:
                if st.button(f"🗑 Delete", key=f"delete_{file}"):
                    os.remove(file_path)
                    st.rerun()
    else:
        st.info("No files uploaded yet.")