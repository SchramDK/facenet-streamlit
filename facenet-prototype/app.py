import os
import torch
from PIL import Image
from insightface.app import FaceAnalysis
from facenet_pytorch import InceptionResnetV1
import base64
from io import BytesIO
import streamlit as st
import numpy as np
from torchvision import transforms
import cv2

UPLOAD_DIR = "uploads"

# Clear old face‐detection cache on startup
if os.path.isdir(UPLOAD_DIR):
    for fname in os.listdir(UPLOAD_DIR):
        if fname.startswith(".cache_") and fname.endswith(".pkl"):
            try:
                os.remove(os.path.join(UPLOAD_DIR, fname))
            except OSError:
                pass

def upload_path(filename):
    return os.path.join(UPLOAD_DIR, filename)

def get_image_files():
    if os.path.exists(UPLOAD_DIR):
        return [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return []

def get_cached_file_list():
    if os.path.exists(UPLOAD_DIR):
        return [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return []

@st.cache_data
def get_image_base64_cached(file_path):
    image = Image.open(file_path)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

@st.cache_resource
def load_models():
    retina = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    retina.prepare(ctx_id=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return retina, resnet

@st.cache_data
def detect_faces_grouped(image_files, file_mtimes):
    import torch.nn.functional as F
    face_counts = {}
    retina, resnet = load_models()
    faces_by_group = []
    embeddings = []

    for file in image_files:
        path = upload_path(file)
        image = Image.open(path).convert('RGB')
        image.load()
        image = image.copy()
        print(f"Opened image {file}, mode: {image.mode}, size: {image.size}")
        try:
            img_array = np.array(image)
            max_size = 640
            h, w = img_array.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img_array = cv2.resize(img_array, (int(w * scale), int(h * scale)))
            print(f"Original np.array shape: {img_array.shape}, dtype: {img_array.dtype}")

            min_height, min_width = 480, 480
            h, w = img_array.shape[:2]
            pad_bottom = max(0, min_height - h)
            pad_right = max(0, min_width - w)
            if pad_bottom > 0 or pad_right > 0:
                img_array = cv2.copyMakeBorder(img_array, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            if img_array.ndim == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif img_array.ndim == 3:
                if img_array.shape[2] == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif img_array.shape[2] == 3:  # RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    raise ValueError("Unsupported number of channels in image")
            else:
                raise ValueError("Unsupported image shape")

            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)

            print(f"Processed image shape: {img_array.shape}, dtype: {img_array.dtype}")
        except Exception as e:
            print(f"Error processing image {file}: {e}")
            continue
        faces = retina.get(img_array)
        face_counts[file] = len(faces)
        print(f"{file}: Detected {len(faces)} face(s)")
        print(f"Found {len(faces)} face(s) in {file}")
        if not faces:
            continue
        boxes = []
        for face in faces:
            facial_area = face.bbox.astype(int)
            boxes.append(facial_area)
        if not boxes:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            h, w = img_array.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            face_crop = Image.fromarray(cv2.cvtColor(img_array[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            face_crop = face_crop.resize((160, 160))  # Resize for InceptionResnetV1

            if face_crop.width < 20 or face_crop.height < 20:
                continue

            face_tensor = transforms.ToTensor()(face_crop).unsqueeze(0)
            if not isinstance(face_tensor, torch.Tensor):
                continue
            with torch.no_grad():
                emb = resnet(face_tensor)

            emb = emb.squeeze(0)
            matched = False
            for i, existing_emb in enumerate(embeddings):
                similarity = F.cosine_similarity(emb, existing_emb, dim=0)
                if similarity.item() > 0.7:
                    faces_by_group[i].append((face_crop, file))
                    matched = True
                    break
            if not matched:
                embeddings.append(emb)
                faces_by_group.append([(face_crop, file)])

    return faces_by_group, face_counts

def load_faces():
    image_files = get_image_files()
    file_mtimes = tuple(os.path.getmtime(upload_path(f)) for f in image_files)
    faces_by_group, face_counts = detect_faces_grouped(image_files, file_mtimes)
    valid_files = set(image_files)
    # Fjern person-grupper hvor INGEN af billederne stadig findes
    filtered_faces_by_group = [group for group in faces_by_group if any(f in valid_files for (_, f) in group)]
    return filtered_faces_by_group, face_counts

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

def delete_file(filename):
    try:
        file_path = upload_path(filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        # Remove all face detection cache files to force reprocessing if needed
        for cached_file in os.listdir(UPLOAD_DIR):
            if cached_file.startswith(".cache_"):
                os.remove(os.path.join(UPLOAD_DIR, cached_file))
    except Exception as e:
        st.warning(f"Error deleting {filename}: {e}")

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
    st.markdown("### 🧑 Detected People in Uploaded Images")
    # Ensure face detection has been run in the background
    if "faces_by_group" not in st.session_state or "face_counts" not in st.session_state:
        with st.spinner("Detecting faces..."):
            st.session_state.faces_by_group, st.session_state.face_counts = load_faces()
    faces_by_group = st.session_state.faces_by_group
    face_counts = st.session_state.face_counts

    image_files = get_image_files()
    if not image_files:
        st.warning("No images uploaded. Please upload files in 'All Files' tab.")
        if "faces_by_group" in st.session_state:
            del st.session_state.faces_by_group
        st.stop()

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

        # Show all images belonging to the person (removed individual large images)

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
    file_list = get_cached_file_list()

    files_per_page = 20
    total_pages = max(1, (len(file_list) - 1) // files_per_page + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
    start = (page - 1) * files_per_page
    end = start + files_per_page

    cols = st.columns(5)
    for idx, file in enumerate(file_list[start:end]):
        file_path = upload_path(file)
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                thumb_b64 = get_image_base64_cached(file_path)
            except Exception as e:
                st.warning(f"Could not load image {file}: {e}")
                continue
            modal_id = f"modal_{file}"
            thumb_html = f"""
            <div style='text-align: center;'>
                <img src='data:image/jpeg;base64,{thumb_b64}' style='width: 100px; height: 100px; object-fit: cover; cursor: pointer; border-radius: 4px;' onclick="document.getElementById('{modal_id}').style.display='block'"/>
                <p style='font-size: 12px; color: #555;'>{file}</p>
                <div id='{modal_id}' style='display:none; position:fixed; z-index:1000; left:0; top:0; width:100%; height:100%; background-color: rgba(0,0,0,0.8);'>
                    <span onclick="document.getElementById('{modal_id}').style.display='none'" style='position:absolute;top:20px;right:35px;color:#fff;font-size:40px;font-weight:bold;cursor:pointer;'>&times;</span>
                    <img src='data:image/jpeg;base64,{thumb_b64}' style='margin:auto;display:block;max-width:90%;max-height:90%;position:relative;top:50%;transform:translateY(-50%);'/>
                </div>
            </div>
            """
            with cols[idx % 5]:
                st.markdown(thumb_html, unsafe_allow_html=True)
                if st.button(f"🗑 Delete", key=f"delete_{file}"):
                    delete_file(file)
                    st.rerun()
        else:
            with cols[idx % 5]:
                st.markdown(f"<p>📄 {file}</p>")
                if st.button("🗑 Delete", key=f"delete_{file}"):
                    delete_file(file)
                    if "view_file" in st.session_state:
                        del st.session_state["view_file"]
                    st.rerun()

    if "view_file" in st.session_state:
        file_to_view = st.session_state["view_file"]
        full_path = upload_path(file_to_view)
        if os.path.exists(full_path):
            st.markdown("---")
            st.image(full_path, caption=f"Viewing: {file_to_view}", use_container_width=True)