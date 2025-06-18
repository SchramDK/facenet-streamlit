import os
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
from facenet_pytorch import MTCNN, InceptionResnetV1
import base64
from io import BytesIO

# CSS styling til personbokse
st.markdown("""
<style>
.person-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 175px;
    width: 140px;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 15px;
    cursor: pointer;
    user-select: none;
    position: relative;
}
.person-box img {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    object-fit: cover;
    margin-bottom: 10px;
}
.person-box strong {
    text-align: center;
    display: block;
}
.person-box span {
    color: gray;
    text-align: center;
    display: block;
}
/* Gør knappen usynlig men klikbar */
.person-btn {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    opacity: 0;
    border: none;
    cursor: pointer;
    background: transparent;
    z-index: 1;
}
</style>
""", unsafe_allow_html=True)

if "menu" not in st.session_state:
    st.session_state.menu = "All Files"
if "person" not in st.session_state:
    st.session_state.person = None

def set_menu():
    st.session_state.menu = st.session_state.menu_radio
    st.session_state.person = None

def select_person(idx):
    st.session_state.person = idx + 1
    st.session_state.menu = "People"

st.sidebar.radio(
    "Navigation",
    ["All Files", "People"],
    index=["All Files", "People"].index(st.session_state.menu),
    key="menu_radio",
    on_change=set_menu,
)

menu = st.session_state.menu

if st.session_state.get("person") is None and "person_select" in st.experimental_get_query_params():
    st.session_state.person = int(st.experimental_get_query_params()["person_select"][0]) + 1
    st.session_state.menu = "People"

if menu == "People":
    st.title("🧑 Detected People in Uploaded Images")

    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    if os.path.exists("uploads"):
        file_list = os.listdir("uploads")
        image_files = [f for f in file_list if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_files = []

    @st.cache_resource
    def detect_faces_grouped(image_files):
        faces_by_group = []
        embeddings = []

        for file in image_files:
            path = os.path.join("uploads", file)
            image = Image.open(path).convert('RGB')
            boxes, _ = mtcnn.detect(image)
            if boxes is None or len(boxes) == 0:
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

    faces_by_group = detect_faces_grouped(image_files)

    selected_person = st.session_state.person
    if selected_person is not None and 1 <= selected_person <= len(faces_by_group):
        idx = selected_person - 1
        group_faces = faces_by_group[idx]

        profile_face = group_faces[0][0]
        buf = BytesIO()
        profile_face.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px; width: 100%;">
            <img src="data:image/jpeg;base64,{b64}" style="border-radius: 50%; width: 80px; height: 80px; object-fit: cover;" />
            <div>
                <h2 style="margin: 0;">Person {selected_person}</h2>
                <p style="margin: 0; color: gray;">{len(group_faces)} photo(s)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⬅ Back to overview"):
            st.session_state.person = None

        unique_files = sorted(set([src_file for (_, src_file) in group_faces]))
        st.markdown("### Photo Gallery")
        gallery_cols = st.columns(4)
        for i, file in enumerate(unique_files):
            full_path = os.path.join("uploads", file)
            gallery_cols[i % 4].image(full_path, caption=file, use_container_width=True)

    else:
        st.markdown("## People Overview")

        cols = st.columns(5)
        for idx, group_faces in enumerate(faces_by_group):
            profile_face = group_faces[0][0]
            photo_count = len(group_faces)
            display_name = f"Person {idx+1}"

            buf = BytesIO()
            profile_face.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            with cols[idx % 5]:
                st.markdown(f"""
                    <form method="post">
                        <button name="person_select" value="{idx}" class="person-box" type="submit">
                            <img src="data:image/jpeg;base64,{b64}" alt="{display_name}" />
                            <strong>{display_name}</strong>
                            <span>Photos: {photo_count}</span>
                        </button>
                    </form>
                """, unsafe_allow_html=True)

elif menu == "All Files":
    st.title("📂 All Files")

    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
    if uploaded_files:
        os.makedirs("uploads", exist_ok=True)
        for file in uploaded_files:
            file_path = os.path.join("uploads", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"{len(uploaded_files)} file(s) uploaded.")

    st.markdown("### Uploaded Files")
    if os.path.exists("uploads"):
        file_list = os.listdir("uploads")
        for file in file_list:
            file_path = os.path.join("uploads", file)
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