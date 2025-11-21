


import sys, os, pandas as pd
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QOpenGLWidget, QSlider, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QWidget
)
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from PIL import Image
import trimesh


import argparse
import random


def load_obj_with_uvs(obj_path):
    if not os.path.exists(obj_path):
        print(f"Warning: {obj_path} not found. Skipping.")
        return None, None, None, None
    try:
        mesh = trimesh.load(obj_path, force='mesh', skip_materials=True)
    except Exception as e:
        print(f"Failed to load {obj_path}: {e}")
        return None, None, None, None

    if not mesh.is_watertight:
        mesh.fill_holes()
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    normals = np.array(mesh.vertex_normals, dtype=np.float32)

    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uv = np.array(mesh.visual.uv, dtype=np.float32)
        if len(uv) < len(vertices):
            uv_new = np.zeros((len(vertices),2), dtype=np.float32)
            uv_new[:len(uv)] = uv
            uv_new[len(uv):] = uv[-1]
            uv = uv_new
    else:
        uv = np.zeros((len(vertices),2), dtype=np.float32)
    return vertices, uv, normals, faces


def load_texture(tex_path):
    if tex_path is None or not os.path.exists(tex_path):
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        data = np.ones((2,2,3), dtype=np.uint8)*255
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,2,2,0,GL_RGB,GL_UNSIGNED_BYTE,data.tobytes())
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D,0)
        return tex_id

    img = Image.open(tex_path).convert("RGB")
    img = np.array(img)[::-1,:,:]
    h,w = img.shape[:2]

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,w,h,0,GL_RGB,GL_UNSIGNED_BYTE,img.tobytes())
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D,0)
    return tex_id

# ------------------ FaceWidget ------------------
class FaceWidget(QOpenGLWidget):
    def __init__(self, parts_list):
        super().__init__()
        self.parts_list = parts_list
        self.meshes_list = []
        self.rot_x = self.rot_y = self.rot_z = 0.0
        self.step = 10.0
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.last_mouse_pos = None

        self.left_face_rot_y = 14
        self.right_face_rot_y = -15

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING)
        glClearColor(1,1,1,1)

        self.parts_list = [p for p in self.parts_list if len(p) > 0]
        if not self.parts_list:
            return

        # compute scale and center
        all_vertices = []
        for parts in self.parts_list:
            for obj_path in parts:
                verts, _, _, _ = load_obj_with_uvs(obj_path)
                if verts is not None:
                    all_vertices.append(verts)
        if not all_vertices:
            return
        all_vertices = np.vstack(all_vertices)
        center = np.mean(all_vertices, axis=0)
        scale = np.max(np.linalg.norm(all_vertices - center, axis=1))

        self.meshes_list = []
        for parts in self.parts_list:
            face_meshes = []
            for obj_path, tex_path in parts.items():
                verts, uv, norms, faces = load_obj_with_uvs(obj_path)
                if verts is None:
                    continue
                verts = (verts - center)/scale * 1.5
                tex_id = load_texture(tex_path)
                face_meshes.append({
                    'verts': verts,
                    'uv': uv,
                    'norms': norms,
                    'faces': faces,
                    'tex_id': tex_id
                })
            self.meshes_list.append(face_meshes)

    def resizeGL(self, w, h):
        glViewport(0,0,w,h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h if h!=0 else 1, 0.1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0,0,5, 0,0,0, 0,1,0)

        offsets = [-1.5, 1.5]
        for i, face_meshes in enumerate(self.meshes_list):
            glPushMatrix()
            glTranslatef(offsets[i],0,0)
            glRotatef(self.rot_x,1,0,0)
            if i == 0: glRotatef(self.left_face_rot_y,0,1,0)
            else: glRotatef(self.right_face_rot_y,0,1,0)
            glRotatef(self.rot_y,0,1,0)
            glRotatef(self.rot_z,0,0,1)

            for mesh in face_meshes:
                glBindTexture(GL_TEXTURE_2D, mesh['tex_id'])
                glBegin(GL_TRIANGLES)
                for f in mesh['faces']:
                    for idx_v in f:
                        glNormal3fv(mesh['norms'][idx_v])
                        glTexCoord2fv(mesh['uv'][idx_v])
                        glVertex3fv(mesh['verts'][idx_v])
                glEnd()
                glBindTexture(GL_TEXTURE_2D,0)
            glPopMatrix()

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            return
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        self.rot_x += dy
        self.rot_y += dx
        self.last_mouse_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.last_mouse_pos = None

    # ------------------ Arrow Key Rotation ------------------
    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Up:
            self.rot_x -= self.step
        elif key == QtCore.Qt.Key_Down:
            self.rot_x += self.step
        elif key == QtCore.Qt.Key_Left:
            self.rot_y -= self.step
        elif key == QtCore.Qt.Key_Right:
            self.rot_y += self.step
        self.update()


# ------------------ Main Window ------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, face1_parts, face2_parts_list, folder_names, output_file="Responses.xlsx"):
        super().__init__()
        self.setWindowTitle("Face Rating Tool")
        self.setGeometry(100, 100, 1200, 700)

        self.face1_parts = face1_parts
        self.face2_parts_list = face2_parts_list
        self.folder_names = folder_names
        self.current_index = 0
        self.output_file = output_file

        # Load existing Excel if available
        if os.path.exists(output_file):
            self.responses = pd.read_excel(output_file).values.tolist()
        else:
            self.responses = []

        # Track yes/no selection
        self.percept_answer = None

        # OpenGL widget
        self.face_widget = FaceWidget([self.face1_parts, self.face2_parts_list[self.current_index]])

        # --- UI ELEMENTS ---
        self.percept_label = QLabel("Acceptable?")
        self.percept_label.setFixedHeight(20)

        self.yes_button = QPushButton("Yes")
        self.no_button = QPushButton("No")
        self.yes_button.clicked.connect(lambda: self.set_percept("YES"))
        self.no_button.clicked.connect(lambda: self.set_percept("NO"))

        self.slider_label = QLabel("Acceptance Level: 0")
        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(10)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_slider_label)

        # Next button
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_item)

        # Layouts
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.yes_button)
        buttons_layout.addWidget(self.no_button)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.slider)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.face_widget)
        main_layout.addWidget(self.percept_label)
        main_layout.addLayout(buttons_layout)
        main_layout.addLayout(slider_layout)
        main_layout.addWidget(self.next_button)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # --- Handlers ---
    def set_percept(self, val):
        self.percept_answer = val
        print("Perception:", val)

    def update_slider_label(self, value):
        self.slider_label.setText(f"Acceptance Level: {value}")

    def next_item(self):
        if self.percept_answer is None:
            print("Please answer YES/NO.")
            return

        slider_value = self.slider.value()
        folder_name = self.folder_names[self.current_index]

        # Append response immediately to Excel
        self.responses.append([folder_name, slider_value, self.percept_answer])
        df = pd.DataFrame(self.responses, columns=["Folder_Name","Acceptance Level","Perceptually Different"])
        df.to_excel(self.output_file, index=False)
        print("Saved:", folder_name, slider_value, self.percept_answer)

        # Reset yes/no for next item
        self.percept_answer = None

        # Move forward
        self.current_index += 1
        if self.current_index >= len(self.face2_parts_list):
            print("All done. Exiting.")
            QtWidgets.QApplication.quit()
            return

        # Load next pair
        self.face_widget.parts_list = [self.face1_parts, self.face2_parts_list[self.current_index]]
        self.face_widget.initializeGL()
        self.face_widget.update()
        self.slider.setValue(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face Rating Tool - compare face pairs and store responses in Excel"
    )
    parser.add_argument(
        "--folder1", type=str, required=True,
        help="Path to the left face folder (Face 1)"
    )
    parser.add_argument(
        "--folder2", type=str, required=True,
        help="Path to the root folder containing right face subfolders (Face 2)"
    )
    parser.add_argument(
        "--xl_file", type=str, required=True,
        help="Path to the Excel file to store responses"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed value for shuffling the right face folders (default: 42)"
    )

    args = parser.parse_args()

    face1_folder = args.folder1
    face2_root_folder = args.folder2
    output_excel_path = args.xl_file
    SEED_VALUE = args.seed

    def get_face_parts_from_folder(folder_path):
        obj_folder = os.path.join(folder_path, "OBJ")
        textures_root = os.path.join(folder_path, "Textures", "JPG")
        parts = {}
        if not os.path.exists(obj_folder): return parts
        for obj_file in os.listdir(obj_folder):
            if not obj_file.lower().endswith(".obj"): continue
            obj_path = os.path.join(obj_folder, obj_file)
            tex_path = None
            name_lower = obj_file.lower()
            if "head" in name_lower:
                tex_path = os.path.join(textures_root, "Face", "Face_Albedo.jpg")
            elif "teeth" in name_lower:
                tex_path = os.path.join(textures_root, "Mouth", "Teeth_diffuse.jpg")
            elif "eyeball" in name_lower:
                tex_path = os.path.join(textures_root, "Eyes", "Eyes_Balls_Diffuse.jpg")
            if tex_path and os.path.exists(tex_path):
                parts[obj_path] = tex_path
        return parts

    face1_parts = get_face_parts_from_folder(face1_folder)
    if not face1_parts:
        print("No valid left face found. Exiting...")
        sys.exit()

    face2_parts_list = []
    folder_names = []
    subfolders = [os.path.join(face2_root_folder, d)
                  for d in os.listdir(face2_root_folder)
                  if os.path.isdir(os.path.join(face2_root_folder,d))]

    for subf in subfolders:
        parts = get_face_parts_from_folder(subf)
        if parts:
            face2_parts_list.append(parts)
            folder_names.append(os.path.basename(subf))

    if not face2_parts_list:
        print("No valid right face subfolders. Exiting...")
        sys.exit()

    # --- Shuffle the data randomly with a seed ---
    random.seed(SEED_VALUE)
    combined = list(zip(folder_names, face2_parts_list))
    random.shuffle(combined)
    folder_names, face2_parts_list = zip(*combined)
    folder_names = list(folder_names)
    face2_parts_list = list(face2_parts_list)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(face1_parts, face2_parts_list[:105], folder_names, output_file=output_excel_path)
    w.show()
    sys.exit(app.exec_())
