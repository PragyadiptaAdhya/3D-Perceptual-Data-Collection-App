import sys, os, pandas as pd, random, argparse
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QOpenGLWidget, QSlider, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from PIL import Image
import trimesh
import time


class FaceWidget(QOpenGLWidget):
    def __init__(self, parts_list):
        super().__init__()
        self.parts_list = parts_list
        self.meshes_list = []
        self.rot_x = self.rot_y = self.rot_z = 0.0
        self.last_mouse_pos = None


        self.yaw_face1 = 14.0
        self.yaw_face2 = -16.0


        self.auto_rotate_angle = 0.0
        self.auto_rotate_direction = 1
        self.max_yaw = 30  
        self.rotation_speed = 4.5 
        self.rotation_duration = 20 

        
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.update_rotation)
        self.rotation_timer.start(30) 
        self.start_time = time.time()

    def update_rotation(self):
        
        if time.time() - self.start_time > self.rotation_duration:
            self.rotation_timer.stop()
            return

        
        self.auto_rotate_angle += self.auto_rotate_direction * self.rotation_speed
        if self.auto_rotate_angle > self.max_yaw or self.auto_rotate_angle < -self.max_yaw:
            self.auto_rotate_direction *= -1
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING)
        glClearColor(1,1,1,1)
        self.load_meshes()

    def load_meshes(self):
        self.meshes_list = []
        all_vertices = []
        for parts in self.parts_list:
            for obj_path in parts:
                verts, _, _, _ = self.load_obj_with_uvs(obj_path)
                if verts is not None:
                    all_vertices.append(verts)
        if not all_vertices: return
        all_vertices = np.vstack(all_vertices)
        center = np.mean(all_vertices, axis=0)
        scale = np.max(np.linalg.norm(all_vertices - center, axis=1))
        for parts in self.parts_list:
            face_meshes = []
            for obj_path, tex_path in parts.items():
                verts, uv, norms, faces = self.load_obj_with_uvs(obj_path)
                if verts is None: continue
                verts = (verts - center)/scale * 1.5
                tex_id = self.load_texture(tex_path)
                face_meshes.append({'verts': verts,'uv': uv,'norms': norms,'faces': faces,'tex_id': tex_id})
            self.meshes_list.append(face_meshes)

    def load_obj_with_uvs(self, obj_path):
        if not os.path.exists(obj_path): return None,None,None,None
        try:
            mesh = trimesh.load(obj_path, force='mesh', skip_materials=True)
        except:
            return None,None,None,None
        if not mesh.is_watertight: mesh.fill_holes()
        vertices = np.array(mesh.vertices,dtype=np.float32)
        faces = np.array(mesh.faces,dtype=np.int32)
        normals = np.array(mesh.vertex_normals,dtype=np.float32)
        uv = np.array(mesh.visual.uv,dtype=np.float32) if hasattr(mesh.visual,'uv') and mesh.visual.uv is not None else np.zeros((len(vertices),2),dtype=np.float32)
        return vertices, uv, normals, faces

    def load_texture(self, tex_path):
        if tex_path is None or not os.path.exists(tex_path):
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            data = np.ones((2,2,3),dtype=np.uint8)*255
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

    def resizeGL(self,w,h):
        glViewport(0,0,w,h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45,w/h if h!=0 else 1,0.1,100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0,0,5,0,0,0,0,1,0)
        offsets = [-1.5,1.5]
        for i, face_meshes in enumerate(self.meshes_list):
            glPushMatrix()
            glTranslatef(offsets[i],0,0)
            glRotatef(self.rot_x,1,0,0)

        
            if i == 0:
                glRotatef(self.yaw_face1 + self.auto_rotate_angle,0,1,0)
            else:
                glRotatef(self.yaw_face2 + self.auto_rotate_angle,0,1,0)

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

    def mousePressEvent(self,event):
        self.last_mouse_pos = event.pos()
    def mouseMoveEvent(self,event):
        if self.last_mouse_pos is None: return
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        self.rot_x += dy
        self.rot_y += dx
        self.last_mouse_pos = event.pos()
        self.update()
    def mouseReleaseEvent(self,event):
        self.last_mouse_pos = None



class LabeledSlider(QSlider):
    def __init__(self, labels, colors=None, *args, **kwargs):
        super().__init__(Qt.Horizontal, *args, **kwargs)
        self.labels = labels
        self.colors = colors if colors else ["black"] * len(labels)
        self.setMinimum(-50)
        self.setMaximum(50)
        self.setSingleStep(1)
        self.setPageStep(1)
        self.setTickPosition(QSlider.NoTicks)
        self.setTracking(True)
        self.left_padding = 20
        self.right_padding = 20
        self.setFixedHeight(80)
        self.setFixedWidth(600)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        slider_length = self.width() - self.left_padding - self.right_padding
        step = slider_length / (len(self.labels) - 1)
        for i, text in enumerate(self.labels):
            x = self.left_padding + i * step
            painter.setPen(QColor(self.colors[i]))
            rect = QRectF(x - 30, self.height() - 35, 60, 40)
            painter.drawText(rect, Qt.AlignCenter, text)

    def get_value(self):
        return self.value() / 10.0

    def set_float_value(self, val):
        self.setValue(int(val*10))



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

        if os.path.exists(output_file):
            try:
                self.responses_df = pd.read_excel(output_file)
            except:
                self.responses_df = pd.DataFrame(columns=["Folder_Name","Rating"])
        else:
            self.responses_df = pd.DataFrame(columns=["Folder_Name","Rating"])

        self.face_widget = FaceWidget([self.face1_parts, self.face2_parts_list[self.current_index]])

        likert_labels = ["Awful", "Poor", "Bad", "Neutral", "Fair", "Good", "Awsome"]
        colors = ["darkred", "red", "orange", "white", "lightblue", "green", "darkgreen"]
        self.slider = LabeledSlider(likert_labels, colors=colors)
        self.slider.set_float_value(0.0)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_item)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20,20,20,20)
        main_layout.addWidget(self.face_widget, stretch=5)

        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.slider, alignment=Qt.AlignCenter)
        slider_layout.addWidget(self.next_button, alignment=Qt.AlignCenter)
        main_layout.addLayout(slider_layout)
        main_layout.addStretch(1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.likert_labels = likert_labels

    def next_item(self):
        folder_name = self.folder_names[self.current_index]
        slider_val = self.slider.get_value()
        self.responses_df = pd.concat([
            self.responses_df,
            pd.DataFrame([[folder_name, slider_val]], columns=["Folder_Name","Rating"])
        ], ignore_index=True)
        self.responses_df.to_excel(self.output_file, index=False)

        self.current_index += 1
        if self.current_index >= len(self.face2_parts_list):
            QtWidgets.QApplication.quit()
            return

        
        self.face_widget.parts_list = [self.face1_parts, self.face2_parts_list[self.current_index]]
        self.face_widget.load_meshes()
        self.face_widget.update()

        
        self.face_widget.auto_rotate_angle = 0.0
        self.face_widget.auto_rotate_direction = 1
        self.face_widget.start_time = time.time()
        self.face_widget.rotation_timer.start(30)

        self.slider.set_float_value(0.0)
        self.slider.update()




def get_face_parts_from_folder(folder_path):
    obj_folder = os.path.join(folder_path,"OBJ")
    textures_root = os.path.join(folder_path,"Textures","JPG")
    parts = {}
    if not os.path.exists(obj_folder): return parts
    for obj_file in os.listdir(obj_folder):
        if not obj_file.lower().endswith(".obj"): continue
        obj_path = os.path.join(obj_folder,obj_file)
        tex_path = None
        name_lower = obj_file.lower()
        if "head" in name_lower: tex_path=os.path.join(textures_root,"Face","Face_Albedo.jpg")
        elif "teeth" in name_lower: tex_path=os.path.join(textures_root,"Mouth","Teeth_diffuse.jpg")
        elif "eyeball" in name_lower: tex_path=os.path.join(textures_root,"Eyes","Eyes_Balls_Diffuse.jpg")
        if tex_path and os.path.exists(tex_path): parts[obj_path]=tex_path
    return parts

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Face Rating Tool")
    parser.add_argument("--folder1", type=str, required=True)
    parser.add_argument("--folder2", type=str, required=True)
    parser.add_argument("--xl_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    face1_parts = get_face_parts_from_folder(args.folder1)
    if not face1_parts: print("No valid left face found."); sys.exit()
    face2_parts_list, folder_names = [], []
    subfolders = [os.path.join(args.folder2,d) for d in os.listdir(args.folder2) if os.path.isdir(os.path.join(args.folder2,d))]
    for subf in subfolders:
        parts = get_face_parts_from_folder(subf)
        if parts:
            face2_parts_list.append(parts)
            folder_names.append(os.path.basename(subf))
    if not face2_parts_list: print("No valid right faces."); sys.exit()
    combined = list(zip(folder_names,face2_parts_list))
    random.shuffle(combined)
    folder_names, face2_parts_list = zip(*combined)
    folder_names = list(folder_names)
    face2_parts_list = list(face2_parts_list)

    output_file = args.xl_file
    if not output_file.lower().endswith(".xlsx"):
        output_file += ".xlsx"

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(face1_parts, face2_parts_list, folder_names, output_file=output_file)
    w.show()
    sys.exit(app.exec_())
