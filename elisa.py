import os
import webbrowser
import whisper
import ollama
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import uuid
import logging
import pygame
import subprocess
import time
import numpy as np
import sys
import stat
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QTextEdit, QLineEdit, QScrollArea)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QMovie, QPixmap, QIcon, QFont, QPalette, QColor, QTextCursor
import librosa


# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'asistente.log')),
        logging.StreamHandler()
    ]
)

def verificar_sistema():
    """Verifica y crea la estructura necesaria de directorios y permisos"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    required_dirs = [
        os.path.join(script_dir, "temp_audio"),
        os.path.join(script_dir, "assets")
    ]
    
    for d in required_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            # Establecer permisos completos
            os.chmod(d, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            logging.info(f"Directorio creado: {d}")
        except Exception as e:
            logging.critical(f"Error creando directorio {d}: {str(e)}")
            sys.exit(1)
    
    try:
        test_file = os.path.join(script_dir, "temp_audio", "permiso_test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        logging.critical(f"ERROR DE PERMISOS: No se puede escribir en temp_audio")
        sys.exit(1)

# Configuración inicial
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_audio_dir = os.path.join(script_dir, "temp_audio")
temp_audio_path = os.path.join(temp_audio_dir, "grabacion.wav")
conversacion_path = os.path.join(script_dir, "conversacion.txt")

verificar_sistema()

# Cargar modelos
try:
    whisper_model = whisper.load_model("small")
except Exception as e:
    logging.critical(f"Error cargando modelo Whisper: {str(e)}")
    sys.exit(1)

model_name = "mistral"
nombre_asistente = "ELISA"
nombre_usuario = None

# Configuración de assets
avatar_quieto_gif = os.path.join(script_dir, "assets", "avatar_quieto.gif")
avatar_hablando_gif = os.path.join(script_dir, "assets", "avatar_hablando.gif")

class Estado:
    QUIETO = 0
    GRABANDO = 1
    HABLANDO = 2

class WorkerGrabacion(QThread):
    finished = pyqtSignal(str)
    update_status = pyqtSignal(str)
    
    def __init__(self, whisper_model, temp_audio_path):
        super().__init__()
        self.whisper_model = whisper_model
        self.temp_audio_path = temp_audio_path
        self._is_running = True
    
    def run(self):
        try:
            samplerate = 44100
            duration = 15
            
            self.update_status.emit(f"Grabando... {duration}s")
            audio = sd.rec(int(duration * samplerate), 
                         samplerate=samplerate, 
                         channels=1,
                         dtype='float32')
            
            for i in range(duration, 0, -1):
                if not self._is_running:
                    return
                self.update_status.emit(f"Grabando... {i}s")
                time.sleep(1)
            
            sd.wait()
            audio = self.mejorar_calidad_audio(audio, samplerate)
            
            try:
                sf.write(self.temp_audio_path, audio, samplerate)
                if not os.path.exists(self.temp_audio_path):
                    raise RuntimeError(f"Archivo no creado: {self.temp_audio_path}")
                logging.info(f"Archivo de audio creado: {self.temp_audio_path} ({os.path.getsize(self.temp_audio_path)} bytes)")
            except Exception as e:
                logging.error(f"Error al guardar audio: {str(e)}")
                raise
            
            texto = self.transcribir_audio()
            self.finished.emit(texto)
        except Exception as e:
            logging.error(f"Error en grabación: {str(e)}", exc_info=True)
            self.finished.emit("")
    
    def mejorar_calidad_audio(self, audio, samplerate):
        try:
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = audio / np.max(np.abs(audio))
            audio = np.convolve(audio, np.ones(5)/5, mode='same')
            return audio
        except Exception as e:
            logging.error(f"Error al mejorar audio: {e}")
            return audio
    
    def transcribir_audio(self):
        try:
            abs_path = os.path.abspath(self.temp_audio_path)
            if not os.path.exists(abs_path):
                logging.error(f"ARCHIVO NO ENCONTRADO: {abs_path}")
                logging.error(f"CONTENIDO DEL DIRECTORIO: {os.listdir(os.path.dirname(abs_path))}")
                return ""
            
            resultado = self.whisper_model.transcribe(
                rf"{abs_path}",
                language="spanish",
                task="transcribe",
                fp16=False,
                temperature=0.2,
                best_of=3,
                beam_size=5
            )
            
            texto = resultado["text"].strip()
            texto = self.limpiar_texto_transcrito(texto)
            return texto
        except Exception as e:
            logging.error(f"Error al transcribir: {str(e)}", exc_info=True)
            return ""
    
    def limpiar_texto_transcrito(self, texto):
        palabras_confusas = ["喝水", "thereel", "谢谢", "gracias", "thank you"]
        return ' '.join([palabra for palabra in texto.split() if palabra.lower() not in palabras_confusas]).capitalize()
    
    def stop(self):
        self._is_running = False
        self.terminate()

class WorkerHablar(QThread):
    finished = pyqtSignal()
    
    def __init__(self, texto, temp_audio_dir):
        super().__init__()
        self.texto = texto
        self.temp_audio_dir = temp_audio_dir
    
    def run(self):
        temp_tts_path = None
        try:
            os.makedirs(self.temp_audio_dir, exist_ok=True)
            temp_tts_path = os.path.join(self.temp_audio_dir, f"respuesta_{uuid.uuid4()}.mp3")
            
            tts = gTTS(text=self.texto, lang="es", slow=False)
            tts.save(temp_tts_path)
            
            if not os.path.exists(temp_tts_path):
                raise FileNotFoundError(f"Archivo de audio no generado: {temp_tts_path}")
            
            pygame.mixer.music.load(temp_tts_path.replace("\\", "/"))
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            
        except Exception as e:
            logging.error(f"ERROR en reproducción: {str(e)}", exc_info=True)
        finally:
            if temp_tts_path and os.path.exists(temp_tts_path):
                try:
                    pygame.time.wait(500)
                    os.remove(temp_tts_path)
                except Exception as e:
                    logging.error(f"Error eliminando archivo temporal: {str(e)}")
            self.finished.emit()

class AsistenteVirtualGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.nombre_asistente = nombre_asistente
        self.nombre_usuario = nombre_usuario
        self.estado_actual = Estado.QUIETO
        self.conversacion = []
        
        pygame.init()
        pygame.mixer.init()
        
        self.setWindowTitle(f"Asistente Virtual {self.nombre_asistente}")
        self.setGeometry(100, 100, 1000, 700)
        self.setup_ui()
        
        mensaje_inicial = f"{self.nombre_asistente}: ¡Hola! Soy {self.nombre_asistente}, tu asistente virtual. ¿Cómo te llamas?"
        self.agregar_mensaje(mensaje_inicial)
        self.hablar(mensaje_inicial.split(": ")[1])
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        left_column = QVBoxLayout()
        left_column.setSpacing(15)
        
        self.avatar_label = QLabel()
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.avatar_label.setFixedSize(300, 500)
        self.cargar_avatar(avatar_quieto_gif)
        
        self.avatar_label.setStyleSheet("""
            QLabel {
                background-color: #eaeaea;
                border: 2px solid #d0d0d0;
                border-radius: 10px;
            }
        """)
        
        left_column.addWidget(self.avatar_label)
        
        self.grabar_button = QPushButton("Grabar Audio")
        self.grabar_button.setObjectName("grabarButton")
        self.grabar_button.setIcon(QIcon.fromTheme("microphone"))
        self.grabar_button.setIconSize(QSize(24, 24))
        self.grabar_button.setFixedHeight(40)
        self.grabar_button.clicked.connect(self.iniciar_grabacion)
        
        left_column.addWidget(self.grabar_button)
        left_column.addStretch()
        
        right_column = QVBoxLayout()
        right_column.setSpacing(15)
        
        title_label = QLabel(f"Conversación con {self.nombre_asistente}")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333333;
                padding-bottom: 10px;
                border-bottom: 2px solid #4b8bbe;
            }
        """)
        right_column.addWidget(title_label)
        
        self.conversacion_text = QTextEdit()
        self.conversacion_text.setReadOnly(True)
        self.conversacion_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                color: #333333;
            }
        """)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.conversacion_text)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none;")
        right_column.addWidget(scroll_area)
        
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Escribe tu mensaje aquí...")
        self.input_line.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
                color: #333333;
            }
        """)
        self.input_line.returnPressed.connect(self.enviar_mensaje)
        right_column.addWidget(self.input_line)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.enviar_button = QPushButton("Enviar")
        self.enviar_button.setIcon(QIcon.fromTheme("mail-send"))
        self.enviar_button.clicked.connect(self.enviar_mensaje)
        button_layout.addWidget(self.enviar_button)
        
        self.limpiar_button = QPushButton("Limpiar")
        self.limpiar_button.setIcon(QIcon.fromTheme("edit-clear"))
        self.limpiar_button.clicked.connect(self.limpiar_conversacion)
        button_layout.addWidget(self.limpiar_button)
        
        self.cerrar_button = QPushButton("Cerrar")
        self.cerrar_button.setObjectName("cerrarButton")
        self.cerrar_button.setIcon(QIcon.fromTheme("window-close"))
        self.cerrar_button.clicked.connect(self.close)
        button_layout.addWidget(self.cerrar_button)
        
        right_column.addLayout(button_layout)
        
        main_layout.addLayout(left_column, 30)
        main_layout.addLayout(right_column, 70)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #4b8bbe;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #6fa8dc;
            }
            QPushButton:pressed {
                background-color: #3a6a94;
            }
            #grabarButton {
                background-color: #2ecc71;
            }
            #grabarButton:hover {
                background-color: #27ae60;
            }
            #grabarButton:pressed {
                background-color: #219653;
            }
            #cerrarButton {
                background-color: #e74c3c;
            }
            #cerrarButton:hover {
                background-color: #c0392b;
            }
        """)
    
    def cargar_avatar(self, gif_path):
        if os.path.exists(gif_path):
            self.avatar_movie = QMovie(gif_path)
            self.avatar_movie.setScaledSize(QSize(300, 500))
            self.avatar_label.setMovie(self.avatar_movie)
            self.avatar_movie.start()
        else:
            pixmap = QPixmap(300, 500)
            pixmap.fill(QColor(234, 234, 234))
            self.avatar_label.setPixmap(pixmap)
    
    def cambiar_estado_avatar(self, estado):
        if estado == Estado.QUIETO:
            self.cargar_avatar(avatar_quieto_gif)
        elif estado in (Estado.GRABANDO, Estado.HABLANDO):
            self.cargar_avatar(avatar_hablando_gif)
    
    def agregar_mensaje(self, mensaje):
        self.conversacion.append(mensaje)
        self.guardar_conversacion(mensaje)
        
        cursor = self.conversacion_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        if mensaje.startswith(f"{self.nombre_asistente}:"):
            color = "#2c3e50"
            prefix = f"<b>{self.nombre_asistente}:</b> "
            texto = mensaje[len(f"{self.nombre_asistente}:"):].strip()
        else:
            color = "#27ae60"
            prefix = "<b>Tú:</b> "
            texto = mensaje[len("Tú:"):].strip() if mensaje.startswith("Tú:") else mensaje
        
        if self.conversacion_text.toPlainText():
            cursor.insertHtml("<hr style='margin: 10px 0; border: 1px solid #eee;'>")
        
        cursor.insertHtml(f"""
            <div style='color: {color}; margin: 5px 0;'>
                {prefix}{texto}
            </div>
        """)
        
        self.conversacion_text.verticalScrollBar().setValue(
            self.conversacion_text.verticalScrollBar().maximum())
    
    def guardar_conversacion(self, mensaje):
        try:
            with open(conversacion_path, "a", encoding="utf-8") as archivo:
                archivo.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {mensaje}\n")
        except Exception as e:
            logging.error(f"Error al guardar conversación: {e}")
    
    def enviar_mensaje(self):
        texto = self.input_line.text().strip()
        if texto:
            self.agregar_mensaje(f"Tú: {texto}")
            self.input_line.clear()
            
            respuesta = self.generar_respuesta(texto)
            self.agregar_mensaje(f"{self.nombre_asistente}: {respuesta}")
            self.hablar(respuesta)
            
            self.ejecutar_comando(texto)
    
    def iniciar_grabacion(self):
        if hasattr(self, 'worker_grabacion') and self.worker_grabacion.isRunning():
            return
        
        self.cambiar_estado_avatar(Estado.GRABANDO)
        self.grabar_button.setEnabled(False)
        self.grabar_button.setText("Grabando...")
        
        self.worker_grabacion = WorkerGrabacion(whisper_model, temp_audio_path)
        self.worker_grabacion.finished.connect(self.finalizar_grabacion)
        self.worker_grabacion.update_status.connect(
            lambda msg: self.agregar_mensaje(f"{self.nombre_asistente}: {msg}"))
        self.worker_grabacion.start()
    
    def finalizar_grabacion(self, texto):
        self.grabar_button.setEnabled(True)
        self.grabar_button.setText("Grabar Audio")
        self.cambiar_estado_avatar(Estado.QUIETO)
        
        if texto:
            self.agregar_mensaje(f"Tú: {texto}")
            
            respuesta = self.generar_respuesta(texto)
            self.agregar_mensaje(f"{self.nombre_asistente}: {respuesta}")
            self.hablar(respuesta)
            
            self.ejecutar_comando(texto)
    
    def generar_respuesta(self, texto):
        texto_lower = texto.lower()
        
        if self.nombre_usuario is None:
            if "me llamo" in texto_lower:
                self.nombre_usuario = texto_lower.split("me llamo")[-1].strip().title()
            elif "mi nombre es" in texto_lower:
                self.nombre_usuario = texto_lower.split("mi nombre es")[-1].strip().title()
            elif "soy" in texto_lower:
                self.nombre_usuario = texto_lower.split("soy")[-1].strip().title()
            
            if self.nombre_usuario and len(self.nombre_usuario) > 1:
                respuesta = f"¡Mucho gusto, {self.nombre_usuario}! ¿En qué puedo ayudarte hoy?"
                logging.info(f"Nombre detectado: {self.nombre_usuario}")
                return respuesta
        
        prompt = (
            f"Eres {self.nombre_asistente}, un asistente virtual en español. "
            f"{f'El usuario {self.nombre_usuario} te dice:' if self.nombre_usuario else 'Usuario:'} {texto}\n"
            f"Responde de manera clara y concisa en español (máximo 50 palabras):"
        )
        
        try:
            respuesta = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={"max_tokens": 50}
            )
            return respuesta["response"]
        except Exception as e:
            logging.error(f"Error al generar respuesta: {e}")
            return "Lo siento, no pude procesar tu solicitud."
    
    def hablar(self, texto):
        self.cambiar_estado_avatar(Estado.HABLANDO)
        
        self.worker_hablar = WorkerHablar(texto, temp_audio_dir)
        self.worker_hablar.finished.connect(
            lambda: self.cambiar_estado_avatar(Estado.QUIETO))
        self.worker_hablar.start()
    
    def ejecutar_comando(self, texto):
        texto = texto.lower()
        
        comandos = {
            "abrir chrome": lambda: subprocess.Popen("chrome.exe"),
            "abrir notepad": lambda: subprocess.Popen("notepad.exe"),
            "abrir calculadora": lambda: subprocess.Popen("calc.exe"),
            "ir a ": lambda url: webbrowser.open(f"https://{url}" if not url.startswith(("http", "www")) else url),
            "reproducir ": lambda cancion: webbrowser.open(f"https://www.youtube.com/results?search_query={cancion}")
        }
        
        for cmd, accion in comandos.items():
            if texto.startswith(cmd):
                try:
                    parametro = texto[len(cmd):].strip()
                    if parametro:
                        accion(parametro)
                    else:
                        accion()
                    return True
                except Exception as e:
                    logging.error(f"Error al ejecutar comando {cmd}: {e}")
        return False
    
    def limpiar_conversacion(self):
        self.conversacion_text.clear()
        self.conversacion = []
    
    def closeEvent(self, event):
        if hasattr(self, 'worker_grabacion') and self.worker_grabacion.isRunning():
            self.worker_grabacion.stop()
        
        if hasattr(self, 'worker_hablar') and self.worker_hablar.isRunning():
            self.worker_hablar.terminate()
        
        pygame.quit()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    font = QFont("Arial", 12)
    app.setFont(font)
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(51, 51, 51))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(51, 51, 51))
    palette.setColor(QPalette.Text, QColor(51, 51, 51))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(51, 51, 51))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(75, 139, 190))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = AsistenteVirtualGUI()
    window.show()
    sys.exit(app.exec_())