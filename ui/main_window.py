#!/usr/bin/env python3
"""
YouTube Info Analyzer - Main Window UI
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLineEdit, QPushButton, QProgressBar, QTextEdit,
    QLabel, QFrame, QMenuBar, QMenu, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor, QAction
from pathlib import Path
from ui.colors import get_main_stylesheet, get_status_colors
from workers.download_worker import DownloadManager
from workers.download_worker import DownloadWorker
from workers.whisper_worker import WhisperWorker, WhisperManager


class MainWindow(QMainWindow):
    """Hauptfenster der Anwendung"""
    
    def __init__(self):
        super().__init__()
        
        # Download Manager initialisieren
        self.download_manager = DownloadManager(self)
        self.setup_download_connections()
        
        # Whisper Manager initialisieren
        self.whisper_manager = WhisperManager(self)
        self.current_whisper_worker = None
        
        self.init_ui()
        self.connect_signals()
        
    def setup_download_connections(self):
        """Download Manager Signals verbinden"""
        # Download Worker Signals weiterleiten
        self.download_manager.current_worker = None
        
        # Manager-Level Signals
        self.download_manager.analysis_ready.connect(self.on_audio_analysis_ready)
        self.download_manager.storage_ready.connect(self.on_video_storage_ready)
        self.download_manager.download_completed.connect(self.on_download_completed)
        
    def init_ui(self):
        """UI-Komponenten initialisieren"""
        self.setWindowTitle("YouTube Info Analyzer")
        self.setMinimumSize(600, 400)
        self.resize(800, 500)
        
        # Ocean Theme anwenden
        self.setStyleSheet(get_main_stylesheet())
        
        # Status Farben laden
        self.status_colors = get_status_colors()
        
        # Menü erstellen
        self.create_menu_bar()
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)
        
        # URL Input Section
        self.create_url_section(main_layout)
        
        # Progress Section  
        self.create_progress_section(main_layout)
        
        # Status Section
        self.create_status_section(main_layout)
        
        # Spacer am Ende
        main_layout.addStretch()
        
    def create_menu_bar(self):
        """Menü-Bar erstellen"""
        menu_bar = self.menuBar()
        
        # Config Menü
        config_menu = menu_bar.addMenu("⚙️ Konfiguration")
        
        # Config Dialog Action
        show_config_action = QAction("🔧 Einstellungen anzeigen", self)
        show_config_action.setShortcut("Ctrl+,")
        show_config_action.triggered.connect(self.show_config_dialog)
        config_menu.addAction(show_config_action)
        
        config_menu.addSeparator()
        
        # Reload Config Action
        reload_config_action = QAction("🔄 Konfiguration neu laden", self)
        reload_config_action.setShortcut("Ctrl+R")
        reload_config_action.triggered.connect(self.reload_config)
        config_menu.addAction(reload_config_action)
        
        # Secrets Menü
        secrets_menu = menu_bar.addMenu("🔐 Secrets")
        
        # Secret Status Action
        secret_status_action = QAction("📊 Secret Status prüfen", self)
        secret_status_action.triggered.connect(self.check_secret_status)
        secrets_menu.addAction(secret_status_action)
        
        # Hilfe Menü
        help_menu = menu_bar.addMenu("❓ Hilfe")
        
        # About Action
        about_action = QAction("ℹ️ Über", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_url_section(self, parent_layout):
        """URL-Eingabe Sektion erstellen"""
        # Label
        url_label = QLabel("YouTube URL:")
        url_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        parent_layout.addWidget(url_label)
        
        # URL Input + Button Layout
        url_layout = QHBoxLayout()
        url_layout.setSpacing(15)
        
        # URL Input Field
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self.url_input.setMinimumHeight(40)
        url_layout.addWidget(self.url_input)
        
        # Start Button
        self.start_button = QPushButton("🚀 Analyse starten")
        self.start_button.setObjectName("startButton")  # Für CSS-Targeting
        self.start_button.setMinimumHeight(40)
        self.start_button.setMinimumWidth(140)
        url_layout.addWidget(self.start_button)
        
        parent_layout.addLayout(url_layout)
        
    def create_progress_section(self, parent_layout):
        """Progress-Sektion erstellen"""
        # Separator
        separator = QFrame()
        separator.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        separator.setFixedHeight(2)
        parent_layout.addWidget(separator)
        
        # Progress Label
        self.progress_label = QLabel("🌊 Bereit für Analyse")
        self.progress_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        parent_layout.addWidget(self.progress_label)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(30)
        self.progress_bar.setValue(0)
        parent_layout.addWidget(self.progress_bar)
        
    def create_status_section(self, parent_layout):
        """Status-Anzeige Sektion erstellen"""
        # Status Label
        status_label = QLabel("📊 Status:")
        status_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        parent_layout.addWidget(status_label)
        
        # Status Display
        self.status_display = QTextEdit()
        self.status_display.setMaximumHeight(180)
        self.status_display.setReadOnly(True)
        parent_layout.addWidget(self.status_display)
        
        # Initial Status
        self.add_status_message("🌊 Warte auf YouTube-URL...", "info")
        
    def connect_signals(self):
        """Signal-Slot Verbindungen erstellen"""
        self.start_button.clicked.connect(self.on_start_clicked)
        self.url_input.returnPressed.connect(self.on_start_clicked)
        
    def on_start_clicked(self):
        """Start-Button geklickt - Echte Download-Analyse starten"""
        url = self.url_input.text().strip()
        
        if not url:
            self.add_status_message("❌ Bitte geben Sie eine YouTube-URL ein", "error")
            return
            
        if not self.is_valid_youtube_url(url):
            self.add_status_message("❌ Ungültige YouTube-URL", "error")
            return
        
        # UI für Verarbeitung vorbereiten
        self.start_button.setEnabled(False)
        self.start_button.setText("⏳ Läuft...")
        self.progress_bar.setValue(0)
        self.progress_label.setText("🚀 Starte Download...")
        
        self.add_status_message(f"🚀 Starte Analyse für: {url}", "working")
        
        try:
            # Download Manager starten (nur Audio zuerst)
            self.start_download_process(url)
            
        except Exception as e:
            self.handle_download_error(f"Fehler beim Start: {str(e)}")
            
    def start_download_process(self, url: str):
        """Download-Prozess starten"""
        # Worker erstellen und Signals verbinden
        from workers.download_worker import DownloadWorker
        
        if hasattr(self, 'current_worker') and self.current_worker:
            self.current_worker.stop_download()
            
        self.current_worker = DownloadWorker(self)
        
        # Worker Signals verbinden
        self.current_worker.progress_updated.connect(self.on_download_progress)
        self.current_worker.video_info_ready.connect(self.on_video_info_ready)
        self.current_worker.audio_ready.connect(self.on_audio_ready)
        self.current_worker.video_ready.connect(self.on_video_ready)
        self.current_worker.error_occurred.connect(self.handle_download_error)
        self.current_worker.finished.connect(self.on_download_finished)
        
        self.current_worker.set_download_params(url, download_audio=True, download_video=False)
        self.current_worker.start()
        
    def on_download_progress(self, progress: int, message: str):
        """Download-Progress Update"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        self.add_status_message(message, "working")
        
    def on_video_info_ready(self, video_info: dict):
        """Video-Metadaten empfangen"""
        title = video_info.get('title', 'Unknown')
        duration = video_info.get('duration', 0)
        uploader = video_info.get('uploader', 'Unknown')
        
        duration_str = f"{duration//60}:{duration%60:02d}" if duration else "Unknown"
        
        self.add_status_message(f"📋 Video erkannt: {title}", "info")
        self.add_status_message(f"📺 Kanal: {uploader} | ⏱️ Länge: {duration_str}", "info")
        
    def on_audio_ready(self, audio_file: Path):
        """Audio-Download abgeschlossen - Whisper-Transkription starten (GEÄNDERT: Path statt BytesIO)"""
        self.add_status_message("🎵 Audio-Download abgeschlossen", "success")
        self.add_status_message("🎤 Starte Whisper-Transkription...", "working")
        
        try:
            # Whisper Worker erstellen und starten
            self.current_whisper_worker = WhisperWorker(self)
            
            # Whisper Signals verbinden
            self.current_whisper_worker.progress_updated.connect(self.on_whisper_progress)
            self.current_whisper_worker.model_loading.connect(self.on_whisper_model_loading)
            self.current_whisper_worker.model_ready.connect(self.on_whisper_model_ready)
            self.current_whisper_worker.transcription_started.connect(self.on_whisper_transcription_started)
            self.current_whisper_worker.transcript_ready.connect(self.on_transcript_ready)
            self.current_whisper_worker.error_occurred.connect(self.handle_whisper_error)
            self.current_whisper_worker.finished.connect(self.on_whisper_finished)
            
            # Audio-Datei an Whisper übergeben und starten (GEÄNDERT)
            self.current_whisper_worker.set_audio_file(audio_file)
            self.current_whisper_worker.start()
            
        except Exception as e:
            self.handle_whisper_error(f"Fehler beim Starten der Transkription: {str(e)}")

    def on_whisper_progress(self, message: str):
        """Whisper-Progress Update"""
        self.add_status_message(message, "working")
        
    def on_whisper_model_loading(self):
        """Whisper Model wird geladen"""
        self.progress_bar.setValue(65)
        self.progress_label.setText("🎤 Lade Whisper Large-v3 Model...")
        self.add_status_message("📥 Whisper Large-v3 Model wird geladen...", "working")
        
    def on_whisper_model_ready(self, model_info: dict):
        """Whisper Model bereit"""
        device = model_info.get('device', 'unknown')
        gpu_name = model_info.get('gpu_name', 'N/A')
        device_info = f"({gpu_name})" if device == 'cuda' else f"({device.upper()})"
        
        self.add_status_message(f"🎤 Whisper Model geladen {device_info}", "success")
        
    def on_whisper_transcription_started(self):
        """Transkription gestartet"""
        self.progress_bar.setValue(70)
        self.progress_label.setText("🎤 Transkribiere Audio...")
        
    def on_transcript_ready(self, transcript: str):
        """Transkript fertig - KI-Analyse starten"""
        transcript_length = len(transcript)
        self.add_status_message(f"✅ Transkription abgeschlossen ({transcript_length} Zeichen)", "success")
        self.add_status_message("🤖 Starte KI-Analyse...", "working")
        
        self.progress_bar.setValue(80)
        self.progress_label.setText("🤖 KI-Analyse läuft...")
        
        # TODO: Hier später echte KI-Analyse (Ollama) einbauen
        # Für jetzt simulieren wir positive Analyse
        from PySide6.QtCore import QTimer
        self.analysis_timer = QTimer()
        self.analysis_timer.timeout.connect(lambda: self.on_analysis_complete(transcript, 0.85))
        self.analysis_timer.setSingleShot(True)
        self.analysis_timer.start(3000)  # 3 Sekunden Simulation
        
    def handle_whisper_error(self, error_message: str):
        """Whisper-Fehler behandeln"""
        self.add_status_message(error_message, "error")
        self.reset_ui_after_download()
        
    def on_whisper_finished(self):
        """Whisper Worker beendet"""
        self.add_status_message("🎤 Whisper-Worker beendet", "info")

    def on_analysis_complete(self, transcript: str, analysis_score: float):
        """KI-Analyse abgeschlossen"""
        threshold = 0.7  # TODO: Aus Config laden
        
        self.add_status_message(f"🤖 Analyse-Score: {analysis_score:.2f} (Threshold: {threshold:.2f})", "info")
        
        if analysis_score >= threshold:
            self.add_status_message("✅ Analyse positiv - Video wird heruntergeladen", "success")
            self.start_video_download()
        else:
            self.add_status_message("❌ Analyse negativ - Video wird nicht heruntergeladen", "warning")
            self.progress_bar.setValue(100)
            self.progress_label.setText("❌ Analyse beendet (negativ)")
            self.reset_ui_after_download()
            
    def start_video_download(self):
        """Video-Download nach positiver Analyse starten"""
        url = self.url_input.text().strip()
        
        self.add_status_message("📹 Starte Video-Download...", "working")
        self.progress_bar.setValue(85)
        self.progress_label.setText("📹 Video wird heruntergeladen...")
        
        # Neuen Worker für Video erstellen
        self.video_worker = DownloadWorker(self)
        self.video_worker.progress_updated.connect(self.on_download_progress)
        self.video_worker.video_ready.connect(self.on_video_ready)
        self.video_worker.error_occurred.connect(self.handle_download_error)
        self.video_worker.finished.connect(self.on_final_download_finished)
        
        self.video_worker.set_download_params(url, download_audio=False, download_video=True)
        self.video_worker.start()
        
    def on_video_ready(self, video_file):
        """Video-Download abgeschlossen"""
        self.add_status_message(f"📹 Video gespeichert: {video_file.name}", "success")
        self.add_status_message("☁️ Upload zu NextCloud wird gestartet...", "working")
        
        # TODO: Hier später NextCloud-Service aufrufen
        # Für jetzt simulieren
        from PySide6.QtCore import QTimer
        self.upload_timer = QTimer()
        self.upload_timer.timeout.connect(lambda: self.simulate_upload_complete(video_file))
        self.upload_timer.setSingleShot(True)
        self.upload_timer.start(2000)
        
    def simulate_upload_complete(self, video_file):
        """Temporäre Simulation des Upload-Abschlusses"""
        self.add_status_message("☁️ NextCloud-Upload abgeschlossen", "success")
        self.add_status_message("📝 Metadaten in Trilium gespeichert", "success")
        self.add_status_message("🎉 Analyse vollständig abgeschlossen!", "success")
        
        self.progress_bar.setValue(100)
        self.progress_label.setText("🎉 Erfolgreich abgeschlossen!")
        
        # UI zurücksetzen
        self.reset_ui_after_download()
        
    def on_download_finished(self):
        """Audio-Download-Worker beendet"""
        self.add_status_message("🎵 Audio-Phase abgeschlossen", "info")
        
    def on_final_download_finished(self):
        """Video-Download-Worker beendet"""
        self.add_status_message("📹 Video-Phase abgeschlossen", "info")
        
    def handle_download_error(self, error_message: str):
        """Download-Fehler behandeln"""
        self.add_status_message(error_message, "error")
        self.reset_ui_after_download()
        
    def reset_ui_after_download(self):
        """UI nach Download zurücksetzen"""
        self.start_button.setEnabled(True)
        self.start_button.setText("🚀 Analyse starten")
        self.progress_label.setText("🌊 Bereit für nächste Analyse")
        
        # Worker cleanup
        if hasattr(self, 'current_worker') and self.current_worker:
            self.current_worker.stop_download()
            self.current_worker = None
            
        if hasattr(self, 'video_worker') and self.video_worker:
            self.video_worker.stop_download()
            self.video_worker = None
            
        # Whisper Worker cleanup
        if hasattr(self, 'current_whisper_worker') and self.current_whisper_worker:
            self.current_whisper_worker.stop_transcription()
            self.current_whisper_worker = None
            
    def on_audio_analysis_ready(self, audio_file: Path):
        """Audio für Analyse bereit (Download Manager Signal) - GEÄNDERT: Path statt BytesIO"""
        self.add_status_message("🤖 Audio bereit für KI-Analyse", "working")
        
    def on_video_storage_ready(self, video_file, metadata):
        """Video für Storage bereit (Download Manager Signal)"""
        self.add_status_message("☁️ Video bereit für NextCloud-Upload", "working")
        
    def on_download_completed(self, complete_data):
        """Download komplett abgeschlossen (Download Manager Signal)"""
        self.add_status_message("🎉 Download-Prozess vollständig!", "success")
        
    def is_valid_youtube_url(self, url):
        """Einfache YouTube URL Validierung"""
        return ("youtube.com/watch" in url or "youtu.be/" in url)
        
    def add_status_message(self, message, status_type="info"):
        """Status-Nachricht mit Farbcodierung hinzufügen"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Farbcode basierend auf Status-Typ
        color = self.status_colors.get(status_type, self.status_colors['info'])
        
        # HTML-formatierte Nachricht
        formatted_message = f'<span style="color: {color};">[{timestamp}] {message}</span>'
        
        self.status_display.append(formatted_message)
        # Auto-scroll zum Ende
        cursor = self.status_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.status_display.setTextCursor(cursor)
        
    def show_config_dialog(self):
        """Config Dialog anzeigen"""
        try:
            from ui.config_dialog import ConfigDialog
            dialog = ConfigDialog(self)
            dialog.exec()
            self.add_status_message("🔧 Konfiguration angezeigt", "info")
        except Exception as e:
            self.add_status_message(f"❌ Fehler beim Öffnen der Konfiguration: {str(e)}", "error")
            
    def reload_config(self):
        """Konfiguration neu laden"""
        try:
            from config import reload_config
            reload_config()
            self.add_status_message("🔄 Konfiguration neu geladen", "success")
        except Exception as e:
            self.add_status_message(f"❌ Fehler beim Neu-Laden: {str(e)}", "error")
            
    def check_secret_status(self):
        """Secret Status prüfen und anzeigen"""
        try:
            from config import get_secrets_manager
            manager = get_secrets_manager()
            status = manager.check_secrets_availability()
            
            self.add_status_message("🔐 Secret Status:", "info")
            for secret, available in status.items():
                status_icon = "✅" if available else "❌"
                status_text = "Verfügbar" if available else "Nicht verfügbar"
                self.add_status_message(f"  {status_icon} {secret}: {status_text}", "info")
                
        except Exception as e:
            self.add_status_message(f"❌ Fehler bei Secret-Prüfung: {str(e)}", "error")
            
    def show_about(self):
        """About Dialog anzeigen"""
        from PySide6.QtWidgets import QMessageBox
        
        about_text = """
        <h3>🌊 YouTube Info Analyzer</h3>
        <p><b>Version:</b> 0.1.0</p>
        <p><b>Theme:</b> Ocean Theme</p>
        <br>
        <p>Intelligente YouTube-Video-Analyse für persönliches Wissensmanagement</p>
        <br>
        <p><b>Features:</b></p>
        <ul>
        <li>🎤 Faster Whisper Transkription</li>
        <li>🤖 Ollama/Gemma KI-Analyse</li>
        <li>☁️ NextCloud Integration</li>
        <li>📝 Trilium Notes Integration</li>
        <li>🔐 KeePassXC Secret Management</li>
        </ul>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Über YouTube Info Analyzer")
        msg_box.setText(about_text)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.exec()
