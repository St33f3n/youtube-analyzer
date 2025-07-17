"""
Main Window - PySide6 UI mit vollständiger Type-Safety und Result-Types
Vollständig überarbeitet nach Quality-Gate-Standards
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QTimer
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtGui import QFont
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QFrame
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QMenu
from PySide6.QtWidgets import QMenuBar
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QProgressBar
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QTextEdit
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from config.settings import get_config
from config.settings import reload_config
from config.secrets import get_secrets_manager
from services.analysis import get_analysis_service
from services.download import get_download_service
from services.ollama import get_ollama_service
from services.whisper import get_whisper_service
from yt_types import AnalysisDecision
from yt_types import AnalysisResult
from yt_types import AudioMetadata
from yt_types import ConfigurationError
from yt_types import Err
from yt_types import Ok
from yt_types import ProcessingStage
from yt_types import ProcessingStatus
from yt_types import Result
from yt_types import TranscriptionResult
from yt_types import VideoMetadata
from yt_types import validate_youtube_url
from ui.colors import get_main_stylesheet
from ui.colors import get_status_colors
from ui.config_dialog import ConfigDialog
from utils.logging import ComponentLogger
from utils.logging import ProcessingLogger
from utils.logging import log_feature_execution
from utils.logging import log_function_calls
from workers.download_worker import DownloadWorker
from workers.whisper_worker import WhisperWorker


class MainWindow(QMainWindow):
    """Hauptfenster der YouTube Analyzer Anwendung"""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Logging Setup
        self.logger = ComponentLogger("MainWindow")
        self.processing_logger: Optional[ProcessingLogger] = None
        
        # Services
        self.download_service = get_download_service()
        self.whisper_service = get_whisper_service()
        self.ollama_service = get_ollama_service()
        self.analysis_service = get_analysis_service()
        self.secrets_manager = get_secrets_manager()
        
        # Workers
        self.current_download_worker: Optional[DownloadWorker] = None
        self.current_whisper_worker: Optional[WhisperWorker] = None
        
        # State
        self.current_video_metadata: Optional[VideoMetadata] = None
        self.current_transcription: Optional[TranscriptionResult] = None
        self.current_analysis: Optional[AnalysisResult] = None
        self.processing_start_time: Optional[float] = None
        
        # UI Setup
        self.init_ui()
        self.connect_signals()
        
        # Initial status
        self.add_status_message("🌊 YouTube Analyzer bereit", "info")
        
        self.logger.info(
            "Main window initialized",
            window_title=self.windowTitle(),
            services_ready=self.check_services_ready(),
        )
    
    def init_ui(self) -> None:
        """UI-Komponenten initialisieren"""
        self.setWindowTitle("YouTube Info Analyzer")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        # Ocean Theme anwenden
        self.setStyleSheet(get_main_stylesheet())
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
        
        # UI-Sektionen erstellen
        self.create_url_section(main_layout)
        self.create_progress_section(main_layout)
        self.create_status_section(main_layout)
        
        # Spacer am Ende
        main_layout.addStretch()
    
    def create_menu_bar(self) -> None:
        """Menü-Bar erstellen"""
        menu_bar = self.menuBar()
        
        # Datei Menü
        file_menu = menu_bar.addMenu("📁 Datei")
        
        # Neu Action
        new_action = QAction("🆕 Neue Analyse", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.reset_analysis)
        file_menu.addAction(new_action)
        
        file_menu.addSeparator()
        
        # Beenden Action
        quit_action = QAction("❌ Beenden", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Konfiguration Menü
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
        
        # Services Menü
        services_menu = menu_bar.addMenu("🔧 Services")
        
        # Service Status Action
        service_status_action = QAction("📊 Service Status prüfen", self)
        service_status_action.triggered.connect(self.check_service_status)
        services_menu.addAction(service_status_action)
        
        # Secret Status Action
        secret_status_action = QAction("🔐 Secret Status prüfen", self)
        secret_status_action.triggered.connect(self.check_secret_status)
        services_menu.addAction(secret_status_action)
        
        # Hilfe Menü
        help_menu = menu_bar.addMenu("❓ Hilfe")
        
        # About Action
        about_action = QAction("ℹ️ Über YouTube Analyzer", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_url_section(self, parent_layout: QVBoxLayout) -> None:
        """URL-Eingabe Sektion erstellen"""
        # Label
        url_label = QLabel("YouTube URL:")
        url_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        parent_layout.addWidget(url_label)
        
        # URL Input + Button Layout
        url_layout = QHBoxLayout()
        url_layout.setSpacing(15)
        
        # URL Input Field
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self.url_input.setMinimumHeight(45)
        self.url_input.setFont(QFont("Segoe UI", 11))
        url_layout.addWidget(self.url_input)
        
        # Start Button
        self.start_button = QPushButton("🚀 Analyse starten")
        self.start_button.setObjectName("startButton")
        self.start_button.setMinimumHeight(45)
        self.start_button.setMinimumWidth(160)
        self.start_button.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        url_layout.addWidget(self.start_button)
        
        parent_layout.addLayout(url_layout)
    
    def create_progress_section(self, parent_layout: QVBoxLayout) -> None:
        """Progress-Sektion erstellen"""
        # Separator
        separator = QFrame()
        separator.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        separator.setFixedHeight(2)
        parent_layout.addWidget(separator)
        
        # Progress Label
        self.progress_label = QLabel("🌊 Bereit für YouTube-Analyse")
        self.progress_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        parent_layout.addWidget(self.progress_label)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(35)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        parent_layout.addWidget(self.progress_bar)
    
    def create_status_section(self, parent_layout: QVBoxLayout) -> None:
        """Status-Anzeige Sektion erstellen"""
        # Status Label
        status_label = QLabel("📊 Verarbeitungsstatus:")
        status_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        parent_layout.addWidget(status_label)
        
        # Status Display
        self.status_display = QTextEdit()
        self.status_display.setMaximumHeight(200)
        self.status_display.setReadOnly(True)
        self.status_display.setFont(QFont("Consolas", 9))
        parent_layout.addWidget(self.status_display)
    
    def connect_signals(self) -> None:
        """Signal-Slot Verbindungen erstellen"""
        self.start_button.clicked.connect(self.on_start_analysis)
        self.url_input.returnPressed.connect(self.on_start_analysis)
    
    @log_function_calls
    def on_start_analysis(self) -> None:
        """Analyse-Prozess starten"""
        url = self.url_input.text().strip()
        
        if not url:
            self.add_status_message("❌ Bitte YouTube-URL eingeben", "error")
            return
        
        # URL validieren
        url_validation = validate_youtube_url(url)
        if isinstance(url_validation, Err):
            self.add_status_message(f"❌ Ungültige URL: {url_validation.error.message}", "error")
            return
        
        # Services-Bereitschaft prüfen
        if not self.check_services_ready():
            self.add_status_message("❌ Services nicht bereit - prüfe Konfiguration", "error")
            return
        
        # UI für Verarbeitung vorbereiten
        self.prepare_ui_for_processing()
        
        # Processing Logger initialisieren
        self.processing_logger = ProcessingLogger(self.logger, f"video_{hash(url) % 10000}")
        self.processing_start_time = time.time()
        
        # Feature-Level Logging
        with log_feature_execution(
            self.logger,
            "complete_video_analysis",
            url=url,
            start_time=self.processing_start_time,
        ) as feature_logger:
            
            self.add_status_message(f"🚀 Starte Analyse: {url}", "working")
            
            # Download-Worker starten
            self.start_download_worker(url)
    
    @log_function_calls
    def start_download_worker(self, url: str) -> None:
        """Download-Worker starten"""
        try:
            # Vorherigen Worker stoppen
            if self.current_download_worker:
                self.current_download_worker.stop_download()
                self.current_download_worker.wait(3000)
            
            # Neuen Worker erstellen
            self.current_download_worker = DownloadWorker()
            
            # Signals verbinden
            self.current_download_worker.progress_updated.connect(self.on_download_progress)
            self.current_download_worker.video_info_ready.connect(self.on_video_info_ready)
            self.current_download_worker.audio_ready.connect(self.on_audio_ready)
            self.current_download_worker.video_ready.connect(self.on_video_ready)
            self.current_download_worker.error_occurred.connect(self.on_download_error)
            self.current_download_worker.finished.connect(self.on_download_finished)
            
            # Worker starten (nur Audio zuerst)
            self.current_download_worker.set_download_params(
                url=url,
                download_audio=True,
                download_video=False,
            )
            
            self.current_download_worker.start()
            
            # Processing Stage
            if self.processing_logger:
                self.processing_logger.stage_started(ProcessingStage.AUDIO, url=url)
            
        except Exception as e:
            self.logger.error(
                "Download worker startup failed",
                error=e,
                url=url,
            )
            self.on_download_error(f"Download-Worker Fehler: {str(e)}")
    
    def on_download_progress(self, progress: int, message: str) -> None:
        """Download-Progress Update"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        
        if self.processing_logger:
            self.processing_logger.stage_progress(
                message,
                progress,
                raw_progress=progress,
            )
    
    def on_video_info_ready(self, video_info: VideoMetadata) -> None:
        """Video-Metadaten empfangen"""
        self.current_video_metadata = video_info
        
        duration_str = f"{video_info.duration//60}:{video_info.duration%60:02d}" if video_info.duration else "Unknown"
        
        self.add_status_message(f"📋 Video: {video_info.title}", "info")
        self.add_status_message(f"📺 Kanal: {video_info.uploader} | ⏱️ Dauer: {duration_str}", "info")
        
        self.logger.info(
            "Video metadata received",
            video_id=video_info.id,
            title=video_info.title,
            duration=video_info.duration,
            uploader=video_info.uploader,
        )
    
    def on_audio_ready(self, audio_metadata: AudioMetadata) -> None:
        """Audio-Download abgeschlossen - Whisper starten"""
        self.add_status_message("🎵 Audio-Download abgeschlossen", "success")
        
        # Processing Stage abschließen
        if self.processing_logger:
            self.processing_logger.stage_completed(
                audio_file_size=audio_metadata.file_size,
                audio_format=audio_metadata.format,
            )
        
        # Whisper-Worker starten
        self.start_whisper_worker(audio_metadata)
    
    @log_function_calls
    def start_whisper_worker(self, audio_metadata: AudioMetadata) -> None:
        """Whisper-Worker starten"""
        try:
            # Vorherigen Worker stoppen
            if self.current_whisper_worker:
                self.current_whisper_worker.stop_transcription()
                self.current_whisper_worker.wait(5000)
            
            # Neuen Worker erstellen
            self.current_whisper_worker = WhisperWorker()
            
            # Signals verbinden
            self.current_whisper_worker.progress_updated.connect(self.on_whisper_progress)
            self.current_whisper_worker.model_loading.connect(self.on_whisper_model_loading)
            self.current_whisper_worker.model_ready.connect(self.on_whisper_model_ready)
            self.current_whisper_worker.transcription_started.connect(self.on_whisper_transcription_started)
            self.current_whisper_worker.transcript_ready.connect(self.on_transcript_ready)
            self.current_whisper_worker.error_occurred.connect(self.on_whisper_error)
            self.current_whisper_worker.finished.connect(self.on_whisper_finished)
            
            # Worker starten
            self.current_whisper_worker.set_audio_metadata(audio_metadata)
            self.current_whisper_worker.start()
            
            # Processing Stage
            if self.processing_logger:
                self.processing_logger.stage_started(
                    ProcessingStage.TRANSCRIPTION,
                    audio_file=str(audio_metadata.file_path),
                    audio_size=audio_metadata.file_size,
                )
            
        except Exception as e:
            self.logger.error(
                "Whisper worker startup failed",
                error=e,
                audio_file=str(audio_metadata.file_path),
            )
            self.on_whisper_error(f"Whisper-Worker Fehler: {str(e)}")
    
    def on_whisper_progress(self, message: str) -> None:
        """Whisper-Progress Update"""
        self.add_status_message(message, "working")
        
        if self.processing_logger:
            self.processing_logger.stage_progress(message, 70)
    
    def on_whisper_model_loading(self) -> None:
        """Whisper Model wird geladen"""
        self.progress_bar.setValue(65)
        self.progress_label.setText("🎤 Lade Whisper Large-v3 Model...")
        self.add_status_message("📥 Whisper Model wird geladen...", "working")
    
    def on_whisper_model_ready(self, model_info: dict) -> None:
        """Whisper Model bereit"""
        device = model_info.get('device', 'unknown')
        gpu_name = model_info.get('gpu_name', 'N/A')
        device_info = f"({gpu_name})" if device == 'cuda' else f"({device.upper()})"
        
        self.add_status_message(f"🎤 Whisper Model bereit {device_info}", "success")
    
    def on_whisper_transcription_started(self) -> None:
        """Transkription gestartet"""
        self.progress_bar.setValue(75)
        self.progress_label.setText("🎤 Transkribiere Audio...")
        self.add_status_message("🎤 Transkription gestartet", "working")
    
    def on_transcript_ready(self, transcription: TranscriptionResult) -> None:
        """Transkript fertig - KI-Analyse starten"""
        self.current_transcription = transcription
        
        self.add_status_message(
            f"✅ Transkription: {len(transcription.text)} Zeichen ({transcription.language})",
            "success"
        )
        
        # Processing Stage abschließen
        if self.processing_logger:
            self.processing_logger.stage_completed(
                transcript_length=len(transcription.text),
                detected_language=transcription.language,
                confidence=transcription.confidence,
                processing_time=transcription.processing_time,
            )
        
        # KI-Analyse starten
        self.start_analysis_process(transcription)
    
    @log_function_calls
    def start_analysis_process(self, transcription: TranscriptionResult) -> None:
        """KI-Analyse-Prozess starten"""
        try:
            self.add_status_message("🤖 Starte KI-Analyse mit allen Regeln...", "working")
            self.progress_bar.setValue(85)
            self.progress_label.setText("🤖 KI-Analyse läuft...")
            
            # Processing Stage
            if self.processing_logger:
                self.processing_logger.stage_started(
                    ProcessingStage.ANALYSIS,
                    transcript_length=len(transcription.text),
                    rules_count=len(self.analysis_service.rules),
                )
            
            # Analysis in QTimer (um UI responsive zu halten)
            self.analysis_timer = QTimer()
            self.analysis_timer.timeout.connect(
                lambda: self.run_analysis_async(transcription)
            )
            self.analysis_timer.setSingleShot(True)
            self.analysis_timer.start(100)  # 100ms delay
            
        except Exception as e:
            self.logger.error(
                "Analysis process startup failed",
                error=e,
                transcript_length=len(transcription.text),
            )
            self.on_analysis_error(f"Analyse-Fehler: {str(e)}")
    
    def run_analysis_async(self, transcription: TranscriptionResult) -> None:
        """Asynchrone Analyse ausführen"""
        try:
            # Analyse durchführen
            analysis_result = self.analysis_service.analyze_transcript(transcription)
            
            if isinstance(analysis_result, Ok):
                self.on_analysis_complete(analysis_result.value)
            else:
                self.on_analysis_error(f"Analyse fehlgeschlagen: {analysis_result.error.message}")
                
        except Exception as e:
            self.on_analysis_error(f"Analyse-Fehler: {str(e)}")
    
    def on_analysis_complete(self, analysis: AnalysisResult) -> None:
        """KI-Analyse abgeschlossen"""
        self.current_analysis = analysis
        
        self.add_status_message(
            f"🤖 Analyse abgeschlossen: Score {analysis.final_score:.2f}",
            "success"
        )
        
        # Processing Stage abschließen
        if self.processing_logger:
            self.processing_logger.stage_completed(
                final_score=analysis.final_score,
                decision=analysis.decision.value,
                rules_processed=len(analysis.rule_scores),
            )
        
        # Entscheidung verarbeiten
        self.process_analysis_decision(analysis)
    
    def process_analysis_decision(self, analysis: AnalysisResult) -> None:
        """Analyse-Entscheidung verarbeiten"""
        decision = analysis.decision
        score = analysis.final_score
        
        if decision == AnalysisDecision.APPROVE:
            self.add_status_message(
                f"✅ Video genehmigt (Score: {score:.2f}) - Video wird heruntergeladen",
                "success"
            )
            self.start_video_download()
            
        elif decision == AnalysisDecision.REJECT:
            self.add_status_message(
                f"❌ Video abgelehnt (Score: {score:.2f}) - Keine weitere Verarbeitung",
                "warning"
            )
            self.complete_processing(success=False)
            
        else:  # MANUAL_REVIEW
            self.add_status_message(
                f"⚠️ Manuelle Überprüfung erforderlich (Score: {score:.2f})",
                "warning"
            )
            self.complete_processing(success=False)
    
    def start_video_download(self) -> None:
        """Video-Download nach positiver Analyse starten"""
        if not self.current_video_metadata:
            self.on_download_error("Video-Metadaten nicht verfügbar")
            return
        
        url = self.current_video_metadata.webpage_url
        
        self.add_status_message("📹 Starte Video-Download...", "working")
        self.progress_bar.setValue(90)
        self.progress_label.setText("📹 Video wird heruntergeladen...")
        
        # Processing Stage
        if self.processing_logger:
            self.processing_logger.stage_started(
                ProcessingStage.STORAGE,
                video_url=url,
                approved_score=self.current_analysis.final_score if self.current_analysis else 0.0,
            )
        
        # Neuen Download-Worker für Video erstellen
        self.start_download_worker_video_only(url)
    
    def start_download_worker_video_only(self, url: str) -> None:
        """Download-Worker nur für Video starten"""
        try:
            # Vorherigen Worker stoppen
            if self.current_download_worker:
                self.current_download_worker.stop_download()
                self.current_download_worker.wait(3000)
            
            # Neuen Worker erstellen
            self.current_download_worker = DownloadWorker()
            
            # Signals verbinden
            self.current_download_worker.progress_updated.connect(self.on_download_progress)
            self.current_download_worker.video_ready.connect(self.on_video_ready)
            self.current_download_worker.error_occurred.connect(self.on_download_error)
            self.current_download_worker.finished.connect(self.on_video_download_finished)
            
            # Worker starten (nur Video)
            self.current_download_worker.set_download_params(
                url=url,
                download_audio=False,
                download_video=True,
            )
            
            self.current_download_worker.start()
            
        except Exception as e:
            self.logger.error(
                "Video download worker startup failed",
                error=e,
                url=url,
            )
            self.on_download_error(f"Video-Download Fehler: {str(e)}")
    
    def on_video_ready(self, video_path: str) -> None:
        """Video-Download abgeschlossen"""
        self.add_status_message(f"📹 Video gespeichert: {video_path}", "success")
        
        # TODO: Hier später NextCloud-Upload und Trilium-Integration
        self.simulate_storage_upload(video_path)
    
    def simulate_storage_upload(self, video_path: str) -> None:
        """Simuliere Storage-Upload (placeholder)"""
        self.add_status_message("☁️ Simuliere NextCloud-Upload...", "working")
        
        # Simulation Timer
        self.storage_timer = QTimer()
        self.storage_timer.timeout.connect(
            lambda: self.complete_storage_simulation(video_path)
        )
        self.storage_timer.setSingleShot(True)
        self.storage_timer.start(2000)  # 2 Sekunden
    
    def complete_storage_simulation(self, video_path: str) -> None:
        """Storage-Simulation abschließen"""
        self.add_status_message("☁️ NextCloud-Upload abgeschlossen", "success")
        self.add_status_message("📝 Metadaten in Trilium gespeichert", "success")
        
        # Processing Stage abschließen
        if self.processing_logger:
            self.processing_logger.stage_completed(
                video_path=video_path,
                storage_simulation=True,
            )
        
        # Verarbeitung abschließen
        self.complete_processing(success=True)
    
    def complete_processing(self, success: bool) -> None:
        """Verarbeitung abschließen"""
        if success:
            self.add_status_message("🎉 Analyse vollständig abgeschlossen!", "success")
            self.progress_bar.setValue(100)
            self.progress_label.setText("🎉 Erfolgreich abgeschlossen!")
        else:
            self.add_status_message("⚠️ Verarbeitung beendet (nicht genehmigt)", "warning")
            self.progress_bar.setValue(100)
            self.progress_label.setText("⚠️ Verarbeitung beendet")
        
        # Processing Logger abschließen
        if self.processing_logger:
            self.processing_logger.process_completed(
                success=success,
                total_time=time.time() - self.processing_start_time if self.processing_start_time else 0,
                final_decision=self.current_analysis.decision.value if self.current_analysis else "unknown",
            )
        
        # UI zurücksetzen
        self.reset_ui_after_processing()
    
    def on_download_error(self, error_message: str) -> None:
        """Download-Fehler behandeln"""
        self.add_status_message(error_message, "error")
        self.reset_ui_after_processing()
    
    def on_whisper_error(self, error_message: str) -> None:
        """Whisper-Fehler behandeln"""
        self.add_status_message(error_message, "error")
        self.reset_ui_after_processing()
    
    def on_analysis_error(self, error_message: str) -> None:
        """Analyse-Fehler behandeln"""
        self.add_status_message(error_message, "error")
        self.reset_ui_after_processing()
    
    def on_download_finished(self) -> None:
        """Download-Worker beendet"""
        self.logger.debug("Download worker finished")
    
    def on_video_download_finished(self) -> None:
        """Video-Download-Worker beendet"""
        self.logger.debug("Video download worker finished")
    
    def on_whisper_finished(self) -> None:
        """Whisper-Worker beendet"""
        self.logger.debug("Whisper worker finished")
    
    def prepare_ui_for_processing(self) -> None:
        """UI für Verarbeitung vorbereiten"""
        self.start_button.setEnabled(False)
        self.start_button.setText("⏳ Verarbeitung läuft...")
        self.progress_bar.setValue(0)
        self.progress_label.setText("🚀 Initialisiere...")
        
        # Status-Display leeren
        self.status_display.clear()
    
    def reset_ui_after_processing(self) -> None:
        """UI nach Verarbeitung zurücksetzen"""
        self.start_button.setEnabled(True)
        self.start_button.setText("🚀 Analyse starten")
        self.progress_label.setText("🌊 Bereit für nächste Analyse")
        
        # Worker cleanup
        self.cleanup_workers()
    
    def reset_analysis(self) -> None:
        """Analyse zurücksetzen"""
        self.cleanup_workers()
        self.reset_ui_after_processing()
        
        # State zurücksetzen
        self.current_video_metadata = None
        self.current_transcription = None
        self.current_analysis = None
        self.processing_start_time = None
        self.processing_logger = None
        
        # URL Input leeren
        self.url_input.clear()
        self.status_display.clear()
        
        self.add_status_message("🔄 Analyse zurückgesetzt", "info")
    
    def cleanup_workers(self) -> None:
        """Worker aufräumen"""
        if self.current_download_worker:
            self.current_download_worker.stop_download()
            self.current_download_worker.wait(3000)
            self.current_download_worker = None
        
        if self.current_whisper_worker:
            self.current_whisper_worker.stop_transcription()
            self.current_whisper_worker.wait(3000)
            self.current_whisper_worker = None
    
    def cleanup(self) -> None:
        """Cleanup beim Beenden"""
        self.logger.info("Main window cleanup started")
        
        # Worker cleanup
        self.cleanup_workers()
        
        # Services cleanup
        self.download_service.cleanup()
        self.whisper_service.cleanup()
        
        self.logger.info("Main window cleanup completed")
    
    def check_services_ready(self) -> bool:
        """Prüfe ob alle Services bereit sind"""
        return (
            self.analysis_service.is_ready() and
            self.ollama_service.is_ready()
        )
    
    def add_status_message(self, message: str, status_type: str = "info") -> None:
        """Status-Nachricht mit Farbcodierung hinzufügen"""
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
        
        # Auch ins Log schreiben
        self.logger.info(f"UI Status: {message}", status_type=status_type)
    
    # =============================================================================
    # MENU ACTIONS
    # =============================================================================
    
    def show_config_dialog(self) -> None:
        """Config Dialog anzeigen"""
        try:
            dialog = ConfigDialog(self)
            dialog.exec()
            self.add_status_message("🔧 Konfiguration angezeigt", "info")
        except Exception as e:
            self.add_status_message(f"❌ Config-Dialog Fehler: {str(e)}", "error")
    
    def reload_config(self) -> None:
        """Konfiguration neu laden"""
        try:
            reload_config()
            self.add_status_message("🔄 Konfiguration neu geladen", "success")
        except Exception as e:
            self.add_status_message(f"❌ Reload-Fehler: {str(e)}", "error")
    
    def check_service_status(self) -> None:
        """Service-Status prüfen und anzeigen"""
        try:
            self.add_status_message("🔧 Prüfe Service-Status...", "info")
            
            # Download Service
            download_info = self.download_service.get_service_info()
            self.add_status_message(
                f"📥 Download: {download_info.get('status', 'unknown')}",
                "info"
            )
            
            # Whisper Service
            whisper_status = self.whisper_service.get_service_status()
            self.add_status_message(
                f"🎤 Whisper: {whisper_status.status} - {whisper_status.message}",
                "info"
            )
            
            # Ollama Service
            ollama_status = self.ollama_service.get_service_status()
            self.add_status_message(
                f"🤖 Ollama: {ollama_status.status} - {ollama_status.message}",
                "info"
            )
            
            # Analysis Service
            analysis_status = self.analysis_service.get_service_status()
            self.add_status_message(
                f"📊 Analysis: {analysis_status.status} - {analysis_status.message}",
                "info"
            )
            
        except Exception as e:
            self.add_status_message(f"❌ Service-Status Fehler: {str(e)}", "error")
    
    def check_secret_status(self) -> None:
        """Secret-Status prüfen und anzeigen"""
        try:
            status = self.secrets_manager.check_secrets_availability()
            
            self.add_status_message("🔐 Secret-Status:", "info")
            for secret, available in status.items():
                status_icon = "✅" if available else "❌"
                self.add_status_message(f"  {status_icon} {secret}", "info")
                
        except Exception as e:
            self.add_status_message(f"❌ Secret-Status Fehler: {str(e)}", "error")
    
    def show_about(self) -> None:
        """About Dialog anzeigen"""
        about_text = """
        <h3>🌊 YouTube Info Analyzer</h3>
        <p><b>Version:</b> 0.1.0</p>
        <p><b>Theme:</b> Ocean Theme</p>
        <br>
        <p>Intelligente YouTube-Video-Analyse für persönliches Wissensmanagement</p>
        <br>
        <p><b>Features:</b></p>
        <ul>
        <li>🎤 Faster Whisper Transkription (Large-v3)</li>
        <li>🤖 Ollama/Gemma2 KI-Analyse</li>
        <li>📊 Regel-basierte Bewertung</li>
        <li>☁️ NextCloud Integration (geplant)</li>
        <li>📝 Trilium Notes Integration (geplant)</li>
        <li>🔐 KeePassXC Secret Management</li>
        </ul>
        <br>
        <p><b>Entwickelt mit:</b> Python 3.11+, PySide6, Result-Types</p>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Über YouTube Info Analyzer")
        msg_box.setText(about_text)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setMinimumSize(400, 300)
        msg_box.exec()
    
    def closeEvent(self, event) -> None:
        """Beim Schließen des Fensters"""
        self.cleanup()
        event.accept()
