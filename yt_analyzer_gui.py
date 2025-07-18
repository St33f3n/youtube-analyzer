"""
YouTube Analyzer - PySide6 GUI with Ocean Theme
Simple Interface mit Pipeline-Status-Anzeige und Config-Validation
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QFrame, QStatusBar, QDialog,
    QGridLayout, QScrollArea, QSizePolicy
)
from PySide6.QtCore import QThread, Signal, QTimer, Qt, QSize
from PySide6.QtGui import QFont, QPalette, QColor, QIcon
from PySide6.QtCore import QObject

# Import our core libraries
from core_types import Result, Ok, Err, is_ok, unwrap_ok, unwrap_err
from yt_analyzer_core import ProcessObject  # For type hints
from logging_plus import get_logger, log_feature
from yt_analyzer_config import SecureConfigManager, AppConfig

# =============================================================================
# OCEAN THEME COLORS
# =============================================================================

class OceanTheme:
    """Ocean-themed color palette for dark UI"""
    
    # Primary Colors
    DEEP_OCEAN = "#edf2f7"
    OCEAN_CURRENT = "#e6e6e6"
    SEA_FOAM = "#6b8e6b"
    DEEP_TEAL = "#4a6b4a"
    KELP_GREEN = "#3d5a3d"
    
    # Coral Accent Colors
    CORAL_PINK = "#c4766a"
    CORAL_ORANGE = "#d4634a"
    CORAL_RED = "#c84a2c"
    CORAL_LIGHT = "#d49284"
    CORAL_DEEP = "#b85347"
    
    # Ocean Accent Colors
    BIOLUMINESCENT = "#5a9a9a"
    CORAL_BLUE = "#6b8db3"
    ARCTIC_BLUE = "#8cb8e8"
    DEEP_SAPPHIRE = "#4a5f7a"
    MARINE_BLUE = "#5c7ba3"
    
    # Anthracite Neutral Colors
    ABYSS = "#1e1e1e"
    DEEP_WATER = "#262626"
    MIDNIGHT = "#2e2e2e"
    SURFACE = "#363636"
    SEAFOAM_GRAY = "#8a9a8a"
    
    # Text Colors
    TEXT_PRIMARY = "#f7f2e3"
    TEXT_SECONDARY = "#e5dcc8"
    TEXT_MUTED = "#c4b89f"
    
    # Special Colors
    YELLOW = "#f0c674"
    GOLDEN_BEIGE = "#d4c5a9"

# FIXED: Pipeline Status synchronized with Pipeline Manager
@dataclass
class PipelineStatus:
    """Status der verschiedenen Pipeline-Stufen - SYNC mit Pipeline Manager"""
    audio_download_queue: int = 0  # FIXED: Renamed from download_queue
    transcription_queue: int = 0
    analysis_queue: int = 0
    video_download_queue: int = 0
    upload_queue: int = 0
    processing_queue: int = 0
    
    total_queued: int = 0
    total_completed: int = 0
    total_failed: int = 0
    
    current_stage: str = "Idle"
    current_video: Optional[str] = None
    
    def is_active(self) -> bool:
        """Prüft ob Pipeline aktiv ist"""
        return (self.audio_download_queue + self.transcription_queue + 
                self.analysis_queue + self.video_download_queue + 
                self.upload_queue + self.processing_queue) > 0

# =============================================================================
# PIPELINE STATUS WIDGET
# =============================================================================

class PipelineStatusWidget(QFrame):
    """Widget für Pipeline-Status-Anzeige mit Queue-Counters"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("PipelineStatusWidget")
        self.setup_ui()
        self.apply_ocean_theme()
    
    def setup_ui(self):
        """Setup der UI-Komponenten"""
        self.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Pipeline Status")
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Current Status
        self.current_label = QLabel("Status: Idle")
        self.current_label.setAlignment(Qt.AlignCenter)
        self.current_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.current_label)
        
        # Queue Grid
        grid_layout = QGridLayout()
        
        # Queue Labels und Values - FIXED: Synchronized with Pipeline Manager
        self.queue_labels = {}
        self.queue_values = {}
        
        queue_names = [
            ("Audio Download", "audio_download_queue"),  # FIXED: Corrected name
            ("Transcription", "transcription_queue"),
            ("Analysis", "analysis_queue"),
            ("Video Download", "video_download_queue"),
            ("Upload", "upload_queue"),
            ("Processing", "processing_queue")
        ]
        
        for i, (display_name, queue_attr) in enumerate(queue_names):
            row = i // 2
            col = (i % 2) * 2
            
            label = QLabel(f"{display_name}:")
            value = QLabel("0")
            value.setAlignment(Qt.AlignCenter)
            value.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            
            grid_layout.addWidget(label, row, col)
            grid_layout.addWidget(value, row, col + 1)
            
            self.queue_labels[queue_attr] = label
            self.queue_values[queue_attr] = value
        
        layout.addLayout(grid_layout)
        
        # Summary Stats
        summary_layout = QHBoxLayout()
        
        self.total_label = QLabel("Total: 0")
        self.completed_label = QLabel("Completed: 0")
        self.failed_label = QLabel("Failed: 0")
        
        for label in [self.total_label, self.completed_label, self.failed_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setFont(QFont("Arial", 9))
            summary_layout.addWidget(label)
        
        layout.addLayout(summary_layout)
        
        # Current Video (if any)
        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFont(QFont("Arial", 8))
        self.video_label.setWordWrap(True)
        layout.addWidget(self.video_label)
    
    def update_status(self, status: PipelineStatus):
        """Aktualisiert Status-Anzeige - ENHANCED LOGGING"""
        # Debug logging für Status-Updates
        self.logger.debug(
            f"GUI Status Update: Total={status.total_queued}, Completed={status.total_completed}, Failed={status.total_failed}",
            extra={
                'total_queued': status.total_queued,
                'total_completed': status.total_completed,
                'total_failed': status.total_failed,
                'current_stage': status.current_stage,
                'audio_download_queue': status.audio_download_queue,
                'transcription_queue': status.transcription_queue,
                'analysis_queue': status.analysis_queue,
                'is_active': status.is_active()
            }
        )
        
        # Current Stage
        self.current_label.setText(f"Status: {status.current_stage}")
        
        # Queue Values - FIXED: Correct attribute mapping
        for queue_attr, value_label in self.queue_values.items():
            count = getattr(status, queue_attr, 0)
            value_label.setText(str(count))
            
            # Color coding für aktive Queues
            if count > 0:
                value_label.setStyleSheet(f"color: {OceanTheme.BIOLUMINESCENT}; font-weight: bold;")
            else:
                value_label.setStyleSheet(f"color: {OceanTheme.TEXT_MUTED};")
        
        # Summary Stats
        self.total_label.setText(f"Total: {status.total_queued}")
        self.completed_label.setText(f"Completed: {status.total_completed}")
        self.failed_label.setText(f"Failed: {status.total_failed}")
        
        # Enhanced progress calculation
        progress_text = f"Progress: {status.total_completed + status.total_failed}/{status.total_queued}"
        if status.total_queued > 0:
            progress_percent = (status.total_completed + status.total_failed) / status.total_queued * 100
            progress_text += f" ({progress_percent:.1f}%)"
        
        # Update total label with progress info
        self.total_label.setText(progress_text)
        
        # Current Video
        if status.current_video:
            video_text = f"Current: {status.current_video[:50]}..." if len(status.current_video) > 50 else f"Current: {status.current_video}"
            self.video_label.setText(video_text)
            self.video_label.setStyleSheet(f"color: {OceanTheme.ARCTIC_BLUE};")
        else:
            self.video_label.setText("")
    
    def apply_ocean_theme(self):
        """Wendet Ocean-Theme auf Widget an"""
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {OceanTheme.DEEP_WATER};
                border: 2px solid {OceanTheme.SURFACE};
                border-radius: 8px;
                padding: 10px;
            }}
            QLabel {{
                color: {OceanTheme.TEXT_PRIMARY};
                background: transparent;
            }}
        """)

# =============================================================================
# CONFIG VALIDATION WINDOW
# =============================================================================

class ConfigValidationWindow(QDialog):
    """Separates Fenster für Config-Validation-Anzeige"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger("ConfigValidationWindow")
        self.config_manager = SecureConfigManager()
        self.setup_ui()
        self.apply_ocean_theme()
        self.load_and_validate_config()
    
    def setup_ui(self):
        """Setup der UI-Komponenten"""
        self.setWindowTitle("Configuration Validation")
        self.setMinimumSize(600, 400)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Configuration Status")
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Scroll Area für Config-Items
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.config_widget = QWidget()
        self.config_layout = QVBoxLayout(self.config_widget)
        
        scroll.setWidget(self.config_widget)
        layout.addWidget(scroll)
        
        # Close Button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setFixedHeight(40)
        layout.addWidget(close_btn)
    
    def add_validation_item(self, title: str, status: str, details: str = "", is_error: bool = False):
        """Fügt Validation-Item zur Anzeige hinzu"""
        item_frame = QFrame()
        item_frame.setFrameStyle(QFrame.StyledPanel)
        
        item_layout = QVBoxLayout(item_frame)
        
        # Title + Status
        header_layout = QHBoxLayout()
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        
        status_label = QLabel(status)
        status_label.setFont(QFont("Arial", 10))
        if is_error:
            status_label.setStyleSheet(f"color: {OceanTheme.CORAL_RED}; font-weight: bold;")
        else:
            status_label.setStyleSheet(f"color: {OceanTheme.SEA_FOAM}; font-weight: bold;")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(status_label)
        
        item_layout.addLayout(header_layout)
        
        # Details (if any)
        if details:
            details_label = QLabel(details)
            details_label.setFont(QFont("Arial", 8))
            details_label.setStyleSheet(f"color: {OceanTheme.TEXT_MUTED};")
            details_label.setWordWrap(True)
            item_layout.addWidget(details_label)
        
        # Style
        if is_error:
            item_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {OceanTheme.MIDNIGHT};
                    border: 1px solid {OceanTheme.CORAL_DEEP};
                    border-radius: 5px;
                    padding: 8px;
                    margin: 2px;
                }}
            """)
        else:
            item_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {OceanTheme.MIDNIGHT};
                    border: 1px solid {OceanTheme.DEEP_TEAL};
                    border-radius: 5px;
                    padding: 8px;
                    margin: 2px;
                }}
            """)
        
        self.config_layout.addWidget(item_frame)
    
    def load_and_validate_config(self):
        """Lädt und validiert Konfiguration"""
        # Clear existing items
        for i in reversed(range(self.config_layout.count())):
            self.config_layout.itemAt(i).widget().setParent(None)
        
        # Load Config
        config_result = self.config_manager.load_config()
        
        if isinstance(config_result, Err):
            self.add_validation_item(
                "Configuration Loading",
                "❌ FAILED",
                str(config_result.error.message),
                is_error=True
            )
            return
        
        config = unwrap_ok(config_result)
        self.add_validation_item("Configuration Loading", "✅ SUCCESS")
        
        # Validate Secrets
        trilium_result = self.config_manager.get_trilium_token()
        if isinstance(trilium_result, Err):
            self.add_validation_item(
                "Trilium Secret Access",
                "❌ FAILED",
                f"Service: {config.secrets.trilium_service}, User: {config.secrets.trilium_username}",
                is_error=True
            )
        else:
            token = unwrap_ok(trilium_result)
            self.add_validation_item(
                "Trilium Secret Access",
                "✅ SUCCESS",
                f"Token length: {len(token)} chars"
            )
        
        nextcloud_result = self.config_manager.get_nextcloud_password()
        if isinstance(nextcloud_result, Err):
            self.add_validation_item(
                "Nextcloud Secret Access",
                "❌ FAILED",
                f"Service: {config.secrets.nextcloud_service}, User: {config.secrets.nextcloud_username}",
                is_error=True
            )
        else:
            password = unwrap_ok(nextcloud_result)
            self.add_validation_item(
                "Nextcloud Secret Access",
                "✅ SUCCESS",
                f"Password length: {len(password)} chars"
            )
        
        # Validate Rule Files
        enabled_rules = config.rules.get_enabled_rules()
        for rule_name, rule_config in enabled_rules.items():
            rule_path = Path(rule_config.file)
            if rule_path.exists():
                self.add_validation_item(
                    f"Rule: {rule_name}",
                    "✅ FOUND",
                    f"Weight: {rule_config.weight}, Path: {rule_config.file}"
                )
            else:
                self.add_validation_item(
                    f"Rule: {rule_name}",
                    "❌ MISSING",
                    f"File not found: {rule_config.file}",
                    is_error=True
                )
        
        # Validate Directories
        temp_dir = Path(config.processing.temp_folder)
        if temp_dir.exists():
            self.add_validation_item("Temp Directory", "✅ EXISTS", str(temp_dir))
        else:
            self.add_validation_item(
                "Temp Directory",
                "⚠️ MISSING",
                f"Will be created: {temp_dir}"
            )
        
        # Config Summary
        total_weight = config.rules.get_total_weight()
        self.add_validation_item(
            "Rules Configuration",
            "✅ VALID",
            f"Enabled rules: {len(enabled_rules)}, Total weight: {total_weight:.2f}"
        )
    
    def apply_ocean_theme(self):
        """Wendet Ocean-Theme an"""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {OceanTheme.ABYSS};
                color: {OceanTheme.TEXT_PRIMARY};
            }}
            QLabel {{
                color: {OceanTheme.TEXT_PRIMARY};
            }}
            QPushButton {{
                background-color: {OceanTheme.DEEP_WATER};
                color: {OceanTheme.TEXT_PRIMARY};
                border: 2px solid {OceanTheme.SURFACE};
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {OceanTheme.SURFACE};
                border-color: {OceanTheme.SEA_FOAM};
            }}
            QScrollArea {{
                background-color: {OceanTheme.DEEP_WATER};
                border: 1px solid {OceanTheme.SURFACE};
                border-radius: 5px;
            }}
        """)

# =============================================================================
# MAIN WINDOW
# =============================================================================

class YouTubeAnalyzerMainWindow(QMainWindow):
    """Haupt-Fenster der YouTube Analyzer GUI"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("MainWindow")
        self.pipeline_status = PipelineStatus()
        self.config_window = None
        
        self.setup_ui()
        self.apply_ocean_theme()
        self.setup_status_timer()
    
    def setup_ui(self):
        """Setup der Haupt-UI"""
        self.setWindowTitle("YouTube Analyzer")
        self.setMinimumSize(800, 600)
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("YouTube Video Analyzer")
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # URL Input Section
        input_section = self.create_input_section()
        layout.addWidget(input_section)
        
        # Pipeline Status Section
        self.status_widget = PipelineStatusWidget()
        layout.addWidget(self.status_widget)
        
        # Control Buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Analysis")
        self.start_button.setFixedHeight(50)
        self.start_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.start_button.clicked.connect(self.start_analysis)
        
        self.config_button = QPushButton("Check Configuration")
        self.config_button.setFixedHeight(50)
        self.config_button.setFont(QFont("Arial", 10))
        self.config_button.clicked.connect(self.show_config_window)
        
        button_layout.addWidget(self.start_button, 3)
        button_layout.addWidget(self.config_button, 1)
        
        layout.addLayout(button_layout)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def create_input_section(self) -> QFrame:
        """Erstellt URL-Input-Sektion"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(frame)
        
        # Label
        label = QLabel("YouTube URLs (one per line):")
        label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(label)
        
        # Text Input
        self.url_input = QTextEdit()
        self.url_input.setPlaceholderText("Paste YouTube URLs here, one per line...\n\nhttps://youtube.com/watch?v=...\nhttps://youtu.be/...")
        self.url_input.setMinimumHeight(150)
        self.url_input.setFont(QFont("Consolas", 10))
        layout.addWidget(self.url_input)
        
        return frame
    
    def setup_status_timer(self):
        """Setup Timer für Status-Updates - ACCELERATED"""
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_pipeline_status)
        self.status_timer.start(100)  # IMPROVED: Update every 100ms (was 1000ms)
    
    def update_pipeline_status(self):
        """Aktualisiert Pipeline-Status-Anzeige"""
        # ENHANCED: Only run if no real pipeline is connected
        if not hasattr(self, 'pipeline_manager'):
            # Fallback Mock-Update (nur wenn keine echte Pipeline)
            self.status_widget.update_status(self.pipeline_status)
        
        # Button-State basierend auf Pipeline-Status
        if self.pipeline_status.is_active():
            self.start_button.setText("Processing...")
            self.start_button.setEnabled(False)
            self.status_bar.showMessage(f"Processing - {self.pipeline_status.current_stage}")
        else:
            self.start_button.setText("Start Analysis")
            self.start_button.setEnabled(True)
            self.status_bar.showMessage("Ready")
    
    def start_analysis(self):
        """Startet YouTube-Analyse"""
        urls_text = self.url_input.toPlainText().strip()
        
        if not urls_text:
            self.status_bar.showMessage("Please enter YouTube URLs")
            return
        
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        if not urls:
            self.status_bar.showMessage("No valid URLs found")
            return
        
        self.logger.info(f"Starting analysis for {len(urls)} URLs")
        
        # ENHANCED: Check if real pipeline is available
        if hasattr(self, 'pipeline_manager'):
            # Real pipeline will be called by integrate_pipeline_with_gui
            self.logger.info("Using real pipeline manager")
        else:
            # Fallback Mock-Demo
            self.logger.info("Using mock pipeline demo")
            self.pipeline_status.audio_download_queue = len(urls)  # FIXED: Correct attribute
            self.pipeline_status.total_queued = len(urls)
            self.pipeline_status.current_stage = "Extracting Metadata"
        
        self.status_bar.showMessage(f"Started analysis for {len(urls)} videos")
    
    def show_config_window(self):
        """Zeigt Config-Validation-Fenster"""
        if self.config_window is None:
            self.config_window = ConfigValidationWindow(self)
        
        self.config_window.load_and_validate_config()
        self.config_window.show()
    
    def apply_ocean_theme(self):
        """Wendet Ocean-Theme auf Hauptfenster an"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {OceanTheme.ABYSS};
                color: {OceanTheme.TEXT_PRIMARY};
            }}
            
            QWidget {{
                background-color: {OceanTheme.ABYSS};
                color: {OceanTheme.TEXT_PRIMARY};
            }}
            
            QLabel {{
                color: {OceanTheme.TEXT_PRIMARY};
                background: transparent;
            }}
            
            QFrame {{
                background-color: {OceanTheme.DEEP_WATER};
                border: 2px solid {OceanTheme.SURFACE};
                border-radius: 8px;
                padding: 15px;
            }}
            
            QTextEdit {{
                background-color: {OceanTheme.MIDNIGHT};
                color: {OceanTheme.TEXT_PRIMARY};
                border: 2px solid {OceanTheme.SURFACE};
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', monospace;
            }}
            
            QTextEdit:focus {{
                border-color: {OceanTheme.BIOLUMINESCENT};
            }}
            
            QPushButton {{
                background-color: {OceanTheme.SEA_FOAM};
                color: {OceanTheme.TEXT_PRIMARY};
                border: 2px solid {OceanTheme.DEEP_TEAL};
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
            }}
            
            QPushButton:hover {{
                background-color: {OceanTheme.DEEP_TEAL};
                border-color: {OceanTheme.BIOLUMINESCENT};
            }}
            
            QPushButton:pressed {{
                background-color: {OceanTheme.KELP_GREEN};
            }}
            
            QPushButton:disabled {{
                background-color: {OceanTheme.SEAFOAM_GRAY};
                color: {OceanTheme.TEXT_MUTED};
                border-color: {OceanTheme.SURFACE};
            }}
            
            QStatusBar {{
                background-color: {OceanTheme.MIDNIGHT};
                color: {OceanTheme.TEXT_SECONDARY};
                border-top: 1px solid {OceanTheme.SURFACE};
            }}
        """)

# =============================================================================
# APPLICATION SETUP
# =============================================================================

def setup_ocean_application(app: QApplication):
    """Konfiguriert Ocean-Theme für gesamte Anwendung"""
    
    # Application-wide dark palette
    palette = QPalette()
    
    # Window colors
    palette.setColor(QPalette.Window, QColor(OceanTheme.ABYSS))
    palette.setColor(QPalette.WindowText, QColor(OceanTheme.TEXT_PRIMARY))
    
    # Base colors (input fields)
    palette.setColor(QPalette.Base, QColor(OceanTheme.MIDNIGHT))
    palette.setColor(QPalette.AlternateBase, QColor(OceanTheme.DEEP_WATER))
    
    # Text colors
    palette.setColor(QPalette.Text, QColor(OceanTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.BrightText, QColor(OceanTheme.ARCTIC_BLUE))
    
    # Button colors
    palette.setColor(QPalette.Button, QColor(OceanTheme.DEEP_WATER))
    palette.setColor(QPalette.ButtonText, QColor(OceanTheme.TEXT_PRIMARY))
    
    # Highlight colors
    palette.setColor(QPalette.Highlight, QColor(OceanTheme.BIOLUMINESCENT))
    palette.setColor(QPalette.HighlightedText, QColor(OceanTheme.TEXT_PRIMARY))
    
    app.setPalette(palette)

def main():
    """Hauptfunktion für GUI-Anwendung"""
    from logging_plus import setup_logging
    from yt_analyzer_config import SecureConfigManager
    from yt_pipeline_manager import integrate_pipeline_with_gui
    
    # Setup Logging
    setup_logging("youtube_analyzer_gui", "INFO")
    
    # Create Application
    app = QApplication(sys.argv)
    app.setApplicationName("YouTube Analyzer")
    app.setApplicationVersion("1.0")
    
    # Setup Ocean Theme
    setup_ocean_application(app)
    
    # Load Configuration
    config_manager = SecureConfigManager()
    config_result = config_manager.load_config()
    
    # Create and show main window
    window = YouTubeAnalyzerMainWindow()
    
    # Integrate Pipeline if config loaded successfully
    if isinstance(config_result, Ok):
        config = unwrap_ok(config_result)
        integrate_pipeline_with_gui(window, config)
        window.status_bar.showMessage("Pipeline ready - Configuration loaded successfully")
    else:
        error = unwrap_err(config_result)
        window.status_bar.showMessage(f"Configuration error: {error.message}")
        window.start_button.setEnabled(False)  # Disable until config fixed
    
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
