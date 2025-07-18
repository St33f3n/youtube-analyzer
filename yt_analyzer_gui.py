"""
YouTube Analyzer - Enhanced PySide6 GUI with Full Pipeline Manager Compatibility
Simple Interface mit Pipeline-Status-Anzeige und Config-Validation
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

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

# FIXED: Import PipelineStatus from Pipeline Manager (no local definition)
try:
    from yt_pipeline_manager import PipelineStatus, integrate_pipeline_with_gui
    PIPELINE_MANAGER_AVAILABLE = True
except ImportError:
    PIPELINE_MANAGER_AVAILABLE = False
    # Fallback minimal definition for development
    from dataclasses import dataclass, field
    
    @dataclass
    class PipelineStatus:
        audio_download_queue: int = 0
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
        
        # Enhanced fields (fallback)
        active_workers: List[str] = field(default_factory=list)
        pipeline_health: str = "healthy"
        estimated_completion: Optional[datetime] = None
        
        def is_active(self) -> bool:
            return (self.audio_download_queue + self.transcription_queue + 
                   self.analysis_queue + self.video_download_queue + 
                   self.upload_queue + self.processing_queue) > 0

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

# =============================================================================
# ENHANCED PIPELINE STATUS WIDGET
# =============================================================================

class PipelineStatusWidget(QFrame):
    """Enhanced Pipeline-Status-Widget mit korrekte Reihenfolge und erweiterten Features"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("PipelineStatusWidget")
        self.setup_ui()
        self.apply_ocean_theme()
    
    def setup_ui(self):
        """Setup der UI-Komponenten mit korrekter Grid-Reihenfolge"""
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
        
        # Queue Grid - CORRECTED ORDER: Links oben->unten, dann rechts oben->unten
        grid_layout = QGridLayout()
        
        # Queue Labels und Values
        self.queue_labels = {}
        self.queue_values = {}
        
        # FIXED: Korrekte Reihenfolge - Links oben->unten, dann rechts oben->unten
        queue_mapping = [
            # Linke Spalte (col=0)
            ("Metadata", "metadata_queue", 0, 0),
            ("Audio Download", "audio_download_queue", 1, 0), 
            ("Transcription", "transcription_queue", 2, 0),
            ("Analysis", "analysis_queue", 3, 0),
            
            # Rechte Spalte (col=2)  
            ("Video Download", "video_download_queue", 0, 2),
            ("Upload", "upload_queue", 1, 2),
            ("Processing", "processing_queue", 2, 2)
        ]
        
        for display_name, queue_attr, row, col in queue_mapping:
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
        
        # ENHANCED: Pipeline Health Indicator
        self.health_label = QLabel("Health: Healthy")
        self.health_label.setAlignment(Qt.AlignCenter)
        self.health_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        layout.addWidget(self.health_label)
        
        # Current Video (if any)
        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFont(QFont("Arial", 8))
        self.video_label.setWordWrap(True)
        layout.addWidget(self.video_label)
        
        # ENHANCED: Active Workers Display
        self.workers_label = QLabel("")
        self.workers_label.setAlignment(Qt.AlignCenter)
        self.workers_label.setFont(QFont("Arial", 8))
        self.workers_label.setWordWrap(True)
        layout.addWidget(self.workers_label)
        
        # ENHANCED: Estimated Completion Time
        self.eta_label = QLabel("")
        self.eta_label.setAlignment(Qt.AlignCenter)
        self.eta_label.setFont(QFont("Arial", 8))
        layout.addWidget(self.eta_label)
    
    def update_status(self, status: PipelineStatus):
        """Enhanced Status-Update mit allen neuen Features"""
        # Debug logging für Status-Updates
        self.logger.debug(
            f"Enhanced GUI Status Update",
            extra={
                'total_queued': status.total_queued,
                'total_completed': status.total_completed,
                'total_failed': status.total_failed,
                'current_stage': status.current_stage,
                'active_workers': getattr(status, 'active_workers', []),
                'pipeline_health': getattr(status, 'pipeline_health', 'unknown'),
                'is_active': status.is_active()
            }
        )
        
        # Current Stage
        self.current_label.setText(f"Status: {status.current_stage}")
        
        # Queue Values mit enhanced Color-Coding
        for queue_attr, value_label in self.queue_values.items():
            count = getattr(status, queue_attr, 0)
            value_label.setText(str(count))
            
            # Enhanced Color coding für aktive Queues
            if count > 0:
                value_label.setStyleSheet(f"color: {OceanTheme.BIOLUMINESCENT}; font-weight: bold;")
            else:
                value_label.setStyleSheet(f"color: {OceanTheme.TEXT_MUTED};")
        
        # Summary Stats mit enhanced Progress-Calculation
        total_processed = status.total_completed + status.total_failed
        
        self.total_label.setText(f"Total: {status.total_queued}")
        self.completed_label.setText(f"Completed: {status.total_completed}")
        self.failed_label.setText(f"Failed: {status.total_failed}")
        
        # Enhanced progress display
        if status.total_queued > 0:
            progress_percent = total_processed / status.total_queued * 100
            self.total_label.setText(f"Progress: {total_processed}/{status.total_queued} ({progress_percent:.1f}%)")
        
        # ENHANCED: Pipeline Health mit Color-Coding
        pipeline_health = getattr(status, 'pipeline_health', 'unknown')
        health_color = {
            "healthy": OceanTheme.SEA_FOAM,
            "degraded": OceanTheme.YELLOW, 
            "failed": OceanTheme.CORAL_RED,
            "unknown": OceanTheme.TEXT_MUTED
        }.get(pipeline_health, OceanTheme.TEXT_MUTED)
        
        self.health_label.setText(f"Health: {pipeline_health.title()}")
        self.health_label.setStyleSheet(f"color: {health_color}; font-weight: bold;")
        
        # Current Video mit enhanced Display
        if status.current_video:
            video_text = (f"Current: {status.current_video[:50]}..." 
                         if len(status.current_video) > 50 
                         else f"Current: {status.current_video}")
            self.video_label.setText(video_text)
            self.video_label.setStyleSheet(f"color: {OceanTheme.ARCTIC_BLUE};")
        else:
            self.video_label.setText("")
        
        # ENHANCED: Active Workers Display
        active_workers = getattr(status, 'active_workers', [])
        if active_workers:
            workers_text = f"Active: {', '.join(active_workers)}"
            self.workers_label.setText(workers_text)
            self.workers_label.setStyleSheet(f"color: {OceanTheme.BIOLUMINESCENT};")
        else:
            self.workers_label.setText("")
        
        # ENHANCED: Estimated Completion Time
        estimated_completion = getattr(status, 'estimated_completion', None)
        if estimated_completion:
            eta_text = f"ETA: {estimated_completion.strftime('%H:%M:%S')}"
            self.eta_label.setText(eta_text)
            self.eta_label.setStyleSheet(f"color: {OceanTheme.GOLDEN_BEIGE};")
        else:
            self.eta_label.setText("")
    
    def apply_ocean_theme(self):
        """Wendet Enhanced Ocean-Theme auf Widget an"""
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
# CONFIG VALIDATION WINDOW (unchanged, but enhanced theme)
# =============================================================================

class ConfigValidationWindow(QDialog):
    """Enhanced Config-Validation-Window"""
    
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
                str(unwrap_err(config_result).message),
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
        
        # Pipeline Manager Availability Check
        if PIPELINE_MANAGER_AVAILABLE:
            self.add_validation_item(
                "Pipeline Manager",
                "✅ AVAILABLE",
                "Enhanced pipeline with state management"
            )
        else:
            self.add_validation_item(
                "Pipeline Manager",
                "⚠️ FALLBACK",
                "Using fallback mode - some features limited",
                is_error=False
            )
        
        # Config Summary
        total_weight = config.rules.get_total_weight()
        self.add_validation_item(
            "Rules Configuration",
            "✅ VALID",
            f"Enabled rules: {len(enabled_rules)}, Total weight: {total_weight:.2f}"
        )
    
    def apply_ocean_theme(self):
        """Wendet Enhanced Ocean-Theme an"""
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
# ENHANCED MAIN WINDOW
# =============================================================================

class YouTubeAnalyzerMainWindow(QMainWindow):
    """Enhanced Haupt-Fenster mit vollständiger Pipeline-Integration"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("MainWindow")
        self.pipeline_status = PipelineStatus()  # Fallback status
        self.config_window = None
        
        self.setup_ui()
        self.apply_ocean_theme()
        self.setup_status_timer()
    
    def setup_ui(self):
        """Setup der Enhanced Haupt-UI"""
        self.setWindowTitle("YouTube Analyzer - Enhanced Edition")
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
        
        # Enhanced Pipeline Status Section
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
        
        # ENHANCED: Stop Pipeline Button
        self.stop_button = QPushButton("Stop Pipeline")
        self.stop_button.setFixedHeight(50)
        self.stop_button.setFont(QFont("Arial", 10))
        self.stop_button.clicked.connect(self.stop_pipeline)
        self.stop_button.setEnabled(False)  # Initially disabled
        
        button_layout.addWidget(self.start_button, 3)
        button_layout.addWidget(self.stop_button, 1)
        button_layout.addWidget(self.config_button, 1)
        
        layout.addLayout(button_layout)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Enhanced GUI loaded")
    
    def create_input_section(self) -> QFrame:
        """Erstellt Enhanced URL-Input-Sektion"""
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
        """Setup Timer für Status-Updates - nur als Fallback"""
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_pipeline_status)
        self.status_timer.start(1000)  # Fallback Timer für Mock-Mode
    
    def update_pipeline_status(self):
        """Aktualisiert Pipeline-Status-Anzeige - nur Fallback für Mock-Mode"""
        # ENHANCED: Only run timer updates if no real pipeline is connected
        if not hasattr(self, 'pipeline_manager'):
            # Fallback Mock-Update (nur wenn keine echte Pipeline)
            self.status_widget.update_status(self.pipeline_status)
        
        # Enhanced Button-State Management
        if hasattr(self, 'pipeline_manager'):
            # Real pipeline mode
            pipeline_active = hasattr(self.pipeline_manager, 'state') and self.pipeline_manager.state.name == "RUNNING"
        else:
            # Fallback mode
            pipeline_active = self.pipeline_status.is_active()
        
        if pipeline_active:
            self.start_button.setText("Processing...")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_bar.showMessage(f"Processing - {self.pipeline_status.current_stage}")
        else:
            self.start_button.setText("Start Analysis")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_bar.showMessage("Ready")
    
    def start_analysis(self):
        """Enhanced Startet YouTube-Analyse"""
        urls_text = self.url_input.toPlainText().strip()
        
        if not urls_text:
            self.status_bar.showMessage("Please enter YouTube URLs")
            return
        
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        if not urls:
            self.status_bar.showMessage("No valid URLs found")
            return
        
        self.logger.info(f"Starting enhanced analysis for {len(urls)} URLs")
        
        # ENHANCED: Check if real pipeline is available
        if hasattr(self, 'pipeline_manager') and PIPELINE_MANAGER_AVAILABLE:
            # Real pipeline will be called by integrate_pipeline_with_gui
            self.logger.info("Using enhanced pipeline manager")
            # The enhanced_start_analysis method will be set by integration
        else:
            # Fallback Mock-Demo
            self.logger.info("Using fallback mock pipeline demo")
            self.pipeline_status.audio_download_queue = len(urls)
            self.pipeline_status.total_queued = len(urls)
            self.pipeline_status.current_stage = "Extracting Metadata"
            self.pipeline_status.pipeline_health = "healthy"
            self.pipeline_status.active_workers = ["Metadata Extractor"]
        
        self.status_bar.showMessage(f"Started enhanced analysis for {len(urls)} videos")
    
    def stop_pipeline(self):
        """Enhanced Stop Pipeline"""
        if hasattr(self, 'pipeline_manager'):
            self.logger.info("Stopping pipeline manager")
            self.pipeline_manager.stop_pipeline()
        else:
            # Fallback mock stop
            self.logger.info("Stopping mock pipeline")
            self.pipeline_status = PipelineStatus()  # Reset to idle
        
        self.status_bar.showMessage("Pipeline stopped")
    
    def show_config_window(self):
        """Zeigt Enhanced Config-Validation-Fenster"""
        if self.config_window is None:
            self.config_window = ConfigValidationWindow(self)
        
        self.config_window.load_and_validate_config()
        self.config_window.show()
    
    def apply_ocean_theme(self):
        """Wendet Enhanced Ocean-Theme auf Hauptfenster an"""
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
# ENHANCED APPLICATION SETUP
# =============================================================================

def setup_ocean_application(app: QApplication):
    """Konfiguriert Enhanced Ocean-Theme für gesamte Anwendung"""
    
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
    """Enhanced Hauptfunktion für GUI-Anwendung"""
    from logging_plus import setup_logging
    from yt_analyzer_config import SecureConfigManager
    
    # Setup Enhanced Logging
    setup_logging("youtube_analyzer_gui", "INFO")
    logger = get_logger("main")
    
    # Create Application
    app = QApplication(sys.argv)
    app.setApplicationName("YouTube Analyzer Enhanced")
    app.setApplicationVersion("2.0")
    
    # Setup Enhanced Ocean Theme
    setup_ocean_application(app)
    
    # Load Configuration
    config_manager = SecureConfigManager()
    config_result = config_manager.load_config()
    
    # Create and show main window
    window = YouTubeAnalyzerMainWindow()
    
    # ENHANCED: Integrate Pipeline if available
    if isinstance(config_result, Ok) and PIPELINE_MANAGER_AVAILABLE:
        config = unwrap_ok(config_result)
        logger.info("Integrating enhanced pipeline manager")
        integrate_pipeline_with_gui(window, config)
        window.status_bar.showMessage("Enhanced Pipeline ready - Configuration loaded successfully")
    elif isinstance(config_result, Ok):
        logger.warning("Config loaded but pipeline manager not available - using fallback mode")
        window.status_bar.showMessage("Fallback mode - Configuration loaded, limited pipeline features")
    else:
        error = unwrap_err(config_result)
        logger.error(f"Configuration error: {error.message}")
        window.status_bar.showMessage(f"Configuration error: {error.message}")
        window.start_button.setEnabled(False)  # Disable until config fixed
    
    window.show()
    
    # Start event loop
    logger.info("Starting enhanced GUI application")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
