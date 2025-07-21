"""
YouTube Analyzer - Corrected GUI with Fork-Join Architecture
Korrekte Darstellung der Fork-Join Pipeline-Architektur
"""

import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QFrame,
    QStatusBar,
    QDialog,
    QGridLayout,
    QScrollArea,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QPalette, QColor

# Import our core libraries
from core_types import Ok, Err, unwrap_ok, unwrap_err
from logging_plus import get_logger
from yt_analyzer_config import SecureConfigManager

# Import PipelineStatus from Pipeline Manager
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
        llm_processing_queue: int = 0
        trilium_upload_queue: int = 0

        total_queued: int = 0
        total_completed: int = 0
        total_failed: int = 0

        current_stage: str = "Idle"
        current_video: Optional[str] = None

        active_workers: List[str] = field(default_factory=list)
        pipeline_health: str = "healthy"
        estimated_completion: Optional[datetime] = None

        def is_active(self) -> bool:
            return (
                self.audio_download_queue
                + self.transcription_queue
                + self.analysis_queue
                + self.video_download_queue
                + self.upload_queue
                + self.llm_processing_queue
                + self.trilium_upload_queue
            ) > 0

# =============================================================================
# CONFIG VALIDATION WINDOW (vollständige Implementation)
# =============================================================================


class ConfigValidationWindow(QDialog):
    """Config-Validation-Window"""

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

    def add_validation_item(
        self, title: str, status: str, details: str = "", is_error: bool = False
    ):
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
            status_label.setStyleSheet(
                f"color: {OceanTheme.CORAL_RED}; font-weight: bold;"
            )
        else:
            status_label.setStyleSheet(
                f"color: {OceanTheme.SEA_FOAM}; font-weight: bold;"
            )

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
            child = self.config_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        # Load Config
        config_result = self.config_manager.load_config()

        if isinstance(config_result, Err):
            self.add_validation_item(
                "Configuration Loading",
                "❌ FAILED",
                str(unwrap_err(config_result).message),
                is_error=True,
            )
            return

        config = unwrap_ok(config_result)
        self.add_validation_item("Configuration Loading", "✅ SUCCESS")

        # Validate Secrets
        self.validate_secrets(config)

        # Validate Rules
        self.validate_rules(config)

        # Validate Processing Settings
        self.validate_processing_settings(config)

        # Validate LLM Settings (NEW)
        self.validate_llm_settings(config)

        # Add stretch to push items to top
        self.config_layout.addStretch()

    def validate_secrets(self, config):
        """Validiert Secret-Zugriff mit korrekten SecureConfigManager Methoden"""

        # Trilium Secret
        trilium_result = self.config_manager.get_trilium_token()
        if isinstance(trilium_result, Err):
            error = unwrap_err(trilium_result)
            self.add_validation_item(
                "Trilium Secret Access",
                "❌ FAILED",
                f"Service: {config.secrets.trilium_service}, Username: {config.secrets.trilium_username}\nError: {error.message}",
                is_error=True,
            )
        else:
            token = unwrap_ok(trilium_result)
            self.add_validation_item(
                "Trilium Secret Access",
                "✅ SUCCESS",
                f"Token length: {len(token)} characters",
            )

        # Nextcloud Secret
        nextcloud_result = self.config_manager.get_nextcloud_password()
        if isinstance(nextcloud_result, Err):
            error = unwrap_err(nextcloud_result)
            self.add_validation_item(
                "Nextcloud Secret Access",
                "❌ FAILED",
                f"Service: {config.secrets.nextcloud_service}, Username: {config.secrets.nextcloud_username}\nError: {error.message}",
                is_error=True,
            )
        else:
            password = unwrap_ok(nextcloud_result)
            self.add_validation_item(
                "Nextcloud Secret Access",
                "✅ SUCCESS",
                f"Password length: {len(password)} characters",
            )

        # LLM API Keys mit den speziellen Methoden
        llm_methods = {
            "openai": self.config_manager.get_openai_api_key,
            "anthropic": self.config_manager.get_anthropic_api_key,
            "google": self.config_manager.get_google_api_key,
        }

        for provider, method in llm_methods.items():
            try:
                key_result = method()
                if isinstance(key_result, Err):
                    error = unwrap_err(key_result)
                    # Get service info from config for better error messages
                    service_attr = f"{provider}_service"
                    username_attr = f"{provider}_username"
                    service_name = getattr(
                        config.secrets, service_attr, f"{provider}_service"
                    )
                    username = getattr(config.secrets, username_attr, "api_key")

                    self.add_validation_item(
                        f"{provider.title()} API Key",
                        "❌ NOT FOUND",
                        f"Service: {service_name}, Username: {username}\nError: {error.message}",
                        is_error=True,
                    )
                else:
                    api_key = unwrap_ok(key_result)
                    self.add_validation_item(
                        f"{provider.title()} API Key",
                        "✅ FOUND",
                        f"Key length: {len(api_key)} characters",
                    )
            except AttributeError:
                # Method doesn't exist - this provider not configured
                self.add_validation_item(
                    f"{provider.title()} API Key",
                    "⚠️ NOT CONFIGURED",
                    f"Add {provider}_service and {provider}_username to secrets section",
                )

    def validate_rules(self, config):
        """Validiert Rule-Chain-Konfiguration"""
        try:
            from yt_rulechain import load_rules_from_config

            rules_result = load_rules_from_config(config)
            if isinstance(rules_result, Err):
                self.add_validation_item(
                    "Rules Configuration",
                    "❌ FAILED",
                    str(unwrap_err(rules_result).message),
                    is_error=True,
                )
                return

            rules = unwrap_ok(rules_result)
            enabled_rules = [rule for rule in rules if rule.enabled]
            total_weight = sum(rule.weight for rule in enabled_rules)

            self.add_validation_item(
                "Rules Configuration",
                "✅ VALID",
                f"Enabled rules: {len(enabled_rules)}, Total weight: {total_weight:.2f}",
            )

        except ImportError:
            self.add_validation_item(
                "Rules Configuration",
                "⚠️ UNAVAILABLE",
                "Rule chain module not available",
                is_error=False,
            )

    def validate_processing_settings(self, config):
        """Validiert Processing-Einstellungen"""
        # Temp Folder
        temp_path = Path(config.processing.temp_folder)
        if temp_path.exists() and temp_path.is_dir():
            self.add_validation_item("Temp Folder", "✅ VALID", f"Path: {temp_path}")
        else:
            self.add_validation_item(
                "Temp Folder", "❌ NOT FOUND", f"Path: {temp_path}", is_error=True
            )

        # Whisper Settings
        if config.whisper.enabled:
            self.add_validation_item(
                "Whisper Transcription",
                "✅ ENABLED",
                f"Model: {config.whisper.model}, Language: {config.whisper.language}",
            )
        else:
            self.add_validation_item(
                "Whisper Transcription", "⚠️ DISABLED", "Transcription will be mocked"
            )

    def validate_llm_settings(self, config):
        """Validiert LLM-Einstellungen (NEW)"""
        if hasattr(config, "llm_processing"):
            provider = config.llm_processing.provider
            model = config.llm_processing.model
            prompt_file = Path(config.llm_processing.system_prompt_file)

            self.add_validation_item(
                "LLM Configuration",
                "✅ CONFIGURED",
                f"Provider: {provider}, Model: {model}",
            )

            # System Prompt File
            if prompt_file.exists():
                prompt_content = prompt_file.read_text(encoding="utf-8")
                self.add_validation_item(
                    "LLM System Prompt",
                    "✅ FOUND",
                    f"File: {prompt_file}, Length: {len(prompt_content)} characters",
                )
            else:
                self.add_validation_item(
                    "LLM System Prompt",
                    "❌ NOT FOUND",
                    f"File: {prompt_file}",
                    is_error=True,
                )
        else:
            self.add_validation_item(
                "LLM Configuration",
                "❌ NOT CONFIGURED",
                "Add llm_processing section to config.yaml",
                is_error=True,
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
# CORRECTED PIPELINE STATUS WIDGET (Fork-Join Architecture)
# =============================================================================


class PipelineStatusWidget(QFrame):
    """Corrected Pipeline Status Widget mit korrekter Fork-Join Darstellung"""

    def __init__(self):
        super().__init__()
        self.logger = get_logger("PipelineStatusWidget")
        self.setup_ui()
        self.apply_ocean_theme()

    def setup_ui(self):
        """Setup UI entsprechend echter Fork-Join Architektur"""
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

        # SECTION 1: Sequential Pipeline (bis Fork-Point)
        self.setup_sequential_pipeline(layout)

        # SECTION 2: Parallel Streams (nach Fork)
        self.setup_parallel_streams(layout)

        # SECTION 3: LLM Metrics
        self.setup_llm_metrics(layout)

        # SECTION 4: Final Results (vereinfacht)
        self.setup_final_results(layout)

        # SECTION 5: System Status
        self.setup_system_status(layout)

    def setup_sequential_pipeline(self, layout):
        """Sequential Pipeline: Audio → Trans → Analysis (bis Fork-Point)"""
        sequential_frame = QFrame()
        sequential_frame.setFrameStyle(QFrame.StyledPanel)
        sequential_layout = QVBoxLayout(sequential_frame)

        # Title
        title = QLabel("Sequential Pipeline")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        sequential_layout.addWidget(title)

        # Horizontal Flow: Audio → Trans → Analysis
        flow_layout = QHBoxLayout()

        # Audio Download
        audio_layout = QVBoxLayout()
        audio_layout.addWidget(QLabel("Audio Download"))
        self.audio_download_value = QLabel("0")
        self.audio_download_value.setAlignment(Qt.AlignCenter)
        self.audio_download_value.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        audio_layout.addWidget(self.audio_download_value)
        flow_layout.addLayout(audio_layout)

        # Arrow
        flow_layout.addWidget(QLabel("→"))

        # Transcription
        trans_layout = QVBoxLayout()
        trans_layout.addWidget(QLabel("Transcription"))
        self.transcription_value = QLabel("0")
        self.transcription_value.setAlignment(Qt.AlignCenter)
        self.transcription_value.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        trans_layout.addWidget(self.transcription_value)
        flow_layout.addLayout(trans_layout)

        # Arrow
        flow_layout.addWidget(QLabel("→"))

        # Analysis (Fork Point)
        analysis_layout = QVBoxLayout()
        analysis_layout.addWidget(QLabel("Analysis"))
        self.analysis_value = QLabel("0")
        self.analysis_value.setAlignment(Qt.AlignCenter)
        self.analysis_value.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        analysis_layout.addWidget(self.analysis_value)
        flow_layout.addLayout(analysis_layout)

        sequential_layout.addLayout(flow_layout)
        layout.addWidget(sequential_frame)

    def setup_parallel_streams(self, layout):
        """Parallel Streams nach Fork-Point"""
        streams_frame = QFrame()
        streams_frame.setFrameStyle(QFrame.StyledPanel)
        streams_layout = QVBoxLayout(streams_frame)

        # Title
        title = QLabel("🔀 Parallel Streams (after Fork)")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        streams_layout.addWidget(title)

        # Horizontal Layout für beide Streams
        streams_horizontal = QHBoxLayout()

        # Stream A: Video Processing
        stream_a_frame = QFrame()
        stream_a_frame.setFrameStyle(QFrame.Box)
        stream_a_layout = QVBoxLayout(stream_a_frame)

        stream_a_title = QLabel("🎥 Stream A (Video)")
        stream_a_title.setAlignment(Qt.AlignCenter)
        stream_a_title.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        stream_a_layout.addWidget(stream_a_title)

        # Video Download → Upload
        video_flow = QHBoxLayout()

        video_dl_layout = QVBoxLayout()
        video_dl_layout.addWidget(QLabel("Video Download"))
        self.video_download_value = QLabel("0")
        self.video_download_value.setAlignment(Qt.AlignCenter)
        self.video_download_value.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        video_dl_layout.addWidget(self.video_download_value)
        video_flow.addLayout(video_dl_layout)

        video_flow.addWidget(QLabel("→"))

        upload_layout = QVBoxLayout()
        upload_layout.addWidget(QLabel("Upload"))
        self.upload_value = QLabel("0")
        self.upload_value.setAlignment(Qt.AlignCenter)
        self.upload_value.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        upload_layout.addWidget(self.upload_value)
        video_flow.addLayout(upload_layout)

        stream_a_layout.addLayout(video_flow)
        streams_horizontal.addWidget(stream_a_frame)

        # Stream B: LLM Processing
        stream_b_frame = QFrame()
        stream_b_frame.setFrameStyle(QFrame.Box)
        stream_b_layout = QVBoxLayout(stream_b_frame)

        stream_b_title = QLabel("📄 Stream B (Transcript)")
        stream_b_title.setAlignment(Qt.AlignCenter)
        stream_b_title.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        stream_b_layout.addWidget(stream_b_title)

        # LLM Processing → Trilium Upload
        llm_flow = QHBoxLayout()

        llm_layout = QVBoxLayout()
        llm_layout.addWidget(QLabel("LLM Processing"))
        self.llm_processing_value = QLabel("0")
        self.llm_processing_value.setAlignment(Qt.AlignCenter)
        self.llm_processing_value.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        llm_layout.addWidget(self.llm_processing_value)
        llm_flow.addLayout(llm_layout)

        llm_flow.addWidget(QLabel("→"))

        trilium_layout = QVBoxLayout()
        trilium_layout.addWidget(QLabel("Trilium Upload"))
        self.trilium_upload_value = QLabel("0")
        self.trilium_upload_value.setAlignment(Qt.AlignCenter)
        self.trilium_upload_value.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        trilium_layout.addWidget(self.trilium_upload_value)
        llm_flow.addLayout(trilium_layout)

        stream_b_layout.addLayout(llm_flow)
        streams_horizontal.addWidget(stream_b_frame)

        streams_layout.addLayout(streams_horizontal)
        layout.addWidget(streams_frame)

    def setup_llm_metrics(self, layout):
        """LLM Metrics Panel"""
        llm_frame = QFrame()
        llm_frame.setFrameStyle(QFrame.StyledPanel)
        llm_layout = QVBoxLayout(llm_frame)

        # Title
        title = QLabel("🤖 LLM Metrics")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        llm_layout.addWidget(title)

        # Metrics Grid
        metrics_grid = QGridLayout()

        # Provider
        metrics_grid.addWidget(QLabel("Provider:"), 0, 0)
        self.llm_provider_value = QLabel("None")
        self.llm_provider_value.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        metrics_grid.addWidget(self.llm_provider_value, 0, 1)

        # Total Tokens
        metrics_grid.addWidget(QLabel("Tokens:"), 0, 2)
        self.llm_tokens_value = QLabel("0")
        self.llm_tokens_value.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        metrics_grid.addWidget(self.llm_tokens_value, 0, 3)

        # Total Cost
        metrics_grid.addWidget(QLabel("Cost:"), 1, 0)
        self.llm_cost_value = QLabel("$0.000")
        self.llm_cost_value.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        metrics_grid.addWidget(self.llm_cost_value, 1, 1)

        # Average Processing Time
        metrics_grid.addWidget(QLabel("Avg Time:"), 1, 2)
        self.llm_avg_time_value = QLabel("0.0s")
        self.llm_avg_time_value.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        metrics_grid.addWidget(self.llm_avg_time_value, 1, 3)

        llm_layout.addLayout(metrics_grid)
        layout.addWidget(llm_frame)

    def setup_final_results(self, layout):
        """Final Results - Vereinfachte Anzeige"""
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.StyledPanel)
        results_layout = QVBoxLayout(results_frame)

        # Title
        title = QLabel("📊 Final Results")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        results_layout.addWidget(title)

        # Results Grid
        results_grid = QGridLayout()

        # Total Archived Successfully
        results_grid.addWidget(QLabel("Total Archived:"), 0, 0)
        self.total_archived_value = QLabel("0")
        self.total_archived_value.setAlignment(Qt.AlignCenter)
        self.total_archived_value.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        results_grid.addWidget(self.total_archived_value, 0, 1)

        # Failed at Stream A
        results_grid.addWidget(QLabel("Failed Stream A:"), 1, 0)
        self.failed_stream_a_value = QLabel("0")
        self.failed_stream_a_value.setAlignment(Qt.AlignCenter)
        self.failed_stream_a_value.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        results_grid.addWidget(self.failed_stream_a_value, 1, 1)

        # Failed at Stream B
        results_grid.addWidget(QLabel("Failed Stream B:"), 2, 0)
        self.failed_stream_b_value = QLabel("0")
        self.failed_stream_b_value.setAlignment(Qt.AlignCenter)
        self.failed_stream_b_value.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        results_grid.addWidget(self.failed_stream_b_value, 2, 1)

        results_layout.addLayout(results_grid)
        layout.addWidget(results_frame)

    def setup_system_status(self, layout):
        """System Status (unverändert)"""
        # Pipeline Health
        self.health_label = QLabel("Health: Healthy")
        self.health_label.setAlignment(Qt.AlignCenter)
        self.health_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        layout.addWidget(self.health_label)

        # Current Video
        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFont(QFont("Arial", 8))
        self.video_label.setWordWrap(True)
        layout.addWidget(self.video_label)

        # Active Workers
        self.workers_label = QLabel("")
        self.workers_label.setAlignment(Qt.AlignCenter)
        self.workers_label.setFont(QFont("Arial", 8))
        self.workers_label.setWordWrap(True)
        layout.addWidget(self.workers_label)

        # ETA
        self.eta_label = QLabel("")
        self.eta_label.setAlignment(Qt.AlignCenter)
        self.eta_label.setFont(QFont("Arial", 8))
        layout.addWidget(self.eta_label)

    def update_status(self, status: PipelineStatus):
        """Update Status mit korrekter Fork-Join Logik"""

        # Sequential Pipeline Updates
        self.audio_download_value.setText(
            str(getattr(status, "audio_download_queue", 0))
        )
        self.transcription_value.setText(str(getattr(status, "transcription_queue", 0)))
        self.analysis_value.setText(str(getattr(status, "analysis_queue", 0)))

        # Stream A (Video) Updates
        self.video_download_value.setText(
            str(getattr(status, "video_download_queue", 0))
        )
        self.upload_value.setText(str(getattr(status, "upload_queue", 0)))

        # Stream B (LLM) Updates
        self.llm_processing_value.setText(
            str(getattr(status, "llm_processing_queue", 0))
        )
        self.trilium_upload_value.setText(
            str(getattr(status, "trilium_upload_queue", 0))
        )

        # Color-coding für aktive Queues
        for queue_attr, value_widget in [
            ("audio_download_queue", self.audio_download_value),
            ("transcription_queue", self.transcription_value),
            ("analysis_queue", self.analysis_value),
            ("video_download_queue", self.video_download_value),
            ("upload_queue", self.upload_value),
            ("llm_processing_queue", self.llm_processing_value),
            ("trilium_upload_queue", self.trilium_upload_value),
        ]:
            count = getattr(status, queue_attr, 0)
            if count > 0:
                value_widget.setStyleSheet(
                    f"color: {OceanTheme.BIOLUMINESCENT}; font-weight: bold;"
                )
            else:
                value_widget.setStyleSheet(f"color: {OceanTheme.TEXT_MUTED};")

        # LLM Metrics Updates
        self.llm_provider_value.setText(getattr(status, "active_llm_provider", "None"))
        self.llm_tokens_value.setText(f"{getattr(status, 'total_llm_tokens', 0):,}")
        self.llm_cost_value.setText(f"${getattr(status, 'total_llm_cost', 0.0):.3f}")

        # Calculate average processing time
        total_processed = getattr(status, "transcript_stream_completed", 0)
        if total_processed > 0:
            # This would be calculated from real metrics in production
            avg_time = 3.2  # Placeholder
            self.llm_avg_time_value.setText(f"{avg_time:.1f}s")
        else:
            self.llm_avg_time_value.setText("0.0s")

        # Final Results Updates (vereinfacht)
        total_archived = getattr(status, "final_archived", 0)
        failed_stream_a = getattr(status, "video_stream_failed", 0)
        failed_stream_b = getattr(status, "transcript_stream_failed", 0)

        self.total_archived_value.setText(str(total_archived))
        self.failed_stream_a_value.setText(str(failed_stream_a))
        self.failed_stream_b_value.setText(str(failed_stream_b))

        # Color coding für Results
        if total_archived > 0:
            self.total_archived_value.setStyleSheet(
                f"color: {OceanTheme.SEA_FOAM}; font-weight: bold;"
            )

        if failed_stream_a > 0:
            self.failed_stream_a_value.setStyleSheet(
                f"color: {OceanTheme.CORAL_ORANGE}; font-weight: bold;"
            )

        if failed_stream_b > 0:
            self.failed_stream_b_value.setStyleSheet(
                f"color: {OceanTheme.CORAL_ORANGE}; font-weight: bold;"
            )

        # System Status Updates (unverändert)
        self.current_label.setText(f"Status: {status.current_stage}")

        pipeline_health = getattr(status, "pipeline_health", "unknown")
        health_color = {
            "healthy": OceanTheme.SEA_FOAM,
            "degraded": OceanTheme.YELLOW,
            "failed": OceanTheme.CORAL_RED,
            "unknown": OceanTheme.TEXT_MUTED,
        }.get(pipeline_health, OceanTheme.TEXT_MUTED)

        self.health_label.setText(f"Health: {pipeline_health.title()}")
        self.health_label.setStyleSheet(f"color: {health_color}; font-weight: bold;")

        # Current Video
        if status.current_video:
            video_text = (
                f"Current: {status.current_video[:50]}..."
                if len(status.current_video) > 50
                else f"Current: {status.current_video}"
            )
            self.video_label.setText(video_text)
            self.video_label.setStyleSheet(f"color: {OceanTheme.ARCTIC_BLUE};")
        else:
            self.video_label.setText("")

        # Active Workers
        active_workers = getattr(status, "active_workers", [])
        if active_workers:
            workers_text = f"Active: {', '.join(active_workers)}"
            self.workers_label.setText(workers_text)
            self.workers_label.setStyleSheet(f"color: {OceanTheme.BIOLUMINESCENT};")
        else:
            self.workers_label.setText("")

        # Debug Logging
        self.logger.debug(
            "GUI Status Update",
            extra={
                "total_queued": status.total_queued,
                "sequential_active": status.audio_download_queue
                + status.transcription_queue
                + status.analysis_queue,
                "stream_a_active": getattr(status, "video_download_queue", 0)
                + getattr(status, "upload_queue", 0),
                "stream_b_active": getattr(status, "llm_processing_queue", 0)
                + getattr(status, "trilium_upload_queue", 0),
                "final_archived": total_archived,
                "failed_a": failed_stream_a,
                "failed_b": failed_stream_b,
                "llm_provider": getattr(status, "active_llm_provider", "None"),
                "llm_tokens": getattr(status, "total_llm_tokens", 0),
                "llm_cost": getattr(status, "total_llm_cost", 0.0),
            },
        )

    def apply_ocean_theme(self):
        """Wendet Ocean-Theme-Styling an"""
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {OceanTheme.DEEP_WATER};
                border: 1px solid {OceanTheme.SURFACE};
                border-radius: 5px;
                margin: 2px;
                padding: 5px;
            }}
            QLabel {{
                color: {OceanTheme.TEXT_PRIMARY};
                background-color: transparent;
                border: none;
            }}
        """)


# =============================================================================
# MAIN WINDOW CLASS (unverändert bis auf PipelineStatusWidget)
# =============================================================================


class YouTubeAnalyzerMainWindow(QMainWindow):
    """Hauptfenster der YouTube Analyzer Anwendung"""

    def __init__(self):
        super().__init__()
        self.logger = get_logger("YouTubeAnalyzerMainWindow")
        self.pipeline_status = PipelineStatus()  # Mock status für Fallback
        self.config_window = None  # Config validation window

        self.setup_ui()
        self.apply_ocean_theme()
        self.setup_status_timer()

    def setup_ui(self):
        """Setup der Benutzeroberfläche"""
        self.setWindowTitle("YouTube Analyzer")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QHBoxLayout(central_widget)

        # Left Panel: Input
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 2)

        # Right Panel: Status
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

    def create_left_panel(self) -> QWidget:
        """Erstellt linkes Panel mit Input-Feldern"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Input Section
        input_section = self.create_input_section()
        layout.addWidget(input_section)

        # Buttons
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Analysis")
        self.start_button.setFixedHeight(50)
        self.start_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.start_button.clicked.connect(self.start_analysis)

        self.config_button = QPushButton("Check Configuration")
        self.config_button.setFixedHeight(50)
        self.config_button.setFont(QFont("Arial", 10))
        self.config_button.clicked.connect(self.show_config_window)

        self.stop_button = QPushButton("Stop Pipeline")
        self.stop_button.setFixedHeight(50)
        self.stop_button.setFont(QFont("Arial", 10))
        self.stop_button.clicked.connect(self.stop_pipeline)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button, 3)
        button_layout.addWidget(self.stop_button, 1)
        button_layout.addWidget(self.config_button, 1)

        layout.addLayout(button_layout)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        return panel

    def create_right_panel(self) -> QWidget:
        """Erstellt rechtes Panel mit Status-Anzeige"""
        panel = QWidget()
        self.right_panel_layout = QVBoxLayout(panel)

        # Corrected Pipeline Status Widget
        self.status_widget = PipelineStatusWidget()
        self.right_panel_layout.addWidget(self.status_widget)

        return panel

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
        self.url_input.setPlaceholderText(
            "Paste YouTube URLs here, one per line...\n\nhttps://youtube.com/watch?v=...\nhttps://youtu.be/..."
        )
        self.url_input.setMinimumHeight(150)
        self.url_input.setFont(QFont("Consolas", 10))
        layout.addWidget(self.url_input)

        return frame

    def setup_status_timer(self):
        """Setup Timer für Status-Updates - nur als Fallback"""
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_pipeline_status)
        self.status_timer.start(1000)

    def update_pipeline_status(self):
        """Aktualisiert Pipeline-Status-Anzeige - nur Fallback"""
        if not hasattr(self, "pipeline_manager"):
            self.status_widget.update_status(self.pipeline_status)

        # Button-State Management
        if hasattr(self, "pipeline_manager"):
            is_active = self.pipeline_manager.state.value == "running"
            self.start_button.setEnabled(not is_active)
            self.stop_button.setEnabled(is_active)

    def start_analysis(self):
        """Startet die Pipeline-Analyse - Fallback"""
        urls_text = self.url_input.toPlainText().strip()

        if not urls_text:
            self.status_bar.showMessage("Please enter YouTube URLs")
            return

        self.status_bar.showMessage("Fallback mode - No pipeline manager available")
        self.logger.warning("No pipeline manager available")

    def stop_pipeline(self):
        """Stoppt die Pipeline - Fallback"""
        if hasattr(self, "pipeline_manager"):
            self.pipeline_manager.stop_pipeline()
            self.status_bar.showMessage("Pipeline stopping...")
        else:
            self.status_bar.showMessage("No active pipeline")

    def show_config_window(self):
        """Zeigt Config-Validation-Fenster"""
        if self.config_window is None:
            self.config_window = ConfigValidationWindow(self)

        self.config_window.load_and_validate_config()
        self.config_window.show()

    def apply_ocean_theme(self):
        """Wendet Ocean-Theme auf gesamtes Fenster an"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {OceanTheme.ABYSS};
                color: {OceanTheme.TEXT_PRIMARY};
            }}
            QWidget {{
                background-color: {OceanTheme.ABYSS};
                color: {OceanTheme.TEXT_PRIMARY};
            }}
            QFrame {{
                background-color: {OceanTheme.DEEP_WATER};
                border: 1px solid {OceanTheme.SURFACE};
                border-radius: 5px;
                margin: 2px;
                padding: 10px;
            }}
            QTextEdit {{
                background-color: {OceanTheme.MIDNIGHT};
                border: 1px solid {OceanTheme.SURFACE};
                border-radius: 3px;
                padding: 5px;
                color: {OceanTheme.TEXT_PRIMARY};
            }}
            QPushButton {{
                background-color: {OceanTheme.DEEP_TEAL};
                border: 1px solid {OceanTheme.SEA_FOAM};
                border-radius: 5px;
                padding: 8px;
                color: {OceanTheme.TEXT_PRIMARY};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {OceanTheme.SEA_FOAM};
            }}
            QPushButton:pressed {{
                background-color: {OceanTheme.KELP_GREEN};
            }}
            QPushButton:disabled {{
                background-color: {OceanTheme.SURFACE};
                color: {OceanTheme.TEXT_MUTED};
            }}
            QLabel {{
                color: {OceanTheme.TEXT_PRIMARY};
                background-color: transparent;
            }}
            QStatusBar {{
                background-color: {OceanTheme.DEEP_WATER};
                color: {OceanTheme.TEXT_PRIMARY};
                border-top: 1px solid {OceanTheme.SURFACE};
            }}
        """)


# =============================================================================
# APPLICATION SETUP
# =============================================================================


def setup_ocean_application(app: QApplication):
    """Konfiguriert Ocean-Theme für gesamte Anwendung"""
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

    # Setup Logging
    setup_logging("youtube_analyzer_gui", "INFO")
    logger = get_logger("main")

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

    # Integrate Pipeline if available
    if isinstance(config_result, Ok) and PIPELINE_MANAGER_AVAILABLE:
        config = unwrap_ok(config_result)
        logger.info("Integrating pipeline manager")
        integrate_pipeline_with_gui(window, config)
        window.status_bar.showMessage(
            "Pipeline ready - Configuration loaded successfully"
        )
    elif isinstance(config_result, Ok):
        logger.warning("Config loaded but pipeline manager not available")
        window.status_bar.showMessage(
            "Fallback mode - Configuration loaded, limited pipeline features"
        )
    else:
        error = unwrap_err(config_result)
        logger.error(f"Configuration error: {error.message}")
        window.status_bar.showMessage(f"Configuration error: {error.message}")
        window.start_button.setEnabled(False)

    window.show()

    # Start event loop
    logger.info("Starting GUI application")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
