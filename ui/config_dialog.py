"""
Config Dialog - Regel-Übersicht und Einstellungen anzeigen
KORRIGIERTE VERSION - Korrekte Imports und Variable-Referenzen
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QGroupBox, QPushButton, QHeaderView, QFrame, QTextEdit,
    QScrollArea, QWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from pathlib import Path
from typing import Dict, Any

# KORRIGIERTE IMPORTS
from config.settings import get_config, get_config_loader, reload_config
from ui.colors import get_main_stylesheet, get_status_colors
from config.secrets import get_secrets_manager


class ConfigDialog(QDialog):
    """Config-Anzeige Dialog für Regeln und Einstellungen"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = None
        self.config_loader = get_config_loader()  # Config-Loader richtig initialisieren
        self.secrets_manager = get_secrets_manager()
        self.status_colors = get_status_colors()
        
        self.init_ui()
        self.load_config_data()
        
    def init_ui(self):
        """UI initialisieren"""
        self.setWindowTitle("🔧 Konfiguration & Regeln")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        # Ocean Theme anwenden
        self.setStyleSheet(get_main_stylesheet())
        
        # Main Layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Header
        self.create_header(main_layout)
        
        # Scroll Area für Content
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Config Sections
        self.create_rules_section(scroll_layout)
        self.create_scoring_section(scroll_layout)
        self.create_secrets_section(scroll_layout)
        self.create_storage_section(scroll_layout)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        # Button Bar
        self.create_button_bar(main_layout)
        
    def create_header(self, parent_layout):
        """Header mit Titel und Status"""
        header_layout = QHBoxLayout()
        
        # Titel
        title_label = QLabel("⚙️ YouTube Analyzer - Konfiguration")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        # Status
        self.status_label = QLabel("📊 Lade Konfiguration...")
        self.status_label.setFont(QFont("Segoe UI", 10))
        header_layout.addWidget(self.status_label)
        header_layout.addStretch()
        
        parent_layout.addLayout(header_layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        parent_layout.addWidget(separator)
        
    def create_rules_section(self, parent_layout):
        """Regeln-Sektion erstellen"""
        # Group Box
        rules_group = QGroupBox("📋 Analyse-Regeln")
        rules_group.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        rules_layout = QVBoxLayout(rules_group)
        
        # Rules Table
        self.rules_table = QTableWidget()
        self.rules_table.setColumnCount(5)
        self.rules_table.setHorizontalHeaderLabels([
            "Status", "Regel Name", "Prompt-Datei", "Gewichtung", "Verfügbar"
        ])
        
        # Table Styling
        header = self.rules_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Status
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Name
        header.setSectionResizeMode(2, QHeaderView.Stretch)           # Datei
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Gewichtung
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Verfügbar
        
        self.rules_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.rules_table.setAlternatingRowColors(False)  # Einheitlicher Hintergrund
        self.rules_table.setMinimumHeight(200)
        
        rules_layout.addWidget(self.rules_table)
        parent_layout.addWidget(rules_group)
        
    def create_scoring_section(self, parent_layout):
        """Scoring-Sektion erstellen"""
        scoring_group = QGroupBox("🎯 Bewertungseinstellungen")
        scoring_group.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        scoring_layout = QVBoxLayout(scoring_group)
        
        # Info Layout
        info_layout = QHBoxLayout()
        
        # Threshold
        self.threshold_label = QLabel("📊 Threshold: Lade...")
        self.threshold_label.setFont(QFont("Segoe UI", 11))
        info_layout.addWidget(self.threshold_label)
        
        # Confidence
        self.confidence_label = QLabel("🎯 Min. Confidence: Lade...")
        self.confidence_label.setFont(QFont("Segoe UI", 11))
        info_layout.addWidget(self.confidence_label)
        
        # Total Weight
        self.weight_label = QLabel("⚖️ Gesamt-Gewichtung: Lade...")
        self.weight_label.setFont(QFont("Segoe UI", 11))
        info_layout.addWidget(self.weight_label)
        
        info_layout.addStretch()
        scoring_layout.addLayout(info_layout)
        parent_layout.addWidget(scoring_group)
        
    def create_secrets_section(self, parent_layout):
        """Secrets-Sektion erstellen"""
        secrets_group = QGroupBox("🔐 Secret Status")
        secrets_group.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        secrets_layout = QVBoxLayout(secrets_group)
        
        # Secrets Status Layout
        secrets_info_layout = QHBoxLayout()
        
        self.trilium_status_label = QLabel("📝 Trilium API: Prüfe...")
        self.trilium_status_label.setFont(QFont("Segoe UI", 11))
        secrets_info_layout.addWidget(self.trilium_status_label)
        
        self.nextcloud_status_label = QLabel("☁️ NextCloud: Prüfe...")
        self.nextcloud_status_label.setFont(QFont("Segoe UI", 11))
        secrets_info_layout.addWidget(self.nextcloud_status_label)
        
        secrets_info_layout.addStretch()
        secrets_layout.addLayout(secrets_info_layout)
        parent_layout.addWidget(secrets_group)
        
    def create_storage_section(self, parent_layout):
        """Storage-Sektion erstellen"""
        storage_group = QGroupBox("💾 Speicher-Einstellungen")
        storage_group.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        storage_layout = QVBoxLayout(storage_group)
        
        # Storage Info
        storage_info_layout = QVBoxLayout()
        
        self.nextcloud_path_label = QLabel("☁️ NextCloud Pfad: Lade...")
        self.nextcloud_path_label.setFont(QFont("Segoe UI", 10))
        storage_info_layout.addWidget(self.nextcloud_path_label)
        
        self.trilium_note_label = QLabel("📝 Trilium Parent: Lade...")
        self.trilium_note_label.setFont(QFont("Segoe UI", 10))
        storage_info_layout.addWidget(self.trilium_note_label)
        
        self.sqlite_path_label = QLabel("🗄️ SQLite DB: Lade...")
        self.sqlite_path_label.setFont(QFont("Segoe UI", 10))
        storage_info_layout.addWidget(self.sqlite_path_label)
        
        storage_layout.addLayout(storage_info_layout)
        parent_layout.addWidget(storage_group)
        
    def create_button_bar(self, parent_layout):
        """Button Bar am Ende"""
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Reload Button
        reload_button = QPushButton("🔄 Neu laden")
        reload_button.setMinimumWidth(120)
        reload_button.clicked.connect(self.reload_config)
        button_layout.addWidget(reload_button)
        
        # Close Button
        close_button = QPushButton("✅ Schließen")
        close_button.setMinimumWidth(120)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        parent_layout.addLayout(button_layout)
        
    def load_config_data(self):
        """Lade und zeige Konfigurationsdaten"""
        try:
            # Config laden mit korrekte get_config() Aufruf
            self.config = get_config()
            
            # Rules Table füllen
            self.populate_rules_table()
            
            # Scoring Info
            self.update_scoring_info()
            
            # Secrets Status
            self.update_secrets_status()
            
            # Storage Info
            self.update_storage_info()
            
            # Status Update
            total_rules = len(self.config.rules)
            active_rules = sum(1 for r in self.config.rules.values() if r.enabled)
            self.status_label.setText(f"✅ {total_rules} Regeln geladen ({active_rules} aktiv)")
            
        except Exception as e:
            self.status_label.setText(f"❌ Config-Fehler: {str(e)}")
            
            # Debug-Information anzeigen
            error_details = f"Config-Loading Fehler:\n{str(e)}\n\nType: {type(e).__name__}"
            
            # Fallback-Anzeige
            self.threshold_label.setText("📊 Threshold: Fehler beim Laden")
            self.confidence_label.setText("🎯 Min. Confidence: Fehler beim Laden")
            self.weight_label.setText("⚖️ Gesamt-Gewichtung: Fehler beim Laden")
            
            print(f"Config Dialog Error: {error_details}")  # Debug-Output
            
    def populate_rules_table(self):
        """Rules Table mit Daten füllen"""
        if not self.config:
            return
            
        try:
            rules = self.config.rules
            self.rules_table.setRowCount(len(rules))
            
            for row, (name, rule) in enumerate(rules.items()):
                # Status (Aktiv/Inaktiv)
                status_icon = "🟢" if rule.enabled else "🔴"
                status_item = QTableWidgetItem(status_icon)
                status_item.setTextAlignment(Qt.AlignCenter)
                self.rules_table.setItem(row, 0, status_item)
                
                # Name
                name_item = QTableWidgetItem(name)
                name_item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                self.rules_table.setItem(row, 1, name_item)
                
                # Prompt-Datei
                file_item = QTableWidgetItem(rule.file)
                self.rules_table.setItem(row, 2, file_item)
                
                # Gewichtung
                weight_item = QTableWidgetItem(f"{rule.weight:.2f}")
                weight_item.setTextAlignment(Qt.AlignCenter)
                self.rules_table.setItem(row, 3, weight_item)
                
                # Verfügbarkeit
                file_exists = Path(rule.file).exists()
                available_icon = "✅" if file_exists else "❌"
                available_item = QTableWidgetItem(available_icon)
                available_item.setTextAlignment(Qt.AlignCenter)
                self.rules_table.setItem(row, 4, available_item)
                
        except Exception as e:
            print(f"Rules table population error: {e}")
            
    def update_scoring_info(self):
        """Scoring-Informationen aktualisieren"""
        if not self.config:
            return
            
        try:
            threshold = self.config.scoring.threshold
            confidence = self.config.scoring.min_confidence
            
            # KORRIGIERT: Verwende self.config statt undefined config_loader
            total_weight = self.config.get_total_weight()
            
            self.threshold_label.setText(f"📊 Threshold: {threshold:.2f}")
            self.confidence_label.setText(f"🎯 Min. Confidence: {confidence:.2f}")
            
            # Gewichtung mit Farbe
            weight_color = self.status_colors['success'] if abs(total_weight - 1.0) < 0.01 else self.status_colors['warning']
            self.weight_label.setText(f'<span style="color: {weight_color};">⚖️ Gesamt-Gewichtung: {total_weight:.3f}</span>')
            
        except Exception as e:
            print(f"Scoring info update error: {e}")
            self.threshold_label.setText("📊 Threshold: Fehler")
            self.confidence_label.setText("🎯 Min. Confidence: Fehler")
            self.weight_label.setText("⚖️ Gesamt-Gewichtung: Fehler")
        
    def update_secrets_status(self):
        """Secrets-Status aktualisieren"""
        try:
            status = self.secrets_manager.check_secrets_availability()
            
            # Config für Usernames verwenden
            if self.config:
                trilium_user = self.config.secrets.trilium_username
                nextcloud_user = self.config.secrets.nextcloud_username
                
                # Trilium
                trilium_icon = "✅" if status.get('trilium_api_key', False) else "❌"
                trilium_text = f"Verfügbar ({trilium_user})" if status.get('trilium_api_key', False) else f"Nicht verfügbar ({trilium_user})"
                self.trilium_status_label.setText(f"📝 Trilium API: {trilium_icon} {trilium_text}")
                
                # NextCloud
                nextcloud_icon = "✅" if status.get('nextcloud_credentials', False) else "❌"
                nextcloud_text = f"Verfügbar ({nextcloud_user})" if status.get('nextcloud_credentials', False) else f"Nicht verfügbar ({nextcloud_user})"
                self.nextcloud_status_label.setText(f"☁️ NextCloud: {nextcloud_icon} {nextcloud_text}")
            else:
                # Fallback ohne Usernames
                trilium_icon = "✅" if status.get('trilium_api_key', False) else "❌"
                trilium_text = "Verfügbar" if status.get('trilium_api_key', False) else "Nicht verfügbar"
                self.trilium_status_label.setText(f"📝 Trilium API: {trilium_icon} {trilium_text}")
                
                nextcloud_icon = "✅" if status.get('nextcloud_credentials', False) else "❌"
                nextcloud_text = "Verfügbar" if status.get('nextcloud_credentials', False) else "Nicht verfügbar"
                self.nextcloud_status_label.setText(f"☁️ NextCloud: {nextcloud_icon} {nextcloud_text}")
                
        except Exception as e:
            print(f"Secrets status update error: {e}")
            self.trilium_status_label.setText("📝 Trilium API: ❌ Fehler beim Prüfen")
            self.nextcloud_status_label.setText("☁️ NextCloud: ❌ Fehler beim Prüfen")
        
    def update_storage_info(self):
        """Storage-Informationen aktualisieren"""
        if not self.config:
            return
            
        try:
            storage = self.config.storage
            
            self.nextcloud_path_label.setText(f"☁️ NextCloud Pfad: {storage.nextcloud_path}")
            self.trilium_note_label.setText(f"📝 Trilium Parent: {storage.trilium_parent_note}")
            self.sqlite_path_label.setText(f"🗄️ SQLite DB: {storage.sqlite_path}")
            
        except Exception as e:
            print(f"Storage info update error: {e}")
            self.nextcloud_path_label.setText("☁️ NextCloud Pfad: Fehler beim Laden")
            self.trilium_note_label.setText("📝 Trilium Parent: Fehler beim Laden")
            self.sqlite_path_label.setText("🗄️ SQLite DB: Fehler beim Laden")
        
    def reload_config(self):
        """Konfiguration neu laden"""
        try:
            self.status_label.setText("🔄 Lade Konfiguration neu...")
            
            # Config neu laden mit korrektem reload_config Aufruf
            reload_config()
            
            # UI aktualisieren
            self.load_config_data()
            
        except Exception as e:
            self.status_label.setText(f"❌ Reload-Fehler: {str(e)}")
            print(f"Config reload error: {e}")
