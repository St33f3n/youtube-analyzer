#!/usr/bin/env python3
"""
YouTube Info Analyzer - UI Prototyp
"""

import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    """Haupteinstiegspunkt der Anwendung"""
    app = QApplication(sys.argv)
    app.setApplicationName("YouTube Info Analyzer")
    app.setApplicationVersion("0.1.0")
    
    # Main Window erstellen und anzeigen
    window = MainWindow()
    window.show()
    
    # Event Loop starten
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
