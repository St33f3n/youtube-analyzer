"""
Ocean Theme - Farbschema für YouTube Info Analyzer
"""

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


def get_main_stylesheet():
    """Hauptstylesheet für die Anwendung"""
    return f"""
    /* Globales Styling */
    QMainWindow {{
        background-color: {ABYSS};
        color: {TEXT_PRIMARY};
    }}
    
    QWidget {{
        background-color: {ABYSS};
        color: {TEXT_PRIMARY};
        font-family: 'Segoe UI', Arial, sans-serif;
    }}
    
    /* Labels */
    QLabel {{
        color: {TEXT_PRIMARY};
        font-weight: bold;
    }}
    
    /* Input Fields */
    QLineEdit {{
        background-color: {DEEP_WATER};
        border: 2px solid {SURFACE};
        border-radius: 8px;
        padding: 8px 12px;
        color: {TEXT_PRIMARY};
        font-size: 11pt;
    }}
    
    QLineEdit:focus {{
        border-color: {BIOLUMINESCENT};
        background-color: {MIDNIGHT};
    }}
    
    QLineEdit::placeholder {{
        color: {TEXT_MUTED};
    }}
    
    /* Buttons */
    QPushButton {{
        background-color: {SEA_FOAM};
        color: {TEXT_PRIMARY};
        border: none;
        border-radius: 8px;
        font-weight: bold;
        font-size: 11pt;
        padding: 8px 16px;
    }}
    
    QPushButton:hover {{
        background-color: {DEEP_TEAL};
    }}
    
    QPushButton:pressed {{
        background-color: {KELP_GREEN};
    }}
    
    QPushButton:disabled {{
        background-color: {SURFACE};
        color: {TEXT_MUTED};
    }}
    
    /* Start Button spezielle Farben */
    QPushButton#startButton {{
        background-color: {CORAL_ORANGE};
    }}
    
    QPushButton#startButton:hover {{
        background-color: {CORAL_RED};
    }}
    
    QPushButton#startButton:pressed {{
        background-color: {CORAL_DEEP};
    }}
    
    /* Progress Bar */
    QProgressBar {{
        border: 2px solid {SURFACE};
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        color: {TEXT_PRIMARY};
        background-color: {DEEP_WATER};
    }}
    
    QProgressBar::chunk {{
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 {BIOLUMINESCENT},
            stop: 0.5 {CORAL_BLUE},
            stop: 1 {ARCTIC_BLUE}
        );
        border-radius: 6px;
        margin: 1px;
    }}
    
    /* Text Area */
    QTextEdit {{
        background-color: {DEEP_WATER};
        border: 2px solid {SURFACE};
        border-radius: 8px;
        color: {TEXT_SECONDARY};
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 9pt;
        padding: 8px;
    }}
    
    QTextEdit:focus {{
        border-color: {SEAFOAM_GRAY};
    }}
    
    /* Separator */
    QFrame[frameShape="4"] {{
        color: {SURFACE};
        background-color: {SURFACE};
    }}
    
    /* Scrollbars */
    QScrollBar:vertical {{
        background-color: {DEEP_WATER};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background-color: {SEAFOAM_GRAY};
        border-radius: 6px;
        min-height: 20px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background-color: {BIOLUMINESCENT};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}
    """


def get_status_colors():
    """Status-spezifische Farben"""
    return {
        'success': ARCTIC_BLUE,
        'error': CORAL_RED,
        'warning': YELLOW,
        'info': BIOLUMINESCENT,
        'working': CORAL_LIGHT
    }
