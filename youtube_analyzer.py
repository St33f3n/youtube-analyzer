#!/usr/bin/env python3
"""
YouTube Analyzer - Main Entry Point
Startet die GUI-Anwendung mit Ocean-Theme
"""

import sys
from pathlib import Path

# Ensure project directory is in Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def main():
    """Startet YouTube Analyzer GUI"""
    try:
        # Import GUI module
        from yt_analyzer_gui import main as gui_main
        
        print("🌊 Starting YouTube Analyzer...")
        print("=================================")
        print("GUI loading with Ocean Theme...")
        
        # Start GUI
        gui_main()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

