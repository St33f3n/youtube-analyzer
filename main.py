"""
Main Application - Entry Point für YouTube Analyzer
Vollständig überarbeitet mit Result-Types und vollständigen Type-Hints
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn
from typing import Optional

from PySide6.QtWidgets import QApplication

from config.settings import get_config
from yt_types import ConfigurationError
from yt_types import Err
from yt_types import Ok
from yt_types import Result
from ui.main_window import MainWindow
from utils.logging import ComponentLogger
from utils.logging import get_development_config
from utils.logging import get_production_config
from utils.logging import setup_logging


class YouTubeAnalyzerApp:
    """Hauptanwendung für YouTube Analyzer"""
    
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.logger = ComponentLogger("YouTubeAnalyzerApp")
        self.app: Optional[QApplication] = None
        self.main_window: Optional[MainWindow] = None
        
        # Logging Setup
        self._setup_logging()
        
        self.logger.info(
            "YouTube Analyzer application initialized",
            debug_mode=self.debug,
            python_version=sys.version,
            platform=sys.platform,
        )
    
    def _setup_logging(self) -> None:
        """Setup Logging basierend auf Debug-Modus"""
        if self.debug:
            config = get_development_config()
        else:
            config = get_production_config()
        
        setup_logging(config)
    
    def _validate_environment(self) -> Result[None, ConfigurationError]:
        """Validiere Laufzeitumgebung"""
        try:
            # Python Version prüfen
            if sys.version_info < (3, 11):
                return Err(ConfigurationError(
                    f"Python 3.11+ required, got {sys.version_info.major}.{sys.version_info.minor}",
                    {
                        'current_version': f"{sys.version_info.major}.{sys.version_info.minor}",
                        'required_version': '3.11+',
                    }
                ))
            
            # Arbeitsverzeichnis prüfen
            cwd = Path.cwd()
            if not cwd.exists():
                return Err(ConfigurationError(
                    f"Current working directory does not exist: {cwd}",
                    {'cwd': str(cwd)}
                ))
            
            # Schreibrechte prüfen
            try:
                test_file = cwd / '.write_test'
                test_file.touch()
                test_file.unlink()
            except Exception:
                return Err(ConfigurationError(
                    f"No write permissions in current directory: {cwd}",
                    {'cwd': str(cwd)}
                ))
            
            self.logger.info(
                "Environment validation successful",
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
                working_directory=str(cwd),
            )
            
            return Ok(None)
        
        except Exception as e:
            return Err(ConfigurationError(
                f"Environment validation failed: {str(e)}",
                {
                    'error_type': type(e).__name__,
                    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
                }
            ))
    
    def _validate_configuration(self) -> Result[None, ConfigurationError]:
        """Validiere Anwendungskonfiguration"""
        try:
            # Konfiguration laden
            config = get_config()
            
            # Kritische Pfade prüfen
            missing_prompts = []
            for rule_name, rule_config in config.rules.items():
                if rule_config.enabled:
                    prompt_path = Path(rule_config.file)
                    if not prompt_path.exists():
                        missing_prompts.append(rule_config.file)
            
            if missing_prompts:
                return Err(ConfigurationError(
                    f"Missing prompt files: {', '.join(missing_prompts)}",
                    {
                        'missing_files': missing_prompts,
                        'suggestion': 'Create missing prompt files or disable rules',
                    }
                ))
            
            # Gewichtungen validieren
            total_weight = config.get_total_weight()
            if abs(total_weight - 1.0) > 0.1:  # Toleranz für größere Abweichungen
                return Err(ConfigurationError(
                    f"Rule weights sum to {total_weight:.3f}, expected ~1.0",
                    {
                        'total_weight': total_weight,
                        'enabled_rules': config.get_rule_names(),
                        'suggestion': 'Adjust rule weights to sum to 1.0',
                    }
                ))
            
            self.logger.info(
                "Configuration validation successful",
                total_rules=len(config.rules),
                enabled_rules=len(config.get_enabled_rules()),
                total_weight=total_weight,
                threshold=config.scoring.threshold,
            )
            
            return Ok(None)
        
        except Exception as e:
            return Err(ConfigurationError(
                f"Configuration validation failed: {str(e)}",
                {
                    'error_type': type(e).__name__,
                    'suggestion': 'Check config.yaml syntax and content',
                }
            ))
    
    def _initialize_qt_application(self) -> Result[QApplication, ConfigurationError]:
        """Initialisiere Qt-Anwendung"""
        try:
            # QApplication erstellen
            self.app = QApplication(sys.argv)
            
            # Anwendungsmetadaten setzen
            self.app.setApplicationName("YouTube Info Analyzer")
            self.app.setApplicationVersion("0.1.0")
            self.app.setOrganizationName("YouTube Analyzer")
            self.app.setOrganizationDomain("youtube-analyzer.local")

            
            self.logger.info(
                "Qt application initialized",
                app_name=self.app.applicationName(),
                app_version=self.app.applicationVersion(),
                high_dpi_enabled=True,
            )
            
            return Ok(self.app)
        
        except Exception as e:
            return Err(ConfigurationError(
                f"Qt application initialization failed: {str(e)}",
                {
                    'error_type': type(e).__name__,
                    'suggestion': 'Check PySide6 installation and display configuration',
                }
            ))
    
    def _initialize_main_window(self) -> Result[MainWindow, ConfigurationError]:
        """Initialisiere Hauptfenster"""
        try:
            if not self.app:
                return Err(ConfigurationError(
                    "Qt application not initialized",
                    {'suggestion': 'Call _initialize_qt_application() first'}
                ))
            
            # Hauptfenster erstellen
            self.main_window = MainWindow()
            
            # Fenster-Eigenschaften setzen
            self.main_window.setWindowTitle("YouTube Info Analyzer")
            self.main_window.setMinimumSize(800, 600)
            self.main_window.resize(1000, 700)
            
            self.logger.info(
                "Main window initialized",
                window_title=self.main_window.windowTitle(),
                window_size=f"{self.main_window.width()}x{self.main_window.height()}",
            )
            
            return Ok(self.main_window)
        
        except Exception as e:
            return Err(ConfigurationError(
                f"Main window initialization failed: {str(e)}",
                {
                    'error_type': type(e).__name__,
                    'suggestion': 'Check UI components and styling',
                }
            ))
    
    def run(self) -> Result[int, ConfigurationError]:
        """Starte die Anwendung"""
        try:
            self.logger.info("Starting YouTube Analyzer application")
            
            # 1. Umgebung validieren
            env_result = self._validate_environment()
            if isinstance(env_result, Err):
                return Err(env_result.error)
            
            # 2. Konfiguration validieren
            config_result = self._validate_configuration()
            if isinstance(config_result, Err):
                return Err(config_result.error)
            
            # 3. Qt-Anwendung initialisieren
            app_result = self._initialize_qt_application()
            if isinstance(app_result, Err):
                return Err(app_result.error)
            
            app = app_result.value
            
            # 4. Hauptfenster initialisieren
            window_result = self._initialize_main_window()
            if isinstance(window_result, Err):
                return Err(window_result.error)
            
            window = window_result.value
            
            # 5. Fenster anzeigen
            window.show()
            
            self.logger.info(
                "Application started successfully",
                status="running",
                window_visible=window.isVisible(),
            )
            
            # 6. Event Loop starten
            exit_code = app.exec()
            
            self.logger.info(
                "Application exited",
                exit_code=exit_code,
                status="terminated",
            )
            
            return Ok(exit_code)
        
        except Exception as e:
            error_msg = f"Application failed to start: {str(e)}"
            self.logger.error(
                "Application startup failed",
                error=e,
                error_type=type(e).__name__,
            )
            
            return Err(ConfigurationError(
                error_msg,
                {
                    'error_type': type(e).__name__,
                    'suggestion': 'Check logs for detailed error information',
                }
            ))
    
    def cleanup(self) -> None:
        """Cleanup-Operationen beim Beenden"""
        try:
            self.logger.info("Starting application cleanup")
            
            # Main Window cleanup
            if self.main_window:
                self.main_window.cleanup()
                self.main_window.close()
                self.main_window = None
            
            # Qt Application cleanup
            if self.app:
                self.app.quit()
                self.app = None
            
            self.logger.info("Application cleanup completed")
        
        except Exception as e:
            self.logger.error(
                "Application cleanup failed",
                error=e,
                error_type=type(e).__name__,
            )


def create_application(debug: bool = False) -> YouTubeAnalyzerApp:
    """Factory-Funktion für YouTube Analyzer App"""
    return YouTubeAnalyzerApp(debug)


def main() -> NoReturn:
    """Haupteinstiegspunkt der Anwendung"""
    # Debug-Modus aus Argumenten ermitteln
    debug = '--debug' in sys.argv or '-d' in sys.argv
    
    # Anwendung erstellen
    app = create_application(debug)
    
    try:
        # Anwendung starten
        result = app.run()
        
        if isinstance(result, Ok):
            exit_code = result.value
        else:
            # Fehler ausgeben
            error = result.error
            print(f"❌ Application Error: {error.message}", file=sys.stderr)
            
            if error.context:
                print(f"   Context: {error.context}", file=sys.stderr)
            
            exit_code = 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Application interrupted by user", file=sys.stderr)
        exit_code = 130  # SIGINT exit code
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}", file=sys.stderr)
        exit_code = 1
    
    finally:
        # Cleanup
        app.cleanup()
    
    sys.exit(exit_code)


def gui_main() -> NoReturn:
    """GUI-Einstiegspunkt (ohne CLI-Argumente)"""
    app = create_application(debug=False)
    
    try:
        result = app.run()
        
        if isinstance(result, Ok):
            exit_code = result.value
        else:
            # Bei GUI-Anwendung: Fehler in MessageBox anzeigen
            from PySide6.QtWidgets import QMessageBox
            
            error = result.error
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("YouTube Analyzer - Error")
            msg.setText(f"Application Error: {error.message}")
            msg.setDetailedText(f"Context: {error.context}")
            msg.exec()
            
            exit_code = 1
    
    except Exception as e:
        from PySide6.QtWidgets import QMessageBox
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("YouTube Analyzer - Critical Error")
        msg.setText(f"Unexpected error: {str(e)}")
        msg.exec()
        
        exit_code = 1
    
    finally:
        app.cleanup()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
