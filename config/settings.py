"""
Configuration System - YAML Loading & Pydantic Models
Vollständig überarbeitet mit Result-Types und vollständigen Type-Hints
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import yaml
from pydantic import BaseModel
from pydantic import Field
from pydantic import root_validator
from pydantic import validator

from yt_types import ConfigurationError
from yt_types import Err
from yt_types import Ok
from yt_types import Result
from yt_types import ValidationError
from utils.logging import ComponentLogger
from utils.logging import log_function_calls


class SecretConfig(BaseModel):
    """Secret Service Konfiguration für KeePassXC"""
    
    trilium_service: str = Field(
        ...,
        description="Service Name für Trilium API Key in KeePassXC",
        min_length=1,
    )
    trilium_username: str = Field(
        ...,
        description="Username für Trilium Secret in KeePassXC",
        min_length=1,
    )
    nextcloud_service: str = Field(
        ...,
        description="Service Name für NextCloud Credentials in KeePassXC",
        min_length=1,
    )
    nextcloud_username: str = Field(
        ...,
        description="Username für NextCloud Secret in KeePassXC",
        min_length=1,
    )
    
    class Config:
        frozen = True


class WhisperConfig(BaseModel):
    """Whisper Service Konfiguration"""
    
    enabled: bool = Field(
        default=True,
        description="Whisper-Transkription aktivieren",
    )
    model_name: str = Field(
        default="large-v3",
        description="Whisper Model Name",
    )
    device: Optional[str] = Field(
        default=None,
        description="Device für Whisper (auto, cuda, cpu)",
    )
    
    @validator('device')
    def validate_device(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ['auto', 'cuda', 'cpu']:
            raise ValueError("Device must be 'auto', 'cuda', or 'cpu'")
        return v
    
    class Config:
        frozen = True


class OllamaConfig(BaseModel):
    """Ollama Service Konfiguration"""
    
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama Server URL",
    )
    model_name: str = Field(
        default="gemma2:2b",
        description="Ollama Model Name",
    )
    timeout: int = Field(
        default=120,
        description="Request Timeout in Sekunden",
        ge=30,
        le=300,
    )
    
    @validator('base_url')
    def validate_base_url(cls, v: str) -> str:
        if not v.startswith('http'):
            raise ValueError("Base URL must start with http:// or https://")
        return v
    
    class Config:
        frozen = True


class RuleConfig(BaseModel):
    """Einzelne Analyse-Regel Konfiguration"""
    
    file: str = Field(
        ...,
        description="Pfad zur .md Prompt-Datei",
        min_length=1,
    )
    weight: float = Field(
        ...,
        description="Gewichtung für finale Score-Berechnung",
        ge=0.0,
        le=1.0,
    )
    enabled: bool = Field(
        default=True,
        description="Regel aktiviert",
    )
    
    @validator('file')
    def validate_file_path(cls, v: str) -> str:
        """Validiere Prompt-Datei-Pfad"""
        path = Path(v)
        
        if not path.suffix == '.md':
            raise ValueError(f"Prompt file must have .md extension: {v}")
        
        # Warnung wenn Datei nicht existiert (nicht fatal)
        if not path.exists():
            import warnings
            warnings.warn(f"Prompt file not found: {v}")
        
        return v
    
    class Config:
        frozen = True


class ScoringConfig(BaseModel):
    """Scoring System Konfiguration"""
    
    threshold: float = Field(
        ...,
        description="Mindest-Score für Video Download",
        ge=0.0,
        le=1.0,
    )
    min_confidence: float = Field(
        default=0.6,
        description="Mindest-Konfidenz für Analyse-Entscheidung",
        ge=0.0,
        le=1.0,
    )
    
    @validator('threshold', 'min_confidence')
    def validate_score_range(cls, v: float) -> float:
        """Validiere Score-Bereiche"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score values must be between 0.0 and 1.0")
        return v
    
    class Config:
        frozen = True


class StorageConfig(BaseModel):
    """Storage Services Konfiguration"""
    
    nextcloud_path: str = Field(
        default="/YouTube-Archive",
        description="NextCloud Upload-Pfad",
    )
    trilium_parent_note: str = Field(
        default="YouTube Knowledge Base",
        description="Trilium Parent Note ID oder Title",
    )
    sqlite_path: str = Field(
        default="data/youtube_analyzer.db",
        description="SQLite Datenbank Pfad",
    )
    
    @validator('nextcloud_path')
    def validate_nextcloud_path(cls, v: str) -> str:
        if not v.startswith('/'):
            raise ValueError("NextCloud path must start with '/'")
        return v
    
    @validator('sqlite_path')
    def validate_sqlite_path(cls, v: str) -> str:
        path = Path(v)
        
        # Stelle sicher dass Verzeichnis existiert
        path.parent.mkdir(parents=True, exist_ok=True)
        
        return v
    
    class Config:
        frozen = True


class AppConfig(BaseModel):
    """Haupt-Anwendungskonfiguration"""
    
    secrets: SecretConfig = Field(
        ...,
        description="Secret Management Konfiguration",
    )
    whisper: WhisperConfig = Field(
        default_factory=WhisperConfig,
        description="Whisper Service Konfiguration",
    )
    ollama: OllamaConfig = Field(
        default_factory=OllamaConfig,
        description="Ollama Service Konfiguration",
    )
    rules: Dict[str, RuleConfig] = Field(
        ...,
        description="Analyse-Regeln Definitionen",
    )
    scoring: ScoringConfig = Field(
        ...,
        description="Scoring System Konfiguration",
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage Services Konfiguration",
    )
    
    @root_validator
    def validate_rules_weights(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validiere Regel-Gewichtungen"""
        rules = values.get('rules', {})
        
        if not rules:
            raise ValueError("At least one rule must be defined")
        
        # Berechne Gesamt-Gewichtung aktivierter Regeln
        enabled_rules = {name: rule for name, rule in rules.items() if rule.enabled}
        
        if not enabled_rules:
            raise ValueError("At least one rule must be enabled")
        
        total_weight = sum(rule.weight for rule in enabled_rules.values())
        
        # Warnung bei unausgewogenen Gewichtungen
        if abs(total_weight - 1.0) > 0.01:
            import warnings
            warnings.warn(
                f"Rule weights sum to {total_weight:.3f}, not 1.0. "
                f"This may affect score interpretation."
            )
        
        return values
    
    def get_enabled_rules(self) -> Dict[str, RuleConfig]:
        """Hole nur aktivierte Regeln"""
        return {name: rule for name, rule in self.rules.items() if rule.enabled}
    
    def get_total_weight(self) -> float:
        """Berechne Gesamt-Gewichtung aktivierter Regeln"""
        return sum(rule.weight for rule in self.get_enabled_rules().values())
    
    def get_rule_names(self) -> list[str]:
        """Hole Namen aller aktivierten Regeln"""
        return list(self.get_enabled_rules().keys())
    
    class Config:
        frozen = True


class ConfigLoader:
    """YAML Configuration Loader mit Validierung"""
    
    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path or self._find_config_file()
        self.logger = ComponentLogger("ConfigLoader")
        self._config: Optional[AppConfig] = None
    
    def _find_config_file(self) -> Path:
        """Suche config.yaml in verschiedenen Pfaden"""
        search_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd() / "youtube_analyzer" / "config.yaml",
            Path.home() / ".config" / "youtube_analyzer" / "config.yaml",
            Path("/etc/youtube_analyzer/config.yaml"),
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        # Fallback: Aktuelles Verzeichnis
        return Path.cwd() / "config.yaml"
    
    @log_function_calls
    def load_config(self) -> Result[AppConfig, ConfigurationError]:
        """Lade und validiere Konfiguration"""
        try:
            if not self.config_path.exists():
                return Err(ConfigurationError(
                    f"Configuration file not found: {self.config_path}",
                    {
                        'config_path': str(self.config_path),
                        'search_paths': [str(p) for p in [
                            Path.cwd() / "config.yaml",
                            Path.home() / ".config" / "youtube_analyzer" / "config.yaml",
                        ]],
                    }
                ))
            
            # YAML laden
            with open(self.config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            if not yaml_data:
                return Err(ConfigurationError(
                    f"Empty or invalid YAML file: {self.config_path}",
                    {'config_path': str(self.config_path)}
                ))
            
            # Pydantic Validierung
            self._config = AppConfig(**yaml_data)
            
            self.logger.info(
                "Configuration loaded successfully",
                config_path=str(self.config_path),
                total_rules=len(self._config.rules),
                enabled_rules=len(self._config.get_enabled_rules()),
                total_weight=self._config.get_total_weight(),
            )
            
            return Ok(self._config)
        
        except yaml.YAMLError as e:
            error_msg = f"YAML parsing error: {str(e)}"
            self.logger.error(
                "YAML parsing failed",
                error=e,
                config_path=str(self.config_path),
            )
            
            return Err(ConfigurationError(
                error_msg,
                {
                    'config_path': str(self.config_path),
                    'error_type': 'yaml_error',
                }
            ))
        
        except Exception as e:
            error_msg = f"Configuration loading failed: {str(e)}"
            self.logger.error(
                "Configuration loading failed",
                error=e,
                config_path=str(self.config_path),
            )
            
            return Err(ConfigurationError(
                error_msg,
                {
                    'config_path': str(self.config_path),
                    'error_type': type(e).__name__,
                }
            ))
    
    @log_function_calls
    def get_config(self) -> Result[AppConfig, ConfigurationError]:
        """Hole Konfiguration (mit Lazy Loading)"""
        if self._config is None:
            return self.load_config()
        
        return Ok(self._config)
    
    @log_function_calls
    def reload_config(self) -> Result[AppConfig, ConfigurationError]:
        """Konfiguration neu laden"""
        self.logger.info(
            "Reloading configuration",
            config_path=str(self.config_path),
        )
        
        self._config = None
        return self.load_config()
    
    @log_function_calls
    def validate_prompt_files(self) -> Result[Dict[str, bool], ConfigurationError]:
        """Validiere alle Prompt-Dateien"""
        config_result = self.get_config()
        if isinstance(config_result, Err):
            return Err(config_result.error)
        
        config = config_result.value
        results = {}
        
        for rule_name, rule_config in config.rules.items():
            prompt_path = Path(rule_config.file)
            results[rule_name] = prompt_path.exists() and prompt_path.is_file()
        
        self.logger.info(
            "Prompt file validation completed",
            total_files=len(results),
            existing_files=sum(results.values()),
            missing_files=len(results) - sum(results.values()),
        )
        
        return Ok(results)
    
    def get_config_info(self) -> Dict[str, Any]:
        """Hole Konfigurationsinformationen für Monitoring"""
        try:
            config_result = self.get_config()
            if isinstance(config_result, Err):
                return {
                    'status': 'error',
                    'error': config_result.error.message,
                    'config_path': str(self.config_path),
                }
            
            config = config_result.value
            
            return {
                'status': 'loaded',
                'config_path': str(self.config_path),
                'total_rules': len(config.rules),
                'enabled_rules': len(config.get_enabled_rules()),
                'rule_names': config.get_rule_names(),
                'total_weight': config.get_total_weight(),
                'threshold': config.scoring.threshold,
                'min_confidence': config.scoring.min_confidence,
                'whisper_enabled': config.whisper.enabled,
                'ollama_model': config.ollama.model_name,
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'config_path': str(self.config_path),
            }


# =============================================================================
# ENVIRONMENT SUPPORT
# =============================================================================

def load_from_environment() -> Dict[str, Any]:
    """Lade Konfiguration aus Umgebungsvariablen"""
    env_config = {}
    
    # Ollama Service
    if ollama_url := os.getenv('OLLAMA_URL'):
        env_config.setdefault('ollama', {})['base_url'] = ollama_url
    
    if ollama_model := os.getenv('OLLAMA_MODEL'):
        env_config.setdefault('ollama', {})['model_name'] = ollama_model
    
    # Whisper Service
    if whisper_device := os.getenv('WHISPER_DEVICE'):
        env_config.setdefault('whisper', {})['device'] = whisper_device
    
    if whisper_model := os.getenv('WHISPER_MODEL'):
        env_config.setdefault('whisper', {})['model_name'] = whisper_model
    
    # Scoring
    if threshold := os.getenv('ANALYSIS_THRESHOLD'):
        try:
            env_config.setdefault('scoring', {})['threshold'] = float(threshold)
        except ValueError:
            pass
    
    # Storage
    if nextcloud_path := os.getenv('NEXTCLOUD_PATH'):
        env_config.setdefault('storage', {})['nextcloud_path'] = nextcloud_path
    
    if sqlite_path := os.getenv('SQLITE_PATH'):
        env_config.setdefault('storage', {})['sqlite_path'] = sqlite_path
    
    return env_config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge zwei Konfigurationen"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


# =============================================================================
# SINGLETON FACTORY
# =============================================================================

_config_loader_instance: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Singleton Factory für ConfigLoader"""
    global _config_loader_instance
    
    if _config_loader_instance is None:
        _config_loader_instance = ConfigLoader()
    
    return _config_loader_instance


def get_config() -> AppConfig:
    """Convenience-Funktion für Konfigurationszugriff"""
    config_loader = get_config_loader()
    config_result = config_loader.get_config()
    
    if isinstance(config_result, Err):
        raise config_result.error
    
    return config_result.value


def reload_config() -> AppConfig:
    """Convenience-Funktion für Konfiguration neu laden"""
    config_loader = get_config_loader()
    config_result = config_loader.reload_config()
    
    if isinstance(config_result, Err):
        raise config_result.error
    
    return config_result.value


def create_config_loader(config_path: Path) -> ConfigLoader:
    """Factory für ConfigLoader mit spezifischem Pfad"""
    return ConfigLoader(config_path)


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def test_config_system() -> None:
    """Test-Funktion für Configuration System"""
    from youtube_analyzer.utils.logging import get_development_config
    from youtube_analyzer.utils.logging import setup_logging
    
    # Setup logging für Test
    setup_logging(get_development_config())
    
    loader = get_config_loader()
    logger = ComponentLogger("ConfigSystemTest")
    
    logger.info("Starting configuration system test")
    
    # Test Config Loading
    config_result = loader.get_config()
    
    if isinstance(config_result, Ok):
        config = config_result.value
        logger.info(
            "✅ Config loading test passed",
            total_rules=len(config.rules),
            enabled_rules=len(config.get_enabled_rules()),
            total_weight=config.get_total_weight(),
        )
    else:
        logger.error(
            "❌ Config loading test failed",
            error=config_result.error.message,
        )
        return
    
    # Test Prompt File Validation
    validation_result = loader.validate_prompt_files()
    
    if isinstance(validation_result, Ok):
        validation = validation_result.value
        existing_files = sum(validation.values())
        total_files = len(validation)
        
        logger.info(
            "✅ Prompt validation test passed",
            existing_files=existing_files,
            total_files=total_files,
            missing_files=total_files - existing_files,
        )
    else:
        logger.error(
            "❌ Prompt validation test failed",
            error=validation_result.error.message,
        )
    
    # Test Config Info
    config_info = loader.get_config_info()
    logger.info(
        "Config info",
        status=config_info['status'],
        config_path=config_info['config_path'],
        details=config_info,
    )
    
    logger.info("✅ Configuration system test completed")


if __name__ == "__main__":
    test_config_system()
