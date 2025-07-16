"""
Config System - YAML Loading & Pydantic Models
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field, validator
from loguru import logger


class SecretConfig(BaseModel):
    """Secret Service Konfiguration"""
    trilium_service: str = Field(..., description="Service Name für Trilium API Key")
    trilium_username: str = Field(..., description="Username für Trilium Secret")
    nextcloud_service: str = Field(..., description="Service Name für NextCloud Credentials")
    nextcloud_username: str = Field(..., description="Username für NextCloud Secret")


class RuleConfig(BaseModel):
    """Einzelne Regel Konfiguration"""
    file: str = Field(..., description="Pfad zur .md Prompt-Datei")
    weight: float = Field(..., ge=0.0, le=1.0, description="Gewichtung (0.0-1.0)")
    enabled: bool = Field(default=True, description="Regel aktiviert")
    
    @validator('file')
    def validate_file_path(cls, v):
        """Validiere dass Prompt-Datei existiert"""
        path = Path(v)
        if not path.exists():
            logger.warning(f"Prompt-Datei nicht gefunden: {v}")
        if not path.suffix == '.md':
            raise ValueError(f"Prompt-Datei muss .md Endung haben: {v}")
        return v


class ScoringConfig(BaseModel):
    """Scoring System Konfiguration"""
    threshold: float = Field(..., ge=0.0, le=1.0, description="Mindest-Score für Video Download")
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Mindest-Konfidenz")
    
    @validator('threshold', 'min_confidence')
    def validate_scores(cls, v):
        """Validiere Score-Bereiche"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Scores müssen zwischen 0.0 und 1.0 liegen")
        return v


class StorageConfig(BaseModel):
    """Storage Konfiguration"""
    nextcloud_path: str = Field(default="/videos", description="NextCloud Upload-Pfad")
    trilium_parent_note: str = Field(default="YouTube Archive", description="Trilium Parent Note")
    sqlite_path: str = Field(default="data/youtube_analyzer.db", description="SQLite Datenbank Pfad")


class AppConfig(BaseModel):
    """Haupt-Anwendungskonfiguration"""
    secrets: SecretConfig
    rules: Dict[str, RuleConfig] = Field(..., description="Regel-Definitionen")
    scoring: ScoringConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    @validator('rules')
    def validate_rules_weights(cls, v):
        """Validiere dass Gewichtungen sinnvoll sind"""
        if not v:
            raise ValueError("Mindestens eine Regel muss definiert sein")
            
        total_weight = sum(rule.weight for rule in v.values() if rule.enabled)
        if total_weight == 0:
            raise ValueError("Mindestens eine Regel muss aktiviert sein")
            
        if abs(total_weight - 1.0) > 0.01:  # Toleranz für Float-Arithmetik
            logger.warning(f"Gesamt-Gewichtung ist {total_weight:.3f}, nicht 1.0")
            
        return v


class ConfigLoader:
    """YAML Config Loader mit Validierung"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Optional[AppConfig] = None
        
    def load_config(self) -> AppConfig:
        """Lade und validiere Konfiguration"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config-Datei nicht gefunden: {self.config_path}")
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
                
            # Pydantic Validierung
            self._config = AppConfig(**yaml_data)
            
            logger.info(f"Konfiguration erfolgreich geladen: {self.config_path}")
            logger.info(f"Regeln geladen: {len(self._config.rules)}")
            logger.info(f"Aktive Regeln: {sum(1 for r in self._config.rules.values() if r.enabled)}")
            
            return self._config
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Config: {e}")
            raise
            
    def get_config(self) -> AppConfig:
        """Hole geladene Konfiguration (lazy loading)"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
        
    def reload_config(self) -> AppConfig:
        """Konfiguration neu laden"""
        logger.info("Lade Konfiguration neu...")
        self._config = None
        return self.load_config()
        
    def validate_prompt_files(self) -> Dict[str, bool]:
        """Validiere alle Prompt-Dateien"""
        results = {}
        if self._config:
            for name, rule in self._config.rules.items():
                path = Path(rule.file)
                results[name] = path.exists() and path.is_file()
        return results
        
    def get_total_weight(self) -> float:
        """Berechne Gesamt-Gewichtung aktiver Regeln"""
        if self._config:
            return sum(rule.weight for rule in self._config.rules.values() if rule.enabled)
        return 0.0


# Globale Config-Instanz
config_loader = ConfigLoader()


def get_config() -> AppConfig:
    """Convenience-Funktion für Config-Zugriff"""
    return config_loader.get_config()


def reload_config() -> AppConfig:
    """Convenience-Funktion für Config-Reload"""
    return config_loader.reload_config()
