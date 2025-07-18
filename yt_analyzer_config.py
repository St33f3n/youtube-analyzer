"""
YouTube Analyzer - Configuration System & Rule Analysis
Secure Config with Keyring + Flexible Rule System
"""

from __future__ import annotations
import yaml
import keyring
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import json

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext
from logging_plus import get_logger, log_feature

# =============================================================================
# SECURE SECRET MANAGEMENT
# =============================================================================

class SecretManager:
    """Secure Secret Management via KeePassXC D-Bus"""
    
    def __init__(self):
        self.logger = get_logger("SecretManager")
        self._setup_keepassxc_backend()
    
    def _setup_keepassxc_backend(self) -> None:
        """Konfiguriert KeePassXC als explizites Keyring-Backend"""
        try:
            import keyring.backends.SecretService
            keyring.set_keyring(keyring.backends.SecretService.Keyring())
            backend = keyring.get_keyring()
            if hasattr(backend, 'scheme'):
                backend.scheme = 'KeePassXC'
            self.logger.debug("KeePassXC SecretService backend configured")
        except ImportError:
            self.logger.warning("SecretService backend not available, using default keyring")
        except Exception as e:
            self.logger.warning(f"Failed to set KeePassXC backend: {e}, using default")
    
    def get_secret(self, service_name: str, username: str) -> Result[str, CoreError]:
        """
        Holt Secret aus System-Keyring
        
        Args:
            service_name: Service-Name (z.B. "TrilliumToken")
            username: Username für Service (z.B. "token")
            
        Returns:
            Ok(str): Secret-Wert
            Err: Keyring-Fehler
        """
        try:
            secret = keyring.get_password(service_name, username)
            
            if secret is None:
                context = ErrorContext.create(
                    "get_secret",
                    input_data={'service': service_name, 'username': username},
                    suggestions=[
                        "Check if secret exists in keyring",
                        "Verify KeePassXC is running and unlocked",
                        "Use 'keyring set <service> <username>' to store secret"
                    ]
                )
                return Err(CoreError(f"Secret not found: {service_name}/{username}", context))
            
            self.logger.debug(
                f"Secret retrieved successfully",
                extra={
                    'service': service_name,
                    'username': username,
                    'secret_length': len(secret)
                }
            )
            
            return Ok(secret)
            
        except Exception as e:
            context = ErrorContext.create(
                "get_secret",
                input_data={'service': service_name, 'username': username},
                suggestions=[
                    "Check keyring backend availability",
                    "Verify D-Bus connection to KeePassXC",
                    "Install python-keyring package"
                ]
            )
            return Err(CoreError(f"Keyring access failed: {e}", context))
    
    def test_secret_access(self, service_name: str, username: str) -> Result[bool, CoreError]:
        """Testet ob spezifisches Secret lesbar ist (read-only test)"""
        try:
            secret = keyring.get_password(service_name, username)
            
            if secret is None:
                return Err(CoreError(f"Secret not accessible: {service_name}/{username}"))
            
            self.logger.debug(f"Secret access test successful for {service_name}/{username}")
            return Ok(True)
            
        except Exception as e:
            context = ErrorContext.create(
                "test_secret_access",
                input_data={'service': service_name, 'username': username},
                suggestions=[
                    "Check if KeePassXC is running and unlocked",
                    "Verify secret exists in KeePassXC",
                    "Check D-Bus connection"
                ]
            )
            return Err(CoreError(f"Secret access test failed: {e}", context))

# =============================================================================
# PYDANTIC CONFIG MODELS (Updated)
# =============================================================================

class SecretsConfig(BaseModel):
    """Secret-Service-Konfiguration für Keyring-Zugriff"""
    trilium_service: str = Field(description="Keyring Service-Name für Trilium Token")
    trilium_username: str = Field(description="Keyring Username für Trilium Token")
    nextcloud_service: str = Field(description="Keyring Service-Name für Nextcloud Password")
    nextcloud_username: str = Field(description="Keyring Username für Nextcloud Password")

class WhisperConfig(BaseModel):
    """Whisper-Transkription-Konfiguration"""
    enabled: bool = Field(default=True, description="Whisper-Transkription aktivieren")
    model: str = Field(default="large-v3", description="Whisper Model")
    device: str = Field(default="cuda", description="Device (cuda/cpu)")
    language: Optional[str] = Field(default=None, description="Sprache forcieren (None = auto-detect)")
    compute_type: str = Field(default="float16", description="Compute Type für GPU")

class AnalysisRule(BaseModel):
    """Einzelne Analyse-Regel mit Konfiguration"""
    file: str = Field(description="Pfad zur Prompt-Datei")
    weight: float = Field(ge=0.0, le=1.0, description="Gewichtung der Regel")
    enabled: bool = Field(default=True, description="Regel aktiviert")
    
    def load_prompt_content(self, base_path: Path) -> Result[str, CoreError]:
        """Lädt Prompt-Content aus Datei"""
        try:
            prompt_path = base_path / self.file
            if not prompt_path.exists():
                context = ErrorContext.create(
                    "load_prompt",
                    input_data={'prompt_path': str(prompt_path)},
                    suggestions=["Check file path", "Create missing prompt file"]
                )
                return Err(CoreError(f"Prompt file not found: {prompt_path}", context))
            
            content = prompt_path.read_text(encoding='utf-8')
            return Ok(content)
            
        except Exception as e:
            context = ErrorContext.create(
                "load_prompt",
                input_data={'file': self.file},
                suggestions=["Check file permissions", "Verify file encoding"]
            )
            return Err(CoreError(f"Failed to load prompt: {e}", context))

class RulesConfig(BaseModel):
    """Konfiguration aller Analyse-Regeln"""
    fachinhalt: AnalysisRule
    qualität: AnalysisRule  
    länge_tiefe: AnalysisRule
    relevanz: AnalysisRule
    
    def get_enabled_rules(self) -> Dict[str, AnalysisRule]:
        """Gibt alle aktivierten Regeln zurück"""
        return {
            name: rule for name, rule in self.__dict__.items() 
            if isinstance(rule, AnalysisRule) and rule.enabled
        }
    
    def get_total_weight(self) -> float:
        """Berechnet Gesamt-Gewichtung aller aktivierten Regeln"""
        return sum(rule.weight for rule in self.get_enabled_rules().values())
    
    @validator('fachinhalt', 'qualität', 'länge_tiefe', 'relevanz')
    def validate_weights_sum(cls, v, values):
        """Validiert dass Gewichtungen sinnvoll sind"""
        # Note: Diese Validierung läuft pro Feld, nicht am Ende
        # Für Gesamt-Validierung siehe validate_total_weights unten
        return v

class ScoringConfig(BaseModel):
    """Bewertungs- und Entscheidungs-Konfiguration"""
    threshold: float = Field(ge=0.0, le=1.0, description="Mindest-Score für Video Download")
    min_confidence: float = Field(ge=0.0, le=1.0, description="Mindest-Konfidenz der KI-Bewertung")

class StorageConfig(BaseModel):
    """Speicher-Pfad-Konfiguration"""
    nextcloud_path: str = Field(description="Upload-Pfad in Nextcloud")
    trilium_parent_note: str = Field(description="Parent Note in Trilium für YouTube-Inhalte")
    sqlite_path: str = Field(description="Pfad zur SQLite-Datenbank")

class OllamaConfig(BaseModel):
    """Ollama/LLM-Konfiguration"""
    host: str = Field(default="http://localhost:11434", description="Ollama API Host")
    model: str = Field(default="gemma2", description="LLM Model Name")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model Temperature")
    max_tokens: int = Field(default=4000, gt=0, description="Max Output Tokens")
    timeout: int = Field(default=300, gt=0, description="Request Timeout (seconds)")

class ProcessingConfig(BaseModel):
    """Video-Processing-Konfiguration"""
    temp_folder: Path = Field(default=Path("/tmp/youtube_analyzer"), description="Temp-Ordner für Downloads")
    max_video_length: int = Field(default=7200, gt=0, description="Max Video-Länge in Sekunden")
    audio_format: str = Field(default="mp3", description="Audio-Format")
    video_format: str = Field(default="mp4", description="Video-Format")
    cleanup_temp_files: bool = Field(default=True, description="Temp-Files löschen")

class AppConfig(BaseModel):
    """Haupt-Konfiguration mit Secure Secrets"""
    # Secret Management
    secrets: SecretsConfig
    
    # Processing Components
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    rules: RulesConfig
    scoring: ScoringConfig
    storage: StorageConfig
    
    # Technical Components
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    @validator('rules')
    def validate_total_weights(cls, v):
        """Validiert dass Gesamt-Gewichtung der aktivierten Regeln <= 1.0"""
        total_weight = v.get_total_weight()
        if total_weight > 1.0:
            raise ValueError(f"Total weight of enabled rules ({total_weight}) exceeds 1.0")
        return v

# =============================================================================
# SECURE CONFIG MANAGER
# =============================================================================

class SecureConfigManager:
    """Configuration Manager mit Secure Secret Integration"""
    
    def __init__(self, config_path: Path = Path("config.yaml")):
        self.config_path = config_path
        self.config: Optional[AppConfig] = None
        self.secret_manager = SecretManager()
        self.logger = get_logger("SecureConfigManager")
    
    def load_config(self) -> Result[AppConfig, CoreError]:
        """Lädt Konfiguration mit Secret-Validation"""
        try:
            # Load YAML Config first
            if not self.config_path.exists():
                self._generate_example_config()
                return Err(CoreError(f"Config file created at {self.config_path}. Please edit and add secrets to KeePassXC."))
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Parse Config
            self.config = AppConfig(**config_data)
            
            # Validate Secrets Exist in KeePassXC
            secrets_result = self._validate_secrets()
            if isinstance(secrets_result, Err):
                return secrets_result
            
            self.logger.info(
                f"Secure configuration loaded successfully",
                extra={
                    'config_path': str(self.config_path),
                    'enabled_rules': list(self.config.rules.get_enabled_rules().keys()),
                    'total_rule_weight': self.config.rules.get_total_weight(),
                    'whisper_enabled': self.config.whisper.enabled
                }
            )
            
            return Ok(self.config)
            
        except Exception as e:
            context = ErrorContext.create(
                "load_secure_config",
                input_data={'config_path': str(self.config_path)},
                suggestions=[
                    "Check YAML syntax",
                    "Verify all required sections exist",
                    "Run keyring setup for secrets"
                ]
            )
            return Err(CoreError(f"Failed to load secure config: {e}", context))
    
    def _validate_secrets(self) -> Result[None, CoreError]:
        """Validiert dass alle benötigten Secrets in KeePassXC verfügbar sind"""
        if not self.config:
            return Err(CoreError("Config not loaded"))
        
        # Test Trilium Secret Access
        trilium_result = self.secret_manager.test_secret_access(
            self.config.secrets.trilium_service,
            self.config.secrets.trilium_username
        )
        if isinstance(trilium_result, Err):
            self.logger.error(f"Trilium secret not accessible: {trilium_result.error.message}")
            return trilium_result
        
        # Test Nextcloud Secret Access
        nextcloud_result = self.secret_manager.test_secret_access(
            self.config.secrets.nextcloud_service,
            self.config.secrets.nextcloud_username
        )
        if isinstance(nextcloud_result, Err):
            self.logger.error(f"Nextcloud secret not accessible: {nextcloud_result.error.message}")
            return nextcloud_result
        
        self.logger.info("All secrets accessible in KeePassXC")
        return Ok(None)
    
    def get_trilium_token(self) -> Result[str, CoreError]:
        """Holt Trilium API Token aus Keyring"""
        if not self.config:
            return Err(CoreError("Config not loaded"))
        
        return self.secret_manager.get_secret(
            self.config.secrets.trilium_service,
            self.config.secrets.trilium_username
        )
    
    def get_nextcloud_password(self) -> Result[str, CoreError]:
        """Holt Nextcloud Password aus Keyring"""
        if not self.config:
            return Err(CoreError("Config not loaded"))
        
        return self.secret_manager.get_secret(
            self.config.secrets.nextcloud_service,
            self.config.secrets.nextcloud_username
        )
    
    def _generate_example_config(self) -> None:
        """Generiert Beispiel-Konfiguration"""
        example_config = {
            'secrets': {
                'trilium_service': 'TrilliumToken',
                'trilium_username': 'token',
                'nextcloud_service': 'NextcloudPW',
                'nextcloud_username': 'steefen'
            },
            'whisper': {
                'enabled': True,
                'model': 'large-v3',
                'device': 'cuda',
                'language': None,
                'compute_type': 'float16'
            },
            'rules': {
                'fachinhalt': {
                    'file': 'prompts/fachinhalt.md',
                    'weight': 0.35,
                    'enabled': True
                },
                'qualität': {
                    'file': 'prompts/qualität.md',
                    'weight': 0.25,
                    'enabled': True
                },
                'länge_tiefe': {
                    'file': 'prompts/länge_tiefe.md',
                    'weight': 0.20,
                    'enabled': True
                },
                'relevanz': {
                    'file': 'prompts/relevanz.md',
                    'weight': 0.20,
                    'enabled': True
                }
            },
            'scoring': {
                'threshold': 0.7,
                'min_confidence': 0.6
            },
            'storage': {
                'nextcloud_path': '/YouTube-Archive',
                'trilium_parent_note': 'YouTube Knowledge Base',
                'sqlite_path': 'data/youtube_analyzer.db'
            },
            'ollama': {
                'host': 'http://localhost:11434',
                'model': 'gemma2',
                'temperature': 0.1,
                'max_tokens': 4000,
                'timeout': 300
            },
            'processing': {
                'temp_folder': '/tmp/youtube_analyzer',
                'max_video_length': 7200,
                'audio_format': 'mp3',
                'video_format': 'mp4',
                'cleanup_temp_files': True
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(example_config, f, default_flow_style=False, allow_unicode=True)

# =============================================================================
# ENHANCED RULE SYSTEM  
# =============================================================================

class EnhancedRuleSystem:
    """Rule System mit gewichteter Bewertung und flexibler Konfiguration"""
    
    def __init__(self, config: AppConfig, base_path: Path = Path(".")):
        self.config = config
        self.base_path = base_path
        self.loaded_rules: Dict[str, str] = {}
        self.logger = get_logger("EnhancedRuleSystem")
    
    def load_rule_prompts(self) -> Result[Dict[str, str], CoreError]:
        """Lädt alle aktivierten Regel-Prompts"""
        enabled_rules = self.config.rules.get_enabled_rules()
        loaded_prompts = {}
        
        for rule_name, rule_config in enabled_rules.items():
            prompt_result = rule_config.load_prompt_content(self.base_path)
            if isinstance(prompt_result, Err):
                return prompt_result
            
            loaded_prompts[rule_name] = prompt_result.value
        
        self.loaded_rules = loaded_prompts
        
        self.logger.info(
            f"Loaded {len(loaded_prompts)} rule prompts",
            extra={
                'rules': list(loaded_prompts.keys()),
                'total_weight': self.config.rules.get_total_weight()
            }
        )
        
        return Ok(loaded_prompts)
    
    def generate_analysis_prompt(self, transcript: str) -> Result[str, CoreError]:
        """Generiert gewichteten Analysis-Prompt"""
        if not self.loaded_rules:
            load_result = self.load_rule_prompts()
            if isinstance(load_result, Err):
                return load_result
        
        enabled_rules = self.config.rules.get_enabled_rules()
        rules_section = []
        
        for rule_name, rule_content in self.loaded_rules.items():
            rule_config = enabled_rules[rule_name]
            
            rules_section.append(f"""
## Regel: {rule_name.upper()}
**Gewichtung:** {rule_config.weight} ({rule_config.weight * 100:.1f}%)
**Status:** {'✅ Aktiviert' if rule_config.enabled else '❌ Deaktiviert'}

{rule_content}
""")
        
        prompt = f"""Du bist ein Experte für Content-Analyse. Analysiere den folgenden YouTube-Transkript basierend auf den gewichteten Regeln.

# ANALYSE-REGELN (Gesamt-Gewichtung: {self.config.rules.get_total_weight()})
{''.join(rules_section)}

# TRANSKRIPT ZU ANALYSIEREN
{transcript}

# AUFGABE
Analysiere den Transkript systematisch anhand jeder aktivierten Regel:

1. **Regel-Bewertung** (pro Regel):
   - Erfüllung: true/false
   - Konfidenz: 0.0-1.0 (wie sicher bist du?)
   - Score: 0.0-1.0 (Qualität der Erfüllung)
   - Begründung: Kurze Erklärung

2. **Gesamt-Bewertung**:
   - Gewichteter Gesamtscore (basierend auf Regel-Gewichtungen)
   - Gesamtkonfidenz (Durchschnitt aller Regel-Konfidenzen)
   - Empfehlung: DOWNLOAD oder SKIP

# OUTPUT-FORMAT
Antworte ausschließlich mit folgendem JSON-Format:

{{
    "rules_analysis": [
        {{
            "rule_name": "{rule_name}",
            "fulfilled": true,
            "confidence": 0.85,
            "score": 0.90,
            "reasoning": "Begründung..."
        }}
    ],
    "overall_score": 0.82,
    "overall_confidence": 0.88,
    "recommendation": "DOWNLOAD",
    "summary": "Kurze Zusammenfassung der Analyse"
}}

# SCORING-KRITERIEN
- Threshold für DOWNLOAD: {self.config.scoring.threshold}
- Minimum Confidence: {self.config.scoring.min_confidence}
- Gewichtete Berechnung: Σ(regel_score × regel_weight)
"""
        
        return Ok(prompt)

# =============================================================================
# BEISPIEL-SETUP & USAGE
# =============================================================================

def generate_example_prompts(prompts_dir: Path) -> None:
    """Generiert Beispiel-Prompt-Dateien"""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # Fachinhalt Prompt
    (prompts_dir / "fachinhalt.md").write_text("""
# Fachinhalt-Bewertung

Bewerte die **fachliche Korrektheit und Tiefe** des Inhalts:

## Kriterien:
1. **Technische Korrektheit**: Sind die Fakten richtig?
2. **Fachliche Tiefe**: Wird das Thema umfassend behandelt?
3. **Aktualität**: Sind die Informationen aktuell?
4. **Quellenqualität**: Werden seriöse Quellen referenziert?

## Bewertung:
- **1.0**: Exzellenter Fachinhalt, keine Fehler erkennbar
- **0.8**: Guter Fachinhalt mit kleinen Ungenauigkeiten
- **0.6**: Okay, aber oberflächlich oder einzelne Fehler
- **0.4**: Schwache fachliche Basis
- **0.2**: Viele Fehler oder sehr oberflächlich
- **0.0**: Fachlich inkorrekt oder irreführend
""", encoding='utf-8')
    
    # Qualität Prompt
    (prompts_dir / "qualität.md").write_text("""
# Qualitäts-Bewertung

Bewerte die **Produktions- und Inhaltsqualität**:

## Kriterien:
1. **Struktur**: Klarer Aufbau und logische Gliederung
2. **Verständlichkeit**: Gut erklärt und nachvollziehbar
3. **Audio-Qualität**: Deutliche Sprache, wenig Störgeräusche
4. **Professionalität**: Vorbereitung und Präsentation

## Bewertung:
- **1.0**: Professionelle, hochwertige Produktion
- **0.8**: Gute Qualität mit kleinen Schwächen
- **0.6**: Durchschnittliche Qualität
- **0.4**: Niedrige Qualität, aber verwendbar
- **0.2**: Schlechte Qualität
- **0.0**: Unbrauchbar schlecht
""", encoding='utf-8')
    
    # Länge/Tiefe Prompt
    (prompts_dir / "länge_tiefe.md").write_text("""
# Länge/Tiefe-Bewertung

Bewerte das **Verhältnis von Videolänge zu Informationsdichte**:

## Kriterien:
1. **Informationsdichte**: Viel Inhalt pro Zeiteinheit
2. **Redundanz**: Wenig Wiederholungen oder Füllwörter
3. **Fokus**: Bleibt beim Thema, wenig Abschweifen
4. **Effizienz**: Kompakte, präzise Vermittlung

## Bewertung:
- **1.0**: Perfekte Balance, sehr informationsdicht
- **0.8**: Gute Informationsdichte
- **0.6**: Akzeptables Verhältnis
- **0.4**: Etwas langatmig oder oberflächlich
- **0.2**: Schlecht strukturiert, viel Füllmaterial
- **0.0**: Zeitverschwendung, kaum Inhalt
""", encoding='utf-8')
    
    # Relevanz Prompt
    (prompts_dir / "relevanz.md").write_text("""
# Relevanz-Bewertung

Bewerte die **Relevanz und den Nutzen** des Inhalts:

## Kriterien:
1. **Praktischer Nutzen**: Anwendbare Erkenntnisse
2. **Zeitlose Relevanz**: Nicht nur kurzfristig interessant
3. **Zielgruppen-Fit**: Passt zu meinen Interessen
4. **Lernwert**: Erweitert Wissen oder Fähigkeiten

## Bewertung:
- **1.0**: Extrem relevant und nützlich
- **0.8**: Sehr relevant
- **0.6**: Interessant und brauchbar
- **0.4**: Begrenzte Relevanz
- **0.2**: Kaum relevant
- **0.0**: Irrelevant oder unnütz
""", encoding='utf-8')

if __name__ == "__main__":
    from logging_plus import setup_logging
    
    # Setup
    setup_logging("secure_config_demo", "DEBUG")
    
    # Generiere Beispiel-Config
    config_manager = SecureConfigManager(Path("secure_config.yaml"))
    
    # Generiere Beispiel-Prompts
    prompts_dir = Path("prompts")
    generate_example_prompts(prompts_dir)
    
    print("🚀 Secure Configuration System Setup Complete!")
    print("==============================================")
    print("Generated files:")
    print("- secure_config.yaml (example configuration)")
    print("- prompts/*.md (example rule prompts)")
    print("")
    print("Next steps:")
    print("1. Edit secure_config.yaml with your settings")
    print("2. Add secrets manually to KeePassXC:")
    print("   - Service: 'TrilliumToken', Username: 'token'")
    print("   - Service: 'NextcloudPW', Username: 'steefen'")
    print("3. Customize prompts in prompts/ directory")
    print("4. Ensure KeePassXC is running and unlocked")
