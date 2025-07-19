"""
YouTube Analyzer - Enhanced Configuration System
Erweitert um LLM Processing und Trilium Upload für Fork-Join Architecture
"""

from __future__ import annotations
import yaml
import keyring
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import json

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext
from logging_plus import get_logger, log_feature

# =============================================================================
# SECURE SECRET MANAGEMENT (Enhanced)
# =============================================================================

class SecretManager:
    """Secure Secret Management via KeePassXC D-Bus - Enhanced for LLM APIs"""
    
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
            service_name: Service-Name (z.B. "OpenAI_API_Key")
            username: Username für Service (z.B. "api_key")
            
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
# ENHANCED PYDANTIC CONFIG MODELS
# =============================================================================

class SecretsConfig(BaseModel):
    """Enhanced Secret-Service-Konfiguration für alle APIs"""
    # Existing secrets
    trilium_service: str = Field(description="Keyring Service-Name für Trilium Token")
    trilium_username: str = Field(description="Keyring Username für Trilium Token")
    nextcloud_service: str = Field(description="Keyring Service-Name für Nextcloud Password")
    nextcloud_username: str = Field(description="Keyring Username für Nextcloud Password")
    
    # NEW: LLM API Secrets
    openai_service: str = Field(description="Keyring Service-Name für OpenAI API Key")
    openai_username: str = Field(description="Keyring Username für OpenAI API Key")
    anthropic_service: str = Field(description="Keyring Service-Name für Anthropic API Key")
    anthropic_username: str = Field(description="Keyring Username für Anthropic API Key")
    google_service: str = Field(description="Keyring Service-Name für Google API Key")
    google_username: str = Field(description="Keyring Username für Google API Key")

class WhisperConfig(BaseModel):
    """Whisper-Transkription-Konfiguration (unchanged)"""
    enabled: bool = Field(default=True, description="Whisper-Transkription aktivieren")
    model: str = Field(default="large-v3", description="Whisper Model")
    device: str = Field(default="cuda", description="Device (cuda/cpu)")
    language: Optional[str] = Field(default=None, description="Sprache forcieren (None = auto-detect)")
    compute_type: str = Field(default="float16", description="Compute Type für GPU")

class AnalysisRule(BaseModel):
    """Einzelne Analyse-Regel mit Konfiguration (unchanged)"""
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
    """Konfiguration aller Analyse-Regeln (unchanged)"""
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
        return v

class ScoringConfig(BaseModel):
    """Bewertungs- und Entscheidungs-Konfiguration (unchanged)"""
    threshold: float = Field(ge=0.0, le=1.0, description="Mindest-Score für Video Download")
    min_confidence: float = Field(ge=0.0, le=1.0, description="Mindest-Konfidenz der KI-Bewertung")

class StorageConfig(BaseModel):
    """Speicher-Pfad-Konfiguration (unchanged)"""
    nextcloud_base_url: str = Field(description="Full WebDAV base URL für Nextcloud")
    nextcloud_path: str = Field(description="Upload-Pfad in Nextcloud")
    trilium_parent_note: str = Field(description="Parent Note in Trilium für YouTube-Inhalte")
    sqlite_path: str = Field(description="Pfad zur SQLite-Datenbank")

class OllamaConfig(BaseModel):
    """Ollama/LLM-Konfiguration (unchanged)"""
    host: str = Field(default="http://localhost:11434", description="Ollama API Host")
    model: str = Field(default="gemma2", description="LLM Model Name")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model Temperature")
    max_tokens: int = Field(default=4000, gt=0, description="Max Output Tokens")
    timeout: int = Field(default=300, gt=0, description="Request Timeout (seconds)")

class ProcessingConfig(BaseModel):
    """Video-Processing-Konfiguration (unchanged)"""
    temp_folder: Path = Field(default=Path("/tmp/youtube_analyzer"), description="Temp-Ordner für Downloads")
    max_video_length: int = Field(default=7200, gt=0, description="Max Video-Länge in Sekunden")
    audio_format: str = Field(default="mp3", description="Audio-Format")
    video_format: str = Field(default="mp4", description="Video-Format")
    cleanup_temp_files: bool = Field(default=True, description="Temp-Files löschen")

# =============================================================================
# NEW: LLM PROCESSING CONFIGURATION
# =============================================================================

class LLMProcessingConfig(BaseModel):
    """Konfiguration für LLM-basierte Transkript-Verarbeitung"""
    provider: Literal["openai", "anthropic", "google"] = Field(
        default="openai", 
        description="LLM Provider auswählen"
    )
    model: str = Field(
        default="gpt-4", 
        description="Provider-spezifisches Model (gpt-4, claude-3-sonnet, gemini-pro)"
    )
    system_prompt_file: str = Field(
        default="prompts/transcript_processing.md",
        description="Pfad zur System-Prompt-Datei"
    )
    temperature: float = Field(
        default=0.1, 
        ge=0.0, 
        le=2.0, 
        description="Model Temperature für Konsistenz"
    )
    max_tokens: int = Field(
        default=4000, 
        gt=0, 
        description="Maximum Output Tokens"
    )
    retry_attempts: int = Field(
        default=5, 
        gt=0, 
        description="Anzahl Retry-Versuche bei API-Fehlern"
    )
    retry_delay: int = Field(
        default=5, 
        gt=0, 
        description="Sekunden zwischen Retry-Versuchen"
    )
    timeout: int = Field(
        default=300, 
        gt=0, 
        description="Request Timeout in Sekunden"
    )
    
    def get_provider_model_mapping(self) -> Dict[str, List[str]]:
        """Mapping verfügbarer Models pro Provider"""
        return {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
            "google": ["gemini-pro", "gemini-pro-vision"]
        }
    
    def validate_provider_model_combination(self) -> Result[None, CoreError]:
        """Validiert ob Model für Provider verfügbar ist"""
        mapping = self.get_provider_model_mapping()
        available_models = mapping.get(self.provider, [])
        
        if self.model not in available_models:
            context = ErrorContext.create(
                "validate_provider_model",
                input_data={'provider': self.provider, 'model': self.model},
                suggestions=[
                    f"Available models for {self.provider}: {', '.join(available_models)}",
                    "Update model in config.yaml",
                    "Or change provider"
                ]
            )
            return Err(CoreError(f"Model '{self.model}' not available for provider '{self.provider}'", context))
        
        return Ok(None)

# =============================================================================
# NEW: TRILIUM UPLOAD CONFIGURATION
# =============================================================================

class TrilliumUploadConfig(BaseModel):
    """Konfiguration für Trilium-Upload von bearbeiteten Transkripten"""
    enabled: bool = Field(default=True, description="Trilium Upload aktivieren")
    base_url: str = Field(
        default="https://trilium.example.com", 
        description="Trilium Base URL"
    )
    parent_note_id: str = Field(
        default="YouTube Transcripts", 
        description="Parent Note ID oder Titel in Trilium"
    )
    note_template: str = Field(
        default="transcript", 
        description="Template für Note-Erstellung"
    )
    auto_tags: List[str] = Field(
        default_factory=lambda: ["youtube", "transcript", "llm-processed"],
        description="Automatische Tags für alle Notes"
    )
    include_metadata: bool = Field(
        default=True, 
        description="LLM-Metadaten in Note einbetten"
    )
    timeout: int = Field(
        default=60, 
        gt=0, 
        description="Upload Timeout in Sekunden"
    )

# =============================================================================
# ENHANCED MAIN APP CONFIG
# =============================================================================

class AppConfig(BaseModel):
    """Haupt-Konfiguration mit Fork-Join-Features"""
    # Secret Management (Enhanced)
    secrets: SecretsConfig
    
    # Processing Components (Enhanced)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    rules: RulesConfig
    scoring: ScoringConfig
    storage: StorageConfig
    
    # Technical Components (Enhanced)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    # NEW: Fork-Join Components
    llm_processing: LLMProcessingConfig = Field(default_factory=LLMProcessingConfig)
    trilium_upload: TrilliumUploadConfig = Field(default_factory=TrilliumUploadConfig)
    
    @validator('rules')
    def validate_total_weights(cls, v):
        """Validiert dass Gesamt-Gewichtung der aktivierten Regeln <= 1.0"""
        total_weight = v.get_total_weight()
        if total_weight > 1.0:
            raise ValueError(f"Total weight of enabled rules ({total_weight}) exceeds 1.0")
        return v
    
    @validator('llm_processing')
    def validate_llm_config(cls, v):
        """Validiert LLM-Provider-Model-Kombination"""
        validation_result = v.validate_provider_model_combination()
        if isinstance(validation_result, Err):
            raise ValueError(f"LLM config validation failed: {validation_result.error.message}")
        return v

# =============================================================================
# SECURE CONFIG MANAGER (Enhanced)
# =============================================================================

class SecureConfigManager:
    """Configuration Manager mit Enhanced Secret Integration für Fork-Join"""
    
    def __init__(self, config_path: Path = Path("config.yaml")):
        self.config_path = config_path
        self.config: Optional[AppConfig] = None
        self.secret_manager = SecretManager()
        self.logger = get_logger("SecureConfigManager")
    
    def load_config(self) -> Result[AppConfig, CoreError]:
        """Lädt Konfiguration mit Enhanced Secret-Validation"""
        try:
            # Load YAML Config first
            if not self.config_path.exists():
                self._generate_example_config()
                return Err(CoreError(f"Config file created at {self.config_path}. Please edit and add secrets to KeePassXC."))
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Parse Config with Pydantic validation
            self.config = AppConfig(**config_data)
            
            # Validate all secrets exist in KeePassXC
            secrets_result = self._validate_all_secrets()
            if isinstance(secrets_result, Err):
                return secrets_result
            
            self.logger.info(
                f"Enhanced configuration loaded successfully",
                extra={
                    'config_path': str(self.config_path),
                    'enabled_rules': list(self.config.rules.get_enabled_rules().keys()),
                    'total_rule_weight': self.config.rules.get_total_weight(),
                    'whisper_enabled': self.config.whisper.enabled,
                    'llm_provider': self.config.llm_processing.provider,
                    'llm_model': self.config.llm_processing.model,
                    'trilium_enabled': self.config.trilium_upload.enabled,
                    'nextcloud_base_url': self.config.storage.nextcloud_base_url
                }
            )
            
            return Ok(self.config)
            
        except Exception as e:
            context = ErrorContext.create(
                "load_enhanced_config",
                input_data={'config_path': str(self.config_path)},
                suggestions=[
                    "Check YAML syntax",
                    "Verify all required sections exist",
                    "Run keyring setup for LLM API keys",
                    "Check LLM provider-model combination"
                ]
            )
            return Err(CoreError(f"Failed to load enhanced config: {e}", context))
    
    def _validate_all_secrets(self) -> Result[None, CoreError]:
        """Validiert dass alle benötigten Secrets in KeePassXC verfügbar sind"""
        if not self.config:
            return Err(CoreError("Config not loaded"))
    
        # CORE SECRETS (immer required)
        core_secrets_to_test = [
            (self.config.secrets.trilium_service, self.config.secrets.trilium_username, "Trilium"),
            (self.config.secrets.nextcloud_service, self.config.secrets.nextcloud_username, "Nextcloud"),
        ]
    
        failed_secrets = []
    
        # Test core secrets
        for service, username, name in core_secrets_to_test:
            test_result = self.secret_manager.test_secret_access(service, username)
            if isinstance(test_result, Err):
                failed_secrets.append(f"{name}: {service}/{username}")
                self.logger.warning(f"{name} secret not accessible: {test_result.error.message}")
    
        # LLM PROVIDER SECRET (nur aktiven Provider testen)
        active_provider = self.config.llm_processing.provider
        provider_secret_mapping = {
            "openai": (self.config.secrets.openai_service, self.config.secrets.openai_username, "OpenAI"),
            "anthropic": (self.config.secrets.anthropic_service, self.config.secrets.anthropic_username, "Anthropic"),
            "google": (self.config.secrets.google_service, self.config.secrets.google_username, "Google")
        }
    
        if active_provider in provider_secret_mapping:
            service, username, name = provider_secret_mapping[active_provider]
            test_result = self.secret_manager.test_secret_access(service, username)
            if isinstance(test_result, Err):
                failed_secrets.append(f"{name}: {service}/{username}")
                self.logger.error(f"Active LLM provider secret not accessible: {test_result.error.message}")
    
        # Core secrets failure = hard error
        core_failures = [s for s in failed_secrets if any(core in s for core in ["Trilium", "Nextcloud"])]
        if core_failures:
            context = ErrorContext.create(
                "validate_core_secrets",
                input_data={'failed_secrets': core_failures},
                suggestions=[
                    "Configure core secrets in KeePassXC",
                    "keyring set <service> <username>",
                    "Ensure KeePassXC is running and unlocked"
                ]
            )
            return Err(CoreError(f"Core secrets not accessible: {core_failures}", context))
    
        # Active LLM provider failure = hard error  
        llm_failures = [s for s in failed_secrets if any(provider in s for provider in ["OpenAI", "Anthropic", "Google"])]
        if llm_failures:
            context = ErrorContext.create(
                "validate_llm_secrets",
                input_data={'active_provider': active_provider, 'failed_secrets': llm_failures},
                suggestions=[
                    f"Configure {active_provider.title()} API key in KeePassXC",
                    f"keyring set {provider_secret_mapping[active_provider][0]} {provider_secret_mapping[active_provider][1]}",
                    "Or change LLM provider in config"
                ]
            )
            return Err(CoreError(f"Active LLM provider secret not accessible: {active_provider}", context))
    
        self.logger.info(
            "All required secrets accessible",
            extra={
                'core_secrets': ['Trilium', 'Nextcloud'],
                'active_llm_provider': active_provider,
                'total_secrets_checked': len(core_secrets_to_test) + 1
            }
        )
        return Ok(None)
    
    
    # Enhanced secret getters
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
    
    # NEW: LLM API Key getters
    def get_openai_api_key(self) -> Result[str, CoreError]:
        """Holt OpenAI API Key aus Keyring"""
        if not self.config:
            return Err(CoreError("Config not loaded"))
        
        return self.secret_manager.get_secret(
            self.config.secrets.openai_service,
            self.config.secrets.openai_username
        )
    
    def get_anthropic_api_key(self) -> Result[str, CoreError]:
        """Holt Anthropic API Key aus Keyring"""
        if not self.config:
            return Err(CoreError("Config not loaded"))
        
        return self.secret_manager.get_secret(
            self.config.secrets.anthropic_service,
            self.config.secrets.anthropic_username
        )
    
    def get_google_api_key(self) -> Result[str, CoreError]:
        """Holt Google API Key aus Keyring"""
        if not self.config:
            return Err(CoreError("Config not loaded"))
        
        return self.secret_manager.get_secret(
            self.config.secrets.google_service,
            self.config.secrets.google_username
        )
    
    def get_llm_api_key(self, provider: str) -> Result[str, CoreError]:
        """Holt API Key für aktuellen LLM Provider"""
        provider_methods = {
            "openai": self.get_openai_api_key,
            "anthropic": self.get_anthropic_api_key,
            "google": self.get_google_api_key
        }
        
        if provider not in provider_methods:
            return Err(CoreError(f"Unknown LLM provider: {provider}"))
        
        return provider_methods[provider]()
    
    def _generate_example_config(self) -> None:
        """Generiert Enhanced Beispiel-Konfiguration mit Fork-Join-Features"""
        example_config = {
            'secrets': {
                # Existing secrets
                'trilium_service': 'TrilliumToken',
                'trilium_username': 'token',
                'nextcloud_service': 'NextcloudPW',
                'nextcloud_username': 'steefen',
                # NEW: LLM API secrets
                'openai_service': 'OpenAI_API_Key',
                'openai_username': 'api_key',
                'anthropic_service': 'Anthropic_API_Key',
                'anthropic_username': 'api_key',
                'google_service': 'Google_API_Key',
                'google_username': 'api_key'
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
                'nextcloud_base_url': 'https://nextcloud.example.com/remote.php/dav/files/username/',
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
            },
            # NEW: LLM Processing Configuration
            'llm_processing': {
                'provider': 'openai',
                'model': 'gpt-4',
                'system_prompt_file': 'prompts/transcript_processing.md',
                'temperature': 0.1,
                'max_tokens': 4000,
                'retry_attempts': 5,
                'retry_delay': 5,
                'timeout': 300
            },
            # NEW: Trilium Upload Configuration
            'trilium_upload': {
                'enabled': True,
                'base_url': 'https://trilium.example.com',
                'parent_note_id': 'YouTube Transcripts',
                'note_template': 'transcript',
                'auto_tags': ['youtube', 'transcript', 'llm-processed'],
                'include_metadata': True,
                'timeout': 60
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(example_config, f, default_flow_style=False, allow_unicode=True)

# =============================================================================
# ENHANCED RULE SYSTEM (unchanged but compatible)
# =============================================================================

class EnhancedRuleSystem:
    """Rule System mit gewichteter Bewertung (unchanged interface)"""
    
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

# =============================================================================
# LLM PROMPT SYSTEM (NEW)
# =============================================================================

class LLMPromptSystem:
    """System für LLM-Prompt-Management mit Caching"""
    
    def __init__(self, config: AppConfig, base_path: Path = Path(".")):
        self.config = config
        self.base_path = base_path
        self.logger = get_logger("LLMPromptSystem")
        self._cached_system_prompt: Optional[str] = None
        self._cache_timestamp: Optional[float] = None
        self.cache_ttl = 300  # 5 Minuten Cache
    
    def get_system_prompt(self) -> Result[str, CoreError]:
        """Lädt System-Prompt mit Caching (analog zu Rules)"""
        import time
        current_time = time.time()
        
        # Check cache validity
        if (self._cached_system_prompt and 
            self._cache_timestamp and 
            current_time - self._cache_timestamp < self.cache_ttl):
            
            self.logger.debug("Using cached LLM system prompt")
            return Ok(self._cached_system_prompt)
        
        # Load fresh prompt
        try:
            prompt_path = self.base_path / self.config.llm_processing.system_prompt_file
            
            if not prompt_path.exists():
                context = ErrorContext.create(
                    "load_llm_system_prompt",
                    input_data={'prompt_path': str(prompt_path)},
                    suggestions=[
                        "Create system prompt file",
                        f"Expected location: {prompt_path}",
                        "Check llm_processing.system_prompt_file in config"
                    ]
                )
                return Err(CoreError(f"LLM system prompt file not found: {prompt_path}", context))
            
            prompt_content = prompt_path.read_text(encoding='utf-8')
            
            # Cache the result
            self._cached_system_prompt = prompt_content
            self._cache_timestamp = current_time
            
            self.logger.info(
                f"LLM system prompt loaded and cached",
                extra={
                    'prompt_file': str(prompt_path),
                    'prompt_length': len(prompt_content),
                    'cache_ttl': self.cache_ttl
                }
            )
            
            return Ok(prompt_content)
            
        except Exception as e:
            context = ErrorContext.create(
                "load_llm_system_prompt",
                input_data={'prompt_file': self.config.llm_processing.system_prompt_file},
                suggestions=["Check file permissions", "Verify file encoding"]
            )
            return Err(CoreError(f"Failed to load LLM system prompt: {e}", context))

# =============================================================================
# EXAMPLE USAGE & TESTING
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    
    # Setup
    setup_logging("enhanced_config_demo", "DEBUG")
    
    # Test Enhanced Config
    config_manager = SecureConfigManager(Path("enhanced_config.yaml"))
    
    # Generate example config
    if not Path("enhanced_config.yaml").exists():
        config_manager._generate_example_config()
        print("📝 Enhanced configuration file generated: enhanced_config.yaml")
    
    # Load and test config
    config_result = config_manager.load_config()
    
    if isinstance(config_result, Ok):
        config = config_result.value
        
        print("✅ Enhanced configuration loaded successfully!")
        print(f"   LLM Provider: {config.llm_processing.provider}")
        print(f"   LLM Model: {config.llm_processing.model}")
        print(f"   Trilium Enabled: {config.trilium_upload.enabled}")
        print(f"   Enabled Rules: {list(config.rules.get_enabled_rules().keys())}")
        
        # Test LLM prompt loading
        prompt_system = LLMPromptSystem(config)
        prompt_result = prompt_system.get_system_prompt()
        
        if isinstance(prompt_result, Ok):
            prompt = prompt_result.value
            print(f"✅ LLM system prompt loaded: {len(prompt)} characters")
        else:
            print(f"⚠️ LLM system prompt loading failed: {prompt_result.error.message}")
        
        # Test secret access for active provider
        api_key_result = config_manager.get_llm_api_key(config.llm_processing.provider)
        
        if isinstance(api_key_result, Ok):
            api_key = api_key_result.value
            print(f"✅ {config.llm_processing.provider.title()} API key accessible: {len(api_key)} characters")
        else:
            print(f"⚠️ {config.llm_processing.provider.title()} API key not accessible: {api_key_result.error.message}")
    
    else:
        print(f"❌ Configuration loading failed: {config_result.error.message}")
    
    print("\n🚀 Enhanced Configuration System Ready!")
    print("================================")
    print("New Features:")
    print("- LLM Processing Configuration (OpenAI, Anthropic, Google)")
    print("- Trilium Upload Configuration") 
    print("- Enhanced Secret Management")
    print("- System Prompt Loading with Caching")
    print("- Provider-Model Validation")
