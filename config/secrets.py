"""
Secret Management - KeePassXC Integration mit Result-Types
Vollständig überarbeitet mit vollständigen Type-Hints und Error-Handling
"""

from __future__ import annotations

import getpass
from typing import Dict
from typing import Optional

import keyring

from config.settings import get_config
from yt_types import ConfigurationError
from yt_types import Err
from yt_types import Ok
from yt_types import Result
from yt_types import ServiceUnavailableError
from utils.logging import ComponentLogger
from utils.logging import log_function_calls


def setup_keepassxc_backend() -> None:
    """KeePassXC Backend konfigurieren"""
    try:
        backend = keyring.get_keyring()
        if hasattr(backend, 'scheme'):
            backend.scheme = 'KeePassXC'
            
        # Logger für diese Funktion
        logger = ComponentLogger("KeePassXCSetup")
        logger.debug(
            "KeePassXC backend configured",
            backend_type=type(backend).__name__,
        )
    
    except Exception as e:
        logger = ComponentLogger("KeePassXCSetup")
        logger.warning(
            "KeePassXC backend setup failed",
            error=e,
            fallback="Using default keyring backend",
        )


class SecretsManager:
    """Manager für Passwörter und API Keys über KeePassXC/Linux Secret Service"""
    
    def __init__(self) -> None:
        self.logger = ComponentLogger("SecretsManager")
        self._config: Optional[object] = None
        
        # KeePassXC Backend einrichten
        setup_keepassxc_backend()
        
        self.logger.info(
            "Secrets manager initialized",
            backend_type=type(keyring.get_keyring()).__name__,
        )
    
    def _get_config(self) -> Result[object, ConfigurationError]:
        """Lazy-load Konfiguration"""
        if self._config is None:
            try:
                self._config = get_config()
            except Exception as e:
                return Err(ConfigurationError(
                    f"Failed to load configuration: {str(e)}",
                    {'error_type': type(e).__name__}
                ))
        
        return Ok(self._config)
    
    @log_function_calls
    def get_trilium_api_key(self) -> Result[str, ServiceUnavailableError]:
        """Hole Trilium API Key aus Keyring"""
        try:
            config_result = self._get_config()
            if isinstance(config_result, Err):
                return Err(ServiceUnavailableError(
                    f"Configuration error: {config_result.error.message}",
                    {'component': 'config_loader'}
                ))
            
            config = config_result.value
            service_name = config.secrets.trilium_service
            username = config.secrets.trilium_username
            
            self.logger.debug(
                "Retrieving Trilium API key",
                service_name=service_name,
                username=username,
            )
            
            api_key = keyring.get_password(service_name, username)
            
            if not api_key:
                return Err(ServiceUnavailableError(
                    f"Trilium API key not found in keyring",
                    {
                        'service_name': service_name,
                        'username': username,
                        'suggestion': 'Run setup_secrets_interactive() to configure',
                    }
                ))
            
            self.logger.info(
                "Trilium API key retrieved successfully",
                service_name=service_name,
                username=username,
                key_length=len(api_key),
            )
            
            return Ok(api_key)
        
        except Exception as e:
            error_msg = f"Failed to retrieve Trilium API key: {str(e)}"
            self.logger.error(
                "Trilium API key retrieval failed",
                error=e,
                error_type=type(e).__name__,
            )
            
            return Err(ServiceUnavailableError(
                error_msg,
                {
                    'error_type': type(e).__name__,
                    'component': 'keyring',
                }
            ))
    
    @log_function_calls
    def get_nextcloud_credentials(self) -> Result[Dict[str, str], ServiceUnavailableError]:
        """Hole NextCloud Credentials aus Keyring"""
        try:
            config_result = self._get_config()
            if isinstance(config_result, Err):
                return Err(ServiceUnavailableError(
                    f"Configuration error: {config_result.error.message}",
                    {'component': 'config_loader'}
                ))
            
            config = config_result.value
            service_name = config.secrets.nextcloud_service
            username = config.secrets.nextcloud_username
            
            self.logger.debug(
                "Retrieving NextCloud credentials",
                service_name=service_name,
                username=username,
            )
            
            # NextCloud Password aus Keyring
            password = keyring.get_password(service_name, username)
            
            if not password:
                return Err(ServiceUnavailableError(
                    f"NextCloud credentials not found in keyring",
                    {
                        'service_name': service_name,
                        'username': username,
                        'suggestion': 'Run setup_secrets_interactive() to configure',
                    }
                ))
            
            credentials = {
                "username": username,
                "password": password,
            }
            
            self.logger.info(
                "NextCloud credentials retrieved successfully",
                service_name=service_name,
                username=username,
                password_length=len(password),
            )
            
            return Ok(credentials)
        
        except Exception as e:
            error_msg = f"Failed to retrieve NextCloud credentials: {str(e)}"
            self.logger.error(
                "NextCloud credentials retrieval failed",
                error=e,
                error_type=type(e).__name__,
            )
            
            return Err(ServiceUnavailableError(
                error_msg,
                {
                    'error_type': type(e).__name__,
                    'component': 'keyring',
                }
            ))
    
    @log_function_calls
    def set_trilium_api_key(self, api_key: str) -> Result[None, ServiceUnavailableError]:
        """Speichere Trilium API Key in Keyring"""
        try:
            if not api_key or len(api_key.strip()) < 10:
                return Err(ServiceUnavailableError(
                    "Invalid API key: too short or empty",
                    {'min_length': 10, 'provided_length': len(api_key.strip())}
                ))
            
            config_result = self._get_config()
            if isinstance(config_result, Err):
                return Err(ServiceUnavailableError(
                    f"Configuration error: {config_result.error.message}",
                    {'component': 'config_loader'}
                ))
            
            config = config_result.value
            service_name = config.secrets.trilium_service
            username = config.secrets.trilium_username
            
            keyring.set_password(service_name, username, api_key)
            
            self.logger.info(
                "Trilium API key stored successfully",
                service_name=service_name,
                username=username,
                key_length=len(api_key),
            )
            
            return Ok(None)
        
        except Exception as e:
            error_msg = f"Failed to store Trilium API key: {str(e)}"
            self.logger.error(
                "Trilium API key storage failed",
                error=e,
                error_type=type(e).__name__,
            )
            
            return Err(ServiceUnavailableError(
                error_msg,
                {
                    'error_type': type(e).__name__,
                    'component': 'keyring',
                }
            ))
    
    @log_function_calls
    def set_nextcloud_password(self, password: str) -> Result[None, ServiceUnavailableError]:
        """Speichere NextCloud Password in Keyring"""
        try:
            if not password or len(password.strip()) < 3:
                return Err(ServiceUnavailableError(
                    "Invalid password: too short or empty",
                    {'min_length': 3, 'provided_length': len(password.strip())}
                ))
            
            config_result = self._get_config()
            if isinstance(config_result, Err):
                return Err(ServiceUnavailableError(
                    f"Configuration error: {config_result.error.message}",
                    {'component': 'config_loader'}
                ))
            
            config = config_result.value
            service_name = config.secrets.nextcloud_service
            username = config.secrets.nextcloud_username
            
            keyring.set_password(service_name, username, password)
            
            self.logger.info(
                "NextCloud password stored successfully",
                service_name=service_name,
                username=username,
                password_length=len(password),
            )
            
            return Ok(None)
        
        except Exception as e:
            error_msg = f"Failed to store NextCloud password: {str(e)}"
            self.logger.error(
                "NextCloud password storage failed",
                error=e,
                error_type=type(e).__name__,
            )
            
            return Err(ServiceUnavailableError(
                error_msg,
                {
                    'error_type': type(e).__name__,
                    'component': 'keyring',
                }
            ))
    
    @log_function_calls
    def check_secrets_availability(self) -> Dict[str, bool]:
        """Prüfe Verfügbarkeit aller Secrets"""
        try:
            trilium_result = self.get_trilium_api_key()
            nextcloud_result = self.get_nextcloud_credentials()
            
            availability = {
                "trilium_api_key": isinstance(trilium_result, Ok),
                "nextcloud_credentials": isinstance(nextcloud_result, Ok),
            }
            
            self.logger.info(
                "Secrets availability check completed",
                trilium_available=availability["trilium_api_key"],
                nextcloud_available=availability["nextcloud_credentials"],
            )
            
            return availability
        
        except Exception as e:
            self.logger.error(
                "Secrets availability check failed",
                error=e,
                error_type=type(e).__name__,
            )
            
            return {
                "trilium_api_key": False,
                "nextcloud_credentials": False,
            }
    
    @log_function_calls
    def delete_trilium_api_key(self) -> Result[None, ServiceUnavailableError]:
        """Lösche Trilium API Key aus Keyring"""
        try:
            config_result = self._get_config()
            if isinstance(config_result, Err):
                return Err(ServiceUnavailableError(
                    f"Configuration error: {config_result.error.message}",
                    {'component': 'config_loader'}
                ))
            
            config = config_result.value
            service_name = config.secrets.trilium_service
            username = config.secrets.trilium_username
            
            keyring.delete_password(service_name, username)
            
            self.logger.info(
                "Trilium API key deleted successfully",
                service_name=service_name,
                username=username,
            )
            
            return Ok(None)
        
        except Exception as e:
            error_msg = f"Failed to delete Trilium API key: {str(e)}"
            self.logger.error(
                "Trilium API key deletion failed",
                error=e,
                error_type=type(e).__name__,
            )
            
            return Err(ServiceUnavailableError(
                error_msg,
                {
                    'error_type': type(e).__name__,
                    'component': 'keyring',
                }
            ))
    
    @log_function_calls
    def delete_nextcloud_password(self) -> Result[None, ServiceUnavailableError]:
        """Lösche NextCloud Password aus Keyring"""
        try:
            config_result = self._get_config()
            if isinstance(config_result, Err):
                return Err(ServiceUnavailableError(
                    f"Configuration error: {config_result.error.message}",
                    {'component': 'config_loader'}
                ))
            
            config = config_result.value
            service_name = config.secrets.nextcloud_service
            username = config.secrets.nextcloud_username
            
            keyring.delete_password(service_name, username)
            
            self.logger.info(
                "NextCloud password deleted successfully",
                service_name=service_name,
                username=username,
            )
            
            return Ok(None)
        
        except Exception as e:
            error_msg = f"Failed to delete NextCloud password: {str(e)}"
            self.logger.error(
                "NextCloud password deletion failed",
                error=e,
                error_type=type(e).__name__,
            )
            
            return Err(ServiceUnavailableError(
                error_msg,
                {
                    'error_type': type(e).__name__,
                    'component': 'keyring',
                }
            ))
    
    def get_secrets_status(self) -> Dict[str, Dict[str, any]]:
        """Detaillierter Secrets-Status für Monitoring"""
        try:
            config_result = self._get_config()
            if isinstance(config_result, Err):
                return {
                    'status': 'error',
                    'error': config_result.error.message,
                    'secrets': {},
                }
            
            config = config_result.value
            
            # Trilium Status
            trilium_result = self.get_trilium_api_key()
            trilium_status = {
                'service_name': config.secrets.trilium_service,
                'username': config.secrets.trilium_username,
                'available': isinstance(trilium_result, Ok),
                'error': trilium_result.error.message if isinstance(trilium_result, Err) else None,
            }
            
            # NextCloud Status
            nextcloud_result = self.get_nextcloud_credentials()
            nextcloud_status = {
                'service_name': config.secrets.nextcloud_service,
                'username': config.secrets.nextcloud_username,
                'available': isinstance(nextcloud_result, Ok),
                'error': nextcloud_result.error.message if isinstance(nextcloud_result, Err) else None,
            }
            
            return {
                'status': 'ready',
                'backend': type(keyring.get_keyring()).__name__,
                'secrets': {
                    'trilium': trilium_status,
                    'nextcloud': nextcloud_status,
                },
            }
        
        except Exception as e:
            self.logger.error(
                "Secrets status check failed",
                error=e,
                error_type=type(e).__name__,
            )
            
            return {
                'status': 'error',
                'error': str(e),
                'secrets': {},
            }


# =============================================================================
# SINGLETON FACTORY
# =============================================================================

_secrets_manager_instance: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Singleton Factory für Secrets-Manager"""
    global _secrets_manager_instance
    
    if _secrets_manager_instance is None:
        _secrets_manager_instance = SecretsManager()
    
    return _secrets_manager_instance


def create_secrets_manager() -> SecretsManager:
    """Factory für neuen Secrets-Manager"""
    return SecretsManager()


# =============================================================================
# INTERACTIVE SETUP
# =============================================================================

def setup_secrets_interactive() -> None:
    """Interaktive Einrichtung der Secrets für CLI"""
    print("🔐 Secrets Setup für YouTube Info Analyzer")
    print("=" * 50)
    
    # KeePassXC Backend setup
    setup_keepassxc_backend()
    
    manager = get_secrets_manager()
    logger = ComponentLogger("SecretsSetup")
    
    try:
        # Aktuelle Konfiguration anzeigen
        config = get_config()
        
        print(f"\n📋 Aktuelle Konfiguration:")
        print(f"   Trilium: {config.secrets.trilium_service}/{config.secrets.trilium_username}")
        print(f"   NextCloud: {config.secrets.nextcloud_service}/{config.secrets.nextcloud_username}")
        
        # Aktuelle Verfügbarkeit prüfen
        availability = manager.check_secrets_availability()
        
        print(f"\n🔍 Aktuelle Verfügbarkeit:")
        for secret_name, available in availability.items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {secret_name}")
        
        # Trilium API Key Setup
        print(f"\n📝 Trilium API Key Setup:")
        print(f"   Service: {config.secrets.trilium_service}")
        print(f"   Username: {config.secrets.trilium_username}")
        
        if not availability["trilium_api_key"]:
            api_key = getpass.getpass("Trilium API Key eingeben: ").strip()
            if api_key:
                result = manager.set_trilium_api_key(api_key)
                if isinstance(result, Ok):
                    print("✅ Trilium API Key gespeichert")
                else:
                    print(f"❌ Fehler beim Speichern: {result.error.message}")
        else:
            print("✅ Trilium API Key bereits verfügbar")
            
            overwrite = input("Überschreiben? (y/N): ").strip().lower()
            if overwrite == 'y':
                api_key = getpass.getpass("Neuer Trilium API Key: ").strip()
                if api_key:
                    result = manager.set_trilium_api_key(api_key)
                    if isinstance(result, Ok):
                        print("✅ Trilium API Key aktualisiert")
                    else:
                        print(f"❌ Fehler beim Aktualisieren: {result.error.message}")
        
        # NextCloud Password Setup
        print(f"\n☁️ NextCloud Password Setup:")
        print(f"   Service: {config.secrets.nextcloud_service}")
        print(f"   Username: {config.secrets.nextcloud_username}")
        
        if not availability["nextcloud_credentials"]:
            password = getpass.getpass("NextCloud Password eingeben: ").strip()
            if password:
                result = manager.set_nextcloud_password(password)
                if isinstance(result, Ok):
                    print("✅ NextCloud Password gespeichert")
                else:
                    print(f"❌ Fehler beim Speichern: {result.error.message}")
        else:
            print("✅ NextCloud Password bereits verfügbar")
            
            overwrite = input("Überschreiben? (y/N): ").strip().lower()
            if overwrite == 'y':
                password = getpass.getpass("Neues NextCloud Password: ").strip()
                if password:
                    result = manager.set_nextcloud_password(password)
                    if isinstance(result, Ok):
                        print("✅ NextCloud Password aktualisiert")
                    else:
                        print(f"❌ Fehler beim Aktualisieren: {result.error.message}")
        
        # Finale Überprüfung
        print(f"\n🔍 Finale Überprüfung:")
        final_availability = manager.check_secrets_availability()
        
        all_available = all(final_availability.values())
        
        for secret_name, available in final_availability.items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {secret_name}")
        
        if all_available:
            print(f"\n🎉 Secrets Setup erfolgreich abgeschlossen!")
            logger.info("Interactive secrets setup completed successfully")
        else:
            print(f"\n⚠️ Einige Secrets sind nicht verfügbar. Prüfe die Konfiguration.")
            logger.warning("Interactive secrets setup completed with missing secrets")
    
    except Exception as e:
        print(f"\n❌ Fehler beim Setup: {str(e)}")
        logger.error(
            "Interactive secrets setup failed",
            error=e,
            error_type=type(e).__name__,
        )


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def test_secrets_system() -> None:
    """Test-Funktion für Secrets System"""
    from youtube_analyzer.utils.logging import get_development_config
    from youtube_analyzer.utils.logging import setup_logging
    
    # Setup logging für Test
    setup_logging(get_development_config())
    
    manager = get_secrets_manager()
    logger = ComponentLogger("SecretsSystemTest")
    
    logger.info("Starting secrets system test")
    
    # Test Secrets Status
    status = manager.get_secrets_status()
    logger.info(
        "Secrets status",
        status=status['status'],
        backend=status.get('backend', 'unknown'),
        secrets=status.get('secrets', {}),
    )
    
    # Test Availability Check
    availability = manager.check_secrets_availability()
    logger.info(
        "Secrets availability",
        trilium_available=availability["trilium_api_key"],
        nextcloud_available=availability["nextcloud_credentials"],
    )
    
    # Test mit Mock-Daten (nur wenn keine echten Secrets vorhanden)
    if not availability["trilium_api_key"]:
        logger.info("Testing with mock Trilium API key...")
        
        # Mock API Key setzen
        mock_result = manager.set_trilium_api_key("mock_api_key_for_testing_12345")
        if isinstance(mock_result, Ok):
            logger.info("✅ Mock Trilium API key stored")
            
            # Wieder abrufen
            retrieve_result = manager.get_trilium_api_key()
            if isinstance(retrieve_result, Ok):
                logger.info("✅ Mock Trilium API key retrieved")
                
                # Wieder löschen
                delete_result = manager.delete_trilium_api_key()
                if isinstance(delete_result, Ok):
                    logger.info("✅ Mock Trilium API key deleted")
                else:
                    logger.error("❌ Mock Trilium API key deletion failed")
            else:
                logger.error("❌ Mock Trilium API key retrieval failed")
        else:
            logger.error("❌ Mock Trilium API key storage failed")
    
    logger.info("✅ Secrets system test completed")


if __name__ == "__main__":
    # Direkte Ausführung für Interactive Setup
    setup_secrets_interactive()
