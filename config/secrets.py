"""
Secret Management - Keyring Integration für KeePassXC
"""

import keyring
from typing import Optional, Dict, Any
from loguru import logger
from config.settings import get_config


def setup_keepassxc_backend():
    """KeePassXC Backend konfigurieren"""
    try:
        backend = keyring.get_keyring()
        if hasattr(backend, 'scheme'):
            backend.scheme = 'KeePassXC'
            logger.debug("KeePassXC scheme gesetzt")
    except Exception as e:
        logger.warning(f"KeePassXC Backend Setup fehlgeschlagen: {e}")


class SecretsManager:
    """Manager für Passwörter und API Keys über Linux Secret Service"""
    
    def __init__(self):
        self.config = None
        # KeePassXC Backend einrichten
        setup_keepassxc_backend()
        
    def _get_config(self):
        """Lazy-load config"""
        if self.config is None:
            self.config = get_config()
        return self.config
        
    def get_trilium_api_key(self) -> Optional[str]:
        """Hole Trilium API Key aus Keyring"""
        try:
            config = self._get_config()
            service_name = config.secrets.trilium_service
            username = config.secrets.trilium_username
            
            api_key = keyring.get_password(service_name, username)
            
            if not api_key:
                logger.warning(f"Trilium API Key nicht in Keyring gefunden: {service_name}/{username}")
                return None
                
            logger.debug("Trilium API Key erfolgreich geladen")
            return api_key
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Trilium API Keys: {e}")
            return None
            
    def get_nextcloud_credentials(self) -> Optional[Dict[str, str]]:
        """Hole NextCloud Credentials aus Keyring"""
        try:
            config = self._get_config()
            service_name = config.secrets.nextcloud_service
            username = config.secrets.nextcloud_username
            
            # NextCloud Password aus Keyring
            password = keyring.get_password(service_name, username)
            
            if not password:
                logger.warning(f"NextCloud Password nicht in Keyring gefunden: {service_name}/{username}")
                return None
                
            logger.debug("NextCloud Credentials erfolgreich geladen")
            return {
                "username": username,  # Username aus Config
                "password": password   # Password aus Keyring
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der NextCloud Credentials: {e}")
            return None
            
    def set_trilium_api_key(self, api_key: str) -> bool:
        """Speichere Trilium API Key in Keyring"""
        try:
            config = self._get_config()
            service_name = config.secrets.trilium_service
            username = config.secrets.trilium_username
            
            keyring.set_password(service_name, username, api_key)
            logger.info(f"Trilium API Key gespeichert in: {service_name}/{username}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Trilium API Keys: {e}")
            return False
            
    def set_nextcloud_password(self, password: str) -> bool:
        """Speichere NextCloud Password in Keyring"""
        try:
            config = self._get_config()
            service_name = config.secrets.nextcloud_service
            username = config.secrets.nextcloud_username
            
            keyring.set_password(service_name, username, password)
            logger.info(f"NextCloud Password gespeichert in: {service_name}/{username}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern des NextCloud Passwords: {e}")
            return False
            
    def check_secrets_availability(self) -> Dict[str, bool]:
        """Prüfe Verfügbarkeit aller Secrets"""
        return {
            "trilium_api_key": self.get_trilium_api_key() is not None,
            "nextcloud_credentials": self.get_nextcloud_credentials() is not None
        }
        
    def delete_trilium_api_key(self) -> bool:
        """Lösche Trilium API Key aus Keyring"""
        try:
            config = self._get_config()
            service_name = config.secrets.trilium_service
            username = config.secrets.trilium_username
            
            keyring.delete_password(service_name, username)
            logger.info("Trilium API Key gelöscht")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Löschen des Trilium API Keys: {e}")
            return False
            
    def delete_nextcloud_password(self) -> bool:
        """Lösche NextCloud Password aus Keyring"""
        try:
            config = self._get_config()
            service_name = config.secrets.nextcloud_service
            username = config.secrets.nextcloud_username
            
            keyring.delete_password(service_name, username)
            logger.info("NextCloud Password gelöscht")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Löschen des NextCloud Passwords: {e}")
            return False


# Globale Secrets-Manager Instanz
secrets_manager = SecretsManager()


def get_secrets_manager() -> SecretsManager:
    """Convenience-Funktion für Secrets-Manager Zugriff"""
    return secrets_manager


def setup_secrets_interactive():
    """Interaktive Einrichtung der Secrets (für CLI)"""
    print("🔐 Secrets Setup für YouTube Info Analyzer (KeePassXC)")
    print("=" * 60)
    
    # KeePassXC Backend setup
    setup_keepassxc_backend()
    
    manager = get_secrets_manager()
    
    # Config laden für Service/Username Info
    try:
        config = manager._get_config()
        print(f"📋 Konfiguration geladen:")
        print(f"   Trilium: {config.secrets.trilium_service}/{config.secrets.trilium_username}")
        print(f"   NextCloud: {config.secrets.nextcloud_service}/{config.secrets.nextcloud_username}")
    except Exception as e:
        print(f"❌ Fehler beim Laden der Config: {e}")
        return
    
    # Trilium API Key
    print(f"\n📝 Trilium API Key für '{config.secrets.trilium_username}':")
    trilium_key = input("API Key eingeben: ").strip()
    if trilium_key:
        if manager.set_trilium_api_key(trilium_key):
            print("✅ Trilium API Key gespeichert")
        else:
            print("❌ Fehler beim Speichern")
    
    # NextCloud Password
    print(f"\n☁️ NextCloud Password für '{config.secrets.nextcloud_username}':")
    nc_password = input("Password eingeben: ").strip()
    
    if nc_password:
        if manager.set_nextcloud_password(nc_password):
            print("✅ NextCloud Password gespeichert")
        else:
            print("❌ Fehler beim Speichern")
    
    # Status prüfen
    print("\n🔍 Secret Status:")
    status = manager.check_secrets_availability()
    for secret, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"{status_icon} {secret}: {'Verfügbar' if available else 'Nicht verfügbar'}")


if __name__ == "__main__":
    # Direkte Ausführung für Secret-Setup
    setup_secrets_interactive()
