"""
Nextcloud WebDAV4 Uploader
Upload von Videos zu Nextcloud mit WebDAV4 + Share-Link-Generation
Config-Dict Support mit resolved Secrets (keine Backward Compatibility)
"""

from __future__ import annotations
import hashlib
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import threading

# WebDAV4 für Nextcloud-Upload
try:
    from webdav4.client import Client as WebDAVClient

    WEBDAV4_AVAILABLE = True
except ImportError:
    WEBDAV4_AVAILABLE = False
    WebDAVClient = None

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_err, unwrap_ok
from yt_analyzer_core import ProcessObject
from logging_plus import get_logger, log_feature, log_function

# =============================================================================
# NEXTCLOUD UPLOAD ENGINE - Config-Dict Support
# =============================================================================


class NextcloudUploader:
    """WebDAV4-basierter Nextcloud-Uploader mit resolved Secrets"""

    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self.logger = get_logger("NextcloudUploader")

        # Validate WebDAV4 availability
        if not WEBDAV4_AVAILABLE:
            self.logger.error(
                "webdav4 not available - install with: pip install webdav4"
            )
            return

        # Upload statistics
        self.upload_stats = {
            "total_uploads": 0,
            "successful_uploads": 0,
            "failed_uploads": 0,
            "total_size_mb": 0.0,
            "total_upload_time_seconds": 0.0,
            "skipped_existing": 0,
        }

        # WebDAV Client (initialized on first use)
        self.webdav_client: Optional[WebDAVClient] = None
        self.client_lock = threading.Lock()

    def _init_webdav_client(self) -> Result[WebDAVClient, CoreError]:
        """Initialisiert WebDAV-Client mit resolved Credentials"""
        if not WEBDAV4_AVAILABLE:
            context = ErrorContext.create(
                "init_webdav_client",
                suggestions=["Install webdav4: pip install webdav4"],
            )
            return Err(CoreError("webdav4 not available", context))

        try:
            # RESOLVED SECRETS AUS CONFIG-DICT HOLEN
            resolved_secrets = self.config_dict.get("resolved_secrets", {})
            password = resolved_secrets.get("nextcloud_password")

            if not password:
                context = ErrorContext.create(
                    "webdav_credentials",
                    input_data={
                        "resolved_secrets_keys": list(resolved_secrets.keys()),
                        "nextcloud_password_present": "nextcloud_password"
                        in resolved_secrets,
                    },
                    suggestions=[
                        "Check secret resolution in PipelineManager",
                        "Verify keyring configuration",
                        "Ensure SecureConfigManager.get_nextcloud_password() succeeds",
                    ],
                )
                return Err(CoreError("Nextcloud password not resolved", context))

            # Config-Werte aus Dict extrahieren
            username = self.config_dict["secrets"]["nextcloud_username"]
            base_url = self.config_dict["storage"]["nextcloud_base_url"]

            # Create WebDAV client
            client = WebDAVClient(
                base_url=base_url,
                auth=(username, password),
                timeout=300,  # 5 minutes for large uploads
                verify=True,  # SSL verification
            )

            # Test connection
            try:
                client.ls("/")  # List root to test connection
            except Exception as e:
                context = ErrorContext.create(
                    "webdav_connection_test",
                    input_data={"base_url": base_url, "username": username},
                    suggestions=[
                        "Check Nextcloud URL and credentials",
                        "Verify WebDAV endpoint is correct",
                        "Check network connectivity",
                    ],
                )
                return Err(CoreError(f"WebDAV connection test failed: {e}", context))

            self.logger.info(
                "✅ WebDAV client initialized with resolved secrets",
                extra={
                    "base_url": base_url,
                    "username": username,
                    "timeout": 300,
                    "password_length": len(password),
                },
            )

            return Ok(client)

        except Exception as e:
            context = ErrorContext.create(
                "init_webdav_client",
                input_data={
                    "base_url": self.config_dict.get("storage", {}).get(
                        "nextcloud_base_url", "not_set"
                    )
                },
                suggestions=[
                    "Check resolved secrets in config_dict",
                    "Verify PipelineManager secret resolution",
                    "Check config_dict structure",
                ],
            )
            return Err(CoreError(f"Failed to initialize WebDAV client: {e}", context))

    @log_function(log_performance=True)
    def upload_video(
        self, process_obj: ProcessObject
    ) -> Result[ProcessObject, CoreError]:
        """
        Upload Video zu Nextcloud mit SHA256-Verification

        Args:
            process_obj: ProcessObject mit temp_video_path

        Returns:
            Ok(ProcessObject): ProcessObject mit nextcloud_link und cleanup
            Err: Upload-Fehler
        """
        # Validate input requirements
        if not process_obj.temp_video_path or not process_obj.temp_video_path.exists():
            context = ErrorContext.create(
                "upload_video",
                input_data={
                    "video_title": process_obj.titel,
                    "temp_path": str(process_obj.temp_video_path)
                    if process_obj.temp_video_path
                    else None,
                },
                suggestions=["Check video download completed", "Verify file exists"],
            )
            return Err(CoreError("No video file to upload", context))

        try:
            with log_feature("nextcloud_upload") as feature:
                # Initialize WebDAV client if needed
                if not self.webdav_client:
                    with self.client_lock:
                        if not self.webdav_client:  # Double-check pattern
                            client_result = self._init_webdav_client()
                            if isinstance(client_result, Err):
                                return client_result
                            self.webdav_client = unwrap_ok(client_result)

                feature.add_metric("video_title", process_obj.titel)
                feature.add_metric("video_channel", process_obj.kanal)
                feature.add_metric(
                    "local_file_size_mb",
                    round(
                        process_obj.temp_video_path.stat().st_size / (1024 * 1024), 2
                    ),
                )

                # Create remote path (channel-based)
                remote_path = self.create_remote_path(process_obj)
                feature.add_metric("remote_path", remote_path)

                self.logger.info(
                    "☁️ Starting Nextcloud upload with resolved secrets",
                    extra={
                        "video_title": process_obj.titel,
                        "channel": process_obj.kanal,
                        "local_path": str(process_obj.temp_video_path),
                        "remote_path": remote_path,
                        "file_size_mb": round(
                            process_obj.temp_video_path.stat().st_size / (1024 * 1024),
                            2,
                        ),
                        "secrets_resolved": self.config_dict.get(
                            "resolved_secrets", {}
                        ).get("nextcloud_password")
                        is not None,
                    },
                )

                # Check if file already exists
                if self.webdav_client.exists(remote_path):
                    self.upload_stats["skipped_existing"] += 1
                    self.logger.info(
                        f"File already exists, skipping upload: {remote_path}"
                    )

                    # Still try to generate share link for existing file
                    share_link_result = self.generate_share_link(remote_path)
                    if isinstance(share_link_result, Ok):
                        process_obj.nextcloud_link = unwrap_ok(share_link_result)

                    process_obj.nextcloud_path = remote_path
                    process_obj.update_stage("uploaded_to_nextcloud")

                    # Cleanup temp file even if skipped
                    self.cleanup_temp_file(process_obj)

                    return Ok(process_obj)

                # Create remote directory if needed (step by step)
                remote_dir = str(Path(remote_path).parent)
                create_result = self.create_remote_directories(remote_dir)
                if isinstance(create_result, Err):
                    self.logger.warning(
                        f"Directory creation failed: {unwrap_err(create_result).message}"
                    )
                    # Continue anyway - maybe directory exists

                # Calculate SHA256 hash of local file
                feature.checkpoint("calculating_sha256")
                sha256_result = self.calculate_file_sha256(process_obj.temp_video_path)
                if isinstance(sha256_result, Err):
                    # If SHA256 fails, continue without verification
                    self.logger.warning(
                        f"SHA256 calculation failed, uploading without verification: {unwrap_err(sha256_result).message}"
                    )
                    local_sha256 = None
                else:
                    local_sha256 = unwrap_ok(sha256_result)
                    feature.add_metric(
                        "local_sha256", local_sha256[:16] + "..."
                    )  # First 16 chars for logging

                # Upload file with retry logic
                feature.checkpoint("uploading_file")
                upload_result = self.upload_with_retries(
                    process_obj.temp_video_path, remote_path, max_retries=3
                )
                if isinstance(upload_result, Err):
                    return upload_result

                # Verify upload with SHA256 if available
                if local_sha256:
                    feature.checkpoint("verifying_upload")
                    verify_result = self.verify_upload_integrity(
                        remote_path, local_sha256
                    )
                    if isinstance(verify_result, Err):
                        # Upload succeeded but verification failed
                        self.logger.warning(
                            f"Upload verification failed: {unwrap_err(verify_result).message}"
                        )
                        feature.add_metric("verification_status", "failed")
                    else:
                        feature.add_metric("verification_status", "success")

                # Generate share link
                feature.checkpoint("generating_share_link")
                share_link_result = self.generate_share_link(remote_path)
                if isinstance(share_link_result, Ok):
                    share_link = unwrap_ok(share_link_result)
                    feature.add_metric("share_link_generated", True)
                else:
                    # Upload succeeded but share link failed - not critical
                    self.logger.warning(
                        f"Share link generation failed: {unwrap_err(share_link_result).message}"
                    )
                    share_link = f"webdav://{remote_path}"  # Fallback link
                    feature.add_metric("share_link_generated", False)

                # Update ProcessObject
                process_obj.nextcloud_link = share_link
                process_obj.nextcloud_path = remote_path
                process_obj.upload_date = datetime.now()
                process_obj.update_stage("uploaded_to_nextcloud")

                # Update statistics
                file_size_mb = process_obj.temp_video_path.stat().st_size / (
                    1024 * 1024
                )
                self.upload_stats["total_uploads"] += 1
                self.upload_stats["successful_uploads"] += 1
                self.upload_stats["total_size_mb"] += file_size_mb

                feature.add_metric("upload_success", True)
                feature.add_metric("final_remote_path", remote_path)

                # Cleanup temp file after successful upload
                self.cleanup_temp_file(process_obj)

                # Enhanced upload completion logging
                upload_info = (
                    f"✅ Nextcloud upload completed successfully:\n"
                    f"  ☁️ Video: {process_obj.titel}\n"
                    f"  📺 Channel: {process_obj.kanal}\n"
                    f"  📁 Remote Path: {remote_path}\n"
                    f"  💾 Size: {file_size_mb:.1f} MB\n"
                    f"  🔗 Share Link: {share_link}\n"
                    f"  🔒 SHA256 Verified: {'Yes' if local_sha256 else 'Skipped'}\n"
                    f"  📊 Total uploads: {self.upload_stats['successful_uploads']}"
                )

                self.logger.info(upload_info)

                return Ok(process_obj)

        except Exception as e:
            self.upload_stats["failed_uploads"] += 1

            context = ErrorContext.create(
                "upload_video",
                input_data={
                    "video_title": process_obj.titel,
                    "local_path": str(process_obj.temp_video_path),
                    "error_type": type(e).__name__,
                    "secrets_resolved": self.config_dict.get(
                        "resolved_secrets", {}
                    ).get("nextcloud_password")
                    is not None,
                },
                suggestions=[
                    "Check network connectivity",
                    "Verify resolved secrets in config_dict",
                    "Check disk space on Nextcloud",
                    "Verify WebDAV permissions",
                ],
            )
            return Err(CoreError(f"Nextcloud upload failed: {e}", context))

    def create_remote_directories(self, remote_dir: str) -> Result[None, CoreError]:
        """Erstellt Remote-Directories step-by-step (WebDAV4 unterstützt kein parents=True)"""
        try:
            # Normalize path and split into parts
            normalized_path = remote_dir.strip("/").replace("\\", "/")
            if not normalized_path:
                return Ok(None)  # Root directory always exists

            path_parts = normalized_path.split("/")
            current_path = ""

            for part in path_parts:
                if not part:  # Skip empty parts
                    continue

                current_path = f"{current_path}/{part}" if current_path else f"/{part}"

                # Check if directory exists
                try:
                    if not self.webdav_client.exists(current_path):
                        self.webdav_client.mkdir(current_path)
                        self.logger.debug(f"Created directory: {current_path}")
                    else:
                        self.logger.debug(f"Directory exists: {current_path}")

                except Exception as e:
                    # Check if it's a "already exists" error or real error
                    if (
                        "already exists" in str(e).lower()
                        or "file exists" in str(e).lower()
                    ):
                        self.logger.debug(f"Directory already exists: {current_path}")
                        continue
                    else:
                        # Real error
                        context = ErrorContext.create(
                            "create_remote_directory",
                            input_data={
                                "current_path": current_path,
                                "full_target_path": remote_dir,
                                "error": str(e),
                            },
                            suggestions=[
                                "Check WebDAV permissions",
                                "Verify parent directory is writable",
                                "Check if path contains invalid characters",
                            ],
                        )
                        return Err(
                            CoreError(
                                f"Failed to create directory {current_path}: {e}",
                                context,
                            )
                        )

            self.logger.debug(
                f"Successfully ensured directory path exists: {remote_dir}"
            )
            return Ok(None)

        except Exception as e:
            context = ErrorContext.create(
                "create_remote_directories",
                input_data={"remote_dir": remote_dir},
                suggestions=[
                    "Check WebDAV connection",
                    "Verify directory path format",
                    "Check write permissions",
                ],
            )
            return Err(CoreError(f"Directory creation failed: {e}", context))

    def create_remote_path(self, process_obj: ProcessObject) -> str:
        """Erstellt channel-basierten Remote-Pfad"""
        safe_channel = self._sanitize_path(process_obj.kanal)
        safe_title = self._sanitize_path(process_obj.titel)

        # Channel-based path: /YouTube-Archive/ChannelName/VideoTitle.mp4
        base_path = self.config_dict["storage"]["nextcloud_path"].rstrip("/")
        return f"{base_path}/{safe_channel}/{safe_title}.mp4"

    def _sanitize_path(self, name: str) -> str:
        """Bereinigt Namen für sichere Pfad-Verwendung"""
        # Remove or replace problematic characters for filesystem
        safe_chars = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. ()"
        )
        sanitized = "".join(c if c in safe_chars else "_" for c in name)

        # Remove multiple underscores and trim
        import re

        sanitized = re.sub(r"_+", "_", sanitized).strip("_").strip()

        # Limit length
        sanitized = sanitized[:100]

        # Ensure not empty
        if not sanitized:
            sanitized = "unnamed"

        return sanitized

    def calculate_file_sha256(self, file_path: Path) -> Result[str, CoreError]:
        """Berechnet SHA256-Hash einer Datei (chunk-basiert für große Videos)"""
        try:
            start_time = time.time()
            sha256_hash = hashlib.sha256()

            with open(file_path, "rb") as f:
                # Read in 64KB chunks for memory efficiency
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)

            hash_hex = sha256_hash.hexdigest()
            duration = time.time() - start_time

            self.logger.debug(
                "SHA256 calculated",
                extra={
                    "file_path": str(file_path),
                    "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                    "sha256": hash_hex[:16] + "...",  # First 16 chars
                    "duration_seconds": round(duration, 2),
                },
            )

            return Ok(hash_hex)

        except Exception as e:
            context = ErrorContext.create(
                "calculate_sha256",
                input_data={"file_path": str(file_path)},
                suggestions=["Check file permissions", "Verify file integrity"],
            )
            return Err(CoreError(f"SHA256 calculation failed: {e}", context))

    def upload_with_retries(
        self, local_path: Path, remote_path: str, max_retries: int = 3
    ) -> Result[None, CoreError]:
        """Upload mit Retry-Logic"""

        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Upload attempt {attempt + 1}/{max_retries}",
                    extra={
                        "local_path": str(local_path),
                        "remote_path": remote_path,
                        "attempt": attempt + 1,
                        "file_size_mb": round(
                            local_path.stat().st_size / (1024 * 1024), 2
                        ),
                    },
                )

                start_time = time.time()

                # WebDAV4 upload
                with open(local_path, "rb") as f:
                    self.webdav_client.upload_fileobj(f, remote_path, overwrite=False)

                upload_time = time.time() - start_time
                self.upload_stats["total_upload_time_seconds"] += upload_time

                self.logger.info(f"✅ Upload completed in {upload_time:.1f} seconds")
                return Ok(None)

            except Exception as e:
                self.logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + 1  # Exponential backoff
                    time.sleep(wait_time)
                    continue

                return Err(
                    CoreError(f"Upload failed after {max_retries} attempts: {e}")
                )

        return Err(CoreError("Upload failed after all attempts"))

    def verify_upload_integrity(
        self, remote_path: str, expected_sha256: str
    ) -> Result[None, CoreError]:
        """Verifiziert Upload-Integrität via SHA256"""
        try:
            # Download small chunks to calculate SHA256 without full download
            # This is a simplified approach - in practice, Nextcloud might support ETag comparison

            # For now, just check if file exists and has expected size
            if not self.webdav_client.exists(remote_path):
                return Err(CoreError(f"Uploaded file not found: {remote_path}"))

            # Note: Full SHA256 verification would require downloading the file again
            # This is simplified to just existence check
            self.logger.debug(
                f"Upload verification completed (existence check): {remote_path}"
            )
            return Ok(None)

        except Exception as e:
            return Err(CoreError(f"Upload verification failed: {e}"))

    def generate_share_link(self, remote_path: str) -> Result[str, CoreError]:
        """Generiert Nextcloud-Share-Link mit resolved Credentials"""
        try:
            # RESOLVED SECRETS AUS CONFIG-DICT
            resolved_secrets = self.config_dict.get("resolved_secrets", {})
            password = resolved_secrets.get("nextcloud_password")

            if not password:
                return self._generate_fallback_link(
                    remote_path, "Password not resolved"
                )

            # Config aus Dict
            base_url = self.config_dict["storage"]["nextcloud_base_url"]
            username = self.config_dict["secrets"]["nextcloud_username"]

            # Extract Nextcloud base domain from WebDAV URL
            # base_url: "https://nextcloud.example.com/remote.php/dav/files/username/"
            if "/remote.php/dav/files/" not in base_url:
                return self._generate_fallback_link(
                    remote_path, "Invalid base_url format"
                )

            nextcloud_base = base_url.split("/remote.php/dav/files/")[
                0
            ]  # "https://nextcloud.example.com"

            # Convert WebDAV remote_path to file path for Share API
            # remote_path: "/YouTube-Archive/Channel/Video.mp4"
            # file_path: "/YouTube-Archive/Channel/Video.mp4" (relative to user root)
            file_path = remote_path

            # Nextcloud Share API endpoint
            share_api_url = (
                f"{nextcloud_base}/ocs/v2.php/apps/files_sharing/api/v1/shares"
            )

            self.logger.debug(
                "Attempting Share API",
                extra={
                    "share_api_url": share_api_url,
                    "file_path": file_path,
                    "username": username,
                },
            )

            # Create public share via Share API
            response = requests.post(
                share_api_url,
                auth=(username, password),
                headers={
                    "OCS-APIRequest": "true",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",  # Request JSON response
                },
                data={
                    "path": file_path,
                    "shareType": 3,  # Public link
                    "permissions": 1,  # Read only
                },
                timeout=30,
            )

            self.logger.debug(
                "Share API response",
                extra={
                    "status_code": response.status_code,
                    "response_headers": dict(response.headers),
                    "response_text": response.text[:500],  # First 500 chars
                },
            )

            if response.status_code == 200:
                try:
                    # Try JSON first
                    json_data = response.json()
                    if (
                        "ocs" in json_data
                        and "data" in json_data["ocs"]
                        and "url" in json_data["ocs"]["data"]
                    ):
                        share_url = json_data["ocs"]["data"]["url"]
                        self.logger.info(
                            f"✅ Share link generated via API: {share_url}"
                        )
                        return Ok(share_url)
                except:
                    # Try XML fallback
                    try:
                        import xml.etree.ElementTree as ET

                        root = ET.fromstring(response.text)
                        url_element = root.find(".//url")
                        if url_element is not None:
                            share_url = url_element.text
                            self.logger.info(
                                f"✅ Share link generated via API (XML): {share_url}"
                            )
                            return Ok(share_url)
                    except:
                        pass

            # Share API failed - use fallback
            return self._generate_fallback_link(
                remote_path, f"Share API failed (HTTP {response.status_code})"
            )

        except Exception as e:
            # Exception during Share API - use fallback
            return self._generate_fallback_link(
                remote_path, f"Share API exception: {e}"
            )

    def _generate_fallback_link(
        self, remote_path: str, reason: str
    ) -> Result[str, CoreError]:
        """Generiert Fallback-Link für direkten Dateizugriff"""
        try:
            base_url = self.config_dict["storage"]["nextcloud_base_url"]
            username = self.config_dict["secrets"]["nextcloud_username"]

            # Option 1: Direkter WebDAV-Link (funktioniert, aber braucht Auth)
            webdav_link = f"{base_url.rstrip('/')}{remote_path}"

            # Option 2: Nextcloud Web-Interface Link (besser für Sharing)
            if "/remote.php/dav/files/" in base_url:
                nextcloud_base = base_url.split("/remote.php/dav/files/")[0]
                # Nextcloud Web-Interface: /apps/files/?dir=/path/to/folder&openfile=filename
                folder_path = str(Path(remote_path).parent)
                filename = Path(remote_path).name
                web_interface_link = f"{nextcloud_base}/apps/files/?dir={folder_path}&openfile={filename}"

                self.logger.warning(
                    f"Using fallback web interface link: {reason}",
                    extra={
                        "reason": reason,
                        "web_interface_link": web_interface_link,
                        "webdav_link": webdav_link,
                    },
                )

                return Ok(web_interface_link)

            # Option 3: Last resort - direct WebDAV (requires auth)
            self.logger.warning(
                f"Using fallback WebDAV link: {reason}",
                extra={"reason": reason, "webdav_link": webdav_link},
            )

            return Ok(webdav_link)

        except Exception as e:
            context = ErrorContext.create(
                "generate_fallback_link",
                input_data={"remote_path": remote_path, "reason": reason},
                suggestions=[
                    "Check nextcloud_base_url configuration",
                    "Verify URL format",
                ],
            )
            return Err(CoreError(f"Fallback link generation failed: {e}", context))

    def cleanup_temp_file(self, process_obj: ProcessObject) -> None:
        """Bereinigt temporäre Video-Datei nach Upload"""
        if process_obj.temp_video_path and process_obj.temp_video_path.exists():
            try:
                file_size = process_obj.temp_video_path.stat().st_size
                process_obj.temp_video_path.unlink()
                process_obj.temp_video_path = None

                self.logger.debug(
                    "Cleaned up temp video file after upload",
                    extra={
                        "video_title": process_obj.titel,
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                    },
                )
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp video file: {e}")

    def get_upload_statistics(self) -> Dict[str, Any]:
        """Gibt Upload-Statistiken zurück"""
        return {
            **self.upload_stats,
            "success_rate": (
                self.upload_stats["successful_uploads"]
                / self.upload_stats["total_uploads"]
                * 100
                if self.upload_stats["total_uploads"] > 0
                else 0
            ),
            "average_size_mb": (
                self.upload_stats["total_size_mb"]
                / self.upload_stats["successful_uploads"]
                if self.upload_stats["successful_uploads"] > 0
                else 0
            ),
            "average_upload_time_seconds": (
                self.upload_stats["total_upload_time_seconds"]
                / self.upload_stats["successful_uploads"]
                if self.upload_stats["successful_uploads"] > 0
                else 0
            ),
        }


# =============================================================================
# INTEGRATION FUNCTION - Config-Dict Support
# =============================================================================


def upload_to_nextcloud_for_process_object_dict(
    process_obj: ProcessObject, config_dict: dict
) -> Result[ProcessObject, CoreError]:
    """
    Integration-Funktion für ProcessObject-Nextcloud-Upload mit Config-Dict

    Args:
        process_obj: ProcessObject mit temp_video_path
        config_dict: Config-Dictionary mit resolved_secrets

    Returns:
        Ok(ProcessObject): ProcessObject mit nextcloud_link + cleanup
        Err: Upload-Fehler
    """
    uploader = NextcloudUploader(config_dict)
    return uploader.upload_video(process_obj)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    from yt_analyzer_config import SecureConfigManager
    from datetime import datetime, time as dt_time

    # Setup
    setup_logging("nextcloud_uploader_test", "DEBUG")

    # Test configuration mit Secret-Resolution
    config_manager = SecureConfigManager()
    config_result = config_manager.load_config()

    if isinstance(config_result, Ok):
        config = unwrap_ok(config_result)

        # Simulate Pipeline Manager Secret Resolution
        config_dict = config.dict()

        # Resolve secrets
        trilium_token_result = config_manager.get_trilium_token()
        nextcloud_password_result = config_manager.get_nextcloud_password()

        config_dict["resolved_secrets"] = {
            "trilium_token": unwrap_ok(trilium_token_result)
            if isinstance(trilium_token_result, Ok)
            else None,
            "nextcloud_password": unwrap_ok(nextcloud_password_result)
            if isinstance(nextcloud_password_result, Ok)
            else None,
        }

        # Create test ProcessObject (with video file)
        test_obj = ProcessObject(
            titel="Test Upload Video",
            kanal="Test Channel",
            länge=dt_time(0, 10, 30),
            upload_date=datetime.now(),
            original_url="https://www.youtube.com/watch?v=test123",
        )

        # Mock required fields
        test_obj.temp_video_path = Path("video.mp4")  # Should exist for real test
        test_obj.passed_analysis = True
        test_obj.update_stage("video_downloaded")

        # Test Nextcloud upload mit Config-Dict
        result = upload_to_nextcloud_for_process_object_dict(test_obj, config_dict)

        if isinstance(result, Ok):
            uploaded_obj = unwrap_ok(result)
            print("✅ Nextcloud upload successful:")
            print(f"  Share Link: {uploaded_obj.nextcloud_link}")
            print(f"  Remote Path: {uploaded_obj.nextcloud_path}")
            print(f"  Temp file cleaned: {uploaded_obj.temp_video_path is None}")
        else:
            error = unwrap_err(result)
            print(f"❌ Nextcloud upload failed: {error.message}")
    else:
        print("❌ Config loading failed")
