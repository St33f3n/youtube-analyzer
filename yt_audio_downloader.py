"""
YouTube Audio Downloader with yt-dlp (OS-Version)
Lädt Audio-Spuren von YouTube-Videos für Transkription
"""

from __future__ import annotations
import time
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional
import threading

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_err, unwrap_ok
from yt_analyzer_core import ProcessObject
from logging_plus import get_logger, log_feature, log_function
from yt_analyzer_config import AppConfig

# =============================================================================
# AUDIO DOWNLOAD ENGINE (OS yt-dlp Version)
# =============================================================================


class YouTubeAudioDownloader:
    """OS yt-dlp Audio-Downloader mit Progress-Tracking und Error-Handling"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger("YouTubeAudioDownloader")

        # Ensure temp directory exists
        self.temp_dir = Path(self.config.processing.temp_folder)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Cookie-Datei finden (wie im URL-Processor)
        self.cookie_file = self._find_cookie_file()
        
        if self.cookie_file:
            self.logger.info(f"Using cookie file for audio downloads: {self.cookie_file}")
        else:
            self.logger.info("Using Firefox browser cookies for audio downloads")

        # Download statistics
        self.download_stats = {
            "total_downloads": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_size_mb": 0.0,
            "total_download_time_seconds": 0.0,
        }

    def _find_cookie_file(self) -> Optional[str]:
        """Sucht nach cookies.txt Datei (identisch mit URL-Processor)"""
        cookie_paths = [
        #    "cookies.txt",
         #   os.path.expanduser("data/yt_analyzer_cookies.txt"),  
          #  os.path.expanduser("yt_analyzer_cookies.txt"),
        ]
        
        for path in cookie_paths:
            if os.path.exists(path):
                return path
        return None

    def _download_audio_with_os_ytdlp(
        self, url: str, output_path: Path, audio_format: str
    ) -> Result[Path, CoreError]:
        """
        OS-Version von yt-dlp für Audio-Download
        """
        try:
            # yt-dlp Kommando für Audio-Download
            cmd = [
                "yt-dlp",
                "--extract-audio",
                "--audio-format", audio_format,
                "--audio-quality", "192K",
                "--output", str(output_path),
                "--no-playlist",
                "--quiet",
            ]
            
            # Cookie-Optionen
            if self.cookie_file:
                cmd.extend(["--cookies", self.cookie_file])
            else:
                cmd.extend(["--cookies-from-browser", "firefox"])
            
            cmd.append(url)
            
            # yt-dlp ausführen mit Timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for audio download
                check=True
            )
            
            # Find the downloaded file
            return self._find_downloaded_audio_file(output_path)
            
        except subprocess.TimeoutExpired:
            context = ErrorContext.create(
                "download_audio_os_ytdlp",
                input_data={"url": url, "timeout": 300},
                suggestions=[
                    "Check network connectivity",
                    "Try again later - YouTube may be throttling",
                    "Verify video has audio stream",
                ],
            )
            return Err(CoreError(f"Audio download timeout after 5 minutes: {url}", context))
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            context = ErrorContext.create(
                "download_audio_os_ytdlp",
                input_data={
                    "url": url, 
                    "return_code": e.returncode,
                    "stderr": error_msg[:200],
                },
                suggestions=[
                    "Check if video exists and has audio",
                    "Verify yt-dlp is up to date: pip install -U yt-dlp",
                    "Check network connectivity",
                    "Verify ffmpeg is installed for audio extraction",
                ],
            )
            return Err(CoreError(f"yt-dlp audio download failed: {error_msg}", context))
            
        except FileNotFoundError:
            context = ErrorContext.create(
                "download_audio_os_ytdlp",
                input_data={"url": url},
                suggestions=[
                    "Install yt-dlp: pip install yt-dlp",
                    "Verify yt-dlp is in PATH",
                    "Check system installation",
                ],
            )
            return Err(CoreError("yt-dlp not found in system PATH", context))

        except Exception as e:
            context = ErrorContext.create(
                "download_audio_os_ytdlp",
                input_data={"url": url, "error_type": type(e).__name__},
                suggestions=[
                    "Check system configuration",
                    "Verify all dependencies (yt-dlp, ffmpeg)",
                    "Try manual yt-dlp command",
                ],
            )
            return Err(CoreError(f"Unexpected error in audio download: {e}", context))

    def _find_downloaded_audio_file(self, expected_path: Path) -> Result[Path, CoreError]:
        """Findet die heruntergeladene Audio-Datei (OS yt-dlp ändert oft Dateinamen)"""
        try:
            # Expected path might have different extension after conversion
            base_path = expected_path.with_suffix('')
            audio_extensions = [".mp3", ".m4a", ".opus", ".wav", ".aac", ".ogg"]
            
            # First try: exact expected filename with different extensions
            for ext in audio_extensions:
                potential_file = base_path.with_suffix(ext)
                if potential_file.exists():
                    self.logger.debug(f"Found audio file (exact match): {potential_file}")
                    return Ok(potential_file)

            # Second try: look in the parent directory for recent audio files
            parent_dir = expected_path.parent
            current_time = time.time()
            
            for file_path in parent_dir.glob("*"):
                if (
                    file_path.suffix.lower() in audio_extensions
                    and file_path.stat().st_mtime > current_time - 600  # 10 minutes window
                ):
                    self.logger.debug(f"Found audio file (recent): {file_path}")
                    return Ok(file_path)

            # Log debugging info
            all_files = list(parent_dir.glob("*"))
            context = ErrorContext.create(
                "find_downloaded_audio_file",
                input_data={
                    "expected_path": str(expected_path),
                    "base_path": str(base_path),
                    "parent_dir": str(parent_dir),
                    "all_files": [f.name for f in all_files],
                    "audio_files": [
                        f.name for f in all_files 
                        if f.suffix.lower() in audio_extensions
                    ],
                },
                suggestions=[
                    "Check if download completed successfully",
                    "Verify ffmpeg is working for audio conversion",
                    "Check disk space and permissions",
                ],
            )
            return Err(CoreError("Downloaded audio file not found", context))

        except Exception as e:
            context = ErrorContext.create(
                "find_downloaded_audio_file",
                input_data={"expected_path": str(expected_path)},
                suggestions=[
                    "Check file system permissions",
                    "Verify disk space",
                ],
            )
            return Err(CoreError(f"Error finding downloaded audio file: {e}", context))

    @log_function(log_performance=True)
    def download_audio(
        self, process_obj: ProcessObject
    ) -> Result[ProcessObject, CoreError]:
        """
        Lädt Audio-Spur für ProcessObject mit OS yt-dlp

        Args:
            process_obj: ProcessObject mit original_url

        Returns:
            Ok(ProcessObject): ProcessObject mit temp_audio_path
            Err: Download-Fehler
        """
        if not process_obj.original_url:
            context = ErrorContext.create(
                "download_audio",
                input_data={"video_title": process_obj.titel},
                suggestions=[
                    "Check metadata extraction", 
                    "Verify original_url is set",
                    "Ensure URL processing completed successfully",
                ],
            )
            return Err(CoreError("No original URL in ProcessObject", context))

        try:
            with log_feature("audio_download_os") as feature:
                # Extract config values with fallbacks
                audio_format = getattr(self.config.processing, "audio_format", "mp3")
                if isinstance(audio_format, dict):
                    self.logger.warning(
                        f"audio_format is dict, using 'mp3': {audio_format}"
                    )
                    audio_format = "mp3"

                temp_folder = getattr(
                    self.config.processing, "temp_folder", "/tmp/youtube_analyzer"
                )
                if isinstance(temp_folder, dict):
                    self.logger.warning(
                        f"temp_folder is dict, using default: {temp_folder}"
                    )
                    temp_folder = "/tmp/youtube_analyzer"

                feature.add_metric("video_title", process_obj.titel)
                feature.add_metric("video_url", process_obj.original_url)
                feature.add_metric("audio_format", audio_format)

                # Generate safe filename
                safe_title = self._sanitize_filename(process_obj.titel)
                audio_filename = f"audio_{safe_title}_{int(time.time())}.%(ext)s"
                output_path = Path(temp_folder) / audio_filename

                self.logger.info(
                    "🎵 Starting OS yt-dlp audio download",
                    extra={
                        "video_title": process_obj.titel,
                        "channel": process_obj.kanal,
                        "video_url": process_obj.original_url,
                        "output_path": str(output_path),
                        "audio_format": audio_format,
                        "temp_dir": str(temp_folder),
                        "cookie_method": 'file' if self.cookie_file else 'browser',
                    },
                )

                # Download with retry logic
                download_result = self._download_with_retries(
                    process_obj.original_url, output_path, audio_format, max_retries=3
                )

                if isinstance(download_result, Err):
                    return download_result

                actual_path = unwrap_ok(download_result)

                # Verify file exists and has content
                if not actual_path.exists():
                    context = ErrorContext.create(
                        "download_audio",
                        input_data={
                            "video_title": process_obj.titel,
                            "expected_path": str(actual_path),
                        },
                        suggestions=[
                            "Check disk space",
                            "Verify yt-dlp installation",
                            "Check network connectivity",
                            "Verify ffmpeg is installed",
                        ],
                    )
                    return Err(
                        CoreError(f"Downloaded audio file not found: {actual_path}", context)
                    )

                file_size = actual_path.stat().st_size
                if file_size < 1000:  # Less than 1KB probably indicates failure
                    context = ErrorContext.create(
                        "download_audio",
                        input_data={
                            "video_title": process_obj.titel,
                            "file_size": file_size,
                            "file_path": str(actual_path),
                        },
                        suggestions=[
                            "Check if video has audio stream",
                            "Verify video is not just images/slideshow",
                            "Check yt-dlp and ffmpeg versions",
                            "Try different audio format",
                        ],
                    )
                    return Err(
                        CoreError(
                            f"Downloaded audio file too small ({file_size} bytes)", context
                        )
                    )

                # Update ProcessObject
                process_obj.temp_audio_path = actual_path
                process_obj.update_stage("audio_downloaded")

                # Update statistics
                self.download_stats["total_downloads"] += 1
                self.download_stats["successful_downloads"] += 1
                self.download_stats["total_size_mb"] += file_size / (1024 * 1024)

                feature.add_metric("download_success", True)
                feature.add_metric("file_size_mb", round(file_size / (1024 * 1024), 2))
                feature.add_metric("actual_path", str(actual_path))
                feature.add_metric("final_extension", actual_path.suffix)

                # Enhanced download completion logging
                self.logger.info(
                    f"✅ OS yt-dlp audio download completed: {process_obj.titel}",
                    extra={
                        "video_title": process_obj.titel,
                        "channel": process_obj.kanal,
                        "file_name": actual_path.name,
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                        "audio_format": audio_format,
                        "final_extension": actual_path.suffix,
                        "total_downloads": self.download_stats['total_downloads'],
                        "cookie_method": 'file' if self.cookie_file else 'browser',
                    }
                )

                return Ok(process_obj)

        except Exception as e:
            self.download_stats["failed_downloads"] += 1

            context = ErrorContext.create(
                "download_audio",
                input_data={
                    "video_title": process_obj.titel,
                    "video_url": process_obj.original_url,
                    "error_type": type(e).__name__,
                },
                suggestions=[
                    "Check network connectivity",
                    "Verify video is available and has audio",
                    "Check disk space in temp directory",
                    "Verify yt-dlp and ffmpeg installation",
                    "Update yt-dlp: pip install -U yt-dlp",
                ],
            )
            return Err(CoreError(f"Audio download failed: {e}", context))

    def _download_with_retries(
        self, url: str, output_path: Path, audio_format: str, max_retries: int = 3
    ) -> Result[Path, CoreError]:
        """Download mit Retry-Logic für OS yt-dlp"""

        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Audio download attempt {attempt + 1}/{max_retries}",
                    extra={
                        "url": url,
                        "attempt": attempt + 1,
                        "output_path": str(output_path),
                        "audio_format": audio_format,
                    },
                )

                start_time = time.time()

                # OS yt-dlp download
                download_result = self._download_audio_with_os_ytdlp(url, output_path, audio_format)
                
                if isinstance(download_result, Ok):
                    download_time = time.time() - start_time
                    self.download_stats["total_download_time_seconds"] += download_time
                    
                    actual_path = unwrap_ok(download_result)
                    self.logger.info(
                        f"✅ Audio download completed in {download_time:.1f} seconds",
                        extra={
                            "attempt": attempt + 1,
                            "download_time": round(download_time, 1),
                            "actual_path": str(actual_path),
                        }
                    )
                    return Ok(actual_path)
                else:
                    # Download failed, check if retryable
                    error = unwrap_err(download_result)
                    error_msg = error.message.lower()
                    
                    if attempt < max_retries - 1 and any(
                        keyword in error_msg
                        for keyword in ["network", "timeout", "connection", "temporary", "throttl"]
                    ):
                        wait_time = (2**attempt) + 1
                        self.logger.warning(
                            f"Retryable error on attempt {attempt + 1}, retrying in {wait_time}s: {error.message}",
                            extra={"attempt": attempt + 1, "wait_time": wait_time}
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        # Non-retryable or final attempt
                        return download_result

            except Exception as e:
                error_msg = str(e)
                
                if attempt < max_retries - 1 and any(
                    keyword in error_msg.lower()
                    for keyword in ["network", "timeout", "connection", "temporary"]
                ):
                    wait_time = (2**attempt) + 1
                    self.logger.warning(
                        f"Download attempt {attempt + 1} failed, retrying in {wait_time}s: {error_msg}",
                        extra={"attempt": attempt + 1, "wait_time": wait_time}
                    )
                    time.sleep(wait_time)
                    continue

                context = ErrorContext.create(
                    "download_with_retries",
                    input_data={
                        "url": url,
                        "attempt": attempt + 1,
                        "error": error_msg,
                        "error_type": type(e).__name__,
                    },
                    suggestions=[
                        "Check network connectivity",
                        "Verify yt-dlp and dependencies",
                        "Check disk space and permissions",
                    ],
                )
                return Err(CoreError(f"Audio download failed: {error_msg}", context))

        # Should never reach here due to return statements in loop
        context = ErrorContext.create(
            "download_with_retries",
            input_data={"url": url, "max_attempts": max_retries},
            suggestions=[
                "Check video accessibility",
                "Verify system configuration",
                "Try manual yt-dlp command",
            ],
        )
        return Err(CoreError("Audio download failed after all retry attempts", context))

    def _sanitize_filename(self, filename: str) -> str:
        """Bereinigt Filename für sicheres Dateisystem"""
        # Remove or replace problematic characters
        safe_chars = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. "
        )
        sanitized = "".join(c if c in safe_chars else "_" for c in filename)

        # Limit length and remove extra spaces
        sanitized = sanitized.strip()[:50]

        # Ensure not empty
        if not sanitized:
            sanitized = "unnamed_video"

        return sanitized

    def cleanup_temp_files(self, process_obj: ProcessObject) -> Result[None, CoreError]:
        """Bereinigt temporäre Audio-Dateien nach Verarbeitung"""
        if not process_obj.temp_audio_path:
            return Ok(None)

        try:
            if process_obj.temp_audio_path.exists():
                file_size = process_obj.temp_audio_path.stat().st_size
                process_obj.temp_audio_path.unlink()

                self.logger.debug(
                    "Cleaned up temp audio file",
                    extra={
                        "video_title": process_obj.titel,
                        "file_path": str(process_obj.temp_audio_path),
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                    },
                )

            return Ok(None)

        except Exception as e:
            # Non-critical error - log but don't fail
            self.logger.warning(
                f"Failed to cleanup temp audio file {process_obj.temp_audio_path}: {e}",
                extra={
                    "video_title": process_obj.titel,
                    "error": str(e),
                }
            )
            return Ok(None)  # Return success anyway

    def get_download_statistics(self) -> Dict[str, Any]:
        """Gibt Download-Statistiken zurück"""
        return {
            **self.download_stats,
            "success_rate": (
                self.download_stats["successful_downloads"]
                / self.download_stats["total_downloads"]
                * 100
                if self.download_stats["total_downloads"] > 0
                else 0
            ),
            "average_size_mb": (
                self.download_stats["total_size_mb"]
                / self.download_stats["successful_downloads"]
                if self.download_stats["successful_downloads"] > 0
                else 0
            ),
            "average_download_time_seconds": (
                self.download_stats["total_download_time_seconds"]
                / self.download_stats["successful_downloads"]
                if self.download_stats["successful_downloads"] > 0
                else 0
            ),
        }


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================


def download_audio_for_process_object(
    process_obj: ProcessObject, config: AppConfig
) -> Result[ProcessObject, CoreError]:
    """
    Standalone-Funktion für ProcessObject-Audio-Download

    Args:
        process_obj: ProcessObject mit original_url
        config: App-Konfiguration

    Returns:
        Ok(ProcessObject): ProcessObject mit temp_audio_path
        Err: Download-Fehler
    """
    downloader = YouTubeAudioDownloader(config)
    return downloader.download_audio(process_obj)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    from yt_analyzer_config import SecureConfigManager
    from datetime import datetime, time as dt_time

    # Setup
    setup_logging("audio_downloader_test", "DEBUG")

    # Test configuration
    config_manager = SecureConfigManager()
    config_result = config_manager.load_config()

    if isinstance(config_result, Ok):
        config = unwrap_ok(config_result)

        # Create test ProcessObject
        test_obj = ProcessObject(
            titel="Test Video",
            kanal="Test Channel",
            länge=dt_time(0, 5, 30),
            upload_date=datetime.now(),
            original_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll for testing
        )

        # Test audio download
        result = download_audio_for_process_object(test_obj, config)

        if isinstance(result, Ok):
            downloaded_obj = unwrap_ok(result)
            print("✅ Audio download successful:")
            print(f"  Audio path: {downloaded_obj.temp_audio_path}")
            print(f"  File exists: {downloaded_obj.temp_audio_path.exists()}")
            if downloaded_obj.temp_audio_path.exists():
                print(
                    f"  File size: {downloaded_obj.temp_audio_path.stat().st_size / (1024 * 1024):.1f} MB"
                )
        else:
            error = unwrap_err(result)
            print(f"❌ Audio download failed: {error.message}")
            if hasattr(error, 'context') and error.context.suggestions:
                print("Suggestions:", error.context.suggestions)
    else:
        print("❌ Config loading failed")
