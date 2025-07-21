"""
YouTube Audio Downloader with yt-dlp
Lädt Audio-Spuren von YouTube-Videos für Transkription
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any
import threading

# yt-dlp für Audio-Download
import yt_dlp
from yt_dlp import YoutubeDL

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_err, unwrap_ok
from yt_analyzer_core import ProcessObject
from logging_plus import get_logger, log_feature, log_function
from yt_analyzer_config import AppConfig

# =============================================================================
# AUDIO DOWNLOAD ENGINE
# =============================================================================


class YouTubeAudioDownloader:
    """yt-dlp Audio-Downloader mit Progress-Tracking und Error-Handling"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger("YouTubeAudioDownloader")

        # Ensure temp directory exists
        self.temp_dir = Path(self.config.processing.temp_folder)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.current_download_progress = {}
        self.progress_lock = threading.Lock()

        # Download statistics
        self.download_stats = {
            "total_downloads": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_size_mb": 0.0,
        }

    def progress_hook(self, d: Dict[str, Any]) -> None:
        """Progress-Hook für yt-dlp Downloads"""
        with self.progress_lock:
            if d["status"] == "downloading":
                # Update progress info
                filename = d.get("filename", "unknown")
                downloaded = d.get("downloaded_bytes", 0)
                total = d.get("total_bytes", 0) or d.get("total_bytes_estimate", 0)

                self.current_download_progress[filename] = {
                    "downloaded_bytes": downloaded,
                    "total_bytes": total,
                    "speed": d.get("speed", 0),
                    "eta": d.get("eta", 0),
                    "percent": (downloaded / total * 100) if total > 0 else 0,
                }

            elif d["status"] == "finished":
                filename = d.get("filename", "unknown")
                file_size = d.get("total_bytes", 0)

                # Update stats
                self.download_stats["successful_downloads"] += 1
                self.download_stats["total_size_mb"] += file_size / (1024 * 1024)

                self.logger.debug(
                    f"Download completed: {Path(filename).name}",
                    extra={
                        "filename": Path(filename).name,
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                        "total_downloads": self.download_stats["successful_downloads"],
                    },
                )

                # Clean up progress tracking
                if filename in self.current_download_progress:
                    del self.current_download_progress[filename]

    @log_function(log_performance=True)
    def download_audio(
        self, process_obj: ProcessObject
    ) -> Result[ProcessObject, CoreError]:
        """
        Lädt Audio-Spur für ProcessObject

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
                suggestions=["Check metadata extraction", "Verify original_url is set"],
            )
            return Err(CoreError("No original URL in ProcessObject", context))

        try:
            with log_feature("audio_download") as feature:
                # Debug config values
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
                audio_filename = f"audio_{safe_title}_{int(time.time())}.{audio_format}"
                audio_path = Path(temp_folder) / audio_filename

                self.logger.info(
                    "🎵 Starting audio download",
                    extra={
                        "video_title": process_obj.titel,
                        "channel": process_obj.kanal,
                        "video_url": process_obj.original_url,
                        "output_path": str(audio_path),
                        "audio_format": audio_format,
                        "temp_dir": str(temp_folder),
                        "audio_path_type": type(audio_path).__name__,
                        "audio_format_type": type(audio_format).__name__,
                    },
                )

                # yt-dlp options for simple audio-only download (ULTRA-SIMPLIFIED!)
                simple_filename = f"audio_{int(time.time())}"
                output_path = Path(temp_folder) / f"{simple_filename}.%(ext)s"

                ydl_opts = {
                    "format": "bestaudio[ext=m4a]/bestaudio/best",
                    "outtmpl": str(output_path),  # Explicit string conversion
                    "quiet": False,  # Enable output to see what's happening
                    "extractaudio": True,
                    "audioformat": audio_format,
                    "audioquality": "192k",  # Explicit bitrate
                    "prefer_ffmpeg": True,
                    "progress_hooks": [self.progress_hook],
                }

                # Debug ydl_opts
                self.logger.info(
                    "yt-dlp options (ultra-simplified)",
                    extra={
                        "outtmpl": ydl_opts["outtmpl"],
                        "outtmpl_type": type(ydl_opts["outtmpl"]).__name__,
                        "format": ydl_opts["format"],
                        "audioformat": ydl_opts["audioformat"],
                        "temp_folder": temp_folder,
                        "simple_filename": simple_filename,
                    },
                )

                # Download with retry logic
                download_result = self._download_with_retries(
                    process_obj.original_url, ydl_opts, max_retries=3
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
                        ],
                    )
                    return Err(
                        CoreError(f"Downloaded file not found: {actual_path}", context)
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
                            "Check video availability",
                            "Verify audio stream exists",
                            "Check yt-dlp version",
                        ],
                    )
                    return Err(
                        CoreError(
                            f"Downloaded file too small ({file_size} bytes)", context
                        )
                    )

                # Update ProcessObject
                process_obj.temp_audio_path = actual_path
                process_obj.update_stage("audio_downloaded")

                # Update statistics
                self.download_stats["total_downloads"] += 1

                feature.add_metric("download_success", True)
                feature.add_metric("file_size_mb", round(file_size / (1024 * 1024), 2))
                feature.add_metric("output_path", str(actual_path))

                # Enhanced download completion logging
                download_info = (
                    f"✅ Audio download completed successfully:\n"
                    f"  🎵 Video: {process_obj.titel}\n"
                    f"  📺 Channel: {process_obj.kanal}\n"
                    f"  📁 File: {actual_path.name}\n"
                    f"  💾 Size: {file_size / (1024 * 1024):.1f} MB\n"
                    f"  🎚️ Format: {audio_format}\n"
                    f"  📊 Total downloads: {self.download_stats['total_downloads']}"
                )

                self.logger.info(download_info)

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
                    "Verify video is available",
                    "Check disk space in temp directory",
                    "Verify yt-dlp and ffmpeg installation",
                ],
            )
            return Err(CoreError(f"Audio download failed: {e}", context))

    def _download_with_retries(
        self, url: str, ydl_opts: Dict[str, Any], max_retries: int = 3
    ) -> Result[Path, CoreError]:
        """Download mit Retry-Logic (DIRECT FILE FINDING!)"""

        temp_folder = Path(ydl_opts["outtmpl"]).parent
        expected_base_name = Path(ydl_opts["outtmpl"]).stem  # e.g., "audio_1752803693"

        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Download attempt {attempt + 1}/{max_retries}",
                    extra={
                        "url": url,
                        "attempt": attempt + 1,
                        "temp_folder": str(temp_folder),
                        "expected_base_name": expected_base_name,
                    },
                )

                with YoutubeDL(ydl_opts) as ydl:
                    # Download
                    ydl.download([url])

                # Find the exact file - we know the base name!
                audio_extensions = [".mp3", ".m4a", ".opus", ".wav", ".aac"]

                # First try: exact expected filename with different extensions
                for ext in audio_extensions:
                    expected_file = temp_folder / f"{expected_base_name}{ext}"
                    if expected_file.exists():
                        self.logger.info(
                            f"✅ Found downloaded file (exact match): {expected_file}"
                        )
                        return Ok(expected_file)

                # Second try: list all files and find our file
                self.logger.debug(
                    f"Exact match not found, listing all files in {temp_folder}"
                )

                all_files = list(temp_folder.glob("*"))
                self.logger.debug(
                    f"All files in temp folder: {[f.name for f in all_files]}"
                )

                # Look for files that start with our expected base name
                for file_path in all_files:
                    if (
                        file_path.name.startswith(expected_base_name)
                        and file_path.suffix.lower() in audio_extensions
                    ):
                        self.logger.info(
                            f"✅ Found downloaded file (prefix match): {file_path}"
                        )
                        return Ok(file_path)

                # Third try: newest audio file (fallback)
                import time

                current_time = time.time()
                newest_file = None
                newest_time = 0

                for file_path in all_files:
                    if (
                        file_path.suffix.lower() in audio_extensions
                        and file_path.stat().st_mtime > current_time - 600
                    ):  # 10 minutes window
                        if file_path.stat().st_mtime > newest_time:
                            newest_time = file_path.stat().st_mtime
                            newest_file = file_path

                if newest_file:
                    self.logger.info(
                        f"✅ Found downloaded file (newest): {newest_file}"
                    )
                    return Ok(newest_file)

                # Log debugging info
                self.logger.error(
                    "No audio file found after download",
                    extra={
                        "temp_folder": str(temp_folder),
                        "expected_base_name": expected_base_name,
                        "all_files": [f.name for f in all_files],
                        "audio_files": [
                            f.name
                            for f in all_files
                            if f.suffix.lower() in audio_extensions
                        ],
                        "recent_files": [
                            f.name
                            for f in all_files
                            if f.stat().st_mtime > current_time - 600
                        ],
                    },
                )

                raise FileNotFoundError(
                    f"Downloaded audio file not found. Expected: {expected_base_name}.*"
                )

            except yt_dlp.DownloadError as e:
                self.logger.warning(f"yt-dlp error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + 1
                    time.sleep(wait_time)
                    continue

                return Err(CoreError(f"yt-dlp download failed: {e}"))

            except Exception as e:
                self.logger.error(
                    f"Download attempt {attempt + 1} failed: {e}",
                    extra={
                        "error_type": type(e).__name__,
                        "expected_base_name": expected_base_name,
                        "temp_folder": str(temp_folder),
                    },
                )
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + 1
                    time.sleep(wait_time)
                    continue

                return Err(CoreError(f"Download failed: {e}"))

        return Err(CoreError("Download failed after all attempts"))

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
                f"Failed to cleanup temp file {process_obj.temp_audio_path}: {e}"
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
    else:
        print("❌ Config loading failed")
