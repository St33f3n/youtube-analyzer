"""
Download Service - yt-dlp Integration mit Result-Types und vollständigen Type-Hints
Vollständig überarbeitet nach Quality-Gate-Standards
REPARIERTE IMPORTS - Konsistente direkte Imports aus Hauptverzeichnis
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import yt_dlp

# REPARIERTE IMPORTS - Direkte Imports aus Hauptverzeichnis
from yt_types import AudioMetadata
from yt_types import DownloadError
from yt_types import Err
from yt_types import Ok
from yt_types import ProgressCallback
from yt_types import Result
from yt_types import VideoMetadata
from yt_types import validate_youtube_url
from utils.logging import ComponentLogger
from utils.logging import log_function_calls
from utils.logging import log_performance


class DownloadProgressHandler:
    """Handler für yt-dlp Progress-Callbacks mit Type-Safety"""
    
    def __init__(self, callback: Optional[ProgressCallback] = None) -> None:
        self.callback = callback
        self.logger = ComponentLogger("DownloadProgressHandler")
    
    def __call__(self, progress_data: Dict[str, Any]) -> None:
        """yt-dlp Progress Hook mit strukturiertem Logging"""
        try:
            # Normalisiere Progress-Daten
            normalized_progress = self._normalize_progress_data(progress_data)
            
            # Debug-Logging für Function-Level
            self.logger.debug(
                "Download progress update",
                progress_data=normalized_progress,
                raw_data_keys=list(progress_data.keys()),
            )
            
            # Callback ausführen falls vorhanden
            if self.callback:
                self.callback(
                    normalized_progress['progress_percent'],
                    normalized_progress['status_message']
                )
        
        except Exception as e:
            self.logger.error(
                "Progress handler failed",
                error=e,
                progress_data=progress_data,
            )
    
    def _normalize_progress_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalisiere yt-dlp Progress-Daten"""
        status = data.get('status', 'unknown')
        total_bytes = data.get('total_bytes', 0)
        downloaded_bytes = data.get('downloaded_bytes', 0)
        speed = data.get('speed', 0)
        
        # Progress-Prozent berechnen
        if total_bytes > 0:
            progress_percent = min(int((downloaded_bytes / total_bytes) * 100), 100)
        else:
            progress_percent = 0
        
        # Status-Message erstellen
        if status == 'downloading':
            speed_text = f" ({speed / (1024 * 1024):.1f} MB/s)" if speed > 0 else ""
            status_message = f"Downloading... {progress_percent}%{speed_text}"
        elif status == 'finished':
            status_message = f"Download completed ({downloaded_bytes / (1024 * 1024):.1f} MB)"
        else:
            status_message = f"Status: {status}"
        
        return {
            'status': status,
            'progress_percent': progress_percent,
            'status_message': status_message,
            'total_bytes': total_bytes,
            'downloaded_bytes': downloaded_bytes,
            'speed': speed,
        }


class DownloadService:
    """Service für YouTube Downloads mit vollständiger Type-Safety"""
    
    def __init__(self, temp_dir: Optional[Path] = None) -> None:
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="youtube_analyzer_"))
        self.logger = ComponentLogger("DownloadService")
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            "Download service initialized",
            temp_dir=str(self.temp_dir),
            service_status="ready",
        )
    
    def __del__(self) -> None:
        """Cleanup beim Beenden"""
        self.cleanup()
    
    @log_function_calls
    def get_video_info(self, url: str) -> Result[VideoMetadata, DownloadError]:
        """Hole Video-Informationen ohne Download"""
        
        # URL-Validierung
        url_validation = validate_youtube_url(url)
        if isinstance(url_validation, Err):
            return Err(DownloadError(f"Invalid URL: {url_validation.error.message}"))
        
        try:
            self.logger.debug(
                "Fetching video info",
                url=url,
                operation="get_video_info",
            )
            
            # yt-dlp Optionen für Info-Extraktion
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
            # VideoMetadata erstellen
            video_metadata = VideoMetadata(
                id=info.get('id', ''),
                title=info.get('title', 'Unknown'),
                uploader=info.get('uploader', 'Unknown'),
                duration=info.get('duration', 0),
                view_count=info.get('view_count', 0),
                upload_date=info.get('upload_date', ''),
                description=info.get('description', ''),
                webpage_url=info.get('webpage_url', url),
                thumbnail_url=info.get('thumbnail', None),
                format_count=len(info.get('formats', [])),
            )
            
            self.logger.info(
                "Video info extracted successfully",
                video_id=video_metadata.id,
                title=video_metadata.title,
                duration=video_metadata.duration,
                uploader=video_metadata.uploader,
            )
            
            return Ok(video_metadata)
        
        except Exception as e:
            error_msg = f"Failed to extract video info: {str(e)}"
            self.logger.error(
                "Video info extraction failed",
                error=e,
                url=url,
                operation="get_video_info",
            )
            return Err(DownloadError(error_msg, {'url': url, 'error_type': type(e).__name__}))
    
    @log_performance
    def download_audio(
        self,
        url: str,
        progress_callback: Optional[ProgressCallback] = None
    ) -> Result[AudioMetadata, DownloadError]:
        """Download Audio zu temporärer Datei"""
        
        # URL-Validierung
        url_validation = validate_youtube_url(url)
        if isinstance(url_validation, Err):
            return Err(DownloadError(f"Invalid URL: {url_validation.error.message}"))
        
        try:
            self.logger.info(
                "Starting audio download",
                url=url,
                operation="download_audio",
            )
            
            # Temporäre Datei für Audio
            audio_file = self.temp_dir / f"audio_{hash(url) % 100000}.wav"
            
            # Progress Handler
            progress_handler = DownloadProgressHandler(progress_callback)
            
            # yt-dlp Optionen für Audio-Download
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(audio_file.with_suffix('.%(ext)s')),
                'quiet': True,
                'no_warnings': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '0',
                }],
                'postprocessor_args': [
                    '-map_metadata', '-1',  # Metadaten entfernen
                    '-fflags', '+bitexact',
                    '-avoid_negative_ts', 'make_zero',
                ],
                'progress_hooks': [progress_handler],
            }
            
            # Download ausführen
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # WAV-Datei finden
            wav_files = list(self.temp_dir.glob(f"audio_{hash(url) % 100000}.wav"))
            
            if not wav_files:
                raise DownloadError("WAV file not found after download")
            
            wav_file = wav_files[0]
            
            # Datei validieren
            if not wav_file.exists() or wav_file.stat().st_size < 1000:
                raise DownloadError(f"Invalid WAV file: {wav_file}")
            
            # AudioMetadata erstellen
            audio_metadata = AudioMetadata(
                file_path=wav_file,
                file_size=wav_file.stat().st_size,
                format='wav',
                duration=0.0,  # TODO: Extract from ffprobe
                sample_rate=44100,  # Default
                channels=1,  # Default
            )
            
            self.logger.info(
                "Audio download completed successfully",
                audio_file=str(wav_file),
                file_size=audio_metadata.file_size,
                operation="download_audio",
            )
            
            return Ok(audio_metadata)
        
        except Exception as e:
            error_msg = f"Audio download failed: {str(e)}"
            self.logger.error(
                "Audio download failed",
                error=e,
                url=url,
                operation="download_audio",
            )
            return Err(DownloadError(error_msg, {'url': url, 'error_type': type(e).__name__}))
    
    @log_performance
    def download_video(
        self,
        url: str,
        progress_callback: Optional[ProgressCallback] = None
    ) -> Result[Path, DownloadError]:
        """Download Video zu temporärer Datei"""
        
        # URL-Validierung
        url_validation = validate_youtube_url(url)
        if isinstance(url_validation, Err):
            return Err(DownloadError(f"Invalid URL: {url_validation.error.message}"))
        
        try:
            self.logger.info(
                "Starting video download",
                url=url,
                operation="download_video",
            )
            
            # Temporäre Datei für Video
            video_file = self.temp_dir / f"video_{hash(url) % 100000}.%(ext)s"
            
            # Progress Handler
            progress_handler = DownloadProgressHandler(progress_callback)
            
            # yt-dlp Optionen für Video-Download
            ydl_opts = {
                'format': 'best[height<=1080]/best',
                'outtmpl': str(video_file),
                'quiet': True,
                'no_warnings': True,
                'progress_hooks': [progress_handler],
            }
            
            # Download ausführen
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Tatsächliche Datei finden
            video_files = list(self.temp_dir.glob(f"video_{hash(url) % 100000}.*"))
            
            if not video_files:
                raise DownloadError("Video file not found after download")
            
            actual_video_file = video_files[0]
            
            # Datei validieren
            if not actual_video_file.exists() or actual_video_file.stat().st_size < 1000:
                raise DownloadError(f"Invalid video file: {actual_video_file}")
            
            self.logger.info(
                "Video download completed successfully",
                video_file=str(actual_video_file),
                file_size=actual_video_file.stat().st_size,
                operation="download_video",
            )
            
            return Ok(actual_video_file)
        
        except Exception as e:
            error_msg = f"Video download failed: {str(e)}"
            self.logger.error(
                "Video download failed",
                error=e,
                url=url,
                operation="download_video",
            )
            return Err(DownloadError(error_msg, {'url': url, 'error_type': type(e).__name__}))
    
    @log_function_calls
    def get_available_formats(self, url: str) -> Result[Dict[str, Any], DownloadError]:
        """Verfügbare Formate abrufen"""
        
        try:
            self.logger.debug(
                "Fetching available formats",
                url=url,
                operation="get_available_formats",
            )
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
            formats = info.get('formats', [])
            
            # Formatiere Formate für bessere Lesbarkeit
            audio_formats = []
            video_formats = []
            
            for fmt in formats:
                if fmt.get('acodec') != 'none' and fmt.get('vcodec') == 'none':
                    # Nur Audio
                    audio_formats.append({
                        'format_id': fmt.get('format_id'),
                        'ext': fmt.get('ext'),
                        'acodec': fmt.get('acodec'),
                        'abr': fmt.get('abr'),
                        'filesize': fmt.get('filesize'),
                    })
                elif fmt.get('vcodec') != 'none':
                    # Video (mit oder ohne Audio)
                    video_formats.append({
                        'format_id': fmt.get('format_id'),
                        'ext': fmt.get('ext'),
                        'vcodec': fmt.get('vcodec'),
                        'acodec': fmt.get('acodec'),
                        'width': fmt.get('width'),
                        'height': fmt.get('height'),
                        'fps': fmt.get('fps'),
                        'filesize': fmt.get('filesize'),
                    })
            
            formats_info = {
                'audio_formats': audio_formats,
                'video_formats': video_formats,
                'total_formats': len(formats),
            }
            
            self.logger.info(
                "Available formats retrieved",
                total_formats=formats_info['total_formats'],
                audio_formats_count=len(audio_formats),
                video_formats_count=len(video_formats),
            )
            
            return Ok(formats_info)
        
        except Exception as e:
            error_msg = f"Failed to get available formats: {str(e)}"
            self.logger.error(
                "Format retrieval failed",
                error=e,
                url=url,
                operation="get_available_formats",
            )
            return Err(DownloadError(error_msg, {'url': url, 'error_type': type(e).__name__}))
    
    @log_function_calls
    def cleanup(self) -> None:
        """Temporäre Dateien aufräumen"""
        try:
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
                
                self.logger.info(
                    "Cleanup completed",
                    temp_dir=str(self.temp_dir),
                    operation="cleanup",
                )
        
        except Exception as e:
            self.logger.error(
                "Cleanup failed",
                error=e,
                temp_dir=str(self.temp_dir),
                operation="cleanup",
            )
    
    def get_service_info(self) -> Dict[str, Any]:
        """Service-Informationen für Monitoring"""
        return {
            'service_name': 'DownloadService',
            'temp_dir': str(self.temp_dir),
            'temp_dir_exists': self.temp_dir.exists(),
            'ytdlp_version': yt_dlp.version.__version__,
            'status': 'ready',
        }


# =============================================================================
# SERVICE FACTORY
# =============================================================================

_download_service_instance: Optional[DownloadService] = None


def get_download_service() -> DownloadService:
    """Singleton Factory für Download-Service"""
    global _download_service_instance
    
    if _download_service_instance is None:
        _download_service_instance = DownloadService()
    
    return _download_service_instance


def create_download_service(temp_dir: Optional[Path] = None) -> DownloadService:
    """Factory für neuen Download-Service mit spezifischem temp_dir"""
    return DownloadService(temp_dir)


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def test_download_service() -> None:
    """Test-Funktion für Download-Service"""
    from utils.logging import get_development_config
    from utils.logging import setup_logging
    
    # Setup logging für Test
    setup_logging(get_development_config())
    
    # Test-URL (Rick Roll für Stabilität)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    service = get_download_service()
    logger = ComponentLogger("DownloadServiceTest")
    
    logger.info("Starting download service test")
    
    # Test Video Info
    info_result = service.get_video_info(test_url)
    if isinstance(info_result, Ok):
        logger.info("✅ Video info test passed", video_title=info_result.value.title)
    else:
        logger.error("❌ Video info test failed", error=info_result.error.message)
        return
    
    # Test Audio Download
    audio_result = service.download_audio(test_url)
    if isinstance(audio_result, Ok):
        logger.info("✅ Audio download test passed", file_size=audio_result.value.file_size)
    else:
        logger.error("❌ Audio download test failed", error=audio_result.error.message)
        return
    
    # Test Video Download
    video_result = service.download_video(test_url)
    if isinstance(video_result, Ok):
        logger.info("✅ Video download test passed", file_size=video_result.value.stat().st_size)
    else:
        logger.error("❌ Video download test failed", error=video_result.error.message)
        return
    
    # Test Cleanup
    service.cleanup()
    logger.info("✅ All tests passed")


if __name__ == "__main__":
    test_download_service()
