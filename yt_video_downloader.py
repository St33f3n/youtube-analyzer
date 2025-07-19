"""
YouTube Video Downloader with yt-dlp
Lädt Video-Dateien von YouTube-Videos für Bibliotheks-Aufbau
"""

from __future__ import annotations
import os
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import threading

# yt-dlp für Video-Download
import yt_dlp
from yt_dlp import YoutubeDL

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_err, unwrap_ok
from yt_analyzer_core import ProcessObject
from logging_plus import get_logger, log_feature, log_function
from yt_analyzer_config import AppConfig

# =============================================================================
# VIDEO DOWNLOAD ENGINE
# =============================================================================

class YouTubeVideoDownloader:
    """yt-dlp Video-Downloader mit Progress-Tracking und Error-Handling"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger("YouTubeVideoDownloader")
        
        # Ensure temp directory exists
        self.temp_dir = Path(self.config.processing.temp_folder)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.current_download_progress = {}
        self.progress_lock = threading.Lock()
        
        # Download statistics
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_size_mb': 0.0,
            'total_download_time_seconds': 0.0
        }
    
    def progress_hook(self, d: Dict[str, Any]) -> None:
        """Progress-Hook für yt-dlp Video-Downloads"""
        with self.progress_lock:
            if d['status'] == 'downloading':
                # Update progress info
                filename = d.get('filename', 'unknown')
                downloaded = d.get('downloaded_bytes', 0)
                total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
                
                self.current_download_progress[filename] = {
                    'downloaded_bytes': downloaded,
                    'total_bytes': total,
                    'speed': d.get('speed', 0),
                    'eta': d.get('eta', 0),
                    'percent': (downloaded / total * 100) if total > 0 else 0
                }
            
            elif d['status'] == 'finished':
                filename = d.get('filename', 'unknown')
                file_size = d.get('total_bytes', 0)
                
                # Update stats
                self.download_stats['successful_downloads'] += 1
                self.download_stats['total_size_mb'] += file_size / (1024 * 1024)
                
                self.logger.debug(
                    f"Video download completed: {Path(filename).name}",
                    extra={
                        'filename': Path(filename).name,
                        'file_size_mb': round(file_size / (1024 * 1024), 2),
                        'total_downloads': self.download_stats['successful_downloads']
                    }
                )
                
                # Clean up progress tracking
                if filename in self.current_download_progress:
                    del self.current_download_progress[filename]
    
    @log_function(log_performance=True)
    def download_video(self, process_obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        """
        Lädt Video-Datei für ProcessObject
        
        Args:
            process_obj: ProcessObject mit original_url und passed_analysis=True
            
        Returns:
            Ok(ProcessObject): ProcessObject mit temp_video_path
            Err: Download-Fehler
        """
        # Validate ProcessObject requirements
        if not process_obj.original_url:
            context = ErrorContext.create(
                "download_video",
                input_data={'video_title': process_obj.titel},
                suggestions=["Check metadata extraction", "Verify original_url is set"]
            )
            return Err(CoreError("No original URL in ProcessObject", context))
        
        if not process_obj.passed_analysis:
            context = ErrorContext.create(
                "download_video", 
                input_data={'video_title': process_obj.titel, 'passed_analysis': process_obj.passed_analysis},
                suggestions=["Only download videos that passed analysis", "Check analysis pipeline"]
            )
            return Err(CoreError("Video did not pass analysis - should not be downloaded", context))
        
        try:
            with log_feature("video_download") as feature:
                # Extract config values with fallbacks
                video_format = getattr(self.config.processing, 'video_format', 'mp4')
                if isinstance(video_format, dict):
                    self.logger.warning(f"video_format is dict, using 'mp4': {video_format}")
                    video_format = 'mp4'
                
                temp_folder = getattr(self.config.processing, 'temp_folder', '/tmp/youtube_analyzer')
                if isinstance(temp_folder, dict):
                    self.logger.warning(f"temp_folder is dict, using default: {temp_folder}")
                    temp_folder = '/tmp/youtube_analyzer'
                
                feature.add_metric("video_title", process_obj.titel)
                feature.add_metric("video_url", process_obj.original_url)
                feature.add_metric("video_format", video_format)
                
                # Generate safe filename
                safe_title = self._sanitize_filename(process_obj.titel)
                video_filename = f"video_{safe_title}_{int(time.time())}.{video_format}"
                video_path = Path(temp_folder) / video_filename
                
                self.logger.info(
                    f"🎬 Starting video download",
                    extra={
                        'video_title': process_obj.titel,
                        'channel': process_obj.kanal,
                        'video_url': process_obj.original_url,
                        'output_path': str(video_path),
                        'video_format': video_format,
                        'temp_dir': str(temp_folder),
                        'analysis_score': process_obj.relevancy,
                        'passed_analysis': process_obj.passed_analysis
                    }
                )
                
                # yt-dlp options for highest quality video download
                simple_filename = f"video_{int(time.time())}"
                output_path = Path(temp_folder) / f"{simple_filename}.%(ext)s"
                
                ydl_opts = {
                    # HIGHEST QUALITY: No limitations
                    'format': 'best',  # Highest available quality
                    'merge_output_format': video_format,  # Unified format
                    'outtmpl': str(output_path),
                    
                    # NO EXTRAS: Only video file
                    'writesubtitles': False,
                    'writeautomaticsub': False, 
                    'writeinfojson': False,
                    'writethumbnail': False,
                    'writedescription': False,
                    
                    # Technical settings
                    'quiet': False,
                    'prefer_ffmpeg': True,
                    'progress_hooks': [self.progress_hook],
                    'socket_timeout': 60,
                }
                
                # Debug yt-dlp options
                self.logger.info(
                    f"yt-dlp options for highest quality",
                    extra={
                        'outtmpl': ydl_opts['outtmpl'],
                        'format': ydl_opts['format'],
                        'merge_output_format': ydl_opts['merge_output_format'],
                        'temp_folder': temp_folder,
                        'simple_filename': simple_filename
                    }
                )
                
                # Download with 2-retry logic
                download_result = self._download_with_retries(process_obj.original_url, ydl_opts, max_retries=2)
                
                if isinstance(download_result, Err):
                    return download_result
                
                actual_path = unwrap_ok(download_result)
                
                # Verify file exists and has content
                if not actual_path.exists():
                    context = ErrorContext.create(
                        "download_video",
                        input_data={
                            'video_title': process_obj.titel,
                            'expected_path': str(actual_path)
                        },
                        suggestions=["Check disk space", "Verify yt-dlp installation", "Check network connectivity"]
                    )
                    return Err(CoreError(f"Downloaded video file not found: {actual_path}", context))
                
                file_size = actual_path.stat().st_size
                if file_size < 100000:  # Less than 100KB probably indicates failure
                    context = ErrorContext.create(
                        "download_video",
                        input_data={
                            'video_title': process_obj.titel,
                            'file_size': file_size,
                            'file_path': str(actual_path)
                        },
                        suggestions=["Check video availability", "Verify video stream exists", "Check yt-dlp version"]
                    )
                    return Err(CoreError(f"Downloaded video file too small ({file_size} bytes)", context))
                
                # Update ProcessObject
                process_obj.temp_video_path = actual_path
                process_obj.update_stage("video_downloaded")
                
                # Optional: Add video metadata
                process_obj.video_size_mb = round(file_size / (1024 * 1024), 2)
                
                # Update statistics
                self.download_stats['total_downloads'] += 1
                
                feature.add_metric("download_success", True)
                feature.add_metric("file_size_mb", round(file_size / (1024 * 1024), 2))
                feature.add_metric("output_path", str(actual_path))
                
                # Enhanced download completion logging
                download_info = (
                    f"✅ Video download completed successfully:\n"
                    f"  🎬 Video: {process_obj.titel}\n"
                    f"  📺 Channel: {process_obj.kanal}\n"
                    f"  📁 File: {actual_path.name}\n"
                    f"  💾 Size: {file_size / (1024 * 1024):.1f} MB\n"
                    f"  🎚️ Format: {video_format}\n"
                    f"  📊 Analysis Score: {process_obj.relevancy:.3f}\n"
                    f"  📈 Total downloads: {self.download_stats['total_downloads']}"
                )
                
                self.logger.info(download_info)
                
                return Ok(process_obj)
                
        except Exception as e:
            self.download_stats['failed_downloads'] += 1
            
            context = ErrorContext.create(
                "download_video",
                input_data={
                    'video_title': process_obj.titel,
                    'video_url': process_obj.original_url,
                    'error_type': type(e).__name__
                },
                suggestions=[
                    "Check network connectivity",
                    "Verify video is available",
                    "Check disk space in temp directory", 
                    "Verify yt-dlp and ffmpeg installation"
                ]
            )
            return Err(CoreError(f"Video download failed: {e}", context))
    
    def _download_with_retries(self, url: str, ydl_opts: Dict[str, Any], max_retries: int = 2) -> Result[Path, CoreError]:
        """Video download with 2-retry logic"""
        
        temp_folder = Path(ydl_opts['outtmpl']).parent
        expected_base_name = Path(ydl_opts['outtmpl']).stem  # e.g., "video_1752803693"
        
        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Video download attempt {attempt + 1}/{max_retries}",
                    extra={
                        'url': url,
                        'attempt': attempt + 1,
                        'temp_folder': str(temp_folder),
                        'expected_base_name': expected_base_name
                    }
                )
                
                start_time = time.time()
                
                with YoutubeDL(ydl_opts) as ydl:
                    # Download
                    ydl.download([url])
                
                download_time = time.time() - start_time
                self.download_stats['total_download_time_seconds'] += download_time
                
                # Find the exact file - video extensions
                video_extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.m4v']
                
                # First try: exact expected filename with different extensions
                for ext in video_extensions:
                    expected_file = temp_folder / f"{expected_base_name}{ext}"
                    if expected_file.exists():
                        self.logger.info(f"✅ Found downloaded video file (exact match): {expected_file}")
                        return Ok(expected_file)
                
                # Second try: list all files and find our video file
                self.logger.debug(f"Exact match not found, listing all files in {temp_folder}")
                
                all_files = list(temp_folder.glob('*'))
                self.logger.debug(f"All files in temp folder: {[f.name for f in all_files]}")
                
                # Look for files that start with our expected base name
                for file_path in all_files:
                    if (file_path.name.startswith(expected_base_name) and 
                        file_path.suffix.lower() in video_extensions):
                        self.logger.info(f"✅ Found downloaded video file (prefix match): {file_path}")
                        return Ok(file_path)
                
                # Third try: newest video file (fallback)
                
                current_time = time.time()
                newest_file = None
                newest_time = 0
                
                for file_path in all_files:
                    if (file_path.suffix.lower() in video_extensions and 
                        file_path.stat().st_mtime > current_time - 600):  # 10 minutes window
                        
                        if file_path.stat().st_mtime > newest_time:
                            newest_time = file_path.stat().st_mtime
                            newest_file = file_path
                
                if newest_file:
                    self.logger.info(f"✅ Found downloaded video file (newest): {newest_file}")
                    return Ok(newest_file)
                
                # Log debugging info
                self.logger.error(
                    f"No video file found after download",
                    extra={
                        'temp_folder': str(temp_folder),
                        'expected_base_name': expected_base_name,
                        'all_files': [f.name for f in all_files],
                        'video_files': [f.name for f in all_files if f.suffix.lower() in video_extensions],
                        'recent_files': [f.name for f in all_files if f.stat().st_mtime > current_time - 600]
                    }
                )
                
                raise FileNotFoundError(f"Downloaded video file not found. Expected: {expected_base_name}.*")
            
            except yt_dlp.DownloadError as e:
                self.logger.warning(f"yt-dlp error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    time.sleep(wait_time)
                    continue
                
                return Err(CoreError(f"yt-dlp video download failed after {max_retries} attempts: {e}"))
            
            except Exception as e:
                self.logger.error(
                    f"Video download attempt {attempt + 1} failed: {e}",
                    extra={
                        'error_type': type(e).__name__,
                        'expected_base_name': expected_base_name,
                        'temp_folder': str(temp_folder)
                    }
                )
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    time.sleep(wait_time)
                    continue
                
                return Err(CoreError(f"Video download failed: {e}"))
        
        return Err(CoreError("Video download failed after all attempts"))
    
    def _sanitize_filename(self, filename: str) -> str:
        """Bereinigt Filename für sicheres Dateisystem"""
        # Remove or replace problematic characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. "
        sanitized = "".join(c if c in safe_chars else "_" for c in filename)
        
        # Limit length and remove extra spaces
        sanitized = sanitized.strip()[:50]
        
        # Ensure not empty
        if not sanitized:
            sanitized = "unnamed_video"
        
        return sanitized
    
    def cleanup_temp_files(self, process_obj: ProcessObject) -> Result[None, CoreError]:
        """Bereinigt temporäre Video-Dateien nach Verarbeitung"""
        if not process_obj.temp_video_path:
            return Ok(None)
        
        try:
            if process_obj.temp_video_path.exists():
                file_size = process_obj.temp_video_path.stat().st_size
                process_obj.temp_video_path.unlink()
                
                self.logger.debug(
                    f"Cleaned up temp video file",
                    extra={
                        'video_title': process_obj.titel,
                        'file_path': str(process_obj.temp_video_path),
                        'file_size_mb': round(file_size / (1024 * 1024), 2)
                    }
                )
            
            return Ok(None)
            
        except Exception as e:
            # Non-critical error - log but don't fail
            self.logger.warning(f"Failed to cleanup temp video file {process_obj.temp_video_path}: {e}")
            return Ok(None)  # Return success anyway
    
    def get_download_statistics(self) -> Dict[str, Any]:
        """Gibt Video-Download-Statistiken zurück"""
        return {
            **self.download_stats,
            'success_rate': (
                self.download_stats['successful_downloads'] / self.download_stats['total_downloads'] * 100
                if self.download_stats['total_downloads'] > 0 else 0
            ),
            'average_size_mb': (
                self.download_stats['total_size_mb'] / self.download_stats['successful_downloads']
                if self.download_stats['successful_downloads'] > 0 else 0
            ),
            'average_download_time_seconds': (
                self.download_stats['total_download_time_seconds'] / self.download_stats['successful_downloads']
                if self.download_stats['successful_downloads'] > 0 else 0
            )
        }

# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def download_video_for_process_object(process_obj: ProcessObject, config: AppConfig) -> Result[ProcessObject, CoreError]:
    """
    Standalone-Funktion für ProcessObject-Video-Download
    
    Args:
        process_obj: ProcessObject mit original_url und passed_analysis=True
        config: App-Konfiguration
        
    Returns:
        Ok(ProcessObject): ProcessObject mit temp_video_path
        Err: Download-Fehler
    """
    downloader = YouTubeVideoDownloader(config)
    return downloader.download_video(process_obj)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    from yt_analyzer_config import SecureConfigManager
    from datetime import datetime, time as dt_time
    
    # Setup
    setup_logging("video_downloader_test", "DEBUG")
    
    # Test configuration
    config_manager = SecureConfigManager()
    config_result = config_manager.load_config()
    
    if isinstance(config_result, Ok):
        config = unwrap_ok(config_result)
        
        # Create test ProcessObject (with passed analysis)
        test_obj = ProcessObject(
            titel="Test Video",
            kanal="Test Channel",
            länge=dt_time(0, 5, 30),
            upload_date=datetime.now(),
            original_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll for testing
        )
        
        # Mock analysis results (required for video download)
        test_obj.passed_analysis = True
        test_obj.relevancy = 0.85
        test_obj.rule_amount = 3
        test_obj.rule_accuracy = 0.9
        test_obj.update_stage("analyzed")
        
        # Test video download
        result = download_video_for_process_object(test_obj, config)
        
        if isinstance(result, Ok):
            downloaded_obj = unwrap_ok(result)
            print(f"✅ Video download successful:")
            print(f"  Video path: {downloaded_obj.temp_video_path}")
            print(f"  File exists: {downloaded_obj.temp_video_path.exists()}")
            if downloaded_obj.temp_video_path.exists():
                print(f"  File size: {downloaded_obj.temp_video_path.stat().st_size / (1024*1024):.1f} MB")
                print(f"  Video size metadata: {getattr(downloaded_obj, 'video_size_mb', 'N/A')} MB")
        else:
            error = unwrap_err(result)
            print(f"❌ Video download failed: {error.message}")
    else:
        print("❌ Config loading failed")
