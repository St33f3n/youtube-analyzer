"""
YouTube URL Processor & Metadata Extractor
Verarbeitet Multiline-URL-Input und extrahiert Metadata via yt-dlp
"""

from __future__ import annotations
import re
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from urllib.parse import urlparse, parse_qs
import traceback

# yt-dlp für YouTube-Metadata-Extraktion
import yt_dlp
from yt_dlp import YoutubeDL

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_err, unwrap_ok, unwrap_or
from yt_analyzer_core import ProcessObject
from logging_plus import get_logger, log_feature, log_function

# =============================================================================
# URL VALIDATION & PARSING
# =============================================================================

class YouTubeURLProcessor:
    """Processor für YouTube-URLs mit Validation und Normalisierung"""
    
    def __init__(self):
        self.logger = get_logger("YouTubeURLProcessor")
        
        # YouTube URL-Patterns
        self.youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/channel/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/@([a-zA-Z0-9_.-]+)',
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.youtube_patterns]
    
    def parse_multiline_input(self, text: str) -> Result[List[str], CoreError]:
        """
        Parst Multiline-Text zu Liste von URLs
        
        Args:
            text: Multiline-Text mit URLs
            
        Returns:
            Ok(List[str]): Liste valider YouTube-URLs
            Err: Parse-Fehler
        """
        try:
            # Split lines and clean
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            if not lines:
                context = ErrorContext.create(
                    "parse_multiline_input",
                    input_data={'text_length': len(text)},
                    suggestions=["Enter at least one URL", "Check input format"]
                )
                return Err(CoreError("No URLs found in input", context))
            
            valid_urls = []
            invalid_lines = []
            
            for line_num, line in enumerate(lines, 1):
                url_result = self.validate_youtube_url(line)
                if isinstance(url_result, Ok):
                    valid_urls.append(unwrap_ok(url_result))
                else:
                    invalid_lines.append(f"Line {line_num}: {line}")
            
            if not valid_urls:
                context = ErrorContext.create(
                    "parse_multiline_input",
                    input_data={'total_lines': len(lines), 'invalid_lines': invalid_lines},
                    suggestions=["Check URL format", "Use full YouTube URLs", "Remove invalid lines"]
                )
                return Err(CoreError(f"No valid YouTube URLs found in {len(lines)} lines", context))
            
            if invalid_lines:
                self.logger.warning(
                    f"Skipped {len(invalid_lines)} invalid URLs",
                    extra={
                        'valid_count': len(valid_urls),
                        'invalid_count': len(invalid_lines),
                        'invalid_lines': invalid_lines[:5]  # Log first 5
                    }
                )
            
            self.logger.info(
                f"Parsed {len(valid_urls)} valid URLs from {len(lines)} lines",
                extra={'valid_urls': len(valid_urls), 'total_lines': len(lines)}
            )
            
            return Ok(valid_urls)
            
        except Exception as e:
            context = ErrorContext.create(
                "parse_multiline_input",
                input_data={'text': text[:200]},  # First 200 chars
                suggestions=["Check text encoding", "Verify input format"]
            )
            return Err(CoreError(f"Failed to parse input: {e}", context))
    
    def validate_youtube_url(self, url: str) -> Result[str, CoreError]:
        """
        Validiert und normalisiert YouTube-URL
        
        Args:
            url: URL zu validieren
            
        Returns:
            Ok(str): Normalisierte YouTube-URL
            Err: Validation-Fehler
        """
        try:
            # Remove whitespace
            url = url.strip()
            
            if not url:
                return Err(CoreError("Empty URL"))
            
            # Add https if missing
            if not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
            
            # Check against patterns
            for pattern in self.compiled_patterns:
                if pattern.search(url):
                    # URL matches YouTube pattern
                    return Ok(url)
            
            # Try to parse as generic URL and check domain
            parsed = urlparse(url)
            if parsed.netloc.lower() in ['youtube.com', 'www.youtube.com', 'youtu.be', 'www.youtu.be']:
                # YouTube domain but non-standard format
                return Ok(url)
            
            context = ErrorContext.create(
                "validate_youtube_url",
                input_data={'url': url, 'domain': parsed.netloc},
                suggestions=["Use youtube.com or youtu.be URLs", "Check URL format", "Copy URL from browser"]
            )
            return Err(CoreError(f"Not a valid YouTube URL: {url}", context))
            
        except Exception as e:
            context = ErrorContext.create(
                "validate_youtube_url",
                input_data={'url': url},
                suggestions=["Check URL format", "Remove special characters"]
            )
            return Err(CoreError(f"URL validation failed: {e}", context))
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extrahiert Video-ID aus YouTube-URL"""
        patterns = [
            r'(?:v=|/)([a-zA-Z0-9_-]{11})',
            r'youtu\.be/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def is_playlist_url(self, url: str) -> bool:
        """Prüft ob URL eine Playlist ist"""
        return 'playlist?list=' in url or '/playlist?' in url

# =============================================================================
# METADATA EXTRACTOR mit yt-dlp
# =============================================================================

class YouTubeMetadataExtractor:
    """Extrahiert Metadata von YouTube-Videos via yt-dlp"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger("YouTubeMetadataExtractor")
        self.config = config or {}
        
        # yt-dlp Konfiguration
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,  # Full metadata extraction
            'writesubtitles': False,
            'writeautomaticsub': False,
            'writedescription': False,
            'writeinfojson': False,
            'writethumbnail': False,
            'socket_timeout': 30,
            'retries': 0,  # Manual retry handling
        }
        
        # Progress tracking für Timeout-Detection
        self.last_progress_time = time.time()
        self.progress_timeout = 60  # 60s without progress = timeout
    
    def progress_hook(self, d: Dict[str, Any]) -> None:
        """Hook für yt-dlp Progress-Tracking"""
        if d['status'] in ['downloading', 'extracting']:
            self.last_progress_time = time.time()
    
    @log_function(log_performance=True)
    def extract_batch_metadata(self, urls: List[str]) -> Result[List[ProcessObject], List[CoreError]]:
        """
        Extrahiert Metadata für URL-Batch mit Playlist-Expansion
        
        Args:
            urls: Liste von YouTube-URLs (kann Playlists enthalten)
            
        Returns:
            Ok(List[ProcessObject]): Erfolgreich extrahierte Videos
            Err(List[CoreError]): Sammlung aller Fehler
        """
        with log_feature("batch_metadata_extraction") as feature:
            all_objects = []
            all_errors = []
            
            feature.add_metric("input_urls", len(urls))
            
            for url in urls:
                extract_result = self.extract_single_or_playlist(url)
                
                if isinstance(extract_result, Ok):
                    objects = unwrap_ok(extract_result)
                    all_objects.extend(objects)
                    feature.add_metric(f"success_{url[:20]}", len(objects))
                else:
                    error = unwrap_err(extract_result)
                    all_errors.append(error)
                    feature.add_metric(f"error_{url[:20]}", 1)
            
            feature.add_metric("total_extracted", len(all_objects))
            feature.add_metric("total_errors", len(all_errors))
            
            if all_objects:
                # Enhanced batch logging mit Metadata-Details
                self.logger.info(
                    f"🎯 Batch extraction completed successfully",
                    extra={
                        'total_videos_extracted': len(all_objects),
                        'total_errors': len(all_errors),
                        'success_rate': round(len(all_objects) / (len(all_objects) + len(all_errors)) * 100, 1),
                        'video_titles': [obj.titel[:50] for obj in all_objects[:3]],  # First 3 titles
                        'channels': list(set([obj.kanal for obj in all_objects])),
                        'total_duration_minutes': sum(
                            (obj.länge.hour * 60 + obj.länge.minute + obj.länge.second / 60) 
                            for obj in all_objects
                        ),
                        'input_urls_count': len(urls)
                    }
                )
                return Ok(all_objects)
            else:
                return Err(all_errors)
    
    def extract_single_or_playlist(self, url: str) -> Result[List[ProcessObject], CoreError]:
        """Extrahiert Metadata für einzelne URL oder Playlist"""
        url_processor = YouTubeURLProcessor()
        
        if url_processor.is_playlist_url(url):
            return self.extract_playlist_metadata(url)
        else:
            single_result = self.extract_single_metadata(url)
            if isinstance(single_result, Ok):
                return Ok([unwrap_ok(single_result)])
            else:
                return single_result
    
    def extract_playlist_metadata(self, playlist_url: str) -> Result[List[ProcessObject], CoreError]:
        """Extrahiert alle Videos aus Playlist"""
        try:
            with log_feature("playlist_extraction") as feature:
                self.logger.info(f"Extracting playlist: {playlist_url}")
                
                # First pass: Extract playlist info
                playlist_opts = self.ydl_opts.copy()
                playlist_opts['extract_flat'] = True  # Only get video URLs
                
                with YoutubeDL(playlist_opts) as ydl:
                    playlist_info = ydl.extract_info(playlist_url, download=False)
                
                if not playlist_info or 'entries' not in playlist_info:
                    context = ErrorContext.create(
                        "extract_playlist_metadata",
                        input_data={'playlist_url': playlist_url},
                        suggestions=["Check playlist URL", "Verify playlist is public"]
                    )
                    return Err(CoreError(f"No entries found in playlist: {playlist_url}", context))
                
                entries = playlist_info['entries']
                if not entries:
                    return Err(CoreError(f"Empty playlist: {playlist_url}"))
                
                feature.add_metric("playlist_entries", len(entries))
                
                # Second pass: Extract metadata for each video
                extracted_objects = []
                
                for i, entry in enumerate(entries):
                    if not entry or 'url' not in entry:
                        continue
                    
                    video_url = entry['url']
                    single_result = self.extract_single_metadata(video_url)
                    
                    if isinstance(single_result, Ok):
                        extracted_objects.append(unwrap_ok(single_result))
                    else:
                        self.logger.warning(f"Failed to extract video {i+1}/{len(entries)}: {video_url}")
                
                feature.add_metric("extracted_videos", len(extracted_objects))
                
                if extracted_objects:
                    # Enhanced playlist logging
                    self.logger.info(
                        f"🎵 Playlist extraction completed successfully",
                        extra={
                            'playlist_url': playlist_url,
                            'playlist_title': playlist_info.get('title', 'Unknown Playlist'),
                            'playlist_uploader': playlist_info.get('uploader', 'Unknown'),
                            'total_entries': len(entries),
                            'successfully_extracted': len(extracted_objects),
                            'extraction_rate': round(len(extracted_objects) / len(entries) * 100, 1),
                            'video_titles': [obj.titel[:30] for obj in extracted_objects[:5]],  # First 5 titles
                            'unique_channels': len(set([obj.kanal for obj in extracted_objects])),
                            'total_playlist_duration': sum(
                                (obj.länge.hour * 60 + obj.länge.minute + obj.länge.second / 60) 
                                for obj in extracted_objects
                            )
                        }
                    )
                    return Ok(extracted_objects)
                else:
                    context = ErrorContext.create(
                        "extract_playlist_metadata",
                        input_data={'playlist_url': playlist_url, 'total_entries': len(entries)},
                        suggestions=["Check video availability", "Verify playlist permissions"]
                    )
                    return Err(CoreError(f"No videos could be extracted from playlist", context))
        
        except Exception as e:
            context = ErrorContext.create(
                "extract_playlist_metadata",
                input_data={'playlist_url': playlist_url},
                suggestions=["Check playlist URL", "Verify network connection"]
            )
            return Err(CoreError(f"Playlist extraction failed: {e}", context))
    
    def extract_single_metadata(self, url: str) -> Result[ProcessObject, CoreError]:
        """Extrahiert Metadata für einzelnes Video mit Retry-Logic"""
        
        for attempt in range(3):  # 3 retry attempts
            try:
                with log_feature(f"video_metadata_extraction_attempt_{attempt + 1}") as feature:
                    feature.add_metric("url", url)
                    feature.add_metric("attempt", attempt + 1)
                    
                    self.last_progress_time = time.time()
                    
                    # yt-dlp mit Progress-Hook
                    ydl_opts = self.ydl_opts.copy()
                    ydl_opts['progress_hooks'] = [self.progress_hook]
                    
                    with YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                    
                    if not info:
                        raise ValueError("No video information extracted")
                    
                    # Create ProcessObject from extracted info
                    process_obj = self.create_process_object_from_info(info, url)
                    
                    feature.add_metric("success", True)
                    feature.add_metric("duration_seconds", info.get('duration', 0))
                    
                    # Enhanced metadata logging - explicit format
                    metadata_info = (
                        f"✅ Metadata extracted successfully:\n"
                        f"  📹 Title: {process_obj.titel}\n"
                        f"  📺 Channel: {process_obj.kanal}\n"
                        f"  ⏱️ Duration: {process_obj.länge}\n"
                        f"  📅 Upload Date: {process_obj.upload_date.strftime('%Y-%m-%d')}\n"
                        f"  🔗 Video ID: {info.get('id', 'unknown')}\n"
                        f"  👀 Views: {info.get('view_count', 'unknown'):,}\n"
                        f"  👍 Likes: {info.get('like_count', 'unknown'):,}\n"
                        f"  🔄 Attempt: {attempt + 1}"
                    )
                    
                    self.logger.info(metadata_info)
                    
                    return Ok(process_obj)
            
            except yt_dlp.DownloadError as e:
                error_msg = str(e)
                
                # Check for retryable errors
                if attempt < 2 and any(keyword in error_msg.lower() for keyword in ['network', 'timeout', 'connection', 'temporary']):
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                    self.logger.warning(f"Network error on attempt {attempt + 1}, retrying in {wait_time}s: {error_msg}")
                    time.sleep(wait_time)
                    continue
                
                # Non-retryable or final attempt
                context = ErrorContext.create(
                    "extract_single_metadata",
                    input_data={'url': url, 'attempt': attempt + 1, 'error': error_msg},
                    suggestions=["Check video availability", "Verify URL is correct", "Check if video is private/deleted"]
                )
                return Err(CoreError(f"Video extraction failed: {error_msg}", context))
            
            except Exception as e:
                if attempt < 2:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Unexpected error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                
                context = ErrorContext.create(
                    "extract_single_metadata",
                    input_data={'url': url, 'attempt': attempt + 1},
                    suggestions=["Check URL format", "Verify network connection", "Check yt-dlp installation"]
                )
                return Err(CoreError(f"Metadata extraction failed: {e}", context))
        
        # Should never reach here, but just in case
        return Err(CoreError(f"Metadata extraction failed after 3 attempts"))
    
    def create_process_object_from_info(self, info: Dict[str, Any], original_url: str) -> ProcessObject:
        """Erstellt ProcessObject aus yt-dlp info dict"""
        
        # Extract basic info
        title = info.get('title', 'Unknown Title')
        uploader = info.get('uploader', info.get('channel', 'Unknown Channel'))
        duration_seconds = info.get('duration', 0)
        upload_date_str = info.get('upload_date', '')
        
        # Parse duration
        if duration_seconds and duration_seconds > 0:
            hours = duration_seconds // 3600
            minutes = (duration_seconds % 3600) // 60
            seconds = duration_seconds % 60
            länge = dt_time(hours, minutes, seconds)
        else:
            länge = dt_time(0, 0, 0)
        
        # Parse upload date
        upload_date = datetime.now()  # Default fallback
        if upload_date_str:
            try:
                # yt-dlp format: YYYYMMDD
                upload_date = datetime.strptime(upload_date_str, '%Y%m%d')
            except ValueError:
                try:
                    # Alternative format
                    upload_date = datetime.strptime(upload_date_str[:10], '%Y-%m-%d')
                except ValueError:
                    self.logger.warning(f"Could not parse upload date: {upload_date_str}")
        
        # Create ProcessObject
        process_obj = ProcessObject(
            titel=title,
            kanal=uploader,
            länge=länge,
            upload_date=upload_date,
            original_url=original_url  # Store original URL for downloads
        )
        
        # Add additional metadata
        process_obj.update_stage("metadata_extracted")
        
        return process_obj

# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def process_urls_to_objects(urls_text: str, config: Optional[Dict[str, Any]] = None) -> Result[List[ProcessObject], List[CoreError]]:
    """
    Complete Pipeline: URLs Text → ProcessObjects mit Metadata
    
    Args:
        urls_text: Multiline-Text mit YouTube-URLs
        config: Optional configuration
        
    Returns:
        Ok(List[ProcessObject]): Fertige ProcessObjects mit Metadata
        Err(List[CoreError]): Sammlung aller Fehler
    """
    with log_feature("urls_to_objects_pipeline") as feature:
        # Step 1: Parse URLs
        url_processor = YouTubeURLProcessor()
        urls_result = url_processor.parse_multiline_input(urls_text)
        
        if isinstance(urls_result, Err):
            return Err([unwrap_err(urls_result)])
        
        urls = unwrap_ok(urls_result)
        feature.add_metric("parsed_urls", len(urls))
        
        # Step 2: Extract Metadata
        extractor = YouTubeMetadataExtractor(config)
        objects_result = extractor.extract_batch_metadata(urls)
        
        if isinstance(objects_result, Ok):
            objects = unwrap_ok(objects_result)
            feature.add_metric("final_objects", len(objects))
            
            feature.add_metric("success_rate", len(objects) / len(urls) * 100 if urls else 0)
            
            return Ok(objects)
        else:
            errors = unwrap_err(objects_result)
            feature.add_metric("total_errors", len(errors))
            return Err(errors)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    
    # Setup
    setup_logging("url_processor_test", "DEBUG")
    
    # Test URL processing
    test_input = """
    https://www.youtube.com/watch?v=dQw4w9WgXcQ
    https://youtu.be/jNQXAC9IVRw
    https://www.youtube.com/playlist?list=PLqwT4lKYke4xTY1HdHGIHy4NoBOJtDdRi
    not a valid url
    https://www.youtube.com/watch?v=invalid
    """
    
    result = process_urls_to_objects(test_input)
    
    if isinstance(result, Ok):
        objects = unwrap_ok(result)
        print(f"✅ Successfully processed {len(objects)} videos:")
        for obj in objects:
            print(f"  - {obj.titel} by {obj.kanal} ({obj.länge})")
    else:
        errors = unwrap_err(result)
        print(f"❌ Processing failed with {len(errors)} errors:")
        for error in errors:
            print(f"  - {error.message}")
