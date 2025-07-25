"""
YouTube URL Processor & Metadata Extractor
Verarbeitet Multiline-URL-Input und extrahiert Metadata via yt-dlp (OS-Version)
"""

from __future__ import annotations
import re
import time
import os
import subprocess
import json
from datetime import datetime, time as dt_time
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse
from pathlib import Path

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_err, unwrap_ok
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
            r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
            r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
            r"(?:https?://)?(?:www\.)?youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)",
            r"(?:https?://)?(?:www\.)?youtube\.com/channel/([a-zA-Z0-9_-]+)",
            r"(?:https?://)?(?:www\.)?youtube\.com/@([a-zA-Z0-9_.-]+)",
        ]

        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.youtube_patterns
        ]

    @log_function(log_performance=True)
    def parse_multiline_input(self, text: str) -> Result[List[str], CoreError]:
        """
        Parst Multiline-Text zu Liste von URLs

        Args:
            text: Multiline-Text mit URLs

        Returns:
            Ok(List[str]): Liste valider YouTube-URLs
            Err(CoreError): Parse-Fehler
        """
        try:
            with log_feature("parse_multiline_input") as feature:
                # Split lines and clean
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                feature.add_metric("input_lines", len(lines))

                if not lines:
                    context = ErrorContext.create(
                        "parse_multiline_input",
                        input_data={"text_length": len(text)},
                        suggestions=[
                            "Enter at least one URL", 
                            "Check input format",
                            "Ensure URLs are separated by newlines"
                        ],
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

                feature.add_metric("valid_urls", len(valid_urls))
                feature.add_metric("invalid_urls", len(invalid_lines))

                if not valid_urls:
                    context = ErrorContext.create(
                        "parse_multiline_input",
                        input_data={
                            "total_lines": len(lines),
                            "invalid_lines": invalid_lines[:3],  # First 3 for brevity
                        },
                        suggestions=[
                            "Check URL format (youtube.com or youtu.be)",
                            "Use full YouTube URLs",
                            "Remove invalid lines",
                            "Copy URLs directly from browser",
                        ],
                    )
                    return Err(
                        CoreError(
                            f"No valid YouTube URLs found in {len(lines)} lines", context
                        )
                    )

                if invalid_lines:
                    self.logger.warning(
                        f"Skipped {len(invalid_lines)} invalid URLs",
                        extra={
                            "valid_count": len(valid_urls),
                            "invalid_count": len(invalid_lines),
                            "invalid_lines": invalid_lines[:5],  # Log first 5
                            "success_rate": round(len(valid_urls) / len(lines) * 100, 1),
                        },
                    )

                feature.add_metric("success_rate", len(valid_urls) / len(lines) * 100)

                self.logger.info(
                    f"✅ URL parsing completed: {len(valid_urls)}/{len(lines)} valid",
                    extra={
                        "valid_urls": len(valid_urls), 
                        "total_lines": len(lines),
                        "success_rate": round(len(valid_urls) / len(lines) * 100, 1),
                    },
                )

                return Ok(valid_urls)

        except Exception as e:
            context = ErrorContext.create(
                "parse_multiline_input",
                input_data={"text_preview": text[:200]},  # First 200 chars
                suggestions=[
                    "Check text encoding", 
                    "Verify input format",
                    "Remove special characters from URLs",
                ],
            )
            return Err(CoreError(f"Failed to parse input: {e}", context))

    def validate_youtube_url(self, url: str) -> Result[str, CoreError]:
        """
        Validiert und normalisiert YouTube-URL

        Args:
            url: URL zu validieren

        Returns:
            Ok(str): Normalisierte YouTube-URL
            Err(CoreError): Validation-Fehler
        """
        try:
            # Remove whitespace
            url = url.strip()

            if not url:
                context = ErrorContext.create(
                    "validate_youtube_url",
                    input_data={"url": url},
                    suggestions=["Provide non-empty URL"],
                )
                return Err(CoreError("Empty URL", context))

            # Add https if missing
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"

            # Check against patterns
            for pattern in self.compiled_patterns:
                if pattern.search(url):
                    # URL matches YouTube pattern
                    return Ok(url)

            # Try to parse as generic URL and check domain
            parsed = urlparse(url)
            if parsed.netloc.lower() in [
                "youtube.com",
                "www.youtube.com",
                "youtu.be",
                "www.youtu.be",
            ]:
                # YouTube domain but non-standard format
                return Ok(url)

            context = ErrorContext.create(
                "validate_youtube_url",
                input_data={"url": url, "domain": parsed.netloc},
                suggestions=[
                    "Use youtube.com or youtu.be URLs",
                    "Check URL format (should contain video ID or playlist ID)",
                    "Copy URL directly from browser address bar",
                    "Ensure URL is complete and not truncated",
                ],
            )
            return Err(CoreError(f"Not a valid YouTube URL: {url}", context))

        except Exception as e:
            context = ErrorContext.create(
                "validate_youtube_url",
                input_data={"url": url},
                suggestions=[
                    "Check URL format", 
                    "Remove special characters",
                    "Verify URL is not corrupted",
                ],
            )
            return Err(CoreError(f"URL validation failed: {e}", context))

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extrahiert Video-ID aus YouTube-URL"""
        patterns = [r"(?:v=|/)([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})"]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def is_playlist_url(self, url: str) -> bool:
        """Prüft ob URL eine Playlist ist"""
        return "playlist?list=" in url or "/playlist?" in url


# =============================================================================
# METADATA EXTRACTOR mit yt-dlp (OS-Version)
# =============================================================================


class YouTubeMetadataExtractor:
    """Extrahiert Metadata von YouTube-Videos via OS yt-dlp"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger("YouTubeMetadataExtractor")
        self.config = config or {}

        # Cookie-Datei finden
        self.cookie_file = self._find_cookie_file()
        
        if self.cookie_file:
            self.logger.info(f"Using cookie file: {self.cookie_file}")
        else:
            self.logger.info("Using Firefox browser cookies")

    def _find_cookie_file(self) -> Optional[str]:
        """Sucht nach cookies.txt Datei"""
        cookie_paths = [
        #    os.path.expanduser("data/yt_analyzer_cookies.txt"),  
        ]
        
        for path in cookie_paths:
            if os.path.exists(path):
                return path
        return None

    def _extract_metadata_with_os_ytdlp(self, url: str) -> Result[Dict[str, Any], CoreError]:
        """
        OS-Version von yt-dlp für Metadata-Extraktion mit strukturiertem Error-Handling
        """
        try:
            # yt-dlp Kommando zusammenbauen
            cmd = [
                "yt-dlp", 
                "--dump-json",
                "--no-download",
                "--no-warnings",
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
                timeout=60,
                check=True
            )
            
            # JSON parsen
            if not result.stdout.strip():
                context = ErrorContext.create(
                    "extract_metadata_os_ytdlp",
                    input_data={"url": url, "stdout_empty": True},
                    suggestions=[
                        "Check if video is available",
                        "Verify URL format",
                        "Check if video is private or deleted",
                    ],
                )
                return Err(CoreError("yt-dlp returned empty output", context))

            metadata = json.loads(result.stdout.strip())
            return Ok(metadata)
            
        except subprocess.TimeoutExpired:
            context = ErrorContext.create(
                "extract_metadata_os_ytdlp",
                input_data={"url": url, "timeout": 60},
                suggestions=[
                    "Check network connectivity",
                    "Try again later - YouTube may be throttling",
                    "Verify video is accessible",
                ],
            )
            return Err(CoreError(f"yt-dlp timeout after 60s for URL: {url}", context))
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            context = ErrorContext.create(
                "extract_metadata_os_ytdlp",
                input_data={
                    "url": url, 
                    "return_code": e.returncode,
                    "stderr": error_msg[:200],
                },
                suggestions=[
                    "Check if video exists and is public",
                    "Verify yt-dlp is up to date: pip install -U yt-dlp",
                    "Check network connectivity",
                    "Try using cookies for age-restricted content",
                ],
            )
            return Err(CoreError(f"yt-dlp failed: {error_msg}", context))
            
        except json.JSONDecodeError as e:
            context = ErrorContext.create(
                "extract_metadata_os_ytdlp",
                input_data={
                    "url": url,
                    "json_error": str(e),
                    "output_preview": result.stdout[:200] if 'result' in locals() else None,
                },
                suggestions=[
                    "Check yt-dlp output format",
                    "Update yt-dlp to latest version",
                    "Check if output contains error message",
                ],
            )
            return Err(CoreError(f"Failed to parse yt-dlp JSON output: {e}", context))
            
        except FileNotFoundError:
            context = ErrorContext.create(
                "extract_metadata_os_ytdlp",
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
                "extract_metadata_os_ytdlp",
                input_data={"url": url, "error_type": type(e).__name__},
                suggestions=[
                    "Check system configuration",
                    "Verify all dependencies",
                    "Try manual yt-dlp command",
                ],
            )
            return Err(CoreError(f"Unexpected error in yt-dlp extraction: {e}", context))

    @log_function(log_performance=True)
    def extract_playlist_metadata(
        self, playlist_url: str
    ) -> Result[List[ProcessObject], CoreError]:
        """OS-Version der Playlist-Extraktion mit vollständigem Error-Handling"""
        try:
            with log_feature("playlist_extraction_os") as feature:
                self.logger.info(f"🎵 Extracting playlist with OS yt-dlp: {playlist_url}")
                feature.add_metric("playlist_url", playlist_url)

                # OS yt-dlp für Playlist-Info
                cmd = [
                    "yt-dlp",
                    "--dump-json", 
                    "--flat-playlist",  # Nur URLs, keine komplette Metadata
                    "--no-download",
                    "--quiet",
                ]
                
                if self.cookie_file:
                    cmd.extend(["--cookies", self.cookie_file])
                else:
                    cmd.extend(["--cookies-from-browser", "firefox"])
                
                cmd.append(playlist_url)

                # Playlist-Entries extrahieren
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=120, 
                    check=True
                )
                
                if not result.stdout.strip():
                    context = ErrorContext.create(
                        "extract_playlist_metadata_os",
                        input_data={"playlist_url": playlist_url, "stdout_empty": True},
                        suggestions=[
                            "Check if playlist exists and is public",
                            "Verify playlist URL format", 
                            "Check network connectivity",
                        ],
                    )
                    return Err(CoreError("Empty playlist or no accessible videos", context))

                # Jede Zeile ist ein JSON-Objekt (ein Video)
                entries = []
                for line_num, line in enumerate(result.stdout.strip().split('\n'), 1):
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            if entry.get('url'):
                                entries.append(entry)
                        except json.JSONDecodeError as e:
                            self.logger.warning(
                                f"Failed to parse playlist entry line {line_num}: {e}",
                                extra={"line_content": line[:100], "line_number": line_num}
                            )
                            continue

                if not entries:
                    context = ErrorContext.create(
                        "extract_playlist_metadata_os",
                        input_data={
                            "playlist_url": playlist_url,
                            "stdout_lines": len(result.stdout.strip().split('\n')),
                        },
                        suggestions=[
                            "Check if playlist contains accessible videos",
                            "Verify playlist is not private",
                            "Check if playlist contains valid video entries",
                        ],
                    )
                    return Err(CoreError(f"No valid entries found in playlist: {playlist_url}", context))

                feature.add_metric("playlist_entries", len(entries))

                # Metadata für jedes Video extrahieren
                extracted_objects = []
                failed_extractions = []

                for i, entry in enumerate(entries):
                    video_url = entry["url"]
                    single_result = self.extract_single_metadata(video_url)

                    if isinstance(single_result, Ok):
                        extracted_objects.append(unwrap_ok(single_result))
                    else:
                        error = unwrap_err(single_result)
                        failed_extractions.append({
                            "url": video_url,
                            "error": error.message,
                            "position": i + 1
                        })
                        self.logger.warning(
                            f"Failed to extract video {i + 1}/{len(entries)}: {video_url}",
                            extra={"error": error.message, "video_position": i + 1}
                        )

                feature.add_metric("extracted_videos", len(extracted_objects))
                feature.add_metric("failed_extractions", len(failed_extractions))
                feature.add_metric("success_rate", len(extracted_objects) / len(entries) * 100)

                if extracted_objects:
                    self.logger.info(
                        f"✅ Playlist extraction completed: {len(extracted_objects)}/{len(entries)} videos",
                        extra={
                            "playlist_url": playlist_url,
                            "total_entries": len(entries),
                            "successful_extractions": len(extracted_objects),
                            "failed_extractions": len(failed_extractions),
                            "success_rate": round(len(extracted_objects) / len(entries) * 100, 1),
                        }
                    )
                    return Ok(extracted_objects)
                else:
                    context = ErrorContext.create(
                        "extract_playlist_metadata_os",
                        input_data={
                            "playlist_url": playlist_url,
                            "total_entries": len(entries),
                            "failed_extractions": len(failed_extractions),
                        },
                        suggestions=[
                            "Check individual video availability",
                            "Verify network connectivity",
                            "Try extracting individual videos manually",
                        ],
                    )
                    return Err(CoreError("No videos could be extracted from playlist", context))

        except subprocess.TimeoutExpired:
            context = ErrorContext.create(
                "extract_playlist_metadata_os",
                input_data={"playlist_url": playlist_url, "timeout": 120},
                suggestions=[
                    "Check network connectivity",
                    "Try again later - large playlists may take time",
                    "Split large playlists into smaller chunks",
                ],
            )
            return Err(CoreError(f"Playlist extraction timeout: {playlist_url}", context))

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            context = ErrorContext.create(
                "extract_playlist_metadata_os",
                input_data={
                    "playlist_url": playlist_url,
                    "return_code": e.returncode,
                    "stderr": error_msg,
                },
                suggestions=[
                    "Check if playlist exists and is accessible",
                    "Verify yt-dlp installation and version",
                    "Check network connectivity",
                ],
            )
            return Err(CoreError(f"OS yt-dlp playlist extraction failed: {error_msg}", context))

        except Exception as e:
            context = ErrorContext.create(
                "extract_playlist_metadata_os",
                input_data={"playlist_url": playlist_url, "error_type": type(e).__name__},
                suggestions=[
                    "Check system configuration",
                    "Verify yt-dlp installation",
                    "Try manual command execution",
                ],
            )
            return Err(CoreError(f"Playlist extraction failed: {e}", context))

    @log_function(log_performance=True)
    def extract_single_metadata(self, url: str) -> Result[ProcessObject, CoreError]:
        """OS-Version der Single-Metadata-Extraktion mit Retry-Logic"""

        for attempt in range(3):  # 3 retry attempts
            try:
                with log_feature(f"video_metadata_extraction_os_attempt_{attempt + 1}") as feature:
                    feature.add_metric("url", url)
                    feature.add_metric("attempt", attempt + 1)

                    # OS yt-dlp verwenden
                    metadata_result = self._extract_metadata_with_os_ytdlp(url)
                    if isinstance(metadata_result, Err):
                        return metadata_result

                    info = unwrap_ok(metadata_result)

                    if not info:
                        context = ErrorContext.create(
                            "extract_single_metadata_os",
                            input_data={"url": url, "attempt": attempt + 1},
                            suggestions=[
                                "Check if video exists",
                                "Verify URL format",
                                "Check network connectivity",
                            ],
                        )
                        return Err(CoreError("No video information extracted", context))

                    # Create ProcessObject from extracted info 
                    process_obj = self.create_process_object_from_info(info, url)

                    feature.add_metric("success", True)
                    feature.add_metric("duration_seconds", info.get("duration", 0))
                    feature.add_metric("video_title", process_obj.titel)

                    # Enhanced metadata logging
                    self.logger.info(
                        f"✅ Video metadata extracted: {process_obj.titel}",
                        extra={
                            "video_title": process_obj.titel,
                            "channel": process_obj.kanal,
                            "duration": str(process_obj.länge),
                            "upload_date": process_obj.upload_date.strftime('%Y-%m-%d'),
                            "video_id": info.get('id', 'unknown'),
                            "view_count": info.get('view_count', 'unknown'),
                            "attempt": attempt + 1,
                            "cookie_method": 'file' if self.cookie_file else 'browser'
                        }
                    )

                    return Ok(process_obj)

            except Exception as e:
                error_msg = str(e)

                # Check for retryable errors  
                if attempt < 2 and any(
                    keyword in error_msg.lower()
                    for keyword in ["network", "timeout", "connection", "temporary", "sign in", "throttl"]
                ):
                    wait_time = 2**attempt  # Exponential backoff
                    self.logger.warning(
                        f"Retryable error on attempt {attempt + 1}, retrying in {wait_time}s: {error_msg}",
                        extra={"attempt": attempt + 1, "wait_time": wait_time, "url": url}
                    )
                    time.sleep(wait_time)
                    continue

                # Non-retryable or final attempt
                context = ErrorContext.create(
                    "extract_single_metadata_os",
                    input_data={
                        "url": url, 
                        "attempt": attempt + 1, 
                        "error": error_msg,
                        "error_type": type(e).__name__,
                    },
                    suggestions=[
                        "Check video availability and access permissions",
                        "Verify URL is correct and complete", 
                        "Check if video is private, deleted, or age-restricted",
                        "Ensure yt-dlp is installed and up-to-date",
                        "Verify network connectivity and DNS resolution",
                        "Check cookie file/browser cookies for restricted content"
                    ],
                )
                return Err(CoreError(f"Metadata extraction failed: {error_msg}", context))

        # Should never reach here due to return statements in loop
        context = ErrorContext.create(
            "extract_single_metadata_os",
            input_data={"url": url, "max_attempts": 3},
            suggestions=[
                "Check video accessibility",
                "Verify system configuration",
                "Try manual yt-dlp command",
            ],
        )
        return Err(CoreError("Metadata extraction failed after all retry attempts", context))

    def create_process_object_from_info(
        self, info: Dict[str, Any], original_url: str
    ) -> ProcessObject:
        """Erstellt ProcessObject aus yt-dlp info dict mit robustem Parsing"""

        # Extract basic info with fallbacks
        title = info.get("title", "Unknown Title")
        uploader = info.get("uploader", info.get("channel", "Unknown Channel"))
        duration_seconds = info.get("duration", 0)
        upload_date_str = info.get("upload_date", "")

        # Parse duration with error handling
        if duration_seconds and duration_seconds > 0:
            try:
                hours = int(duration_seconds // 3600)
                minutes = int((duration_seconds % 3600) // 60)
                seconds = int(duration_seconds % 60)
                länge = dt_time(hours, minutes, seconds)
            except (ValueError, OverflowError):
                self.logger.warning(f"Invalid duration value: {duration_seconds}, using 0:00:00")
                länge = dt_time(0, 0, 0)
        else:
            länge = dt_time(0, 0, 0)

        # Parse upload date with multiple fallback formats
        upload_date = datetime.now()  # Default fallback
        if upload_date_str:
            for date_format in ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    upload_date = datetime.strptime(upload_date_str[:10], date_format)
                    break
                except ValueError:
                    continue
            else:
                self.logger.warning(
                    f"Could not parse upload date: {upload_date_str}, using current date",
                    extra={"upload_date_str": upload_date_str, "video_title": title}
                )

        # Create ProcessObject
        process_obj = ProcessObject(
            titel=title,
            kanal=uploader,
            länge=länge,
            upload_date=upload_date,
            original_url=original_url,  # Store original URL for downloads
        )

        # Add additional metadata
        process_obj.update_stage("metadata_extracted")

        return process_obj

    def extract_single_or_playlist(
        self, url: str
    ) -> Result[List[ProcessObject], CoreError]:
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


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================


@log_function(log_performance=True)
def process_urls_to_objects(
    urls_text: str, config: Optional[Dict[str, Any]] = None
) -> Result[List[ProcessObject], CoreError]:
    """
    Complete Pipeline: URLs Text → ProcessObjects mit Metadata

    Args:
        urls_text: Multiline-Text mit YouTube-URLs
        config: Optional configuration

    Returns:
        Ok(List[ProcessObject]): Fertige ProcessObjects mit Metadata
        Err(CoreError): Pipeline-Fehler
    """
    logger = get_logger("process_urls_to_objects")
    
    try:
        with log_feature("urls_to_objects_pipeline") as feature:
            # Step 1: Parse URLs
            url_processor = YouTubeURLProcessor()
            urls_result = url_processor.parse_multiline_input(urls_text)

            if isinstance(urls_result, Err):
                return urls_result

            urls = unwrap_ok(urls_result)
            feature.add_metric("parsed_urls", len(urls))

            # Step 2: Extract Metadata für alle URLs
            extractor = YouTubeMetadataExtractor(config)
            all_objects = []
            failed_urls = []

            for i, url in enumerate(urls):
                feature.checkpoint(f"processing_url_{i + 1}")
                
                # Extract metadata (single video or playlist)
                objects_result = extractor.extract_single_or_playlist(url)

                if isinstance(objects_result, Ok):
                    url_objects = unwrap_ok(objects_result)
                    all_objects.extend(url_objects)
                    
                    logger.info(
                        f"✅ URL {i + 1}/{len(urls)} processed: {len(url_objects)} videos",
                        extra={
                            "url": url,
                            "videos_extracted": len(url_objects),
                            "progress": f"{i + 1}/{len(urls)}",
                        }
                    )
                else:
                    error = unwrap_err(objects_result)
                    failed_urls.append({"url": url, "error": error.message})
                    
                    logger.error(
                        f"❌ URL {i + 1}/{len(urls)} failed: {error.message}",
                        extra={
                            "url": url,
                            "error": error.message,
                            "progress": f"{i + 1}/{len(urls)}",
                        }
                    )

            feature.add_metric("final_objects", len(all_objects))
            feature.add_metric("failed_urls", len(failed_urls))
            feature.add_metric("success_rate", 
                              len(all_objects) / len(urls) * 100 if urls else 0)

            if all_objects:
                logger.info(
                    f"🎯 URL processing pipeline completed successfully",
                    extra={
                        "total_urls": len(urls),
                        "total_videos": len(all_objects),
                        "failed_urls": len(failed_urls),
                        "success_rate": round(len(all_objects) / len(urls) * 100, 1) if urls else 0,
                        "unique_channels": len(set([obj.kanal for obj in all_objects])),
                    }
                )
                return Ok(all_objects)
            else:
                context = ErrorContext.create(
                    "process_urls_to_objects",
                    input_data={
                        "total_urls": len(urls),
                        "failed_urls": len(failed_urls),
                        "failed_details": failed_urls[:3],  # First 3 failures
                    },
                    suggestions=[
                        "Check URL validity and accessibility",
                        "Verify network connectivity",
                        "Check if videos are private or deleted",
                        "Update yt-dlp to latest version",
                    ],
                )
                return Err(CoreError("No videos could be extracted from any URL", context))

    except Exception as e:
        context = ErrorContext.create(
            "process_urls_to_objects",
            input_data={
                "urls_text_length": len(urls_text),
                "error_type": type(e).__name__,
            },
            suggestions=[
                "Check input format and encoding",
                "Verify system dependencies",
                "Check available memory and disk space",
            ],
        )
        return Err(CoreError(f"URLs to objects pipeline failed: {e}", context))


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
        error = unwrap_err(result)
        print(f"❌ Processing failed: {error.message}")
        if hasattr(error, 'context') and error.context.suggestions:
            print("Suggestions:", error.context.suggestions)
