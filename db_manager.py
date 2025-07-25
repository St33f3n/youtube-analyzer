#!/usr/bin/env python3
"""
YouTube Analyzer - Enhanced Database Manager CLI
EXTENDED: Pipeline Recovery, Trilium Integration, Nextcloud Filename Updates, Nextcloud Video Fix System
"""

from __future__ import annotations
import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal, Union
from dataclasses import dataclass
from enum import Enum
import re
from urllib.parse import urlparse, parse_qs

# Import core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_ok, unwrap_err
from yt_analyzer_core import ArchiveDatabase, ArchivObject, TranskriptObject, ProcessObject
from yt_analyzer_config import SecureConfigManager, AppConfig
from logging_plus import setup_logging, get_logger, log_feature, log_function
from yt_url_processor import YouTubeMetadataExtractor
from yt_llm_processor import process_transcript_with_llm_dict
from yt_trilium_uploader import upload_to_trilium_dict, TriliumUploader

# ENHANCED IMPORTS für Pipeline Recovery
from yt_nextcloud_uploader import upload_to_nextcloud_for_process_object_dict
from yt_video_downloader import download_video_for_process_object
from yt_audio_downloader import download_audio_for_process_object


# =============================================================================
# ENHANCED TYPE DEFINITIONS
# =============================================================================

@dataclass
class PipelineRecoveryReport:
    """Report für Pipeline-Recovery-Operationen"""
    stream_type: Literal["A", "B", "both"]
    processed_entries: int
    successful_recoveries: int
    failed_recoveries: int
    processing_time_seconds: float
    recovery_details: List[Dict[str, Any]]
    errors: List[str]
    fix_all_mode: bool = False


@dataclass
class TriliumUpdateResult:
    """Result für Trilium-Note-Updates"""
    entry_id: int
    trilium_note_id: Optional[str]
    title_updated: bool
    note_title_updated: bool
    success: bool
    error_message: Optional[str] = None


@dataclass
class NextcloudUpdateResult:
    """Result für Nextcloud-Filename-Updates"""
    entry_id: int
    old_filename: Optional[str]
    new_filename: Optional[str] 
    updated: bool
    error_message: Optional[str] = None


@dataclass
class NextcloudVideoFixReport:
    """NEW: Report für Nextcloud Video Fix System"""
    files_found: int
    matched: int
    processed: int
    fixed: int
    failed: int
    processing_time_seconds: float
    results: List[Dict[str, Any]]
    errors: List[str]
    dry_run: bool


class StreamType(Enum):
    """Pipeline Stream Types"""
    STREAM_A = "video"  # Video processing stream
    STREAM_B = "transcript"  # Transcript processing stream
    BOTH = "both"


# =============================================================================
# ENHANCED EXTENDED DATABASE MANAGER
# =============================================================================

class ExtendedDatabaseManager:
    """Enhanced Database Manager mit Pipeline Recovery, Trilium & Nextcloud Integration + Video Fix System"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.db = ArchiveDatabase(Path(config.storage.sqlite_path))
        self.logger = get_logger("ExtendedDatabaseManager")

    def create_backup(self) -> Result[Path, CoreError]:
        """Erstellt timestamped Backup der Datenbank"""
        try:
            db_path = Path(self.config.storage.sqlite_path)
            if not db_path.exists():
                context = ErrorContext.create(
                    "create_backup",
                    input_data={"db_path": str(db_path)},
                    suggestions=["Check database path in config", "Ensure database exists"]
                )
                return Err(CoreError(f"Database file not found: {db_path}", context))

            # Backup mit timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = db_path.parent / f"backup_{db_path.stem}_{timestamp}.db"
            
            shutil.copy2(db_path, backup_path)
            
            backup_size = backup_path.stat().st_size / (1024 * 1024)  # MB
            self.logger.info(
                f"✅ Database backup created: {backup_path.name}",
                extra={
                    "backup_path": str(backup_path),
                    "backup_size_mb": round(backup_size, 2),
                    "original_path": str(db_path)
                }
            )
            
            return Ok(backup_path)

        except Exception as e:
            context = ErrorContext.create(
                "create_backup",
                input_data={"error": str(e)},
                suggestions=["Check file permissions", "Ensure sufficient disk space"]
            )
            return Err(CoreError(f"Backup creation failed: {e}", context))

    def find_broken_metadata_entries(self, pattern: str = "youtube video #") -> Result[List[Dict[str, Any]], CoreError]:
        """Findet Einträge mit defekten Metadaten"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                
                # Query für Titel die mit Pattern beginnen
                cursor.execute(
                    "SELECT id, titel, kanal, original_url, date_created, nextcloud_link FROM processed_videos WHERE titel LIKE ? ORDER BY date_created DESC",
                    (f"{pattern}%",)
                )
                
                rows = cursor.fetchall()
                
                entries = []
                for row in rows:
                    entries.append({
                        "id": row[0],
                        "titel": row[1],
                        "kanal": row[2], 
                        "original_url": row[3],
                        "date_created": row[4],
                        "nextcloud_link": row[5]  # ENHANCED: Include nextcloud_link
                    })

                self.logger.info(
                    f"Found {len(entries)} entries with pattern '{pattern}'",
                    extra={
                        "pattern": pattern,
                        "count": len(entries),
                        "sample_titles": [e["titel"][:50] for e in entries[:3]]
                    }
                )
                
                return Ok(entries)

        except Exception as e:
            context = ErrorContext.create(
                "find_broken_metadata",
                input_data={"pattern": pattern},
                suggestions=["Check database connection", "Verify table structure"]
            )
            return Err(CoreError(f"Failed to query broken metadata: {e}", context))

    def update_metadata(self, entry_id: int, titel: str, kanal: str) -> Result[None, CoreError]:
        """Aktualisiert Titel und Kanal für einen Eintrag"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                
                # Update nur Titel und Kanal
                cursor.execute(
                    "UPDATE processed_videos SET titel = ?, kanal = ? WHERE id = ?",
                    (titel, kanal, entry_id)
                )
                
                if cursor.rowcount == 0:
                    context = ErrorContext.create(
                        "update_metadata",
                        input_data={"entry_id": entry_id},
                        suggestions=["Check if entry exists", "Verify entry ID"]
                    )
                    return Err(CoreError(f"No entry found with ID {entry_id}", context))
                
                conn.commit()
                
                self.logger.info(
                    f"✅ Metadata updated for entry {entry_id}",
                    extra={
                        "entry_id": entry_id,
                        "new_titel": titel,
                        "new_kanal": kanal
                    }
                )
                
                return Ok(None)

        except Exception as e:
            context = ErrorContext.create(
                "update_metadata", 
                input_data={"entry_id": entry_id, "titel": titel, "kanal": kanal},
                suggestions=["Check database connection", "Verify data types"]
            )
            return Err(CoreError(f"Failed to update metadata: {e}", context))

    def get_entry_by_id(self, entry_id: int) -> Result[Dict[str, Any], CoreError]:
        """Holt vollständigen Eintrag per ID"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM processed_videos WHERE id = ?", (entry_id,))
                row = cursor.fetchone()
                
                if not row:
                    context = ErrorContext.create(
                        "get_entry_by_id",
                        input_data={"entry_id": entry_id},
                        suggestions=["Check if entry exists", "Verify entry ID"]
                    )
                    return Err(CoreError(f"No entry found with ID {entry_id}", context))

                # Get column names
                cursor.execute("PRAGMA table_info(processed_videos)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Create dict from row
                entry = dict(zip(columns, row))
                
                return Ok(entry)

        except Exception as e:
            context = ErrorContext.create(
                "get_entry_by_id",
                input_data={"entry_id": entry_id},
                suggestions=["Check database connection", "Verify table structure"]
            )
            return Err(CoreError(f"Failed to get entry: {e}", context))

    def update_trilium_info(self, entry_id: int, trilium_link: str, trilium_note_id: str) -> Result[None, CoreError]:
        """Aktualisiert Trilium-Links für einen Eintrag"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "UPDATE processed_videos SET trilium_link = ?, trilium_note_id = ?, transcript_stream_success = 1 WHERE id = ?",
                    (trilium_link, trilium_note_id, entry_id)
                )
                
                if cursor.rowcount == 0:
                    context = ErrorContext.create(
                        "update_trilium_info",
                        input_data={"entry_id": entry_id},
                        suggestions=["Check if entry exists", "Verify entry ID"]
                    )
                    return Err(CoreError(f"No entry found with ID {entry_id}", context))
                
                conn.commit()
                
                self.logger.info(
                    f"✅ Trilium info updated for entry {entry_id}",
                    extra={
                        "entry_id": entry_id,
                        "trilium_link": trilium_link,
                        "trilium_note_id": trilium_note_id
                    }
                )
                
                return Ok(None)

        except Exception as e:
            context = ErrorContext.create(
                "update_trilium_info",
                input_data={"entry_id": entry_id},
                suggestions=["Check database connection", "Verify data types"]
            )
            return Err(CoreError(f"Failed to update Trilium info: {e}", context))

    def find_missing_attributes(self) -> Result[List[Dict[str, Any]], CoreError]:
        """
        FIXED: Findet Einträge mit fehlenden Attributen - korrigierte Kategorisierung ohne Dopplungen
        """
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                
                # FIXED: Verbesserte Kategorisierung ohne Überlappungen
                basic_fields = {
                    'titel', 'kanal', 'länge', 'upload_date', 'original_url',
                    'sprache', 'transkript', 'rule_amount', 'rule_accuracy', 
                    'relevancy', 'analysis_results'
                }
                
                stream_a_fields = {
                    'nextcloud_link', 'video_stream_success'
                }
                
                stream_b_fields = {
                    'bearbeiteter_transkript', 'llm_model', 'llm_tokens', 
                    'llm_cost', 'llm_processing_time', 'trilium_link', 
                    'trilium_note_id', 'transcript_stream_success'
                }
                
                final_fields = {
                    'final_success'
                }
                
                all_required_fields = basic_fields | stream_a_fields | stream_b_fields | final_fields
                
                # Query nur für erfolgreich analysierte Videos
                cursor.execute("SELECT * FROM processed_videos WHERE passed_analysis = 1 ORDER BY date_created DESC")
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                missing_attrs = []
                
                for row in rows:
                    entry_dict = dict(zip(columns, row))
                    entry_id = entry_dict['id']
                    entry_missing = []
                    
                    # Prüfe jedes required field
                    for field in all_required_fields:
                        if field not in entry_dict:
                            entry_missing.append(f"{field} (column missing)")
                            continue
                            
                        value = entry_dict.get(field)
                        
                        # NULL oder empty string als fehlend betrachten
                        if value is None or value == "" or value == "null":
                            entry_missing.append(field)
                        
                        # Boolean fields checks
                        elif field in ['video_stream_success', 'transcript_stream_success', 'final_success']:
                            if value == 0 or value is False:
                                entry_missing.append(f"{field} (false/failed)")
                    
                    # Nur Einträge mit fehlenden Attributen hinzufügen
                    if entry_missing:
                        # FIXED: Korrekte Kategorisierung ohne Dopplungen
                        categorized_missing = {
                            'basic': [f for f in entry_missing if f.split(' (')[0] in basic_fields],
                            'stream_a': [f for f in entry_missing if f.split(' (')[0] in stream_a_fields],
                            'stream_b': [f for f in entry_missing if f.split(' (')[0] in stream_b_fields],
                            'final': [f for f in entry_missing if f.split(' (')[0] in final_fields]
                        }
                        
                        missing_attrs.append({
                            'id': entry_id,
                            'titel': str(entry_dict['titel'])[:50] + '...' if len(str(entry_dict['titel'])) > 50 else str(entry_dict['titel']),
                            'kanal': entry_dict['kanal'] or 'Unknown',
                            'missing_attributes': entry_missing,
                            'missing_count': len(entry_missing),
                            'categories': categorized_missing
                        })

                self.logger.info(
                    f"Found {len(missing_attrs)} entries with missing attributes",
                    extra={
                        "total_checked": len(rows),
                        "entries_with_missing": len(missing_attrs)
                    }
                )
                
                return Ok(missing_attrs)

        except Exception as e:
            context = ErrorContext.create(
                "find_missing_attributes",
                suggestions=["Check database connection", "Verify table structure", "Check column names"]
            )
            return Err(CoreError(f"Failed to find missing attributes: {e}", context))

    # =============================================================================
    # NEW: NEXTCLOUD VIDEO FIX SYSTEM - Main Orchestration
    # =============================================================================

    def scan_and_fix_nextcloud_videos(
        self,
        batch_size: int = 20,
        dry_run: bool = True
    ) -> Result[NextcloudVideoFixReport, CoreError]:
        """
        NEW: Main orchestration function for complete Nextcloud Video Fix workflow
        
        Workflow Steps:
        1. Scan WebDAV for youtube video files
        2. Match files with database entries
        3. Process matches in batches
        4. Generate detailed report
        """
        try:
            start_time = time.time()
            
            with log_feature("scan_and_fix_nextcloud_videos") as feature:
                
                # STEP 1: WebDAV Scanning
                self.logger.info("Starting WebDAV scan for youtube video files")
                scan_result = self._scan_webdav_for_youtube_videos()
                if isinstance(scan_result, Err):
                    return scan_result
                
                found_files = unwrap_ok(scan_result)
                feature.add_metric("files_found", len(found_files))
                
                # Early exit if no files found
                if not found_files:
                    return Ok(self._create_empty_nextcloud_report(dry_run))
                
                # STEP 2: Database Matching
                self.logger.info(f"Matching {len(found_files)} files with database")
                match_result = self._match_files_with_database(found_files)
                if isinstance(match_result, Err):
                    return match_result
                
                matched_files = unwrap_ok(match_result)
                feature.add_metric("files_matched", len(matched_files))
                
                # STEP 3: Batch Processing
                processing_result = self._process_nextcloud_matches_in_batches(
                    matched_files, batch_size, dry_run
                )
                if isinstance(processing_result, Err):
                    return processing_result
                
                processing_data = unwrap_ok(processing_result)
                
                # STEP 4: Generate Final Report
                processing_time = time.time() - start_time
                report = self._create_nextcloud_final_report(
                    found_files, matched_files, processing_data, processing_time, dry_run
                )
                
                return Ok(report)
                
        except Exception as e:
            context = ErrorContext.create(
                "scan_and_fix_nextcloud_videos",
                input_data={"batch_size": batch_size, "dry_run": dry_run},
                suggestions=["Check WebDAV connectivity", "Verify database access"]
            )
            return Err(CoreError(f"WebDAV scan and fix failed: {e}", context))

    def _scan_webdav_for_youtube_videos(self) -> Result[List[Dict[str, str]], CoreError]:
        """
        Scan entire WebDAV tree for youtube video files
        
        Process:
        1. Connect to WebDAV with credentials
        2. Start recursive scan from root path
        3. Check each file against pattern
        4. Extract video ID from matching files
        5. Collect file metadata
        """
        try:
            # Import and initialize WebDAV client
            try:
                from webdav4.client import Client as WebDAVClient
            except ImportError:
                return Err(CoreError(
                    "webdav4 not available - install with: pip install webdav4"
                ))
            
            # Load configuration
            config_manager = SecureConfigManager()
            config_result = config_manager.load_config()
            config = unwrap_ok(config_result)
            
            # Get credentials
            username = config.secrets.nextcloud_username
            password = unwrap_ok(config_manager.get_nextcloud_password())
            base_url = config.storage.nextcloud_base_url
            base_path = config.storage.nextcloud_path.rstrip("/")
            
            # Initialize WebDAV client with extended timeout for large scans
            client = WebDAVClient(
                base_url=base_url,
                auth=(username, password),
                timeout=120,  # Extended timeout for large directory scans
                verify=True
            )
            
            found_files = []
            
            # Start recursive scan
            self._scan_webdav_directory_recursive(client, base_path, found_files)
            
            self.logger.info(
                f"WebDAV scan completed: found {len(found_files)} youtube video files",
                extra={
                    "total_files": len(found_files),
                    "base_path": base_path,
                    "sample_files": [f["filename"] for f in found_files[:3]]
                }
            )
            
            return Ok(found_files)
            
        except Exception as e:
            return Err(CoreError(f"WebDAV scan failed: {e}"))

    def _scan_webdav_directory_recursive(
        self, 
        client, 
        directory_path: str, 
        found_files: List[Dict[str, str]]
    ) -> None:
        """
        Recursively scan a single directory
        
        Process for each directory:
        1. List all items in directory
        2. For each item:
           - If directory: recurse into it
           - If file: check against youtube video pattern
        3. Extract video ID from matching files
        4. Store file metadata
        """
        try:
            # List directory contents with detailed metadata
            items = client.ls(directory_path, detail=True)
            
            for item in items:
                item_path = item['name']
                is_directory = item.get('type') == 'directory'
                
                if is_directory:
                    # Recursively scan subdirectory
                    self._scan_webdav_directory_recursive(client, item_path, found_files)
                else:
                    # Extract filename from full path
                    filename = item_path.split('/')[-1]
                    
                    # Check against youtube video pattern
                    # Pattern explanation:
                    # - youtube video _  : Literal prefix
                    # - ([0-9A-Za-z_-]{11}) : Capture group for 11-char video ID
                    # - \.mp4$ : Literal .mp4 extension at end
                    pattern = r'youtube video _([0-9A-Za-z_-]{11})\.mp4$'
                    match = re.match(pattern, filename)
                    
                    if match:
                        video_id = match.group(1)
                        
                        # Store comprehensive file metadata
                        found_files.append({
                            "webdav_path": item_path,           # Full WebDAV path
                            "filename": filename,               # Just the filename
                            "video_id": video_id,              # Extracted video ID
                            "directory": str(Path(item_path).parent), # Parent directory
                            "size": item.get('size', 0),       # File size
                            "modified": item.get('modified'),  # Last modified date
                        })
                        
                        self.logger.debug(
                            f"Found youtube video file: {filename}",
                            extra={"video_id": video_id, "path": item_path}
                        )
                        
        except Exception as e:
            self.logger.warning(f"Failed to scan directory {directory_path}: {e}")
            # Continue scanning other directories even if one fails

    def _match_files_with_database(
        self, 
        found_files: List[Dict[str, str]]
    ) -> Result[List[Dict[str, Any]], CoreError]:
        """
        Match WebDAV files with database entries using video ID
        
        Process:
        1. For each found file, extract video ID
        2. Query database for entries containing this video ID
        3. Match based on original_url field
        4. Collect matched entries with metadata
        """
        try:
            import sqlite3
            
            matched_files = []
            unmatched_count = 0
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                
                for file_info in found_files:
                    video_id = file_info["video_id"]
                    
                    # Database query to find matching entry
                    # Search strategy: Look for video ID anywhere in original_url
                    cursor.execute("""
                        SELECT id, titel, kanal, original_url, nextcloud_link, 
                               video_stream_success, transcript_stream_success
                        FROM processed_videos 
                        WHERE original_url LIKE ?
                        ORDER BY date_created DESC
                        LIMIT 1
                    """, (f"%{video_id}%",))
                    
                    row = cursor.fetchone()
                    
                    if row:
                        # Match found - combine file and database metadata
                        matched_files.append({
                            # Database metadata
                            "entry_id": row[0],
                            "titel": row[1],
                            "kanal": row[2],
                            "original_url": row[3],
                            "nextcloud_link": row[4],
                            "video_stream_success": row[5],
                            "transcript_stream_success": row[6],
                            
                            # File metadata
                            "video_id": video_id,
                            "webdav_path": file_info["webdav_path"],
                            "filename": file_info["filename"],
                            "directory": file_info["directory"],
                            "file_size": file_info["size"],
                        })
                        
                        self.logger.debug(
                            f"Matched file {file_info['filename']} with DB entry {row[0]}",
                            extra={
                                "video_id": video_id,
                                "entry_id": row[0],
                                "titel": row[1][:50]  # Truncate for logging
                            }
                        )
                    else:
                        unmatched_count += 1
                        self.logger.debug(f"No database match found for video ID {video_id}")
            
            self.logger.info(
                f"Database matching completed",
                extra={
                    "total_files": len(found_files),
                    "matched_files": len(matched_files),
                    "unmatched_files": unmatched_count,
                    "match_rate": f"{len(matched_files)/len(found_files)*100:.1f}%" if found_files else "0%"
                }
            )
            
            return Ok(matched_files)
            
        except Exception as e:
            return Err(CoreError(f"Failed to match files with database: {e}"))

    def _process_nextcloud_matches_in_batches(
        self,
        matched_files: List[Dict[str, Any]], 
        batch_size: int,
        dry_run: bool
    ) -> Result[Dict[str, Any], CoreError]:
        """
        Process matched files in configurable batches for performance
        
        Benefits of batch processing:
        - Memory efficiency for large file counts
        - Progress tracking and logging
        - Partial recovery on errors
        - Resource throttling
        """
        try:
            processed = 0
            fixed = 0
            failed = 0
            results = []
            errors = []
            
            total_batches = (len(matched_files) + batch_size - 1) // batch_size
            
            for i in range(0, len(matched_files), batch_size):
                batch = matched_files[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                self.logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)"
                )
                
                # Process each file in current batch
                for file_info in batch:
                    processed += 1
                    
                    try:
                        fix_result = self._fix_single_nextcloud_video(file_info, dry_run)
                        
                        if isinstance(fix_result, Ok):
                            fix_data = unwrap_ok(fix_result)
                            if fix_data["updated"]:
                                fixed += 1
                            
                            results.append({
                                "entry_id": file_info["entry_id"],
                                "video_id": file_info["video_id"],
                                "old_path": file_info["webdav_path"],
                                "new_path": fix_data.get("new_path"),
                                "status": "fixed" if fix_data["updated"] else "skipped",
                                "reason": fix_data.get("reason", "")
                            })
                        else:
                            failed += 1
                            error = unwrap_err(fix_result)
                            error_msg = f"File {file_info['webdav_path']}: {error.message}"
                            errors.append(error_msg)
                            
                            results.append({
                                "entry_id": file_info.get("entry_id"),
                                "video_id": file_info["video_id"],
                                "old_path": file_info["webdav_path"],
                                "status": "failed",
                                "error": error.message
                            })
                    
                    except Exception as e:
                        failed += 1
                        error_msg = f"File {file_info['webdav_path']}: Unexpected error: {e}"
                        errors.append(error_msg)
                        
                        results.append({
                            "video_id": file_info["video_id"],
                            "old_path": file_info["webdav_path"],
                            "status": "failed",
                            "error": str(e)
                        })
            
            return Ok({
                "processed": processed,
                "fixed": fixed,
                "failed": failed,
                "results": results,
                "errors": errors
            })
            
        except Exception as e:
            return Err(CoreError(f"Batch processing failed: {e}"))

    def _fix_single_nextcloud_video(
        self, 
        file_info: Dict[str, Any], 
        dry_run: bool
    ) -> Result[Dict[str, Any], CoreError]:
        """
        Fix single Nextcloud video file
        
        Process:
        1. Construct old and new paths
        2. Verify source file exists  
        3. Create target directory if needed
        4. Perform file move operation
        5. Update database with new link
        """
        try:
            video_id = file_info["video_id"]
            
            # Construct new path using proper title/channel
            path_result = self._construct_new_path(file_info, video_id)
            if isinstance(path_result, Err):
                return path_result
            
            path_data = unwrap_ok(path_result)
            old_path = path_data["old_path"]
            new_path = path_data["new_path"]
            
            result = {
                "video_id": video_id,
                "old_path": old_path,
                "new_path": new_path,
                "updated": False,
                "dry_run": dry_run
            }
            
            # Check if paths are different
            if old_path == new_path:
                result["reason"] = "File path unchanged - no update needed"
                return Ok(result)
            
            if dry_run:
                result["reason"] = "Dry run - would move file"
                result["updated"] = True  # Mark as would-be-updated for reporting
                return Ok(result)
            
            # Load config for WebDAV operations
            config_manager = SecureConfigManager()
            config_result = config_manager.load_config()
            config = unwrap_ok(config_result)
            
            # Perform actual file move
            move_result = self._rename_file_in_nextcloud_webdav(
                old_path, new_path, config
            )
            
            if isinstance(move_result, Ok):
                # Update database with new nextcloud_link if needed
                new_link = new_path.replace(config.storage.nextcloud_path.rstrip("/"), "").lstrip("/")
                self._update_nextcloud_link_in_db(file_info["entry_id"], new_link)
                
                result["updated"] = True
                result["reason"] = "File moved successfully"
                return Ok(result)
            else:
                error = unwrap_err(move_result)
                return Err(error)

        except Exception as e:
            return Err(CoreError(f"Failed to fix single video: {e}"))

    def _construct_new_path(
        self, 
        entry_data: Dict[str, Any], 
        video_id: str
    ) -> Result[Dict[str, str], CoreError]:
        """
        Construct old and new paths for file operation
        
        Process:
        1. Extract metadata from database entry
        2. Construct old path using current file location
        3. Construct new path using proper title/channel
        4. Sanitize all path components for filesystem compatibility
        """
        try:
            # Load configuration for base path
            config_manager = SecureConfigManager()
            config_result = config_manager.load_config()
            config = unwrap_ok(config_result)
            base_path = config.storage.nextcloud_path.rstrip("/")
            
            # Extract metadata
            new_kanal = entry_data.get('kanal', 'Unknown')
            new_titel = entry_data.get('titel', 'Unknown')
            
            # Use current WebDAV path as old path
            old_path = entry_data["webdav_path"]
            
            # Construct NEW path (proper title schema)
            new_safe_kanal = self._sanitize_path_component(new_kanal)
            new_safe_titel = self._sanitize_path_component(new_titel)
            new_remote_path = f"{base_path}/{new_safe_kanal}/{new_safe_titel}.mp4"
            
            return Ok({
                "old_path": old_path,
                "new_path": new_remote_path,
                "old_filename": old_path.split('/')[-1],
                "new_filename": f"{new_safe_kanal}/{new_safe_titel}.mp4",
                "directory_changed": new_safe_kanal not in old_path
            })
            
        except Exception as e:
            return Err(CoreError(f"Path construction failed: {e}"))

    def _extract_video_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract YouTube video ID from various URL formats
        
        Examples:
        https://www.youtube.com/watch?v=hrf6XHgAitE → hrf6XHgAitE
        https://youtu.be/hrf6XHgAitE            → hrf6XHgAitE
        https://youtube.com/watch?v=hrf6XHgAitE → hrf6XHgAitE
        """
        try:
            if not url:
                return None
            
            # Method 1: Standard YouTube watch URL
            if 'youtube.com/watch' in url:
                parsed = urlparse(url)
                query_params = parse_qs(parsed.query)
                if 'v' in query_params:
                    video_id = query_params['v'][0]
                    # Validate video ID format (11 chars, alphanumeric + _ -)
                    if re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
                        return video_id
            
            # Method 2: Short YouTube URL (youtu.be)
            elif 'youtu.be/' in url:
                parsed = urlparse(url)
                video_id = parsed.path.lstrip('/')
                if re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
                    return video_id
            
            # Method 3: Regex fallback for edge cases
            patterns = [
                r'(?:v=|/)([0-9A-Za-z_-]{11})(?:\S+)?',
                r'youtube\.com/embed/([0-9A-Za-z_-]{11})',
                r'youtube\.com/v/([0-9A-Za-z_-]{11})',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract video ID from URL {url}: {e}")
            return None

    def _create_empty_nextcloud_report(self, dry_run: bool) -> NextcloudVideoFixReport:
        """Create empty report when no files found"""
        return NextcloudVideoFixReport(
            files_found=0,
            matched=0,
            processed=0,
            fixed=0,
            failed=0,
            processing_time_seconds=0.0,
            results=[],
            errors=[],
            dry_run=dry_run
        )

    def _create_nextcloud_final_report(
        self,
        found_files: List[Dict[str, str]],
        matched_files: List[Dict[str, Any]],
        processing_data: Dict[str, Any],
        processing_time: float,
        dry_run: bool
    ) -> NextcloudVideoFixReport:
        """Create final comprehensive report"""
        return NextcloudVideoFixReport(
            files_found=len(found_files),
            matched=len(matched_files),
            processed=processing_data["processed"],
            fixed=processing_data["fixed"],
            failed=processing_data["failed"],
            processing_time_seconds=processing_time,
            results=processing_data["results"],
            errors=processing_data["errors"],
            dry_run=dry_run
        )

    def _update_nextcloud_link_in_db(self, entry_id: int, new_link: str) -> None:
        """Update nextcloud_link in database"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE processed_videos SET nextcloud_link = ? WHERE id = ?",
                    (new_link, entry_id)
                )
                conn.commit()
                
                self.logger.debug(
                    f"Updated nextcloud_link for entry {entry_id}",
                    extra={"entry_id": entry_id, "new_link": new_link}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to update nextcloud_link for entry {entry_id}: {e}")

    # =============================================================================
    # NEW: PIPELINE RECOVERY SYSTEM
    # =============================================================================

    def recover_pipeline_stream(
        self, 
        stream: Literal["A", "B", "both"], 
        entry_ids: Optional[List[int]] = None,
        fix_all: bool = False,
        dry_run: bool = True
    ) -> Result[PipelineRecoveryReport, CoreError]:
        """
        ENHANCED: Pipeline Recovery mit Fix-All Option
        
        Args:
            stream: "A" (Video), "B" (Transcript), "both"
            entry_ids: Liste der Datenbank-IDs (optional wenn fix_all=True)
            fix_all: Alle Einträge mit fehlenden Attributen fixen
            dry_run: Simulation ohne echte Verarbeitung
        """
        try:
            start_time = time.time()
            
            with log_feature(f"pipeline_recovery_stream_{stream}") as feature:
                # Determine entries to process
                if fix_all:
                    missing_result = self.find_missing_attributes()
                    if isinstance(missing_result, Err):
                        return missing_result
                    
                    missing_entries = unwrap_ok(missing_result)
                    
                    # Filter by stream type
                    if stream == "A":
                        entry_ids = [e['id'] for e in missing_entries if e['categories']['stream_a']]
                    elif stream == "B":
                        entry_ids = [e['id'] for e in missing_entries if e['categories']['stream_b']]
                    else:  # both
                        entry_ids = [e['id'] for e in missing_entries]
                    
                    self.logger.info(f"Fix-All mode: Found {len(entry_ids)} entries needing Stream {stream} recovery")
                
                elif not entry_ids:
                    return Err(CoreError("Either provide entry_ids or use --fix-all flag"))
                
                # Load config with secrets
                config_manager = SecureConfigManager()
                config_result = config_manager.load_config()
                
                if isinstance(config_result, Err):
                    return config_result
                
                config = unwrap_ok(config_result)
                config_dict = self._prepare_config_dict(config, config_manager)
                
                report = PipelineRecoveryReport(
                    stream_type=stream,
                    processed_entries=0,
                    successful_recoveries=0,
                    failed_recoveries=0,
                    processing_time_seconds=0.0,
                    recovery_details=[],
                    errors=[],
                    fix_all_mode=fix_all
                )
                
                for entry_id in entry_ids:
                    try:
                        recovery_result = self._recover_single_entry(
                            entry_id, stream, config_dict, dry_run
                        )
                        
                        if isinstance(recovery_result, Ok):
                            details = unwrap_ok(recovery_result)
                            report.successful_recoveries += 1
                            report.recovery_details.append(details)
                            
                            self.logger.info(
                                f"Recovery successful for entry {entry_id}",
                                extra={"stream": stream, "entry_id": entry_id}
                            )
                        else:
                            error = unwrap_err(recovery_result)
                            report.failed_recoveries += 1
                            report.errors.append(f"Entry {entry_id}: {error.message}")
                            
                            self.logger.error(
                                f"Recovery failed for entry {entry_id}: {error.message}"
                            )
                    
                    except Exception as e:
                        report.failed_recoveries += 1
                        report.errors.append(f"Entry {entry_id}: Unexpected error: {e}")
                        
                    finally:
                        report.processed_entries += 1
                
                report.processing_time_seconds = time.time() - start_time
                
                self.logger.info(
                    f"Pipeline recovery completed",
                    extra={
                        "stream": stream,
                        "processed": report.processed_entries,
                        "successful": report.successful_recoveries,
                        "failed": report.failed_recoveries,
                        "processing_time": report.processing_time_seconds,
                        "fix_all_mode": fix_all
                    }
                )
                
                return Ok(report)

        except Exception as e:
            context = ErrorContext.create(
                "recover_pipeline_stream",
                input_data={"stream": stream, "fix_all": fix_all}
            )
            return Err(CoreError(f"Pipeline recovery failed: {e}", context))

    def _recover_single_entry(
        self, 
        entry_id: int, 
        stream: Literal["A", "B", "both"], 
        config_dict: dict,
        dry_run: bool
    ) -> Result[Dict[str, Any], CoreError]:
        """Recovery für einzelnen Eintrag"""
        try:
            # Load entry from database
            entry_result = self.get_entry_by_id(entry_id)
            if isinstance(entry_result, Err):
                return entry_result
            
            entry_data = unwrap_ok(entry_result)
            
            recovery_actions = []
            
            if stream in ["A", "both"]:
                # Stream A Recovery: Video processing
                stream_a_result = self._recover_stream_a(entry_data, config_dict, dry_run)
                if isinstance(stream_a_result, Ok):
                    recovery_actions.extend(unwrap_ok(stream_a_result))
            
            if stream in ["B", "both"]:
                # Stream B Recovery: LLM + Trilium processing  
                stream_b_result = self._recover_stream_b(entry_data, config_dict, dry_run)
                if isinstance(stream_b_result, Ok):
                    recovery_actions.extend(unwrap_ok(stream_b_result))
            
            return Ok({
                "entry_id": entry_id,
                "titel": entry_data.get("titel", "Unknown"),
                "stream_type": stream,
                "actions_performed": recovery_actions,
                "dry_run": dry_run
            })

        except Exception as e:
            return Err(CoreError(f"Single entry recovery failed: {e}"))

    def _recover_stream_a(
        self, entry_data: Dict[str, Any], config_dict: dict, dry_run: bool
    ) -> Result[List[str], CoreError]:
        """Stream A (Video) Recovery"""
        actions = []
        
        try:
            # Check missing video attributes
            missing_video_attrs = []
            video_fields = ['nextcloud_link', 'video_stream_success']
            
            for field in video_fields:
                if not entry_data.get(field) or entry_data.get(field) == 0:
                    missing_video_attrs.append(field)
            
            if missing_video_attrs and not dry_run:
                # Reconstruct ProcessObject for video reprocessing
                process_obj = self._reconstruct_process_object(entry_data)
                
                # Re-run nextcloud upload if missing
                if 'nextcloud_link' in missing_video_attrs:
                    upload_result = upload_to_nextcloud_for_process_object_dict(process_obj, config_dict)
                    
                    if isinstance(upload_result, Ok):
                        updated_obj = unwrap_ok(upload_result)
                        # Update database with new nextcloud link
                        self._update_nextcloud_info(entry_data['id'], updated_obj.nextcloud_link)
                        actions.append("nextcloud_upload_completed")
                    else:
                        actions.append("nextcloud_upload_failed")
            
            elif dry_run:
                actions.append(f"would_recover: {missing_video_attrs}")
            
            return Ok(actions)

        except Exception as e:
            return Err(CoreError(f"Stream A recovery failed: {e}"))

    def _recover_stream_b(
        self, entry_data: Dict[str, Any], config_dict: dict, dry_run: bool
    ) -> Result[List[str], CoreError]:
        """Stream B (Transcript) Recovery with LLM + Trilium"""
        actions = []
        
        try:
            # Check missing transcript attributes
            missing_transcript_attrs = []
            transcript_fields = ['bearbeiteter_transkript', 'trilium_note_id', 'trilium_link']
            
            for field in transcript_fields:
                if not entry_data.get(field):
                    missing_transcript_attrs.append(field)
            
            if missing_transcript_attrs and not dry_run:
                # Reconstruct TranskriptObject for transcript reprocessing
                transcript_obj = self._reconstruct_transcript_object(entry_data)
                
                # Re-run LLM processing if bearbeiteter_transkript missing
                if 'bearbeiteter_transkript' in missing_transcript_attrs:
                    llm_result = process_transcript_with_llm_dict(transcript_obj, config_dict)
                    
                    if isinstance(llm_result, Ok):
                        processed_transcript = unwrap_ok(llm_result)
                        # Update database with new LLM results
                        self._update_llm_results(entry_data['id'], processed_transcript)
                        actions.append("llm_processing_completed")
                        transcript_obj = processed_transcript
                    else:
                        actions.append("llm_processing_failed")
                
                # Re-run Trilium upload if trilium_note_id missing
                if 'trilium_note_id' in missing_transcript_attrs and transcript_obj.bearbeiteter_transkript:
                    trilium_result = upload_to_trilium_dict(transcript_obj, config_dict)
                    
                    if isinstance(trilium_result, Ok):
                        trilium_obj = unwrap_ok(trilium_result) 
                        # Update database with Trilium results
                        self.update_trilium_info(entry_data['id'], trilium_obj.trilium_link, trilium_obj.trilium_note_id)
                        actions.append("trilium_upload_completed")
                    else:
                        actions.append("trilium_upload_failed")
            
            elif dry_run:
                actions.append(f"would_recover: {missing_transcript_attrs}")
            
            return Ok(actions)

        except Exception as e:
            return Err(CoreError(f"Stream B recovery failed: {e}"))

    # =============================================================================
    # NEW: UPDATE FINAL SUCCESS STATUS
    # =============================================================================

    def update_final_success_status(
        self,
        batch_size: int = 50,
        dry_run: bool = True
    ) -> Result[Dict[str, Any], CoreError]:
        """
        NEW: Update final_success status based on stream success states
        
        Sets final_success = 1 and processing_stage = 'completed' for entries where:
        - video_stream_success = 1 AND transcript_stream_success = 1
        - Also clears error_messages fields
        
        Args:
            batch_size: Number of entries to process per batch
            dry_run: Simulation mode without actual updates
            
        Returns:
            Report with processing statistics and details
        """
        try:
            start_time = time.time()
            
            with log_feature("update_final_success_status") as feature:
                # Find entries that need final_success update
                candidates_result = self._find_final_success_candidates()
                if isinstance(candidates_result, Err):
                    return candidates_result
                
                candidates = unwrap_ok(candidates_result)
                feature.add_metric("candidates_found", len(candidates))
                
                if not candidates:
                    self.logger.info("No entries need final_success status update")
                    return Ok({
                        "candidates_found": 0,
                        "processed": 0,
                        "updated": 0,
                        "failed": 0,
                        "processing_time_seconds": 0.0,
                        "entries": [],
                        "errors": [],
                        "dry_run": dry_run
                    })
                
                # Process in batches
                processed = 0
                updated = 0
                failed = 0
                results = []
                errors = []
                
                self.logger.info(
                    f"Starting final_success status update for {len(candidates)} entries",
                    extra={
                        "total_candidates": len(candidates),
                        "batch_size": batch_size,
                        "dry_run": dry_run
                    }
                )
                
                for i in range(0, len(candidates), batch_size):
                    batch = candidates[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(candidates) + batch_size - 1) // batch_size
                    
                    self.logger.info(
                        f"Processing batch {batch_num}/{total_batches} ({len(batch)} entries)"
                    )
                    
                    for entry in batch:
                        processed += 1
                        
                        try:
                            if dry_run:
                                # Simulate update for dry run
                                updated += 1
                                results.append({
                                    "entry_id": entry["id"],
                                    "status": "would_update",
                                    "titel": entry["titel"],
                                    "current_final_success": entry["final_success"],
                                    "current_processing_stage": entry.get("processing_stage", "unknown"),
                                    "has_error_messages": bool(entry.get("error_messages"))
                                })
                            else:
                                # Perform actual update
                                update_result = self._update_single_final_success(entry)
                                
                                if isinstance(update_result, Ok):
                                    updated += 1
                                    results.append({
                                        "entry_id": entry["id"],
                                        "status": "updated",
                                        "titel": entry["titel"],
                                        "old_final_success": entry["final_success"],
                                        "new_final_success": 1,
                                        "old_processing_stage": entry.get("processing_stage", "unknown"),
                                        "new_processing_stage": "completed",
                                        "error_messages_cleared": bool(entry.get("error_messages"))
                                    })
                                    
                                    self.logger.debug(
                                        f"Updated final_success for entry {entry['id']}",
                                        extra={
                                            "entry_id": entry["id"],
                                            "titel": entry["titel"]
                                        }
                                    )
                                else:
                                    failed += 1
                                    error = unwrap_err(update_result)
                                    error_msg = f"Entry {entry['id']}: {error.message}"
                                    errors.append(error_msg)
                                    results.append({
                                        "entry_id": entry["id"],
                                        "status": "failed",
                                        "titel": entry["titel"],
                                        "error": error.message
                                    })
                        
                        except Exception as e:
                            failed += 1
                            error_msg = f"Entry {entry['id']}: Unexpected error: {e}"
                            errors.append(error_msg)
                            results.append({
                                "entry_id": entry["id"],
                                "status": "failed",
                                "titel": entry.get("titel", "Unknown"),
                                "error": str(e)
                            })
                            
                            self.logger.error(
                                f"Unexpected error updating entry {entry['id']}: {e}"
                            )
                
                processing_time = time.time() - start_time
                
                # Create summary report
                summary = {
                    "candidates_found": len(candidates),
                    "processed": processed,
                    "updated": updated,
                    "failed": failed,
                    "processing_time_seconds": processing_time,
                    "entries": results,
                    "errors": errors,
                    "dry_run": dry_run
                }
                
                feature.add_metric("processed", processed)
                feature.add_metric("updated", updated)
                feature.add_metric("failed", failed)
                
                self.logger.info(
                    f"✅ Final success status update completed",
                    extra=summary
                )
                
                return Ok(summary)

        except Exception as e:
            context = ErrorContext.create(
                "update_final_success_status",
                input_data={"batch_size": batch_size, "dry_run": dry_run},
                suggestions=["Check database connectivity", "Verify table structure"]
            )
            return Err(CoreError(f"Update final success status failed: {e}", context))

    def _find_final_success_candidates(self) -> Result[List[Dict[str, Any]], CoreError]:
        """Find entries that need final_success status update"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                
                # Query for entries where both streams succeeded but final_success is not 1
                cursor.execute("""
                    SELECT id, titel, kanal, video_stream_success, transcript_stream_success, 
                           final_success, processing_stage, error_messages, date_created
                    FROM processed_videos 
                    WHERE video_stream_success = 1 
                    AND transcript_stream_success = 1
                    AND (final_success != 1 OR final_success IS NULL OR processing_stage != 'completed')
                    ORDER BY date_created DESC
                """)
                
                rows = cursor.fetchall()
                
                candidates = []
                for row in rows:
                    candidates.append({
                        "id": row[0],
                        "titel": row[1],
                        "kanal": row[2],
                        "video_stream_success": row[3],
                        "transcript_stream_success": row[4],
                        "final_success": row[5],
                        "processing_stage": row[6],
                        "error_messages": row[7],
                        "date_created": row[8]
                    })
                
                self.logger.info(
                    f"Found {len(candidates)} candidates for final_success update",
                    extra={
                        "total_candidates": len(candidates),
                        "sample_titles": [c["titel"][:50] for c in candidates[:3]]
                    }
                )
                
                return Ok(candidates)

        except Exception as e:
            context = ErrorContext.create(
                "find_final_success_candidates",
                suggestions=["Check database connection", "Verify table structure"]
            )
            return Err(CoreError(f"Failed to find final_success candidates: {e}", context))

    def _update_single_final_success(self, entry: Dict[str, Any]) -> Result[None, CoreError]:
        """Update final_success status for a single entry"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                
                # Update final_success, processing_stage, and clear error_messages
                cursor.execute("""
                    UPDATE processed_videos 
                    SET final_success = 1, 
                        processing_stage = 'completed',
                        error_messages = NULL
                    WHERE id = ?
                """, (entry["id"],))
                
                if cursor.rowcount == 0:
                    return Err(CoreError(f"No entry found with ID {entry['id']}"))
                
                conn.commit()
                
                self.logger.debug(
                    f"Updated final_success status for entry {entry['id']}",
                    extra={
                        "entry_id": entry["id"],
                        "titel": entry["titel"],
                        "old_final_success": entry["final_success"],
                        "old_processing_stage": entry.get("processing_stage"),
                        "had_error_messages": bool(entry.get("error_messages"))
                    }
                )
                
                return Ok(None)

        except Exception as e:
            return Err(CoreError(f"Failed to update final_success for entry {entry['id']}: {e}"))

    # =============================================================================
    # ENHANCED: AUTOMATIC SUCCESS UPDATE INTEGRATION
    # =============================================================================

    def _auto_update_success_after_operation(
        self, 
        operation_name: str,
        operation_result: Dict[str, Any]
    ) -> None:
        """
        Automatically run success update after major operations
        
        Args:
            operation_name: Name of the operation that just completed
            operation_result: Result data from the operation
        """
        try:
            # Only run if the operation was successful and not a dry run
            if (operation_result.get("successful", 0) > 0 and 
                not operation_result.get("dry_run", False)):
                
                self.logger.info(
                    f"Auto-triggering final_success update after {operation_name}",
                    extra={
                        "operation": operation_name,
                        "successful_operations": operation_result.get("successful", 0)
                    }
                )
                
                # Run success update (small batch size for automatic runs)
                success_result = self.update_final_success_status(
                    batch_size=20, 
                    dry_run=False
                )
                
                if isinstance(success_result, Ok):
                    success_data = unwrap_ok(success_result)
                    self.logger.info(
                        f"Auto final_success update completed",
                        extra={
                            "after_operation": operation_name,
                            "candidates_found": success_data["candidates_found"],
                            "updated": success_data["updated"]
                        }
                    )
                else:
                    error = unwrap_err(success_result)
                    self.logger.warning(
                        f"Auto final_success update failed after {operation_name}: {error.message}"
                    )
            
        except Exception as e:
            self.logger.warning(
                f"Auto final_success update failed after {operation_name}: {e}"
            )

    # =============================================================================
    # NEW: FIX ALL TRILIUM TITLES
    # =============================================================================

    def fix_all_trilium_titles(
        self,
        batch_size: int = 10,
        dry_run: bool = True
    ) -> Result[Dict[str, Any], CoreError]:
        """
        NEW: Fix all Trilium note titles to match database titles
        
        Args:
            batch_size: Number of entries to process per batch
            dry_run: Simulation mode without actual updates
            
        Returns:
            Report with processing statistics and details
        """
        try:
            start_time = time.time()
            
            with log_feature("fix_all_trilium_titles") as feature:
                # Find all entries with trilium_note_id
                entries_result = self._find_entries_with_trilium_notes()
                if isinstance(entries_result, Err):
                    return entries_result
                
                entries = unwrap_ok(entries_result)
                feature.add_metric("total_entries_found", len(entries))
                
                if not entries:
                    self.logger.info("No entries with Trilium notes found")
                    return Ok({
                        "total_found": 0,
                        "processed": 0,
                        "successful": 0,
                        "failed": 0,
                        "skipped": 0,
                        "processing_time_seconds": 0.0,
                        "entries": [],
                        "errors": [],
                        "dry_run": dry_run
                    })
                
                # Process in batches
                processed = 0
                successful = 0
                failed = 0
                skipped = 0
                results = []
                errors = []
                
                self.logger.info(
                    f"Starting Trilium title sync for {len(entries)} entries",
                    extra={
                        "total_entries": len(entries),
                        "batch_size": batch_size,
                        "dry_run": dry_run
                    }
                )
                
                for i in range(0, len(entries), batch_size):
                    batch = entries[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(entries) + batch_size - 1) // batch_size
                    
                    self.logger.info(
                        f"Processing batch {batch_num}/{total_batches} ({len(batch)} entries)"
                    )
                    
                    for entry in batch:
                        processed += 1
                        
                        try:
                            result = self._fix_single_trilium_title(entry, dry_run)
                            
                            if isinstance(result, Ok):
                                update_result = unwrap_ok(result)
                                
                                if update_result["updated"]:
                                    successful += 1
                                    results.append({
                                        "entry_id": entry["id"],
                                        "status": "success",
                                        "trilium_note_id": entry["trilium_note_id"],
                                        "title": entry["titel"],
                                        "updated": True
                                    })
                                else:
                                    skipped += 1
                                    results.append({
                                        "entry_id": entry["id"],
                                        "status": "skipped",
                                        "trilium_note_id": entry["trilium_note_id"],
                                        "reason": update_result.get("reason", "No update needed")
                                    })
                                
                                self.logger.debug(
                                    f"Entry {entry['id']} processed successfully",
                                    extra={
                                        "entry_id": entry["id"],
                                        "trilium_note_id": entry["trilium_note_id"],
                                        "updated": update_result["updated"]
                                    }
                                )
                            else:
                                failed += 1
                                error = unwrap_err(result)
                                error_msg = f"Entry {entry['id']}: {error.message}"
                                errors.append(error_msg)
                                results.append({
                                    "entry_id": entry["id"],
                                    "status": "failed",
                                    "trilium_note_id": entry["trilium_note_id"],
                                    "error": error.message
                                })
                                
                                self.logger.warning(
                                    f"Failed to update Trilium title for entry {entry['id']}",
                                    extra={
                                        "entry_id": entry["id"],
                                        "trilium_note_id": entry["trilium_note_id"],
                                        "error": error.message
                                    }
                                )
                        
                        except Exception as e:
                            failed += 1
                            error_msg = f"Entry {entry['id']}: Unexpected error: {e}"
                            errors.append(error_msg)
                            results.append({
                                "entry_id": entry["id"],
                                "status": "failed",
                                "trilium_note_id": entry.get("trilium_note_id"),
                                "error": str(e)
                            })
                            
                            self.logger.error(
                                f"Unexpected error processing entry {entry['id']}: {e}"
                            )
                
                processing_time = time.time() - start_time
                
                # Create summary report
                summary = {
                    "total_found": len(entries),
                    "processed": processed,
                    "successful": successful,
                    "failed": failed,
                    "skipped": skipped,
                    "processing_time_seconds": processing_time,
                    "entries": results,
                    "errors": errors,
                    "dry_run": dry_run
                }
                
                feature.add_metric("processed", processed)
                feature.add_metric("successful", successful)
                feature.add_metric("failed", failed)
                feature.add_metric("skipped", skipped)
                
                self.logger.info(
                    f"✅ Trilium title sync completed",
                    extra=summary
                )
                
                return Ok(summary)

        except Exception as e:
            context = ErrorContext.create(
                "fix_all_trilium_titles",
                input_data={"batch_size": batch_size, "dry_run": dry_run},
                suggestions=["Check Trilium connectivity", "Verify database access"]
            )
            return Err(CoreError(f"Fix all Trilium titles failed: {e}", context))

    def _find_entries_with_trilium_notes(self) -> Result[List[Dict[str, Any]], CoreError]:
        """Find all database entries that have trilium_note_id"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                
                # Query for entries with trilium_note_id
                cursor.execute("""
                    SELECT id, titel, kanal, trilium_note_id, trilium_link, date_created
                    FROM processed_videos 
                    WHERE trilium_note_id IS NOT NULL 
                    AND trilium_note_id != ''
                    ORDER BY date_created DESC
                """)
                
                rows = cursor.fetchall()
                
                entries = []
                for row in rows:
                    entries.append({
                        "id": row[0],
                        "titel": row[1],
                        "kanal": row[2],
                        "trilium_note_id": row[3],
                        "trilium_link": row[4],
                        "date_created": row[5]
                    })
                
                self.logger.info(
                    f"Found {len(entries)} entries with Trilium notes",
                    extra={
                        "total_entries": len(entries),
                        "sample_titles": [e["titel"][:50] for e in entries[:3]]
                    }
                )
                
                return Ok(entries)

        except Exception as e:
            context = ErrorContext.create(
                "find_entries_with_trilium_notes",
                suggestions=["Check database connection", "Verify table structure"]
            )
            return Err(CoreError(f"Failed to find entries with Trilium notes: {e}", context))

    def _fix_single_trilium_title(
        self, 
        entry: Dict[str, Any], 
        dry_run: bool
    ) -> Result[Dict[str, Any], CoreError]:
        """Fix Trilium note title for a single entry"""
        try:
            trilium_note_id = entry["trilium_note_id"]
            titel = entry["titel"]
            kanal = entry["kanal"]
            
            # Construct new note title: "Title | Channel"
            new_title = f"{titel} | {kanal}"
            
            result = {
                "entry_id": entry["id"],
                "trilium_note_id": trilium_note_id,
                "new_title": new_title,
                "updated": False,
                "dry_run": dry_run
            }
            
            if dry_run:
                result["reason"] = "Dry run - would update title"
                result["updated"] = True  # Mark as would-be-updated for reporting
                return Ok(result)
            
            # Perform actual Trilium note title update
            update_result = self._update_trilium_note_title(
                trilium_note_id, titel, kanal
            )
            
            if isinstance(update_result, Ok):
                result["updated"] = True
                result["reason"] = "Title updated successfully"
                return Ok(result)
            else:
                error = unwrap_err(update_result)
                return Err(error)

        except Exception as e:
            return Err(CoreError(f"Failed to fix single Trilium title: {e}"))

    # =============================================================================
    # NEW: TRILIUM + NEXTCLOUD INTEGRATED METADATA FIXER  
    # =============================================================================

    def fix_metadata_with_integrations(
        self, 
        entry_id: int, 
        titel: str, 
        kanal: str
    ) -> Result[Dict[str, Any], CoreError]:
        """
        ENHANCED: Metadaten-Fix mit Trilium + Nextcloud Integration
        """
        try:
            with log_feature("metadata_fix_with_integrations") as feature:
                # Load current entry
                entry_result = self.get_entry_by_id(entry_id)
                if isinstance(entry_result, Err):
                    return entry_result
                
                entry_data = unwrap_ok(entry_result)
                trilium_note_id = entry_data.get('trilium_note_id')
                nextcloud_link = entry_data.get('nextcloud_link')
                
                results = {
                    "entry_id": entry_id,
                    "database_updated": False,
                    "trilium_updated": False,
                    "nextcloud_updated": False,
                    "trilium_note_id": trilium_note_id,
                    "nextcloud_link": nextcloud_link,
                    "errors": []
                }
                
                # 1. Update database metadata
                db_update_result = self.update_metadata(entry_id, titel, kanal)
                if isinstance(db_update_result, Ok):
                    results["database_updated"] = True
                else:
                    error = unwrap_err(db_update_result)
                    results["errors"].append(f"Database update failed: {error.message}")
                    return Ok(results)  # Return early if DB update fails
                
                # 2. Update Trilium note title if trilium_note_id exists
                if trilium_note_id:
                    trilium_update_result = self._update_trilium_note_title(
                        trilium_note_id, titel, kanal
                    )
                    
                    if isinstance(trilium_update_result, Ok):
                        results["trilium_updated"] = True
                        self.logger.info(f"Trilium note title updated for entry {entry_id}")
                    else:
                        error = unwrap_err(trilium_update_result)
                        results["errors"].append(f"Trilium update failed: {error.message}")
                
                # 3. NEW: Update Nextcloud filename if nextcloud_link exists
                if nextcloud_link:
                    nextcloud_update_result = self._update_nextcloud_filename(
                        entry_data, titel, kanal
                    )
                    
                    if isinstance(nextcloud_update_result, Ok):
                        nc_result = unwrap_ok(nextcloud_update_result)
                        results["nextcloud_updated"] = nc_result.updated
                        results["nextcloud_old_filename"] = nc_result.old_filename
                        results["nextcloud_new_filename"] = nc_result.new_filename
                        
                        if nc_result.updated:
                            self.logger.info(f"Nextcloud filename updated for entry {entry_id}")
                        
                        if nc_result.error_message:
                            results["errors"].append(f"Nextcloud update warning: {nc_result.error_message}")
                    else:
                        error = unwrap_err(nextcloud_update_result)
                        results["errors"].append(f"Nextcloud update failed: {error.message}")
                
                self.logger.info(
                    f"Metadata fix with integrations completed",
                    extra={
                        "entry_id": entry_id,
                        "database_updated": results["database_updated"],
                        "trilium_updated": results["trilium_updated"],
                        "nextcloud_updated": results["nextcloud_updated"],
                        "error_count": len(results["errors"])
                    }
                )
                
                return Ok(results)

        except Exception as e:
            context = ErrorContext.create(
                "fix_metadata_with_integrations",
                input_data={"entry_id": entry_id, "titel": titel[:50]}
            )
            return Err(CoreError(f"Metadata fix with integrations failed: {e}", context))

    def _update_trilium_note_title(
        self, 
        note_id: str, 
        titel: str, 
        kanal: str
    ) -> Result[None, CoreError]:
        """Update Trilium note title"""
        try:
            # Load config for Trilium access
            config_manager = SecureConfigManager()
            config_result = config_manager.load_config()
            
            if isinstance(config_result, Err):
                return config_result
            
            config = unwrap_ok(config_result)
            config_dict = self._prepare_config_dict(config, config_manager)
            
            # Initialize Trilium client
            uploader = TriliumUploader(config_dict)
            client_result = uploader._init_trilium_client()
            
            if isinstance(client_result, Err):
                return client_result
            
            etapi = unwrap_ok(client_result)
            
            # Update note title
            new_title = f"{titel} | {kanal}"
            update_result = etapi.patch_note(
                noteId=note_id,
                title=new_title
            )
            
            if update_result:
                self.logger.debug(
                    f"Trilium note title updated successfully",
                    extra={
                        "note_id": note_id,
                        "new_title": new_title
                    }
                )
                return Ok(None)
            else:
                return Err(CoreError("Failed to update Trilium note title"))

        except Exception as e:
            return Err(CoreError(f"Trilium note title update failed: {e}"))

    def _update_nextcloud_filename(
        self,
        entry_data: Dict[str, Any],
        titel: str,
        kanal: str
    ) -> Result[NextcloudUpdateResult, CoreError]:
        """
        ENHANCED: Update Nextcloud filename via WebDAV based on corrected metadata
        """
        try:
            nextcloud_link = entry_data.get('nextcloud_link')
            if not nextcloud_link:
                return Ok(NextcloudUpdateResult(
                    entry_id=entry_data['id'],
                    old_filename=None,
                    new_filename=None,
                    updated=False,
                    error_message="No Nextcloud link found"
                ))
            
            # Construct old and new paths based on metadata
            old_kanal = entry_data.get('kanal', 'Unknown')
            old_titel = entry_data.get('titel', 'Unknown')
            
            # Sanitize for filesystem compatibility
            old_safe_kanal = self._sanitize_path_component(old_kanal)
            old_safe_titel = self._sanitize_path_component(old_titel)
            new_safe_kanal = self._sanitize_path_component(kanal)
            new_safe_titel = self._sanitize_path_component(titel)
            
            # Construct paths: {nextcloud_path}/{kanal}/{titel}.mp4
            config_manager = SecureConfigManager()
            config_result = config_manager.load_config()
            
            if isinstance(config_result, Err):
                return Err(CoreError("Failed to load config for Nextcloud operation"))
            
            config = unwrap_ok(config_result)
            base_path = config.storage.nextcloud_path.rstrip("/")
            
            old_remote_path = f"{base_path}/{old_safe_kanal}/{old_safe_titel}.mp4"
            new_remote_path = f"{base_path}/{new_safe_kanal}/{new_safe_titel}.mp4"
            
            result = NextcloudUpdateResult(
                entry_id=entry_data['id'],
                old_filename=f"{old_safe_kanal}/{old_safe_titel}.mp4",
                new_filename=f"{new_safe_kanal}/{new_safe_titel}.mp4",
                updated=False
            )
            
            # Only update if paths are actually different
            if old_remote_path != new_remote_path:
                rename_result = self._rename_file_in_nextcloud_webdav(
                    old_remote_path, new_remote_path, config
                )
                
                if isinstance(rename_result, Ok):
                    result.updated = True
                    self.logger.info(
                        f"Nextcloud file moved via WebDAV",
                        extra={
                            "entry_id": entry_data['id'],
                            "old_path": old_remote_path,
                            "new_path": new_remote_path
                        }
                    )
                else:
                    error = unwrap_err(rename_result)
                    result.error_message = error.message
            else:
                result.error_message = "File path unchanged - no update needed"
            
            return Ok(result)

        except Exception as e:
            return Err(CoreError(f"Nextcloud filename update failed: {e}"))

    def _sanitize_path_component(self, name: str) -> str:
        """Sanitize path component for Nextcloud filesystem compatibility"""
        # Remove or replace problematic characters for filesystem
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. "
        sanitized = "".join(c if c in safe_chars else "_" for c in name)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized[:100]  # Limit length

    def _rename_file_in_nextcloud_webdav(
        self, 
        old_remote_path: str, 
        new_remote_path: str,
        config: AppConfig
    ) -> Result[None, CoreError]:
        """
        REAL: Rename/Move file in Nextcloud using WebDAV4
        """
        try:
            # Import WebDAV4 client
            try:
                from webdav4.client import Client as WebDAVClient
            except ImportError:
                return Err(CoreError(
                    "webdav4 not available - install with: pip install webdav4"
                ))
            
            # Get WebDAV credentials
            config_manager = SecureConfigManager()
            config_manager.load_config()
            # Load secrets
            username = config.secrets.nextcloud_username
            password_result = config_manager.get_nextcloud_password()
            
            if isinstance(password_result, Err):
                return Err(CoreError("Failed to get Nextcloud password"))
            
            password = unwrap_ok(password_result)
            base_url = config.storage.nextcloud_base_url
            
            # Initialize WebDAV client
            client = WebDAVClient(
                base_url=base_url,
                auth=(username, password),
                timeout=60,
                verify=True
            )
            
            # Check if old file exists
            if not client.exists(old_remote_path):
                return Err(CoreError(f"Source file not found: {old_remote_path}"))
            
            # Create target directory if needed
            new_dir = str(Path(new_remote_path).parent)
            if new_dir != str(Path(old_remote_path).parent):
                self._create_webdav_directories(client, new_dir)
            
            # Check if target file already exists
            if client.exists(new_remote_path):
                return Err(CoreError(f"Target file already exists: {new_remote_path}"))
            
            # Perform the move operation
            client.move(old_remote_path, new_remote_path)
            
            self.logger.info(
                f"Successfully moved file in Nextcloud",
                extra={
                    "old_path": old_remote_path,
                    "new_path": new_remote_path,
                    "webdav_base": base_url
                }
            )
            
            return Ok(None)
            
        except Exception as e:
            return Err(CoreError(f"WebDAV file move failed: {e}"))

    def _create_webdav_directories(self, client, remote_dir: str) -> None:
        """Create WebDAV directories recursively"""
        try:
            # Normalize path and split into parts
            normalized_path = remote_dir.strip("/").replace("\\", "/")
            if not normalized_path:
                return  # Root directory always exists

            path_parts = normalized_path.split("/")
            current_path = ""

            for part in path_parts:
                if not part:  # Skip empty parts
                    continue

                current_path = f"{current_path}/{part}" if current_path else f"/{part}"

                # Check if directory exists
                try:
                    if not client.exists(current_path):
                        client.mkdir(current_path)
                        self.logger.debug(f"Created WebDAV directory: {current_path}")
                    else:
                        self.logger.debug(f"WebDAV directory exists: {current_path}")

                except Exception as e:
                    # Check if it's an "already exists" error
                    if (
                        "already exists" in str(e).lower()
                        or "file exists" in str(e).lower()
                    ):
                        self.logger.debug(f"WebDAV directory already exists: {current_path}")
                        continue
                    else:
                        # Real error - but don't fail the whole operation
                        self.logger.warning(f"Failed to create WebDAV directory {current_path}: {e}")
                        
        except Exception as e:
            self.logger.warning(f"WebDAV directory creation failed: {e}")

    # =============================================================================
    # HELPER METHODS
    # =============================================================================

    def _prepare_config_dict(self, config: AppConfig, config_manager: SecureConfigManager) -> dict:
        """Prepare config dict with resolved secrets for workers"""
        config_dict = config.dict()
        
        # Resolve secrets
        llm_key_result = config_manager.get_llm_api_key(config.llm_processing.provider)
        trilium_token_result = config_manager.get_trilium_token()
        
        config_dict["resolved_secrets"] = {
            "llm_api_key": unwrap_ok(llm_key_result) if isinstance(llm_key_result, Ok) else None,
            "trilium_token": unwrap_ok(trilium_token_result) if isinstance(trilium_token_result, Ok) else None
        }
        
        return config_dict

    def _reconstruct_process_object(self, entry_data: Dict[str, Any]) -> ProcessObject:
        """Reconstruct ProcessObject from database entry"""
        from datetime import datetime
        
        obj = ProcessObject(
            titel=entry_data.get('titel', 'Unknown'),
            kanal=entry_data.get('kanal', 'Unknown'),
            länge=entry_data.get('länge'),
            upload_date=datetime.fromisoformat(entry_data['upload_date']) 
            if entry_data.get('upload_date') else datetime.now(),
            original_url=entry_data.get('original_url', '')
        )
        
        # Set existing attributes
        if entry_data.get('video_path'):
            obj.video_path = Path(entry_data['video_path'])
        if entry_data.get('audio_path'):
            obj.audio_path = Path(entry_data['audio_path'])
        if entry_data.get('transkript'):
            obj.transkript = entry_data['transkript']
            
        return obj

    def _reconstruct_transcript_object(self, entry_data: Dict[str, Any]) -> TranskriptObject:
        """Reconstruct TranskriptObject from database entry"""  
        process_obj = self._reconstruct_process_object(entry_data)
        transcript_obj = TranskriptObject.from_process_object(process_obj)
        
        # Set transcript-specific attributes
        if entry_data.get('bearbeiteter_transkript'):
            transcript_obj.bearbeiteter_transkript = entry_data['bearbeiteter_transkript']
        if entry_data.get('llm_model'):
            transcript_obj.model = entry_data['llm_model'] 
        if entry_data.get('llm_tokens'):
            transcript_obj.tokens = entry_data['llm_tokens']
        if entry_data.get('llm_cost'):  
            transcript_obj.cost = entry_data['llm_cost']
            
        return transcript_obj

    def _update_llm_results(self, entry_id: int, transcript_obj: TranskriptObject) -> None:
        """Update database with LLM processing results"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE processed_videos 
                    SET bearbeiteter_transkript = ?, llm_model = ?, llm_tokens = ?, llm_cost = ?, llm_processing_time = ?
                    WHERE id = ?
                """, (
                    transcript_obj.bearbeiteter_transkript,
                    transcript_obj.model,
                    transcript_obj.tokens, 
                    transcript_obj.cost,
                    transcript_obj.processing_time,
                    entry_id
                ))
                
                self.logger.debug(f"LLM results updated for entry {entry_id}")

        except Exception as e:
            self.logger.error(f"Failed to update LLM results for entry {entry_id}: {e}")

    def _update_nextcloud_info(self, entry_id: int, nextcloud_link: str) -> None:
        """Update database with Nextcloud upload results"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE processed_videos 
                    SET nextcloud_link = ?, video_stream_success = 1
                    WHERE id = ?
                """, (nextcloud_link, entry_id))
                
                self.logger.debug(f"Nextcloud info updated for entry {entry_id}")

        except Exception as e:
            self.logger.error(f"Failed to update Nextcloud info for entry {entry_id}: {e}")


# =============================================================================
# ENHANCED METADATA FIXER
# =============================================================================

class MetadataFixer:
    """Repariert defekte Video-Metadaten mit erweiterten Integrationen"""

    def __init__(self, config: AppConfig, db_manager: ExtendedDatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.logger = get_logger("MetadataFixer")
        self.extractor = YouTubeMetadataExtractor()

    @log_function(log_performance=True)
    def fix_metadata_batch(self, dry_run: bool = False, batch_size: int = 10, pattern: str = "youtube video #") -> Result[Dict[str, Any], CoreError]:
        """Repariert Metadaten für alle gefundenen Einträge mit Trilium + Nextcloud Integration"""
        try:
            with log_feature("metadata_fix_batch_enhanced") as feature:
                # Backup erstellen (nur bei echtem Run)
                if not dry_run:
                    backup_result = self.db_manager.create_backup()
                    if isinstance(backup_result, Err):
                        return backup_result
                    feature.add_metric("backup_created", str(unwrap_ok(backup_result)))

                # Defekte Einträge finden
                entries_result = self.db_manager.find_broken_metadata_entries(pattern)
                if isinstance(entries_result, Err):
                    return entries_result

                entries = unwrap_ok(entries_result)
                feature.add_metric("entries_found", len(entries))

                if not entries:
                    self.logger.info(f"No entries found with pattern '{pattern}'")
                    return Ok({"processed": 0, "succeeded": 0, "failed": 0, "entries": []})

                # Batch-Verarbeitung
                processed = 0
                succeeded = 0
                failed = 0
                results = []

                for i, entry in enumerate(entries[:batch_size]):
                    feature.checkpoint(f"processing_entry_{i + 1}")
                    
                    self.logger.info(
                        f"Processing entry {i + 1}/{min(len(entries), batch_size)}: {entry['titel'][:50]}...",
                        extra={
                            "entry_id": entry["id"],
                            "original_title": entry["titel"],
                            "progress": f"{i + 1}/{min(len(entries), batch_size)}"
                        }
                    )

                    if not entry.get("original_url"):
                        self.logger.warning(f"Entry {entry['id']} has no original_url, skipping")
                        failed += 1
                        results.append({"entry_id": entry["id"], "status": "failed", "error": "No original URL"})
                        continue

                    # ENHANCED: Neue Metadaten extrahieren und alle Integrationen aktualisieren
                    fix_result = self._fix_single_entry_enhanced(entry, dry_run)
                    processed += 1

                    if isinstance(fix_result, Ok):
                        succeeded += 1
                        results.append({"entry_id": entry["id"], "status": "success", "data": unwrap_ok(fix_result)})
                    else:
                        failed += 1
                        error = unwrap_err(fix_result)
                        results.append({"entry_id": entry["id"], "status": "failed", "error": error.message})

                feature.add_metric("processed", processed)
                feature.add_metric("succeeded", succeeded)
                feature.add_metric("failed", failed)

                summary = {
                    "processed": processed,
                    "succeeded": succeeded, 
                    "failed": failed,
                    "entries": results,
                    "dry_run": dry_run
                }

                self.logger.info(
                    f"✅ Enhanced metadata fix batch completed",
                    extra=summary
                )

                return Ok(summary)

        except Exception as e:
            context = ErrorContext.create(
                "fix_metadata_batch_enhanced",
                input_data={"pattern": pattern, "batch_size": batch_size, "dry_run": dry_run},
                suggestions=["Check database connectivity", "Verify extractor functionality"]
            )
            return Err(CoreError(f"Enhanced batch metadata fix failed: {e}", context))

    def _fix_single_entry_enhanced(self, entry: Dict[str, Any], dry_run: bool) -> Result[Dict[str, Any], CoreError]:
        """ENHANCED: Repariert Metadaten mit Trilium + Nextcloud Integration"""
        try:
            # Neue Metadaten extrahieren
            url = entry["original_url"]
            metadata_result = self.extractor.extract_single_metadata(url)
            
            if isinstance(metadata_result, Err):
                return metadata_result

            process_obj = unwrap_ok(metadata_result)
            new_titel = process_obj.titel
            new_kanal = process_obj.kanal

            self.logger.info(
                f"Extracted new metadata",
                extra={
                    "entry_id": entry["id"],
                    "old_titel": entry["titel"],
                    "new_titel": new_titel,
                    "old_kanal": entry["kanal"],
                    "new_kanal": new_kanal,
                    "dry_run": dry_run
                }
            )

            # ENHANCED: Update mit allen Integrationen (nur wenn nicht dry-run)
            if not dry_run:
                integration_result = self.db_manager.fix_metadata_with_integrations(
                    entry["id"], new_titel, new_kanal
                )
                
                if isinstance(integration_result, Err):
                    return integration_result
                
                integration_data = unwrap_ok(integration_result)
                
                return Ok({
                    "old_titel": entry["titel"],
                    "new_titel": new_titel,
                    "old_kanal": entry["kanal"],
                    "new_kanal": new_kanal,
                    "integrations": integration_data
                })
            else:
                return Ok({
                    "old_titel": entry["titel"],
                    "new_titel": new_titel,
                    "old_kanal": entry["kanal"],
                    "new_kanal": new_kanal,
                    "dry_run": True
                })

        except Exception as e:
            context = ErrorContext.create(
                "fix_single_entry_enhanced",
                input_data={"entry_id": entry.get("id"), "url": entry.get("original_url")},
                suggestions=["Check URL accessibility", "Verify metadata extractor"]
            )
            return Err(CoreError(f"Failed to fix single entry enhanced: {e}", context))


# =============================================================================
# TRANSCRIPT REPROCESSOR (Unchanged but enhanced logging)  
# =============================================================================

class TranscriptReprocessor:
    """Verarbeitet Transkripte erneut mit LLM und sendet zu Trilium"""

    def __init__(self, config: AppConfig, db_manager: ExtendedDatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.logger = get_logger("TranscriptReprocessor")
        
        # Config-Dict für LLM/Trilium
        self.config_dict = self._create_config_dict()

    def _create_config_dict(self) -> Dict[str, Any]:
        """Erstellt config_dict mit resolved secrets"""
        config_manager = SecureConfigManager()
        config_manager.load_config()
        
        config_dict = self.config.dict()
        
        # Resolve secrets
        llm_key_result = config_manager.get_llm_api_key(self.config.llm_processing.provider)
        trilium_token_result = config_manager.get_trilium_token()
        
        config_dict["resolved_secrets"] = {
            "llm_api_key": unwrap_ok(llm_key_result) if isinstance(llm_key_result, Ok) else None,
            "trilium_token": unwrap_ok(trilium_token_result) if isinstance(trilium_token_result, Ok) else None
        }
        
        return config_dict

    @log_function(log_performance=True)
    def reprocess_transcript(self, entry_id: int) -> Result[Dict[str, Any], CoreError]:
        """Verarbeitet Transkript erneut und sendet zu Trilium"""
        try:
            with log_feature("transcript_reprocessing") as feature:
                feature.add_metric("entry_id", entry_id)

                # Entry aus DB holen
                entry_result = self.db_manager.get_entry_by_id(entry_id)
                if isinstance(entry_result, Err):
                    return entry_result

                entry = unwrap_ok(entry_result)
                feature.add_metric("video_title", entry.get("titel", "unknown"))

                # Validierung
                if not entry.get("transkript"):
                    context = ErrorContext.create(
                        "reprocess_transcript",
                        input_data={"entry_id": entry_id},
                        suggestions=["Check if entry has transcript", "Verify transcription was completed"]
                    )
                    return Err(CoreError(f"Entry {entry_id} has no transcript", context))

                # TranskriptObject erstellen
                transcript_obj = self._create_transcript_object_from_entry(entry)
                
                # LLM-Verarbeitung
                self.logger.info(f"Starting LLM processing for entry {entry_id}: {entry['titel']}")
                llm_result = process_transcript_with_llm_dict(transcript_obj, self.config_dict)
                
                if isinstance(llm_result, Err):
                    error = unwrap_err(llm_result)
                    feature.add_metric("llm_success", False)
                    return Err(error)

                processed_transcript = unwrap_ok(llm_result)
                feature.add_metric("llm_success", True)
                feature.add_metric("llm_model", processed_transcript.model)
                feature.add_metric("llm_tokens", processed_transcript.tokens)

                # Trilium-Upload
                self.logger.info(f"Starting Trilium upload for entry {entry_id}")
                trilium_result = upload_to_trilium_dict(processed_transcript, self.config_dict)
                
                if isinstance(trilium_result, Err):
                    error = unwrap_err(trilium_result)
                    feature.add_metric("trilium_success", False)
                    return Err(error)

                uploaded_transcript = unwrap_ok(trilium_result)
                feature.add_metric("trilium_success", True)
                feature.add_metric("trilium_note_id", uploaded_transcript.trilium_note_id)

                # DB aktualisieren mit neuen Trilium-Links
                update_result = self.db_manager.update_trilium_info(
                    entry_id,
                    uploaded_transcript.trilium_link,
                    uploaded_transcript.trilium_note_id
                )
                
                if isinstance(update_result, Err):
                    return update_result

                summary = {
                    "entry_id": entry_id,
                    "video_title": entry["titel"],
                    "llm_model": uploaded_transcript.model,
                    "llm_tokens": uploaded_transcript.tokens,
                    "llm_cost": uploaded_transcript.cost,
                    "trilium_note_id": uploaded_transcript.trilium_note_id,
                    "trilium_link": uploaded_transcript.trilium_link
                }

                self.logger.info(
                    f"✅ Transcript reprocessing completed for entry {entry_id}",
                    extra=summary
                )

                return Ok(summary)

        except Exception as e:
            context = ErrorContext.create(
                "reprocess_transcript",
                input_data={"entry_id": entry_id},
                suggestions=["Check LLM/Trilium configuration", "Verify API keys", "Check network connectivity"]
            )
            return Err(CoreError(f"Transcript reprocessing failed: {e}", context))

    def _create_transcript_object_from_entry(self, entry: Dict[str, Any]) -> TranskriptObject:
        """Erstellt TranskriptObject aus DB-Entry"""
        from datetime import datetime, time as dt_time
        
        # Parse time string back to time object
        länge_str = entry.get("länge", "00:00:00")
        try:
            time_parts = länge_str.split(":")
            if len(time_parts) == 3:
                länge = dt_time(int(time_parts[0]), int(time_parts[1]), int(time_parts[2]))
            else:
                länge = dt_time(0, 0, 0)
        except:
            länge = dt_time(0, 0, 0)

        # Parse upload date
        upload_date = datetime.fromisoformat(entry["upload_date"]) if entry.get("upload_date") else datetime.now()

        return TranskriptObject(
            titel=entry["titel"],
            kanal=entry["kanal"],
            länge=länge,
            upload_date=upload_date,
            original_url=entry["original_url"],
            transkript=entry["transkript"]
        )


# =============================================================================
# ENHANCED CLI ARGUMENT PARSER
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Enhanced Argument Parser mit neuen Commands"""
    parser = argparse.ArgumentParser(
        description="Enhanced YouTube Analyzer Database Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ENHANCED EXAMPLES:

# Basic Operations
  %(prog)s fix-metadata --dry-run                              # Preview metadata fixes
  %(prog)s fix-metadata --batch-size 5                        # Fix max 5 entries with Trilium + Nextcloud
  %(prog)s reprocess-transcript 123                           # Reprocess entry ID 123
  %(prog)s stats                                               # Show database statistics
  %(prog)s check-missing --limit 20                           # Check missing attributes (FIXED categorization)

# NEW: Pipeline Recovery
  %(prog)s recover-pipeline --stream B --entry-ids 45,67,89 --dry-run    # Recover specific LLM/Trilium issues
  %(prog)s recover-pipeline --stream A --fix-all --dry-run               # Fix all Video/Nextcloud issues
  %(prog)s recover-pipeline --stream both --fix-all                      # Complete pipeline recovery

# NEW: Enhanced Metadata Fix
  %(prog)s fix-metadata-integrations --entry-id 123 --titel "New Title" --kanal "Channel"   # Fix with Trilium + Nextcloud

# NEW: Trilium Title Synchronization
  %(prog)s fix-all-trilium-titles --batch-size 20 --dry-run             # Sync all Trilium note titles (preview)
  %(prog)s fix-all-trilium-titles --batch-size 10                       # Execute Trilium title sync

# NEW: Final Success Status Update
  %(prog)s update-success --batch-size 50 --dry-run                     # Preview final_success status updates
  %(prog)s update-success --batch-size 30                               # Update final_success for completed entries

# NEW: WebDAV Auto-Scanning & Fixing
  %(prog)s scan-nextcloud-videos --batch-size 20 --dry-run              # Scan for 'youtube video _{id}.mp4' files
  %(prog)s scan-nextcloud-videos --batch-size 10                        # Auto-fix found youtube video files

PIPELINE STREAMS:
- Stream A: Video processing (download, Nextcloud upload)
- Stream B: Transcript processing (LLM, Trilium upload)  
- Both: Complete pipeline recovery

TRILIUM OPERATIONS:
- fix-all-trilium-titles: Synchronize all Trilium note titles with database titles
- fix-metadata-integrations: Fix single entry with Trilium note title update

NEXTCLOUD OPERATIONS:
- scan-nextcloud-videos: Find and rename 'youtube video _{video_id}.mp4' files to proper titles
- Automatically matches video IDs with database entries
- Renames files to '{channel}/{proper_title}.mp4' format

SUCCESS STATUS:
- update-success: Set final_success=1 for entries where both streams succeeded
- Automatically triggered after successful fix operations
- Clears error_messages and sets processing_stage='completed'

FLAGS:
--fix-all: Automatically find and fix all entries with missing attributes for specified stream
--dry-run: Preview changes without applying them
--batch-size: Control processing batch size for performance tuning
        """
    )

    parser.add_argument(
        "--config", 
        type=Path,
        default=Path("config.yaml"),
        help="Path to config file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Existing commands
    fix_parser = subparsers.add_parser("fix-metadata", help="Fix broken video metadata with Trilium + Nextcloud integration")
    fix_parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    fix_parser.add_argument("--batch-size", type=int, default=10, help="Max entries to process (default: 10)")
    fix_parser.add_argument("--pattern", default="youtube video #", help="Title pattern to match (default: 'youtube video #')")

    reprocess_parser = subparsers.add_parser("reprocess-transcript", help="Reprocess transcript with LLM and upload to Trilium")
    reprocess_parser.add_argument("id", type=int, help="Database entry ID to reprocess")

    subparsers.add_parser("stats", help="Show database statistics")

    list_parser = subparsers.add_parser("list-broken", help="List entries with broken metadata")
    list_parser.add_argument("--limit", type=int, default=20, help="Max entries to show (default: 20)")
    list_parser.add_argument("--pattern", default="youtube video #", help="Title pattern to match (default: 'youtube video #')")

    missing_parser = subparsers.add_parser("check-missing", help="Check for entries with missing attributes (FIXED categorization)")
    missing_parser.add_argument("--limit", type=int, default=50, help="Max entries to show (default: 50)")

    # NEW: Pipeline Recovery Commands
    recovery_parser = subparsers.add_parser("recover-pipeline", help="Recover specific pipeline streams")
    recovery_parser.add_argument(
        "--stream", 
        choices=["A", "B", "both"], 
        required=True,
        help="Pipeline stream to recover: A (video), B (transcript), both"
    )
    
    # Entry selection: either specific IDs or fix-all
    recovery_group = recovery_parser.add_mutually_exclusive_group(required=True)
    recovery_group.add_argument(
        "--entry-ids", 
        help="Comma-separated list of database entry IDs (e.g., 1,2,3)"
    )
    recovery_group.add_argument(
        "--fix-all", 
        action="store_true",
        help="Automatically find and fix all entries with missing attributes for specified stream"
    )
    
    recovery_parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Simulation mode - no actual processing"
    )

    # NEW: Enhanced Metadata Fix with Integrations
    metadata_integrations_parser = subparsers.add_parser(
        "fix-metadata-integrations", 
        help="Fix metadata with Trilium note + Nextcloud filename update"
    )
    metadata_integrations_parser.add_argument("--entry-id", type=int, required=True, help="Database entry ID")
    metadata_integrations_parser.add_argument("--titel", required=True, help="New video title")
    metadata_integrations_parser.add_argument("--kanal", required=True, help="New channel name")

    # NEW: Fix All Trilium Titles
    trilium_titles_parser = subparsers.add_parser(
        "fix-all-trilium-titles",
        help="Fix all Trilium note titles to match database titles"
    )
    trilium_titles_parser.add_argument(
        "--batch-size", 
        type=int, 
        default=10, 
        help="Number of entries to process per batch (default: 10)"
    )
    trilium_titles_parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Simulation mode - preview changes without applying them"
    )

    # NEW: Update Final Success Status
    update_success_parser = subparsers.add_parser(
        "update-success",
        help="Update final_success status based on stream success states"
    )
    update_success_parser.add_argument(
        "--batch-size", 
        type=int, 
        default=50, 
        help="Number of entries to process per batch (default: 50)"
    )
    update_success_parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Simulation mode - preview changes without applying them"
    )

    # NEW: WebDAV Scanning & Auto-Fix
    webdav_scan_parser = subparsers.add_parser(
        "scan-nextcloud-videos",
        help="Scan WebDAV for 'youtube video _{video_id}.mp4' files and auto-fix them"
    )
    webdav_scan_parser.add_argument(
        "--batch-size", 
        type=int, 
        default=20, 
        help="Number of files to process per batch (default: 20)"
    )
    webdav_scan_parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Simulation mode - preview changes without applying them"
    )

    return parser


# =============================================================================
# ENHANCED CLI COMMAND HANDLERS
# =============================================================================

def cmd_scan_nextcloud_videos(args, config: AppConfig) -> int:
    """NEW: Scan and fix Nextcloud videos with youtube video _{id}.mp4 schema"""
    try:
        db_manager = ExtendedDatabaseManager(config)
        
        print(f"\n{'='*70}")
        print(f"SCAN & FIX NEXTCLOUD VIDEOS {'(DRY RUN)' if args.dry_run else ''}")
        print(f"{'='*70}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Mode: {'Simulation' if args.dry_run else 'Execute Changes'}")
        print(f"Pattern: youtube video _{{video_id}}.mp4")
        print(f"Action: Rename to proper video titles and channels")
        
        if args.dry_run:
            print("\n🔍 DRY RUN MODE - No actual file operations will occur")
        else:
            print("\n⚠️  LIVE MODE - Files will be renamed in Nextcloud!")
        
        result = db_manager.scan_and_fix_nextcloud_videos(
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        if isinstance(result, Ok):
            summary = unwrap_ok(result)
            
            print(f"\n✅ NEXTCLOUD VIDEO SCAN & FIX COMPLETED")
            print(f"Files Found: {summary.files_found}")
            print(f"Database Matches: {summary.matched}")
            print(f"Processed: {summary.processed}")
            print(f"Fixed: {summary.fixed}")
            print(f"Failed: {summary.failed}")
            print(f"Processing Time: {summary.processing_time_seconds:.2f}s")
            
            if summary.fixed > 0:
                print(f"\n📋 FIXED FILES:")
                fixed_files = [r for r in summary.results if r['status'] == 'fixed']
                for i, result in enumerate(fixed_files[:10]):  # Show first 10
                    print(f"  ✅ {i+1:2}. Video ID: {result['video_id']}")
                    print(f"      Old: {result['old_path']}")
                    print(f"      New: {result.get('new_path', 'N/A')}")
                
                if len(fixed_files) > 10:
                    print(f"  ... and {len(fixed_files) - 10} more fixed files")
            
            if summary.failed > 0:
                print(f"\n❌ FAILED FILES:")
                failed_files = [r for r in summary.results if r['status'] == 'failed']
                for i, result in enumerate(failed_files[:5]):  # Show first 5
                    print(f"  {i+1}. Video ID: {result['video_id']} | Path: {result['old_path']}")
                    print(f"     Error: {result.get('error', 'Unknown error')}")
                
                if len(failed_files) > 5:
                    print(f"  ... and {len(failed_files) - 5} more failed files")
            
            unmatched_count = summary.files_found - summary.matched
            if unmatched_count > 0:
                print(f"\n📋 UNMATCHED FILES: {unmatched_count}")
                print(f"These files were found but have no matching database entries")
            
            if summary.errors:
                print(f"\n🔍 ERROR DETAILS:")
                for i, error in enumerate(summary.errors[:3]):  # Show first 3 errors
                    print(f"  {i+1}. {error}")
                if len(summary.errors) > 3:
                    print(f"  ... and {len(summary.errors) - 3} more errors")
            
            # Success rate and recommendations
            if summary.processed > 0:
                success_rate = (summary.fixed / summary.processed) * 100
                print(f"\n📊 SUCCESS RATE: {success_rate:.1f}%")
                
                if summary.fixed > 0 and not args.dry_run:
                    print(f"\n💡 RECOMMENDATIONS:")
                    print(f"   - {summary.fixed} files renamed to proper titles")
                    print(f"   - Run 'stats' command to verify database updates")
                    print(f"   - Consider running 'update-success' to clean up status")
            
            return 0
        else:
            error = unwrap_err(result)
            print(f"❌ Nextcloud video scan failed: {error.message}")
            return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

def cmd_update_success(args, config: AppConfig) -> int:
    """NEW: Update final_success status based on stream success"""
    try:
        db_manager = ExtendedDatabaseManager(config)
        
        print(f"\n{'='*70}")
        print(f"UPDATE FINAL SUCCESS STATUS {'(DRY RUN)' if args.dry_run else ''}")
        print(f"{'='*70}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Mode: {'Simulation' if args.dry_run else 'Execute Changes'}")
        print(f"\nCriteria: video_stream_success = 1 AND transcript_stream_success = 1")
        print(f"Actions: Set final_success = 1, processing_stage = 'completed', clear error_messages")
        
        if args.dry_run:
            print("\n🔍 DRY RUN MODE - No actual database updates will occur")
        else:
            print("\n⚠️  LIVE MODE - Database will be updated!")
        
        result = db_manager.update_final_success_status(
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        if isinstance(result, Ok):
            summary = unwrap_ok(result)
            
            print(f"\n✅ FINAL SUCCESS STATUS UPDATE COMPLETED")
            print(f"Candidates Found: {summary['candidates_found']}")
            print(f"Processed: {summary['processed']}")
            print(f"Updated: {summary['updated']}")
            print(f"Failed: {summary['failed']}")
            print(f"Processing Time: {summary['processing_time_seconds']:.2f}s")
            
            if summary['updated'] > 0:
                print(f"\n📋 UPDATED ENTRIES:")
                updated_entries = [e for e in summary['entries'] if e['status'] in ['updated', 'would_update']]
                for i, entry in enumerate(updated_entries[:15]):  # Show first 15
                    status_icon = "✅" if entry['status'] == 'updated' else "🔍"
                    print(f"  {status_icon} {i+1:2}. ID {entry['entry_id']:4} | {entry['titel'][:50]}...")
                    
                    if entry['status'] == 'updated':
                        print(f"      final_success: {entry.get('old_final_success', '?')} → 1")
                        print(f"      stage: {entry.get('old_processing_stage', '?')} → completed")
                        if entry.get('error_messages_cleared'):
                            print(f"      error_messages: cleared")
                    elif entry['status'] == 'would_update':
                        current_stage = entry.get('current_processing_stage', 'unknown')
                        print(f"      Would set: final_success=1, stage=completed")
                        if entry.get('has_error_messages'):
                            print(f"      Would clear: error_messages")
                
                if len(updated_entries) > 15:
                    print(f"  ... and {len(updated_entries) - 15} more entries")
            
            if summary['failed'] > 0:
                print(f"\n❌ FAILED UPDATES:")
                failed_entries = [e for e in summary['entries'] if e['status'] == 'failed']
                for i, entry in enumerate(failed_entries[:5]):  # Show first 5
                    print(f"  {i+1}. ID {entry['entry_id']:4} | {entry.get('titel', 'Unknown')[:40]}... | Error: {entry['error']}")
                
                if len(failed_entries) > 5:
                    print(f"  ... and {len(failed_entries) - 5} more failed updates")
            
            if summary['errors']:
                print(f"\n🔍 ERROR DETAILS:")
                for i, error in enumerate(summary['errors'][:3]):  # Show first 3 errors
                    print(f"  {i+1}. {error}")
                if len(summary['errors']) > 3:
                    print(f"  ... and {len(summary['errors']) - 3} more errors")
            
            # Success rate and recommendations
            if summary['processed'] > 0:
                success_rate = (summary['updated'] / summary['processed']) * 100
                print(f"\n📊 SUCCESS RATE: {success_rate:.1f}%")
                
                if summary['updated'] > 0 and not args.dry_run:
                    print(f"\n💡 RECOMMENDATIONS:")
                    print(f"   - {summary['updated']} entries now marked as final_success=1")
                    print(f"   - These entries should appear as completed in pipeline status")
                    print(f"   - Consider running 'stats' command to verify updated counts")
            
            return 0
        else:
            error = unwrap_err(result)
            print(f"❌ Update final success status failed: {error.message}")
            return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

def cmd_fix_all_trilium_titles(args, config: AppConfig) -> int:
    """NEW: Fix all Trilium note titles to match database"""
    try:
        db_manager = ExtendedDatabaseManager(config)
        
        print(f"\n{'='*70}")
        print(f"FIX ALL TRILIUM NOTE TITLES {'(DRY RUN)' if args.dry_run else ''}")
        print(f"{'='*70}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Mode: {'Simulation' if args.dry_run else 'Execute Changes'}")
        
        if args.dry_run:
            print("\n🔍 DRY RUN MODE - No actual Trilium updates will occur")
        else:
            print("\n⚠️  LIVE MODE - Trilium note titles will be updated!")
        
        result = db_manager.fix_all_trilium_titles(
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        if isinstance(result, Ok):
            summary = unwrap_ok(result)
            
            print(f"\n✅ TRILIUM TITLE SYNC COMPLETED")
            print(f"Total Found: {summary['total_found']}")
            print(f"Processed: {summary['processed']}")
            print(f"Successful: {summary['successful']}")
            print(f"Failed: {summary['failed']}")
            print(f"Skipped: {summary['skipped']}")
            print(f"Processing Time: {summary['processing_time_seconds']:.2f}s")
            
            if summary['successful'] > 0:
                print(f"\n📋 SUCCESSFUL UPDATES:")
                successful_entries = [e for e in summary['entries'] if e['status'] == 'success']
                for i, entry in enumerate(successful_entries[:10]):  # Show first 10
                    print(f"  {i+1:2}. ID {entry['entry_id']:4} | Note: {entry['trilium_note_id'][:8]}... | {entry['title'][:50]}...")
                
                if len(successful_entries) > 10:
                    print(f"  ... and {len(successful_entries) - 10} more successful updates")
            
            if summary['failed'] > 0:
                print(f"\n❌ FAILED UPDATES:")
                failed_entries = [e for e in summary['entries'] if e['status'] == 'failed']
                for i, entry in enumerate(failed_entries[:5]):  # Show first 5
                    print(f"  {i+1}. ID {entry['entry_id']:4} | Note: {entry.get('trilium_note_id', 'N/A')[:8]}... | Error: {entry['error']}")
                
                if len(failed_entries) > 5:
                    print(f"  ... and {len(failed_entries) - 5} more failed updates")
            
            if summary['skipped'] > 0:
                print(f"\n⏭️  SKIPPED ENTRIES:")
                skipped_entries = [e for e in summary['entries'] if e['status'] == 'skipped']
                for i, entry in enumerate(skipped_entries[:5]):  # Show first 5
                    print(f"  {i+1}. ID {entry['entry_id']:4} | Note: {entry.get('trilium_note_id', 'N/A')[:8]}... | Reason: {entry.get('reason', 'Unknown')}")
                
                if len(skipped_entries) > 5:
                    print(f"  ... and {len(skipped_entries) - 5} more skipped entries")
            
            if summary['errors']:
                print(f"\n🔍 ERROR DETAILS:")
                for i, error in enumerate(summary['errors'][:3]):  # Show first 3 errors
                    print(f"  {i+1}. {error}")
                if len(summary['errors']) > 3:
                    print(f"  ... and {len(summary['errors']) - 3} more errors")
            
            # Success rate calculation
            if summary['processed'] > 0:
                success_rate = (summary['successful'] / summary['processed']) * 100
                print(f"\n📊 SUCCESS RATE: {success_rate:.1f}%")
                
                if success_rate < 90 and not args.dry_run:
                    print("⚠️  Lower success rate detected. Consider:")
                    print("   - Check Trilium server connectivity")
                    print("   - Verify authentication credentials") 
                    print("   - Run with smaller batch size")
            
            return 0
        else:
            error = unwrap_err(result)
            print(f"❌ Fix all Trilium titles failed: {error.message}")
            return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

@log_function(log_performance=True)
def cmd_fix_metadata(args, config: AppConfig) -> int:
    """Führt Enhanced Metadata-Fix aus (mit Trilium + Nextcloud)"""
    try:
        db_manager = ExtendedDatabaseManager(config)
        fixer = MetadataFixer(config, db_manager)
        
        result = fixer.fix_metadata_batch(
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            pattern=args.pattern
        )
        
        if isinstance(result, Ok):
            summary = unwrap_ok(result)
            
            print(f"\n{'='*60}")
            print(f"ENHANCED METADATA FIX SUMMARY {'(DRY RUN)' if args.dry_run else ''}")
            print(f"{'='*60}")
            print(f"Processed: {summary['processed']}")
            print(f"Succeeded: {summary['succeeded']}")
            print(f"Failed:    {summary['failed']}")
            
            if summary["entries"]:
                print(f"\nDetails:")
                for entry in summary["entries"][:5]:  # Show first 5
                    status_icon = "✅" if entry["status"] == "success" else "❌"
                    print(f"  {status_icon} Entry {entry['entry_id']}: {entry['status']}")
                    if entry["status"] == "success" and "data" in entry:
                        data = entry["data"]
                        print(f"     Title: {data['old_titel'][:40]}... → {data['new_titel'][:40]}...")
                        
                        # ENHANCED: Show integration results
                        if "integrations" in data:
                            integrations = data["integrations"]
                            print(f"     Database: {'✅' if integrations['database_updated'] else '❌'}")
                            print(f"     Trilium: {'✅' if integrations['trilium_updated'] else '❌'}")
                            print(f"     Nextcloud: {'✅' if integrations['nextcloud_updated'] else '❌'}")
                            
                    elif entry["status"] == "failed":
                        print(f"     Error: {entry['error']}")
                        
                if len(summary["entries"]) > 5:
                    print(f"  ... and {len(summary['entries']) - 5} more entries")
            
            # NEW: Auto-trigger final success update after successful metadata fixes
            if summary['succeeded'] > 0 and not args.dry_run:
                print(f"\n🔄 Auto-triggering final success status update...")
                db_manager._auto_update_success_after_operation(
                    "metadata_fix", 
                    {"successful": summary['succeeded'], "dry_run": args.dry_run}
                )
            
            return 0
        else:
            error = unwrap_err(result)
            print(f"❌ Enhanced metadata fix failed: {error.message}")
            return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def cmd_recover_pipeline(args, config: AppConfig) -> int:
    """NEW: Pipeline Recovery Command"""
    try:
        db_manager = ExtendedDatabaseManager(config)
        
        # Parse entry IDs if provided
        entry_ids = None
        if args.entry_ids:
            entry_ids = [int(x.strip()) for x in args.entry_ids.split(',')]
        
        print(f"\n{'='*70}")
        print(f"ENHANCED PIPELINE RECOVERY - STREAM {args.stream.upper()}")
        print(f"{'='*70}")
        print(f"Stream: {args.stream}")
        print(f"Mode: {'Fix All' if args.fix_all else f'Specific IDs: {entry_ids}'}")
        print(f"Dry Run: {args.dry_run}")
        
        if args.dry_run:
            print("\n🔍 DRY RUN MODE - No actual processing will occur")
        
        result = db_manager.recover_pipeline_stream(
            stream=args.stream,
            entry_ids=entry_ids,
            fix_all=args.fix_all,
            dry_run=args.dry_run
        )
        
        if isinstance(result, Ok):
            report = unwrap_ok(result)
            
            print(f"\n✅ RECOVERY COMPLETED")
            print(f"Mode: {'Fix All' if report.fix_all_mode else 'Specific IDs'}")
            print(f"Processed Entries: {report.processed_entries}")
            print(f"Successful: {report.successful_recoveries}")
            print(f"Failed: {report.failed_recoveries}")
            print(f"Processing Time: {report.processing_time_seconds:.2f}s")
            
            if report.recovery_details:
                print(f"\n📋 RECOVERY DETAILS:")
                for detail in report.recovery_details[:10]:  # Show first 10
                    print(f"  Entry {detail['entry_id']}: {detail['titel'][:50]}...")
                    print(f"    Actions: {', '.join(detail['actions_performed'])}")
                
                if len(report.recovery_details) > 10:
                    print(f"  ... and {len(report.recovery_details) - 10} more entries")
            
            if report.errors:
                print(f"\n❌ ERRORS:")
                for error in report.errors[:5]:  # Show first 5 errors
                    print(f"  {error}")
                if len(report.errors) > 5:
                    print(f"  ... and {len(report.errors) - 5} more errors")
            
            # NEW: Auto-trigger final success update after successful recovery
            if report.successful_recoveries > 0 and not args.dry_run:
                print(f"\n🔄 Auto-triggering final success status update...")
                db_manager._auto_update_success_after_operation(
                    "pipeline_recovery", 
                    {"successful": report.successful_recoveries, "dry_run": args.dry_run}
                )
            
            return 0
        else:
            error = unwrap_err(result)
            print(f"❌ Pipeline recovery failed: {error.message}")
            return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def cmd_fix_metadata_integrations(args, config: AppConfig) -> int:
    """NEW: Enhanced Metadata Fix mit vollständigen Integrationen"""
    try:
        db_manager = ExtendedDatabaseManager(config)
        
        print(f"\n{'='*70}")
        print(f"METADATA FIX WITH COMPLETE INTEGRATIONS")
        print(f"{'='*70}")
        print(f"Entry ID: {args.entry_id}")
        print(f"New Title: {args.titel}")
        print(f"New Channel: {args.kanal}")
        
        result = db_manager.fix_metadata_with_integrations(
            entry_id=args.entry_id,
            titel=args.titel,
            kanal=args.kanal
        )
        
        if isinstance(result, Ok):
            update_results = unwrap_ok(result)
            
            print(f"\n✅ METADATA UPDATE WITH INTEGRATIONS COMPLETED")
            print(f"Database Updated: {'✅' if update_results['database_updated'] else '❌'}")
            print(f"Trilium Note Updated: {'✅' if update_results['trilium_updated'] else '❌'}")
            print(f"Nextcloud Filename Updated: {'✅' if update_results['nextcloud_updated'] else '❌'}")
            
            if update_results.get('trilium_note_id'):
                print(f"Trilium Note ID: {update_results['trilium_note_id']}")
            
            if update_results.get('nextcloud_old_filename') and update_results.get('nextcloud_new_filename'):
                print(f"Nextcloud Filename: {update_results['nextcloud_old_filename']} → {update_results['nextcloud_new_filename']}")
            
            if update_results.get('errors'):
                print(f"\n⚠️  Warnings/Errors:")
                for error in update_results['errors']:
                    print(f"  - {error}")
            
            return 0
        else:
            error = unwrap_err(result)
            print(f"❌ Metadata fix with integrations failed: {error.message}")
            return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


# Keep existing command handlers unchanged
@log_function(log_performance=True)
def cmd_reprocess_transcript(args, config: AppConfig) -> int:
    """Führt Transcript-Reprocessing aus"""
    try:
        db_manager = ExtendedDatabaseManager(config)
        reprocessor = TranscriptReprocessor(config, db_manager)
        
        result = reprocessor.reprocess_transcript(args.id)
        
        if isinstance(result, Ok):
            summary = unwrap_ok(result)
            
            print(f"\n{'='*50}")
            print(f"TRANSCRIPT REPROCESSING SUMMARY")
            print(f"{'='*50}")
            print(f"Entry ID:      {summary['entry_id']}")
            print(f"Video Title:   {summary['video_title']}")
            print(f"LLM Model:     {summary['llm_model']}")
            print(f"LLM Tokens:    {summary['llm_tokens']}")
            print(f"LLM Cost:      ${summary['llm_cost']:.4f}")
            print(f"Trilium Note:  {summary['trilium_note_id']}")
            print(f"Trilium Link:  {summary['trilium_link']}")
            print(f"\n✅ Reprocessing completed successfully!")
            
            # NEW: Auto-trigger final success update after successful transcript reprocessing
            print(f"\n🔄 Auto-triggering final success status update...")
            db_manager._auto_update_success_after_operation(
                "transcript_reprocessing", 
                {"successful": 1, "dry_run": False}
            )
            
            return 0
        else:
            error = unwrap_err(result)
            print(f"❌ Transcript reprocessing failed: {error.message}")
            return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def cmd_stats(args, config: AppConfig) -> int:
    """Zeigt Datenbank-Statistiken"""
    try:
        db_manager = ExtendedDatabaseManager(config)
        stats = db_manager.db.get_statistics()
        
        print(f"\n{'='*50}")
        print(f"DATABASE STATISTICS")
        print(f"{'='*50}")
        print(f"Total Videos:          {stats.get('total_videos', 0)}")
        print(f"Successful Videos:     {stats.get('successful_videos', 0)}")
        print(f"Failed Videos:         {stats.get('failed_videos', 0)}")
        print(f"Video Stream Success:  {stats.get('video_stream_success', 0)}")
        print(f"Transcript Success:    {stats.get('transcript_stream_success', 0)}")
        print(f"Trilium Uploads:       {stats.get('trilium_uploads', 0)}")
        print(f"Nextcloud Uploads:     {stats.get('nextcloud_uploads', 0)}")
        
        # Success rates
        total = stats.get('total_videos', 0)
        if total > 0:
            success_rate = stats.get('successful_videos', 0) / total * 100
            print(f"Success Rate:          {success_rate:.1f}%")
        
        return 0

    except Exception as e:
        print(f"❌ Failed to get statistics: {e}")
        return 1


def cmd_list_broken(args, config: AppConfig) -> int:
    """Listet defekte Metadaten-Einträge"""
    try:
        db_manager = ExtendedDatabaseManager(config)
        result = db_manager.find_broken_metadata_entries(args.pattern)
        
        if isinstance(result, Ok):
            entries = unwrap_ok(result)
            
            print(f"\n{'='*50}")
            print(f"BROKEN METADATA ENTRIES")
            print(f"{'='*50}")
            print(f"Pattern: '{args.pattern}'")
            print(f"Found: {len(entries)} entries")
            
            if entries:
                print(f"\nFirst {min(len(entries), args.limit)} entries:")
                for i, entry in enumerate(entries[:args.limit]):
                    nextcloud_indicator = "📁" if entry.get('nextcloud_link') else "❌"
                    print(f"{i+1:3}. ID: {entry['id']:4} | {entry['titel'][:50]}... | {entry['kanal'][:20]}... | {nextcloud_indicator}")
                    
                if len(entries) > args.limit:
                    print(f"\n... and {len(entries) - args.limit} more entries")
            else:
                print("No broken entries found.")
            
            return 0
        else:
            error = unwrap_err(result)
            print(f"❌ Failed to list broken entries: {error.message}")
            return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def cmd_check_missing(args, config: AppConfig) -> int:
    """ENHANCED: Prüft auf fehlende Attribute mit korrigierter Kategorisierung"""
    try:
        db_manager = ExtendedDatabaseManager(config)
        result = db_manager.find_missing_attributes()
        
        if isinstance(result, Ok):
            entries = unwrap_ok(result)
            
            print(f"\n{'='*80}")
            print(f"MISSING ATTRIBUTES CHECK (FIXED CATEGORIZATION)")
            print(f"{'='*80}")
            print(f"Checking: Basic + Stream A (Video) + Stream B (Transcript) + Final fields")
            print(f"Found: {len(entries)} entries with missing attributes")
            
            if entries:
                print(f"\nFirst {min(len(entries), args.limit)} entries:")
                print(f"{'ID':<5} | {'Title':<35} | {'Channel':<18} | {'Missing':<3} | Missing Attributes")
                print(f"{'-'*5} | {'-'*35} | {'-'*18} | {'-'*7} | {'-'*40}")
                
                for entry in entries[:args.limit]:
                    # Categorize missing attributes with color coding
                    missing_display = []
                    categories = entry['categories']
                    
                    if categories['basic']:
                        missing_display.append(f"BASIC({len(categories['basic'])})")
                    if categories['stream_a']:
                        missing_display.append(f"VIDEO({len(categories['stream_a'])})")
                    if categories['stream_b']:
                        missing_display.append(f"TRANSCRIPT({len(categories['stream_b'])})")
                    if categories['final']:
                        missing_display.append(f"FINAL({len(categories['final'])})")
                    
                    missing_str = " | ".join(missing_display)
                    title_short = entry['titel'][:33] + ".." if len(entry['titel']) > 35 else entry['titel']
                    kanal_short = entry['kanal'][:16] + ".." if len(entry['kanal']) > 18 else entry['kanal']
                    
                    print(f"{entry['id']:<5} | {title_short:<35} | {kanal_short:<18} | {entry['missing_count']:>3} | {missing_str}")
                
                if len(entries) > args.limit:
                    print(f"\n... and {len(entries) - args.limit} more entries")
                
                # ENHANCED: Detailed breakdown by category (FIXED - no duplicates)
                print(f"\n{'='*80}")
                print(f"DETAILED BREAKDOWN BY PIPELINE STAGE (FIXED)")
                print(f"{'='*80}")
                
                # Count by category
                basic_missing = {}
                stream_a_missing = {} 
                stream_b_missing = {}
                final_missing = {}
                
                for entry in entries:
                    for attr in entry['categories']['basic']:
                        basic_missing[attr] = basic_missing.get(attr, 0) + 1
                    for attr in entry['categories']['stream_a']:
                        stream_a_missing[attr] = stream_a_missing.get(attr, 0) + 1
                    for attr in entry['categories']['stream_b']:
                        stream_b_missing[attr] = stream_b_missing.get(attr, 0) + 1
                    for attr in entry['categories']['final']:
                        final_missing[attr] = final_missing.get(attr, 0) + 1
                
                # Display by category
                if basic_missing:
                    print(f"\n🔴 BASIC PROCESSING ISSUES:")
                    for attr, count in sorted(basic_missing.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {attr:<25}: {count:>3} entries")
                
                if stream_a_missing:
                    print(f"\n🟡 STREAM A (VIDEO) ISSUES:")
                    for attr, count in sorted(stream_a_missing.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {attr:<25}: {count:>3} entries")
                
                if stream_b_missing:
                    print(f"\n🟠 STREAM B (TRANSCRIPT) ISSUES:")
                    for attr, count in sorted(stream_b_missing.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {attr:<25}: {count:>3} entries")
                
                if final_missing:
                    print(f"\n🔵 FINAL PROCESSING ISSUES:")
                    for attr, count in sorted(final_missing.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {attr:<25}: {count:>3} entries")
                
                # Summary statistics
                total_with_basic = sum(1 for e in entries if e['categories']['basic'])
                total_with_stream_a = sum(1 for e in entries if e['categories']['stream_a'])
                total_with_stream_b = sum(1 for e in entries if e['categories']['stream_b'])
                total_with_final = sum(1 for e in entries if e['categories']['final'])
                
                print(f"\n{'='*80}")
                print(f"SUMMARY BY PIPELINE STAGE")
                print(f"{'='*80}")
                print(f"Entries with basic issues:      {total_with_basic:>3}")
                print(f"Entries with video issues:      {total_with_stream_a:>3}")
                print(f"Entries with transcript issues: {total_with_stream_b:>3}")
                print(f"Entries with final issues:      {total_with_final:>3}")
                print(f"Total entries with issues:      {len(entries):>3}")
                
                # ENHANCED: Quick fix suggestions
                print(f"\n💡 QUICK FIX SUGGESTIONS:")
                if total_with_stream_a > 0:
                    print(f"  Fix Stream A issues: python db_manager.py recover-pipeline --stream A --fix-all --dry-run")
                if total_with_stream_b > 0:
                    print(f"  Fix Stream B issues: python db_manager.py recover-pipeline --stream B --fix-all --dry-run")
                if total_with_basic > 0 or total_with_final > 0:
                    print(f"  Complete recovery:   python db_manager.py recover-pipeline --stream both --fix-all --dry-run")
                
            else:
                print("✅ No entries with missing attributes found!")
                print("All successfully analyzed videos have complete pipeline data.")
            
            return 0
        else:
            error = unwrap_err(result)
            print(f"❌ Failed to check missing attributes: {error.message}")
            return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


# =============================================================================
# ENHANCED MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """Enhanced Main CLI entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging("db_manager", args.log_level, enable_console=True)
    logger = get_logger("db_manager_main")
    
    try:
        # Load config
        config_manager = SecureConfigManager(args.config)
        config_result = config_manager.load_config()
        
        if isinstance(config_result, Err):
            error = unwrap_err(config_result)
            print(f"❌ Failed to load config: {error.message}")
            return 1
        
        config = unwrap_ok(config_result)
        logger.info(f"Config loaded successfully from {args.config}")
        
        # Enhanced command handlers
        command_handlers = {
            # Existing commands (enhanced)
            "fix-metadata": cmd_fix_metadata,
            "reprocess-transcript": cmd_reprocess_transcript,
            "stats": cmd_stats,
            "list-broken": cmd_list_broken,
            "check-missing": cmd_check_missing,
            
            # NEW enhanced commands
            "recover-pipeline": cmd_recover_pipeline,
            "fix-metadata-integrations": cmd_fix_metadata_integrations,
            "fix-all-trilium-titles": cmd_fix_all_trilium_titles,
            "update-success": cmd_update_success,
            "scan-nextcloud-videos": cmd_scan_nextcloud_videos
        }
        
        handler = command_handlers.get(args.command)
        if handler:
            return handler(args, config)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()

             )
