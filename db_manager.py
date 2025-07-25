#!/usr/bin/env python3
"""
YouTube Analyzer - Database Manager CLI
CLI-Tool für Datenbank-Management: Metadata-Fix und Transcript-Reprocessing
"""

from __future__ import annotations
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_ok, unwrap_err
from yt_analyzer_core import ArchiveDatabase, ArchivObject, TranskriptObject
from yt_analyzer_config import SecureConfigManager, AppConfig
from logging_plus import setup_logging, get_logger, log_feature, log_function
from yt_url_processor import YouTubeMetadataExtractor
from yt_llm_processor import process_transcript_with_llm_dict
from yt_trilium_uploader import upload_to_trilium_dict

# =============================================================================
# EXTENDED DATABASE MANAGER
# =============================================================================


class ExtendedDatabaseManager:
    """Erweiterte DB-Operationen für CLI-Management"""

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
                    "SELECT id, titel, kanal, original_url, date_created FROM processed_videos WHERE titel LIKE ? ORDER BY date_created DESC",
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
                        "date_created": row[4]
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
        """Findet Einträge mit fehlenden Attributen (nur für passed_analysis = 1)"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.config.storage.sqlite_path) as conn:
                cursor = conn.cursor()
                
                # Basic required fields (bis zur Analysis)
                basic_fields = [
                    'titel', 'kanal', 'länge', 'upload_date', 'original_url',
                    'sprache', 'transkript', 'rule_amount', 'rule_accuracy', 
                    'relevancy', 'analysis_results'
                ]
                
                # Stream A fields (Video Processing)
                stream_a_fields = [
                    'nextcloud_link', 'video_stream_success'
                ]
                
                # Stream B fields (Transcript Processing) 
                stream_b_fields = [
                    'bearbeiteter_transkript', 'llm_model', 'llm_tokens', 
                    'llm_cost', 'llm_processing_time', 'trilium_link', 
                    'trilium_note_id', 'transcript_stream_success'
                ]
                
                # Final fields
                final_fields = [
                    'final_success'
                ]
                
                all_required_fields = basic_fields + stream_a_fields + stream_b_fields + final_fields
                
                # Query nur für erfolgreich analysierte Videos - alle Spalten holen
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
                            # Spalte existiert nicht in DB
                            entry_missing.append(f"{field} (column missing)")
                            continue
                            
                        value = entry_dict.get(field)
                        
                        # NULL oder empty string als fehlend betrachten
                        if value is None or value == "" or value == "null":
                            entry_missing.append(field)
                        
                        # Spezielle Checks für numerische Felder
                        elif field in ['rule_amount', 'rule_accuracy', 'relevancy', 'llm_tokens', 'llm_cost', 'llm_processing_time']:
                            if value == 0 and field in ['rule_accuracy', 'relevancy']:
                                # rule_accuracy und relevancy sollten > 0 sein
                                entry_missing.append(f"{field} (zero value)")
                            elif field == 'llm_tokens' and value == 0:
                                entry_missing.append(f"{field} (zero tokens)")
                        
                        # Boolean fields checks
                        elif field in ['video_stream_success', 'transcript_stream_success', 'final_success']:
                            if value == 0 or value is False:
                                entry_missing.append(f"{field} (false/failed)")
                    
                    # Nur Einträge mit fehlenden Attributen hinzufügen
                    if entry_missing:
                        missing_attrs.append({
                            'id': entry_id,
                            'titel': entry_dict['titel'][:50] + '...' if len(str(entry_dict['titel'])) > 50 else str(entry_dict['titel']),
                            'kanal': entry_dict['kanal'] or 'Unknown',
                            'missing_attributes': entry_missing,
                            'missing_count': len(entry_missing),
                            'categories': {
                                'basic': [f for f in entry_missing if any(basic in f for basic in basic_fields)],
                                'stream_a': [f for f in entry_missing if any(sa in f for sa in stream_a_fields)],
                                'stream_b': [f for f in entry_missing if any(sb in f for sb in stream_b_fields)],
                                'final': [f for f in entry_missing if any(final in f for final in final_fields)]
                            }
                        })

                self.logger.info(
                    f"Found {len(missing_attrs)} entries with missing attributes",
                    extra={
                        "total_checked": len(rows),
                        "entries_with_missing": len(missing_attrs),
                        "total_required_fields": len(all_required_fields),
                        "basic_fields": len(basic_fields),
                        "stream_a_fields": len(stream_a_fields), 
                        "stream_b_fields": len(stream_b_fields),
                        "final_fields": len(final_fields)
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
# METADATA FIXER
# =============================================================================


class MetadataFixer:
    """Repariert defekte Video-Metadaten"""

    def __init__(self, config: AppConfig, db_manager: ExtendedDatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.logger = get_logger("MetadataFixer")
        self.extractor = YouTubeMetadataExtractor()

    @log_function(log_performance=True)
    def fix_metadata_batch(self, dry_run: bool = False, batch_size: int = 10, pattern: str = "youtube video #") -> Result[Dict[str, Any], CoreError]:
        """Repariert Metadaten für alle gefundenen Einträge"""
        try:
            with log_feature("metadata_fix_batch") as feature:
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

                    # Neue Metadaten extrahieren
                    fix_result = self._fix_single_entry(entry, dry_run)
                    processed += 1

                    if isinstance(fix_result, Ok):
                        succeeded += 1
                        results.append({"entry_id": entry["id"], "status": "success", "data": unwrap_ok(fix_result)})
                    else:
                        failed += 1
                        error = unwrap_err(fix_result)
                        results.append({"entry_id": entry["id"], "status": "failed", "error": error.message})
                        
                        # Fail-fast bei Fehlern
                        self.logger.error(f"❌ Failed to fix entry {entry['id']}: {error.message}")
                        context = ErrorContext.create(
                            "metadata_fix_batch",
                            input_data={
                                "failed_entry_id": entry["id"],
                                "error": error.message,
                                "progress": f"{processed}/{min(len(entries), batch_size)}"
                            },
                            suggestions=["Check individual entry", "Verify URL accessibility", "Run with smaller batch"]
                        )
                        return Err(CoreError(f"Batch processing stopped due to error in entry {entry['id']}: {error.message}", context))

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
                    f"✅ Metadata fix batch completed",
                    extra=summary
                )

                return Ok(summary)

        except Exception as e:
            context = ErrorContext.create(
                "fix_metadata_batch",
                input_data={"pattern": pattern, "batch_size": batch_size, "dry_run": dry_run},
                suggestions=["Check database connectivity", "Verify extractor functionality"]
            )
            return Err(CoreError(f"Batch metadata fix failed: {e}", context))

    def _fix_single_entry(self, entry: Dict[str, Any], dry_run: bool) -> Result[Dict[str, str], CoreError]:
        """Repariert Metadaten für einen einzelnen Eintrag"""
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

            # Update in DB (nur wenn nicht dry-run)
            if not dry_run:
                update_result = self.db_manager.update_metadata(entry["id"], new_titel, new_kanal)
                if isinstance(update_result, Err):
                    return update_result

            return Ok({
                "old_titel": entry["titel"],
                "new_titel": new_titel,
                "old_kanal": entry["kanal"],
                "new_kanal": new_kanal
            })

        except Exception as e:
            context = ErrorContext.create(
                "fix_single_entry",
                input_data={"entry_id": entry.get("id"), "url": entry.get("original_url")},
                suggestions=["Check URL accessibility", "Verify metadata extractor"]
            )
            return Err(CoreError(f"Failed to fix single entry: {e}", context))


# =============================================================================
# TRANSCRIPT REPROCESSOR  
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
# CLI ARGUMENT PARSER
# =============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """Erstellt CLI-Argument-Parser"""
    parser = argparse.ArgumentParser(
        description="YouTube Analyzer Database Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s fix-metadata --dry-run                    # Preview metadata fixes
  %(prog)s fix-metadata --batch-size 5               # Fix max 5 entries  
  %(prog)s reprocess-transcript 123                  # Reprocess entry ID 123
  %(prog)s stats                                      # Show database statistics
  %(prog)s list-broken --limit 10                    # List first 10 broken entries
  %(prog)s check-missing --limit 20                  # Check for missing attributes in successful entries
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

    # fix-metadata command
    fix_parser = subparsers.add_parser("fix-metadata", help="Fix broken video metadata")
    fix_parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    fix_parser.add_argument("--batch-size", type=int, default=10, help="Max entries to process (default: 10)")
    fix_parser.add_argument("--pattern", default="youtube video #", help="Title pattern to match (default: 'youtube video #')")

    # reprocess-transcript command  
    reprocess_parser = subparsers.add_parser("reprocess-transcript", help="Reprocess transcript with LLM and upload to Trilium")
    reprocess_parser.add_argument("id", type=int, help="Database entry ID to reprocess")

    # stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # list-broken command
    list_parser = subparsers.add_parser("list-broken", help="List entries with broken metadata")
    list_parser.add_argument("--limit", type=int, default=20, help="Max entries to show (default: 20)")
    list_parser.add_argument("--pattern", default="youtube video #", help="Title pattern to match (default: 'youtube video #')")

    # check-missing command
    missing_parser = subparsers.add_parser("check-missing", help="Check for entries with missing attributes (only passed_analysis = true)")
    missing_parser.add_argument("--limit", type=int, default=50, help="Max entries to show (default: 50)")

    return parser


# =============================================================================
# MAIN CLI FUNCTIONS
# =============================================================================


@log_function(log_performance=True)
def cmd_fix_metadata(args, config: AppConfig) -> int:
    """Führt Metadata-Fix aus"""
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
            
            print(f"\n{'='*50}")
            print(f"METADATA FIX SUMMARY {'(DRY RUN)' if args.dry_run else ''}")
            print(f"{'='*50}")
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
                        print(f"     Title: {data['old_titel'][:50]}... → {data['new_titel'][:50]}...")
                    elif entry["status"] == "failed":
                        print(f"     Error: {entry['error']}")
                        
                if len(summary["entries"]) > 5:
                    print(f"  ... and {len(summary['entries']) - 5} more entries")
            
            return 0
        else:
            error = unwrap_err(result)
            print(f"❌ Metadata fix failed: {error.message}")
            return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


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
                    print(f"{i+1:3}. ID: {entry['id']:4} | {entry['titel'][:60]}... | {entry['kanal'][:25]}...")
                    
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
    """Prüft auf fehlende Attribute in erfolgreich analysierten Einträgen"""
    try:
        db_manager = ExtendedDatabaseManager(config)
        result = db_manager.find_missing_attributes()
        
        if isinstance(result, Ok):
            entries = unwrap_ok(result)
            
            print(f"\n{'='*80}")
            print(f"MISSING ATTRIBUTES CHECK (PASSED ANALYSIS ONLY)")
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
                
                # Detailed breakdown by category
                print(f"\n{'='*80}")
                print(f"DETAILED BREAKDOWN BY PIPELINE STAGE")
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
# MAIN ENTRY POINT
# =============================================================================


def main() -> int:
    """Main CLI entry point"""
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
        
        # Route to command handlers
        command_handlers = {
            "fix-metadata": cmd_fix_metadata,
            "reprocess-transcript": cmd_reprocess_transcript,
            "stats": cmd_stats,
            "list-broken": cmd_list_broken,
            "check-missing": cmd_check_missing
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
    sys.exit(main())
