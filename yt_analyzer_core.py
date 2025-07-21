"""
YouTube Analyzer - Enhanced Core Data Structures
Basis für die Fork-Join-Pipeline-Architektur mit ProcessObject, TranskriptObject und ArchivObject
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Optional, List, Union, Dict
import sqlite3
from enum import Enum

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext
from logging_plus import get_logger, log_feature, log_function

# =============================================================================
# CORE DATA OBJECTS - Fork-Join Architecture
# =============================================================================


@dataclass
class ProcessObject:
    """
    Zentrales Objekt für Stream A (Video-Verarbeitung)
    Wandert durch: Audio → Transcription → Analysis → Video-Download → Upload
    Bleibt clean ohne LLM-Pollution
    """

    # Basis Video-Metadaten (von yt-dlp)
    titel: str
    kanal: str
    länge: time
    upload_date: datetime

    # Original URL (für Downloads)
    original_url: Optional[str] = None

    # Audio/Video-Pfade (temporär)
    temp_audio_path: Optional[Path] = None
    temp_video_path: Optional[Path] = None

    # Whisper-Ergebnisse
    sprache: Optional[str] = None
    transkript: Optional[str] = None

    # Regel-Analyse-Ergebnisse (Ollama)
    rule_amount: Optional[int] = None
    rule_accuracy: Optional[float] = None
    relevancy: Optional[float] = None
    analysis_results: Optional[dict] = field(default_factory=dict)
    passed_analysis: Optional[bool] = None  # Fork-Trigger

    # Video-Stream-Ergebnisse (Stream A)
    nextcloud_link: Optional[str] = None
    video_stream_success: Optional[bool] = None  # Stream A final status

    # System-Metadaten
    date_created: datetime = field(default_factory=datetime.now)

    # Pipeline-Status (für Debugging)
    processing_stage: str = "created"
    error_messages: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Fügt Fehler-Message für Debugging hinzu"""
        self.error_messages.append(f"{datetime.now().isoformat()}: {message}")

    def update_stage(self, stage: str) -> None:
        """Aktualisiert Processing-Stage für Monitoring"""
        self.processing_stage = stage

    def get_unique_key(self) -> str:
        """Eindeutiger Key für Archive-Suche (Titel + Kanal)"""
        return f"{self.titel}::{self.kanal}"

    def to_dict(self) -> dict:
        """Converts to dictionary for logging/debugging"""
        return {
            "titel": self.titel,
            "kanal": self.kanal,
            "länge": self.länge.isoformat() if self.länge else None,
            "upload_date": self.upload_date.isoformat(),
            "original_url": self.original_url,
            "temp_audio_path": str(self.temp_audio_path)
            if self.temp_audio_path
            else None,
            "temp_video_path": str(self.temp_video_path)
            if self.temp_video_path
            else None,
            "sprache": self.sprache,
            "transkript_length": len(self.transkript) if self.transkript else 0,
            "rule_amount": self.rule_amount,
            "rule_accuracy": self.rule_accuracy,
            "relevancy": self.relevancy,
            "passed_analysis": self.passed_analysis,
            "nextcloud_link": self.nextcloud_link,
            "video_stream_success": self.video_stream_success,
            "date_created": self.date_created.isoformat(),
            "processing_stage": self.processing_stage,
            "error_count": len(self.error_messages),
        }


@dataclass
class TranskriptObject:
    """
    Objekt für Stream B (Transcript-Verarbeitung)
    Wandert durch: LLM-Processing → Trilium-Upload
    Minimal und fokussiert auf Transcript-Daten
    """

    titel: str  # Primary merge key (matches ProcessObject.titel)
    kanal: str
    länge: time
    upload_date: datetime
    original_url: str
    transkript: str  # Original transcript from ProcessObject
    bearbeiteter_transkript: Optional[str] = None  # LLM-processed transcript
    model: Optional[str] = None  # LLM model used ("gpt-4", "claude-3-sonnet", etc.)
    tokens: Optional[int] = None  # Total token usage (input + output)
    cost: Optional[float] = None  # Estimated processing cost in USD
    success: bool = False  # Processing success flag
    error_message: Optional[str] = None  # Error details if failed
    processing_time: Optional[float] = None  # LLM processing duration in seconds

    # Trilium integration
    trilium_link: Optional[str] = None  # Link to created Trilium note
    trilium_note_id: Optional[str] = None

    # Metadata
    date_created: datetime = field(default_factory=datetime.now)
    processing_stage: str = (
        "created"  # "created", "llm_processing", "trilium_upload", "completed"
    )

    @classmethod
    def from_process_object(cls, obj: ProcessObject) -> "TranskriptObject":
        """Creates TranskriptObject from ProcessObject (data cloning for Stream B)"""
        return cls(
            titel=obj.titel,
            kanal=obj.kanal,
            länge=obj.länge,
            upload_date=obj.upload_date,
            original_url=obj.original_url,
            transkript=obj.transkript,
        )

    def update_stage(self, stage: str) -> None:
        """Updates processing stage for monitoring"""
        self.processing_stage = stage

    def to_dict(self) -> dict:
        """Converts to dictionary for logging/debugging"""
        return {
            "titel": self.titel,
            "kanal": self.kanal,
            "länge": self.time,
            "upload_date": self.upload_date,
            "original_url": self.original_url,
            "transkript_length": len(self.transkript) if self.transkript else 0,
            "bearbeiteter_transkript_length": len(self.bearbeiteter_transkript)
            if self.bearbeiteter_transkript
            else 0,
            "model": self.model,
            "tokens": self.tokens,
            "cost": self.cost,
            "success": self.success,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
            "trilium_link": self.trilium_link,
            "tirlium_note_id": self.trilium_note_id,
            "date_created": self.date_created.isoformat(),
            "processing_stage": self.processing_stage,
        }


@dataclass
class ArchivObject:
    """
    Final archive object combining ProcessObject and TranskriptObject data
    Clean separation - no pollution of core objects
    """

    # Video metadata (from ProcessObject)
    titel: str
    kanal: str
    länge: time
    upload_date: datetime
    original_url: str

    # Processing results (from ProcessObject)
    sprache: str
    transkript: str
    rule_amount: int
    rule_accuracy: float
    relevancy: float
    analysis_results: dict
    passed_analysis: bool

    # Video stream results (from ProcessObject - Stream A)
    nextcloud_link: Optional[str] = None
    video_stream_success: bool = False

    # Transcript stream results (from TranskriptObject - Stream B)
    bearbeiteter_transkript: Optional[str] = None
    llm_model: Optional[str] = None
    llm_tokens: Optional[int] = None
    llm_cost: Optional[float] = None
    llm_processing_time: Optional[float] = None
    trilium_link: Optional[str] = None
    trilium_note_id: Optional[str] = None
    transcript_stream_success: bool = False

    # Final archive metadata
    final_success: bool = False  # AND-logic result
    date_created: datetime = field(default_factory=datetime.now)
    processing_stage: str = "completed"  # "completed" or "failed"
    error_messages: List[str] = field(default_factory=list)

    @classmethod
    def from_process_and_transcript(
        cls, process_obj: ProcessObject, transcript_obj: TranskriptObject
    ) -> "ArchivObject":
        """Creates ArchivObject from both stream results with AND-logic"""
        final_success = process_obj.video_stream_success and transcript_obj.success

        # Combine error messages from both streams
        combined_errors = process_obj.error_messages.copy()
        if transcript_obj.error_message:
            combined_errors.append(f"Transcript stream: {transcript_obj.error_message}")

        return cls(
            # Video metadata (from ProcessObject)
            titel=process_obj.titel,
            kanal=process_obj.kanal,
            länge=process_obj.länge,
            upload_date=process_obj.upload_date,
            original_url=process_obj.original_url,
            # Processing results (from ProcessObject)
            sprache=process_obj.sprache,
            transkript=process_obj.transkript,
            rule_amount=process_obj.rule_amount,
            rule_accuracy=process_obj.rule_accuracy,
            relevancy=process_obj.relevancy,
            analysis_results=process_obj.analysis_results,
            passed_analysis=process_obj.passed_analysis,
            # Video stream results (Stream A)
            nextcloud_link=process_obj.nextcloud_link,
            video_stream_success=process_obj.video_stream_success or False,
            # Transcript stream results (Stream B)
            bearbeiteter_transkript=transcript_obj.bearbeiteter_transkript,
            llm_model=transcript_obj.model,
            llm_tokens=transcript_obj.tokens,
            llm_cost=transcript_obj.cost,
            llm_processing_time=transcript_obj.processing_time,
            trilium_link=transcript_obj.trilium_link,
            trilium_note_id=transcript_obj.trilium_note_id,
            transcript_stream_success=transcript_obj.success,
            # Final archive state
            final_success=final_success,
            processing_stage="completed" if final_success else "failed",
            error_messages=combined_errors,
        )

    def to_dict(self) -> dict:
        """Converts to dictionary for SQLite storage"""
        return {
            "titel": self.titel,
            "kanal": self.kanal,
            "länge": self.länge.isoformat() if self.länge else None,
            "upload_date": self.upload_date.isoformat(),
            "original_url": self.original_url,
            "sprache": self.sprache,
            "transkript": self.transkript,
            "rule_amount": self.rule_amount,
            "rule_accuracy": self.rule_accuracy,
            "relevancy": self.relevancy,
            "analysis_results": str(self.analysis_results)
            if self.analysis_results
            else None,
            "passed_analysis": self.passed_analysis,
            "bearbeiteter_transkript": self.bearbeiteter_transkript,
            "llm_model": self.llm_model,
            "llm_tokens": self.llm_tokens,
            "llm_cost": self.llm_cost,
            "llm_processing_time": self.llm_processing_time,
            "nextcloud_link": self.nextcloud_link,
            "trilium_link": self.trilium_link,
            "trilium_note_id": self.trilium_note_id,
            "video_stream_success": self.video_stream_success,
            "transcript_stream_success": self.transcript_stream_success,
            "final_success": self.final_success,
            "date_created": self.date_created.isoformat(),
            "processing_stage": self.processing_stage,
            "error_messages": "||".join(self.error_messages)
            if self.error_messages
            else None,
        }

    def get_unique_key(self) -> str:
        """Eindeutiger Key für Archive (Titel + Kanal)"""
        return f"{self.titel}::{self.kanal}"


# =============================================================================
# PROCESSING STAGES ENUM (Enhanced)
# =============================================================================


class ProcessingStage(Enum):
    """Definiert alle möglichen Processing-Stufen für Fork-Join-Architecture"""

    # Sequential stages (unchanged)
    CREATED = "created"
    METADATA_EXTRACTED = "metadata_extracted"
    DUPLICATE_CHECKED = "duplicate_checked"
    QUEUED_FOR_DOWNLOAD = "queued_for_download"
    AUDIO_DOWNLOADED = "audio_downloaded"
    QUEUED_FOR_TRANSCRIPTION = "queued_for_transcription"
    TRANSCRIBED = "transcribed"
    QUEUED_FOR_ANALYSIS = "queued_for_analysis"
    ANALYZED = "analyzed"

    # Fork point
    ANALYSIS_FAILED = "analysis_failed"  # Failed Analysis → Archive
    FORKED = "forked"  # Successfully forked into two streams

    # Stream A (Video Processing)
    QUEUED_FOR_VIDEO_DOWNLOAD = "queued_for_video_download"
    VIDEO_DOWNLOADED = "video_downloaded"
    QUEUED_FOR_UPLOAD = "queued_for_upload"
    UPLOADED_TO_NEXTCLOUD = "uploaded_to_nextcloud"
    VIDEO_STREAM_COMPLETED = "video_stream_completed"

    # Stream B (Transcript Processing)
    QUEUED_FOR_LLM_PROCESSING = "queued_for_llm_processing"
    LLM_PROCESSING_COMPLETED = "llm_processing_completed"
    LLM_PROCESSING_FAILED = "llm_processing_failed"
    QUEUED_FOR_TRILIUM = "queued_for_trilium"
    UPLOADED_TO_TRILIUM = "uploaded_to_trilium"
    TRANSCRIPT_STREAM_COMPLETED = "transcript_stream_completed"

    # Final stages
    MERGED_AND_ARCHIVED = "merged_and_archived"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# ENHANCED ARCHIVE DATABASE for ArchivObject
# =============================================================================


class ArchiveDatabase:
    """SQLite-Datenbank für ArchivObject storage (Fork-Join Results)"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.logger = get_logger("ArchiveDatabase")

        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        init_result = self._init_database()
        if isinstance(init_result, Err):
            self.logger.error(
                f"Database initialization failed: {init_result.error.message}"
            )

    def _init_database(self) -> Result[None, CoreError]:
        """Initialize database with updated schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # ✅ EXTENDED: Create enhanced table with trilium_note_id
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processed_videos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        titel TEXT NOT NULL,
                        kanal TEXT NOT NULL,
                        länge TEXT,
                        upload_date DATETIME NOT NULL,
                        original_url TEXT,
                        
                        -- Processing results
                        sprache TEXT,
                        transkript TEXT,
                        rule_amount INTEGER,
                        rule_accuracy REAL,
                        relevancy REAL,
                        analysis_results TEXT,  -- JSON
                        passed_analysis BOOLEAN,
                        
                        -- Stream A results  
                        nextcloud_link TEXT,
                        video_stream_success BOOLEAN DEFAULT 0,
                        
                        -- Stream B results
                        bearbeiteter_transkript TEXT,
                        llm_model TEXT,
                        llm_tokens INTEGER,
                        llm_cost REAL,
                        llm_processing_time REAL,
                        trilium_link TEXT,
                        trilium_note_id TEXT, 
                        transcript_stream_success BOOLEAN DEFAULT 0,
                        
                        -- Final metadata
                        final_success BOOLEAN DEFAULT 0,
                        date_created DATETIME NOT NULL,
                        processing_stage TEXT DEFAULT 'failed',
                        error_messages TEXT,  -- JSON
                        
                        -- Unique constraint
                        UNIQUE(titel, kanal)
                    )
                """)

                # ✅ EXTENDED: Create index for trilium_note_id
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trilium_note_id 
                    ON processed_videos(trilium_note_id)
                """)

                # Create other useful indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_titel_kanal 
                    ON processed_videos(titel, kanal)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_final_success 
                    ON processed_videos(final_success)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_date_created 
                    ON processed_videos(date_created)
                """)

                conn.commit()

                # ✅ EXTENDED: Check if trilium_note_id column exists (migration support)
                cursor.execute("PRAGMA table_info(processed_videos)")
                columns = [column[1] for column in cursor.fetchall()]
                if "trilium_note_id" not in columns:
                    self.logger.info(
                        "Adding trilium_note_id column to existing database"
                    )
                    cursor.execute(
                        "ALTER TABLE processed_videos ADD COLUMN trilium_note_id TEXT"
                    )
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_trilium_note_id 
                        ON processed_videos(trilium_note_id)
                    """)
                    conn.commit()

                self.logger.info(f"Archive database initialized: {self.db_path}")
                return Ok(None)

        except Exception as e:
            return Err(CoreError(f"Database initialization failed: {e}"))

    def check_duplicate(self, process_obj: ProcessObject) -> Result[bool, CoreError]:
        """
        Prüft ob Video bereits ERFOLGREICH verarbeitet wurde

        Returns:
            Ok(True): Video wurde erfolgreich verarbeitet (echtes Duplikat)
            Ok(False): Video ist neu oder vorherige Verarbeitung fehlgeschlagen
            Err: Datenbankfehler
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM processed_videos WHERE unique_key = ? AND final_success = 1",
                    (process_obj.get_unique_key(),),
                )
                count = cursor.fetchone()[0]

                is_duplicate = count > 0

                # Zusätzliche Debug-Info für fehlgeschlagene Versuche
                if not is_duplicate:
                    cursor = conn.execute(
                        "SELECT COUNT(*), processing_stage FROM processed_videos WHERE unique_key = ?",
                        (process_obj.get_unique_key(),),
                    )
                    failed_attempts = cursor.fetchone()
                    failed_count = failed_attempts[0] if failed_attempts else 0
                    last_stage = (
                        failed_attempts[1]
                        if failed_attempts and failed_count > 0
                        else None
                    )
                else:
                    failed_count = 0
                    last_stage = None

                self.logger.debug(
                    "Duplicate check completed",
                    extra={
                        "unique_key": process_obj.get_unique_key(),
                        "is_duplicate": is_duplicate,
                        "successful_count": count,
                        "failed_attempts": failed_count,
                        "last_processing_stage": last_stage,
                    },
                )

                return Ok(is_duplicate)

        except Exception as e:
            context = ErrorContext.create(
                "duplicate_check",
                input_data={"unique_key": process_obj.get_unique_key()},
                suggestions=[
                    "Check database connection",
                    "Verify table exists",
                    "Check final_success column",
                ],
            )
            return Err(CoreError(f"Duplicate check failed: {e}", context))

    @log_function("save_processed_video")
    def save_processed_video(
        self, archive_obj: ArchivObject
    ) -> Result[None, CoreError]:
        """Save ArchivObject to database with trilium_note_id support"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convert to database format
                data = archive_obj.to_dict()

                # Convert complex types to JSON strings
                import json

                data["analysis_results"] = json.dumps(data["analysis_results"])
                data["error_messages"] = json.dumps(data["error_messages"])
                # Insert or replace
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO processed_videos (
                        titel, kanal, länge, upload_date, original_url,
                        sprache, transkript, rule_amount, rule_accuracy, relevancy, 
                        analysis_results, passed_analysis,
                        nextcloud_link, video_stream_success,
                        bearbeiteter_transkript, llm_model, llm_tokens, llm_cost, 
                        llm_processing_time, trilium_link, trilium_note_id, transcript_stream_success,
                        final_success, date_created, processing_stage, error_messages
                    ) VALUES (?, ?, ?, ?,  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        data["titel"],
                        data["kanal"],
                        data["länge"],
                        data["upload_date"],
                        data["original_url"],
                        data["sprache"],
                        data["transkript"],
                        data["rule_amount"],
                        data["rule_accuracy"],
                        data["relevancy"],
                        data["analysis_results"],
                        data["passed_analysis"],
                        data["nextcloud_link"],
                        data["video_stream_success"],
                        data["bearbeiteter_transkript"],
                        data["llm_model"],
                        data["llm_tokens"],
                        data["llm_cost"],
                        data["llm_processing_time"],
                        data["trilium_link"],
                        data["trilium_note_id"],
                        data["transcript_stream_success"],
                        data["final_success"],
                        data["date_created"],
                        data["processing_stage"],
                        data["error_messages"],
                    ),
                )

                conn.commit()

                self.logger.info(
                    f"Saved ArchivObject to database: {archive_obj.titel}",
                    extra={
                        "final_success": archive_obj.final_success,
                        "trilium_note_id": archive_obj.trilium_note_id,  # ✅ EXTENDED
                        "nextcloud_link": bool(archive_obj.nextcloud_link),
                        "trilium_link": bool(archive_obj.trilium_link),
                    },
                )

                return Ok(None)

        except Exception as e:
            context = ErrorContext.create(
                "save_processed_video",
                input_data={"titel": archive_obj.titel, "kanal": archive_obj.kanal},
            )
            return Err(CoreError(f"Failed to save to database: {e}", context))

    def find_by_trilium_note_id(
        self, note_id: str
    ) -> Result[Optional[ArchivObject], CoreError]:
        """✅ EXTENDED: Find ArchivObject by trilium note ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM processed_videos WHERE trilium_note_id = ?",
                    (note_id,),
                )
                row = cursor.fetchone()

                if row:
                    # Convert back to ArchivObject (implementation would need column mapping)
                    self.logger.debug(
                        f"Found archive object with trilium_note_id: {note_id}"
                    )
                    return Ok(self._row_to_archive_object(row))
                else:
                    return Ok(None)

        except Exception as e:
            return Err(CoreError(f"Failed to find by trilium_note_id: {e}"))

    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics including trilium uploads"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM processed_videos")
                total = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM processed_videos WHERE final_success = 1"
                )
                successful = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM processed_videos WHERE video_stream_success = 1"
                )
                video_success = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM processed_videos WHERE transcript_stream_success = 1"
                )
                transcript_success = cursor.fetchone()[0]

                # ✅ EXTENDED: Trilium-specific statistics
                cursor.execute(
                    "SELECT COUNT(*) FROM processed_videos WHERE trilium_note_id IS NOT NULL"
                )
                trilium_uploads = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM processed_videos WHERE nextcloud_link IS NOT NULL"
                )
                nextcloud_uploads = cursor.fetchone()[0]

                return {
                    "total_videos": total,
                    "successful_videos": successful,
                    "video_stream_success": video_success,
                    "transcript_stream_success": transcript_success,
                    "trilium_uploads": trilium_uploads,  # ✅ EXTENDED
                    "nextcloud_uploads": nextcloud_uploads,
                    "failed_videos": total - successful,
                }

        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}

    def _row_to_archive_object(self, row) -> ArchivObject:
        """Convert database row to ArchivObject (simplified implementation)"""
        # This would need proper implementation with column mapping
        # For now, return a minimal object
        return ArchivObject(
            titel=row[1],  # Simplified - would need proper column mapping
            kanal=row[2],
            länge=time.fromisoformat(row[3]) if row[3] else time(),
            upload_date=datetime.fromisoformat(row[4]),
            original_url=row[5] or "",
            sprache=row[6] or "",
            transkript=row[7] or "",
            rule_amount=row[8] or 0,
            rule_accuracy=row[9] or 0.0,
            relevancy=row[10] or 0.0,
            analysis_results={},
            passed_analysis=bool(row[12]) if row[12] is not None else False,
        )


# =============================================================================
# QUEUE SYSTEM FÜR FORK-JOIN-PIPELINE
# =============================================================================

import queue


class ProcessingQueue:
    """Thread-safe Queue für ProcessObject/TranskriptObject-Übertragung"""

    def __init__(self, name: str, maxsize: int = 0):
        self.name = name
        self.queue: queue.Queue[Union[ProcessObject, TranskriptObject]] = queue.Queue(
            maxsize=maxsize
        )
        self.logger = get_logger(f"Queue-{name}")

    def put(
        self,
        obj: Union[ProcessObject, TranskriptObject],
        timeout: Optional[float] = None,
    ) -> Result[None, CoreError]:
        """Fügt Object zur Queue hinzu"""
        try:
            self.queue.put(obj, timeout=timeout)

            obj_type = (
                "ProcessObject"
                if isinstance(obj, ProcessObject)
                else "TranskriptObject"
            )
            self.logger.debug(
                f"Object added to {self.name} queue",
                extra={
                    "queue_name": self.name,
                    "object_type": obj_type,
                    "unique_key": obj.titel,  # Both objects have titel
                    "queue_size": self.queue.qsize(),
                },
            )

            return Ok(None)

        except queue.Full:
            context = ErrorContext.create(
                "queue_put",
                input_data={"queue_name": self.name, "queue_size": self.queue.qsize()},
                suggestions=["Increase queue size", "Check processing bottleneck"],
            )
            return Err(CoreError(f"Queue {self.name} is full", context))

    def get(
        self, timeout: Optional[float] = None
    ) -> Result[Union[ProcessObject, TranskriptObject], CoreError]:
        """Holt Object aus Queue"""
        try:
            obj = self.queue.get(timeout=timeout)

            obj_type = (
                "ProcessObject"
                if isinstance(obj, ProcessObject)
                else "TranskriptObject"
            )
            self.logger.debug(
                f"Object retrieved from {self.name} queue",
                extra={
                    "queue_name": self.name,
                    "object_type": obj_type,
                    "unique_key": obj.titel,
                    "remaining_size": self.queue.qsize(),
                },
            )

            return Ok(obj)

        except queue.Empty:
            context = ErrorContext.create(
                "queue_get",
                input_data={"queue_name": self.name},
                suggestions=["Check if producer is running", "Verify queue input"],
            )
            return Err(CoreError(f"Queue {self.name} is empty", context))

    def task_done(self) -> None:
        """Markiert Task als abgeschlossen"""
        self.queue.task_done()

    def empty(self) -> bool:
        """Prüft ob Queue leer ist"""
        return self.queue.empty()

    def size(self) -> int:
        """Gibt aktuelle Queue-Größe zurück"""
        return self.queue.qsize()


# =============================================================================
# EXAMPLE USAGE & TESTING
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    from datetime import datetime, time as dt_time

    setup_logging("analyzer_core_test", "DEBUG")

    # Test ProcessObject creation
    test_process_obj = ProcessObject(
        titel="Extended Test Video",
        kanal="Extended Test Channel",
        länge=dt_time(0, 5, 30),
        upload_date=datetime.now(),
        original_url="https://www.youtube.com/watch?v=extended123",
    )

    # Mock processing results
    test_process_obj.sprache = "de"
    test_process_obj.transkript = "This is an extended test transcript"
    test_process_obj.rule_amount = 3
    test_process_obj.rule_accuracy = 0.85
    test_process_obj.relevancy = 0.92
    test_process_obj.passed_analysis = True
    test_process_obj.nextcloud_link = "https://nextcloud.example.com/share/abc123"
    test_process_obj.video_stream_success = True

    print(f"Extended ProcessObject Dict: {test_process_obj.to_dict()}")

    # Test TranskriptObject creation
    test_transcript_obj = TranskriptObject.from_process_object(test_process_obj)
    test_transcript_obj.bearbeiteter_transkript = "Test_transkrikpt"
    test_transcript_obj.model = "claude-sonnet-4-20250514"
    test_transcript_obj.tokens = 250
    test_transcript_obj.cost = 0.0045
    test_transcript_obj.processing_time = 3.8
    test_transcript_obj.success = True
    test_transcript_obj.trilium_link = "https://trilium.example.com/#note/sK5fn4T6yZRI"
    test_transcript_obj.trilium_note_id = "sK5fn4T6yZRI"  # ✅ EXTENDED

    print(f"Extended TranskriptObject Dict: {test_transcript_obj.to_dict()}")

    # Test ArchivObject creation
    test_archive_obj = ArchivObject.from_process_and_transcript(
        test_process_obj, test_transcript_obj
    )

    print(f"Extended ArchivObject Final Success: {test_archive_obj.final_success}")
    print(
        f"Extended ArchivObject Trilium Note ID: {test_archive_obj.trilium_note_id}"
    )  # ✅ EXTENDED
    print(f"Extended ArchivObject Dict Keys: {list(test_archive_obj.to_dict().keys())}")

    # Test Extended Archive Database
    with log_feature("extended_database_test"):
        archive = ArchiveDatabase(Path("test_extended_archive.db"))
        print(f"ArchivObject: {test_archive_obj}")
        # Save ArchivObject with trilium_note_id
        save_result = archive.save_processed_video(test_archive_obj)
        if isinstance(save_result, Ok):
            print("✅ Extended ArchivObject saved successfully")

            # Test trilium_note_id lookup
            lookup_result = archive.find_by_trilium_note_id("sK5fn4T6yZRI")
            if isinstance(lookup_result, Ok) and lookup_result.value:
                print("✅ Found ArchivObject by trilium_note_id")
            else:
                print("❌ Trilium note ID lookup failed")
        else:
            print(f"❌ Extended archive save failed: {save_result.error.message}")

        # Test extended statistics
        stats = archive.get_statistics()
        print(f"Extended Database Statistics: {stats}")
