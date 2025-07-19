"""
YouTube Analyzer - Enhanced Core Data Structures
Basis für die Fork-Join-Pipeline-Architektur mit ProcessObject, TranskriptObject und ArchivObject
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Optional, List, Union
import sqlite3
from enum import Enum

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext
from logging_plus import get_logger, log_feature

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
            'titel': self.titel,
            'kanal': self.kanal,
            'länge': self.länge.isoformat() if self.länge else None,
            'upload_date': self.upload_date.isoformat(),
            'original_url': self.original_url,
            'temp_audio_path': str(self.temp_audio_path) if self.temp_audio_path else None,
            'temp_video_path': str(self.temp_video_path) if self.temp_video_path else None,
            'sprache': self.sprache,
            'transkript_length': len(self.transkript) if self.transkript else 0,
            'rule_amount': self.rule_amount,
            'rule_accuracy': self.rule_accuracy,
            'relevancy': self.relevancy,
            'passed_analysis': self.passed_analysis,
            'nextcloud_link': self.nextcloud_link,
            'video_stream_success': self.video_stream_success,
            'date_created': self.date_created.isoformat(),
            'processing_stage': self.processing_stage,
            'error_count': len(self.error_messages)
        }

@dataclass
class TranskriptObject:
    """
    Objekt für Stream B (Transcript-Verarbeitung)  
    Wandert durch: LLM-Processing → Trilium-Upload
    Minimal und fokussiert auf Transcript-Daten
    """
    titel: str                           # Primary merge key (matches ProcessObject.titel)
    transkript: str                      # Original transcript from ProcessObject
    bearbeiteter_transkript: Optional[str] = None  # LLM-processed transcript
    model: Optional[str] = None          # LLM model used ("gpt-4", "claude-3-sonnet", etc.)
    tokens: Optional[int] = None         # Total token usage (input + output)
    cost: Optional[float] = None         # Estimated processing cost in USD
    success: bool = False                # Processing success flag
    error_message: Optional[str] = None  # Error details if failed
    processing_time: Optional[float] = None  # LLM processing duration in seconds
    
    # Trilium integration
    trilium_link: Optional[str] = None   # Link to created Trilium note
    
    # Metadata
    date_created: datetime = field(default_factory=datetime.now)
    processing_stage: str = "created"    # "created", "llm_processing", "trilium_upload", "completed"
    
    @classmethod
    def from_process_object(cls, obj: ProcessObject) -> 'TranskriptObject':
        """Creates TranskriptObject from ProcessObject (data cloning for Stream B)"""
        return cls(
            titel=obj.titel,
            transkript=obj.transkript
        )
    
    def update_stage(self, stage: str) -> None:
        """Updates processing stage for monitoring"""
        self.processing_stage = stage
    
    def to_dict(self) -> dict:
        """Converts to dictionary for logging/debugging"""
        return {
            'titel': self.titel,
            'transkript_length': len(self.transkript) if self.transkript else 0,
            'bearbeiteter_transkript_length': len(self.bearbeiteter_transkript) if self.bearbeiteter_transkript else 0,
            'model': self.model,
            'tokens': self.tokens,
            'cost': self.cost,
            'success': self.success,
            'error_message': self.error_message,
            'processing_time': self.processing_time,
            'trilium_link': self.trilium_link,
            'date_created': self.date_created.isoformat(),
            'processing_stage': self.processing_stage
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
    transcript_stream_success: bool = False
    
    # Final archive metadata
    final_success: bool = False              # AND-logic result
    date_created: datetime = field(default_factory=datetime.now)
    processing_stage: str = "completed"      # "completed" or "failed"
    error_messages: List[str] = field(default_factory=list)
    
    @classmethod
    def from_process_and_transcript(cls, process_obj: ProcessObject, transcript_obj: TranskriptObject) -> 'ArchivObject':
        """Creates ArchivObject from both stream results with AND-logic"""
        final_success = (process_obj.video_stream_success and transcript_obj.success)
        
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
            transcript_stream_success=transcript_obj.success,
            
            # Final archive state
            final_success=final_success,
            processing_stage="completed" if final_success else "failed",
            error_messages=combined_errors
        )
    
    def to_dict(self) -> dict:
        """Converts to dictionary for SQLite storage"""
        return {
            'titel': self.titel,
            'kanal': self.kanal,
            'länge': self.länge.isoformat() if self.länge else None,
            'upload_date': self.upload_date.isoformat(),
            'original_url': self.original_url,
            'sprache': self.sprache,
            'transkript': self.transkript,
            'rule_amount': self.rule_amount,
            'rule_accuracy': self.rule_accuracy,
            'relevancy': self.relevancy,
            'analysis_results': str(self.analysis_results) if self.analysis_results else None,
            'passed_analysis': self.passed_analysis,
            'bearbeiteter_transkript': self.bearbeiteter_transkript,
            'llm_model': self.llm_model,
            'llm_tokens': self.llm_tokens,
            'llm_cost': self.llm_cost,
            'llm_processing_time': self.llm_processing_time,
            'nextcloud_link': self.nextcloud_link,
            'trilium_link': self.trilium_link,
            'video_stream_success': self.video_stream_success,
            'transcript_stream_success': self.transcript_stream_success,
            'final_success': self.final_success,
            'date_created': self.date_created.isoformat(),
            'processing_stage': self.processing_stage,
            'error_messages': '||'.join(self.error_messages) if self.error_messages else None
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
    FORKED = "forked"                    # Successfully forked into two streams
    
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
    
    def __init__(self, db_path: Path = Path("youtube_archive.db")):
        self.db_path = db_path
        self.logger = get_logger("ArchiveDB")
        self._init_database()
    
    def _init_database(self) -> Result[None, CoreError]:
        """Initialisiert SQLite-Schema für ArchivObject"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processed_videos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        unique_key TEXT UNIQUE NOT NULL,           -- titel::kanal combination
                        
                        -- Video metadata (core)
                        titel TEXT NOT NULL,
                        kanal TEXT NOT NULL,
                        länge TEXT,
                        upload_date TEXT NOT NULL,
                        original_url TEXT,
                        
                        -- Processing results (from analysis)
                        sprache TEXT,
                        transkript TEXT,
                        rule_amount INTEGER,
                        rule_accuracy REAL,
                        relevancy REAL,
                        analysis_results TEXT,
                        passed_analysis BOOLEAN,
                        
                        -- Stream A results (video processing)
                        nextcloud_link TEXT,
                        video_stream_success BOOLEAN,
                        
                        -- Stream B results (transcript processing)
                        bearbeiteter_transkript TEXT,
                        llm_model TEXT,
                        llm_tokens INTEGER,
                        llm_cost REAL,
                        llm_processing_time REAL,
                        trilium_link TEXT,
                        transcript_stream_success BOOLEAN,
                        
                        -- Final archive state
                        final_success BOOLEAN,                     -- AND-logic result
                        date_created TEXT NOT NULL,
                        processing_stage TEXT NOT NULL,            -- "completed" or "failed"
                        error_messages TEXT,
                        
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Indexes for efficient operations
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_unique_key 
                    ON processed_videos(unique_key)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_final_success 
                    ON processed_videos(final_success)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_llm_model 
                    ON processed_videos(llm_model)
                """)
                
                conn.commit()
                
                self.logger.info("Archive database initialized for Fork-Join architecture")
                return Ok(None)
                
        except Exception as e:
            context = ErrorContext.create(
                "database_init",
                input_data={'db_path': str(self.db_path)},
                suggestions=["Check file permissions", "Verify SQLite installation"]
            )
            return Err(CoreError(f"Failed to initialize archive database: {e}", context))
    
    def check_duplicate(self, process_obj: ProcessObject) -> Result[bool, CoreError]:
        """
        Prüft ob Video bereits verarbeitet wurde (unchanged interface)
        
        Returns:
            Ok(True): Video existiert bereits (Duplikat)
            Ok(False): Video ist neu
            Err: Datenbankfehler
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM processed_videos WHERE unique_key = ?",
                    (process_obj.get_unique_key(),)
                )
                count = cursor.fetchone()[0]
                
                is_duplicate = count > 0
                
                self.logger.debug(
                    f"Duplicate check completed",
                    extra={
                        'unique_key': process_obj.get_unique_key(),
                        'is_duplicate': is_duplicate,
                        'existing_count': count
                    }
                )
                
                return Ok(is_duplicate)
                
        except Exception as e:
            context = ErrorContext.create(
                "duplicate_check",
                input_data={'unique_key': process_obj.get_unique_key()},
                suggestions=["Check database connection", "Verify table exists"]
            )
            return Err(CoreError(f"Duplicate check failed: {e}", context))
    
    def save_processed_video(self, archive_obj: ArchivObject) -> Result[None, CoreError]:
        """Saves ArchivObject to database (new method for Fork-Join)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = archive_obj.to_dict()
                unique_key = archive_obj.get_unique_key()
                
                conn.execute("""
                    INSERT OR REPLACE INTO processed_videos 
                    (unique_key, titel, kanal, länge, upload_date, original_url, sprache, transkript,
                     rule_amount, rule_accuracy, relevancy, analysis_results, passed_analysis,
                     nextcloud_link, video_stream_success, bearbeiteter_transkript, llm_model,
                     llm_tokens, llm_cost, llm_processing_time, trilium_link, transcript_stream_success,
                     final_success, date_created, processing_stage, error_messages)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_key, data['titel'], data['kanal'], data['länge'],
                    data['upload_date'], data['original_url'], data['sprache'], data['transkript'],
                    data['rule_amount'], data['rule_accuracy'], data['relevancy'],
                    data['analysis_results'], data['passed_analysis'],
                    data['nextcloud_link'], data['video_stream_success'], 
                    data['bearbeiteter_transkript'], data['llm_model'], data['llm_tokens'],
                    data['llm_cost'], data['llm_processing_time'], data['trilium_link'],
                    data['transcript_stream_success'], data['final_success'],
                    data['date_created'], data['processing_stage'], data['error_messages']
                ))
                
                conn.commit()
                
                self.logger.info(
                    f"ArchivObject saved successfully",
                    extra={
                        'unique_key': unique_key,
                        'final_success': archive_obj.final_success,
                        'video_stream_success': archive_obj.video_stream_success,
                        'transcript_stream_success': archive_obj.transcript_stream_success,
                        'llm_model': archive_obj.llm_model,
                        'llm_cost': archive_obj.llm_cost
                    }
                )
                
                return Ok(None)
                
        except Exception as e:
            context = ErrorContext.create(
                "save_archive_object",
                input_data={'unique_key': archive_obj.get_unique_key()},
                suggestions=["Check database permissions", "Verify disk space"]
            )
            return Err(CoreError(f"Failed to save archive object: {e}", context))

# =============================================================================
# QUEUE SYSTEM FÜR FORK-JOIN-PIPELINE
# =============================================================================

import queue
import threading
from typing import Callable

class ProcessingQueue:
    """Thread-safe Queue für ProcessObject/TranskriptObject-Übertragung"""
    
    def __init__(self, name: str, maxsize: int = 0):
        self.name = name
        self.queue: queue.Queue[Union[ProcessObject, TranskriptObject]] = queue.Queue(maxsize=maxsize)
        self.logger = get_logger(f"Queue-{name}")
    
    def put(self, obj: Union[ProcessObject, TranskriptObject], timeout: Optional[float] = None) -> Result[None, CoreError]:
        """Fügt Object zur Queue hinzu"""
        try:
            self.queue.put(obj, timeout=timeout)
            
            obj_type = "ProcessObject" if isinstance(obj, ProcessObject) else "TranskriptObject"
            self.logger.debug(
                f"Object added to {self.name} queue",
                extra={
                    'queue_name': self.name,
                    'object_type': obj_type,
                    'unique_key': obj.titel,  # Both objects have titel
                    'queue_size': self.queue.qsize()
                }
            )
            
            return Ok(None)
            
        except queue.Full:
            context = ErrorContext.create(
                "queue_put",
                input_data={'queue_name': self.name, 'queue_size': self.queue.qsize()},
                suggestions=["Increase queue size", "Check processing bottleneck"]
            )
            return Err(CoreError(f"Queue {self.name} is full", context))
    
    def get(self, timeout: Optional[float] = None) -> Result[Union[ProcessObject, TranskriptObject], CoreError]:
        """Holt Object aus Queue"""
        try:
            obj = self.queue.get(timeout=timeout)
            
            obj_type = "ProcessObject" if isinstance(obj, ProcessObject) else "TranskriptObject"
            self.logger.debug(
                f"Object retrieved from {self.name} queue",
                extra={
                    'queue_name': self.name,
                    'object_type': obj_type,
                    'unique_key': obj.titel,
                    'remaining_size': self.queue.qsize()
                }
            )
            
            return Ok(obj)
            
        except queue.Empty:
            context = ErrorContext.create(
                "queue_get",
                input_data={'queue_name': self.name},
                suggestions=["Check if producer is running", "Verify queue input"]
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
    
    # Logging Setup
    setup_logging("youtube_analyzer_core", "DEBUG")
    
    # Test ProcessObject
    test_process_obj = ProcessObject(
        titel="Test Video",
        kanal="Test Channel",
        länge=time(0, 5, 30),  # 5:30 Minuten
        upload_date=datetime.now(),
        original_url="https://youtube.com/watch?v=test123"
    )
    
    # Mock some processing results
    test_process_obj.sprache = "deutsch"
    test_process_obj.transkript = "Dies ist ein Test-Transkript für das Video."
    test_process_obj.rule_amount = 3
    test_process_obj.rule_accuracy = 0.85
    test_process_obj.relevancy = 0.9
    test_process_obj.passed_analysis = True
    test_process_obj.video_stream_success = True
    test_process_obj.nextcloud_link = "https://nextcloud.example.com/file123"
    
    print(f"ProcessObject Unique Key: {test_process_obj.get_unique_key()}")
    print(f"ProcessObject Dict: {test_process_obj.to_dict()}")
    
    # Test TranskriptObject creation
    test_transcript_obj = TranskriptObject.from_process_object(test_process_obj)
    
    # Mock LLM processing results
    test_transcript_obj.bearbeiteter_transkript = "Bearbeitetes Transkript mit LLM-Verbesserungen."
    test_transcript_obj.model = "gpt-4"
    test_transcript_obj.tokens = 150
    test_transcript_obj.cost = 0.003
    test_transcript_obj.processing_time = 2.5
    test_transcript_obj.success = True
    test_transcript_obj.trilium_link = "https://trilium.example.com/note/abc123"
    
    print(f"TranskriptObject Dict: {test_transcript_obj.to_dict()}")
    
    # Test ArchivObject creation
    test_archive_obj = ArchivObject.from_process_and_transcript(test_process_obj, test_transcript_obj)
    
    print(f"ArchivObject Final Success: {test_archive_obj.final_success}")
    print(f"ArchivObject Unique Key: {test_archive_obj.get_unique_key()}")
    print(f"ArchivObject Dict: {test_archive_obj.to_dict()}")
    
    # Test Archive Database
    with log_feature("database_test"):
        archive = ArchiveDatabase(Path("test_fork_join_archive.db"))
        
        # Save ArchivObject
        save_result = archive.save_processed_video(test_archive_obj)
        if isinstance(save_result, Ok):
            print("✅ ArchivObject saved successfully")
        else:
            print(f"❌ Archive save failed: {save_result.error.message}")
    
    # Test Queue with both object types
    test_queue = ProcessingQueue("test_fork_join_queue")
    
    # Test ProcessObject in queue
    put_result1 = test_queue.put(test_process_obj)
    if isinstance(put_result1, Ok):
        print("✅ ProcessObject added to queue")
    
    # Test TranskriptObject in queue
    put_result2 = test_queue.put(test_transcript_obj)
    if isinstance(put_result2, Ok):
        print("✅ TranskriptObject added to queue")
    
    # Retrieve objects
    get_result1 = test_queue.get()
    if isinstance(get_result1, Ok):
        retrieved_obj = get_result1.value
        obj_type = "ProcessObject" if isinstance(retrieved_obj, ProcessObject) else "TranskriptObject"
        print(f"✅ Retrieved {obj_type}: {retrieved_obj.titel}")
    
    get_result2 = test_queue.get()
    if isinstance(get_result2, Ok):
        retrieved_obj = get_result2.value
        obj_type = "ProcessObject" if isinstance(retrieved_obj, ProcessObject) else "TranskriptObject"
        print(f"✅ Retrieved {obj_type}: {retrieved_obj.titel}")
