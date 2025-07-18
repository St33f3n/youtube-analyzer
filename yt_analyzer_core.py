"""
YouTube Analyzer - Core Data Structures
Basis für die Pipeline-Architektur mit ProcessObject und SQLite-Integration
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Optional, List
import sqlite3
from enum import Enum

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext
from logging_plus import get_logger, log_feature

# =============================================================================
# PROCESS OBJECT - Zentrales Datenmodell
# =============================================================================

@dataclass
class ProcessObject:
    """
    Zentrales Objekt für die gesamte Video-Verarbeitungspipeline
    Wandert durch alle Processing-Stufen und sammelt Daten
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
    analysis_results: Optional[dict] = field(default_factory=dict)  # Detaillierte Regel-Ergebnisse
    passed_analysis: Optional[bool] = None  # True/False für Pipeline-Entscheidung
    
    # Transkript-Bearbeitung (nach Analyse)
    bearbeiteter_transkript: Optional[str] = None
    
    # Finale Links/Speicherorte
    nextcloud_link: Optional[str] = None
    trilium_link: Optional[str] = None
    
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
        """Eindeutiger Key für SQLite-Suche (Titel + Kanal)"""
        return f"{self.titel}::{self.kanal}"
    
    def to_dict(self) -> dict:
        """Konvertiert zu Dictionary für SQLite-Storage"""
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
            'nextcloud_link': self.nextcloud_link,
            'trilium_link': self.trilium_link,
            'date_created': self.date_created.isoformat(),
            'processing_stage': self.processing_stage,
            'error_messages': '||'.join(self.error_messages) if self.error_messages else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ProcessObject:
        """Erstellt ProcessObject aus SQLite-Dictionary"""
        return cls(
            titel=data['titel'],
            kanal=data['kanal'],
            länge=time.fromisoformat(data['länge']) if data['länge'] else None,
            upload_date=datetime.fromisoformat(data['upload_date']),
            original_url=data.get('original_url'),
            sprache=data.get('sprache'),
            transkript=data.get('transkript'),
            rule_amount=data.get('rule_amount'),
            rule_accuracy=data.get('rule_accuracy'),
            relevancy=data.get('relevancy'),
            analysis_results=eval(data['analysis_results']) if data.get('analysis_results') else {},
            passed_analysis=data.get('passed_analysis'),
            bearbeiteter_transkript=data.get('bearbeiteter_transkript'),
            nextcloud_link=data.get('nextcloud_link'),
            trilium_link=data.get('trilium_link'),
            date_created=datetime.fromisoformat(data['date_created']),
            processing_stage=data.get('processing_stage', 'created'),
            error_messages=data['error_messages'].split('||') if data.get('error_messages') else []
        )

# =============================================================================
# PROCESSING STAGES ENUM
# =============================================================================

class ProcessingStage(Enum):
    """Definiert alle möglichen Processing-Stufen"""
    CREATED = "created"
    METADATA_EXTRACTED = "metadata_extracted"
    DUPLICATE_CHECKED = "duplicate_checked"
    QUEUED_FOR_DOWNLOAD = "queued_for_download"
    AUDIO_DOWNLOADED = "audio_downloaded"
    QUEUED_FOR_TRANSCRIPTION = "queued_for_transcription"
    TRANSCRIBED = "transcribed"
    QUEUED_FOR_ANALYSIS = "queued_for_analysis"
    ANALYZED = "analyzed"
    ANALYSIS_FAILED = "analysis_failed"  # Failed Analysis → Archive
    QUEUED_FOR_VIDEO_DOWNLOAD = "queued_for_video_download"
    VIDEO_DOWNLOADED = "video_downloaded"
    QUEUED_FOR_UPLOAD = "queued_for_upload"
    UPLOADED_TO_NEXTCLOUD = "uploaded_to_nextcloud"
    QUEUED_FOR_PROCESSING = "queued_for_processing"
    TRANSCRIPT_PROCESSED = "transcript_processed"
    QUEUED_FOR_PKM = "queued_for_pkm"
    UPLOADED_TO_TRILIUM = "uploaded_to_trilium"
    COMPLETED = "completed"
    FAILED = "failed"

# =============================================================================
# SQLITE ARCHIVE DATABASE
# =============================================================================

class ArchiveDatabase:
    """SQLite-Datenbank für verarbeitete Videos (Duplikat-Check)"""
    
    def __init__(self, db_path: Path = Path("youtube_archive.db")):
        self.db_path = db_path
        self.logger = get_logger("ArchiveDB")
        self._init_database()
    
    def _init_database(self) -> Result[None, CoreError]:
        """Initialisiert SQLite-Schema basierend auf ProcessObject"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processed_videos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        unique_key TEXT UNIQUE NOT NULL,
                        titel TEXT NOT NULL,
                        kanal TEXT NOT NULL,
                        länge TEXT,
                        upload_date TEXT NOT NULL,
                        original_url TEXT,
                        sprache TEXT,
                        transkript TEXT,
                        rule_amount INTEGER,
                        rule_accuracy REAL,
                        relevancy REAL,
                        analysis_results TEXT,
                        passed_analysis BOOLEAN,
                        bearbeiteter_transkript TEXT,
                        nextcloud_link TEXT,
                        trilium_link TEXT,
                        date_created TEXT NOT NULL,
                        processing_stage TEXT NOT NULL,
                        error_messages TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Index für schnelle Duplikat-Suche
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_unique_key 
                    ON processed_videos(unique_key)
                """)
                
                conn.commit()
                
                self.logger.info("Archive database initialized successfully")
                return Ok(None)
                
        except Exception as e:
            context = ErrorContext.create(
                "database_init",
                input_data={'db_path': str(self.db_path)},
                suggestions=["Check file permissions", "Verify SQLite installation"]
            )
            return Err(CoreError(f"Failed to initialize database: {e}", context))
    
    def check_duplicate(self, process_obj: ProcessObject) -> Result[bool, CoreError]:
        """
        Prüft ob Video bereits verarbeitet wurde
        
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
    
    def save_processed_video(self, process_obj: ProcessObject) -> Result[None, CoreError]:
        """Speichert abgeschlossenes ProcessObject in Archiv"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = process_obj.to_dict()
                data['unique_key'] = process_obj.get_unique_key()
                
                conn.execute("""
                    INSERT OR REPLACE INTO processed_videos 
                    (unique_key, titel, kanal, länge, upload_date, original_url, sprache, transkript,
                     rule_amount, rule_accuracy, relevancy, analysis_results, passed_analysis,
                     bearbeiteter_transkript, nextcloud_link, trilium_link,
                     date_created, processing_stage, error_messages)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['unique_key'], data['titel'], data['kanal'], data['länge'],
                    data['upload_date'], data['original_url'], data['sprache'], data['transkript'],
                    data['rule_amount'], data['rule_accuracy'], data['relevancy'],
                    data['analysis_results'], data['passed_analysis'],
                    data['bearbeiteter_transkript'], data['nextcloud_link'], 
                    data['trilium_link'], data['date_created'],
                    data['processing_stage'], data['error_messages']
                ))
                
                conn.commit()
                
                self.logger.info(
                    f"Video saved to archive",
                    extra={
                        'unique_key': process_obj.get_unique_key(),
                        'processing_stage': process_obj.processing_stage
                    }
                )
                
                return Ok(None)
                
        except Exception as e:
            context = ErrorContext.create(
                "save_video",
                input_data={'unique_key': process_obj.get_unique_key()},
                suggestions=["Check database permissions", "Verify disk space"]
            )
            return Err(CoreError(f"Failed to save video: {e}", context))

# =============================================================================
# QUEUE SYSTEM FÜR PIPELINE-STAGES
# =============================================================================

import queue
import threading
from typing import Callable

class ProcessingQueue:
    """Thread-safe Queue für ProcessObject-Übertragung zwischen Pipeline-Stufen"""
    
    def __init__(self, name: str, maxsize: int = 0):
        self.name = name
        self.queue: queue.Queue[ProcessObject] = queue.Queue(maxsize=maxsize)
        self.logger = get_logger(f"Queue-{name}")
    
    def put(self, process_obj: ProcessObject, timeout: Optional[float] = None) -> Result[None, CoreError]:
        """Fügt ProcessObject zur Queue hinzu"""
        try:
            self.queue.put(process_obj, timeout=timeout)
            
            self.logger.debug(
                f"Object added to {self.name} queue",
                extra={
                    'queue_name': self.name,
                    'unique_key': process_obj.get_unique_key(),
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
    
    def get(self, timeout: Optional[float] = None) -> Result[ProcessObject, CoreError]:
        """Holt ProcessObject aus Queue"""
        try:
            process_obj = self.queue.get(timeout=timeout)
            
            self.logger.debug(
                f"Object retrieved from {self.name} queue",
                extra={
                    'queue_name': self.name,
                    'unique_key': process_obj.get_unique_key(),
                    'remaining_size': self.queue.qsize()
                }
            )
            
            return Ok(process_obj)
            
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
# BEISPIEL-USAGE
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    
    # Logging Setup
    setup_logging("youtube_analyzer", "DEBUG")
    
    # Test ProcessObject
    test_obj = ProcessObject(
        titel="Test Video",
        kanal="Test Channel",
        länge=time(0, 5, 30),  # 5:30 Minuten
        upload_date=datetime.now()
    )
    
    print(f"Unique Key: {test_obj.get_unique_key()}")
    print(f"Dict: {test_obj.to_dict()}")
    
    # Test Archive Database
    with log_feature("database_test"):
        archive = ArchiveDatabase(Path("test_archive.db"))
        
        # Duplikat-Check
        dup_result = archive.check_duplicate(test_obj)
        if dup_result.unwrap_or(False):
            print("Video already exists")
        else:
            print("New video - ready for processing")
    
    # Test Queue
    test_queue = ProcessingQueue("test_queue")
    
    put_result = test_queue.put(test_obj)
    if put_result.unwrap_or(None) is None:
        print("Successfully added to queue")
    
    get_result = test_queue.get()
    if isinstance(get_result, Ok):
        retrieved_obj = get_result.value
        print(f"Retrieved: {retrieved_obj.titel}")
