"""
YouTube Analyzer - FIXED Pipeline Manager with Corrected Video-Level Tracking
Orchestriert Video-Processing-Pipeline mit robustem Video-Level State-Management
"""

from __future__ import annotations
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import threading

from PySide6.QtCore import QThread, Signal, QObject, QTimer, QMutex, QMutexLocker
from PySide6.QtWidgets import QMessageBox

# Import our core libraries  
from core_types import Result, Ok, Err, is_ok, unwrap_ok, unwrap_err, CoreError, ErrorContext
from yt_analyzer_core import ProcessObject, ProcessingQueue, ArchiveDatabase, ProcessingStage
from logging_plus import get_logger, log_feature, log_function
from yt_analyzer_config import SecureConfigManager, AppConfig
from yt_url_processor import process_urls_to_objects
from yt_transcription_worker import transcribe_process_object
from yt_audio_downloader import download_audio_for_process_object
from yt_rulechain import analyze_process_object

# =============================================================================
# FIXED VIDEO-LEVEL STATE MANAGEMENT
# =============================================================================

@dataclass
class WorkerState:
    """Detaillierter Worker-Zustand für State-Aggregation"""
    name: str
    is_running: bool = False
    is_processing: bool = False
    queue_size: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    current_object: Optional[str] = None
    
    # FIXED: Separate Worker-Level vs Video-Level Metrics
    worker_completions: int = 0  # How many worker tasks completed
    worker_errors: int = 0       # How many worker tasks failed
    
    def update_activity(self, processing: bool = None, current_obj: str = None):
        """Aktualisiert Worker-Aktivität"""
        self.last_activity = datetime.now()
        if processing is not None:
            self.is_processing = processing
        if current_obj is not None:
            self.current_object = current_obj

@dataclass
class PipelineStatus:
    """FIXED Pipeline-Status mit Queue + Active Workers pro Stage"""
    # FIXED: Each stage shows Queue + Active Workers (total items in stage)
    metadata_queue: int = 0           # Queue + Active Workers
    audio_download_queue: int = 0     # Queue + Active Workers  
    transcription_queue: int = 0      # Queue + Active Workers
    analysis_queue: int = 0           # Queue + Active Workers
    video_download_queue: int = 0     # Queue + Active Workers
    upload_queue: int = 0             # Queue + Active Workers
    processing_queue: int = 0         # Queue + Active Workers
    
    # FIXED: Video-Level Metrics (nicht Worker-Level)
    total_queued: int = 0           # Total videos in pipeline
    total_completed: int = 0        # Videos that finished completely
    total_failed: int = 0           # Videos that failed completely
    total_archived: int = 0         # Videos archived after failed analysis
    
    current_stage: str = "Idle"
    current_video: Optional[str] = None
    
    # Enhanced status info
    active_workers: List[str] = field(default_factory=list)
    pipeline_health: str = "healthy"
    estimated_completion: Optional[datetime] = None
    
    # DEBUG: Worker-Level Metrics (for debugging)
    total_worker_tasks: int = 0     # Total worker completions (debug)
    
    def is_active(self) -> bool:
        """Prüft ob Pipeline aktiv ist (Queues oder Worker)"""
        queue_active = (self.metadata_queue + self.audio_download_queue + 
                       self.transcription_queue + self.analysis_queue + 
                       self.video_download_queue + self.upload_queue + 
                       self.processing_queue) > 0
        worker_active = len(self.active_workers) > 0
        return queue_active or worker_active

@dataclass
class ProcessingError:
    """Error-Information für End-Summary"""
    video_title: str
    video_url: str
    stage: str
    error_message: str
    timestamp: datetime
    process_object: Optional[ProcessObject] = None

class PipelineState(Enum):
    """Pipeline-Zustände"""
    IDLE = "idle"
    RUNNING = "running" 
    STOPPING = "stopping"
    FINISHED = "finished"

# =============================================================================
# FIXED CENTRALIZED STATE AGGREGATOR
# =============================================================================

class PipelineStateAggregator:
    """FIXED Zentrale State-Machine mit korrekter Video-Level-Tracking"""
    
    def __init__(self):
        self.logger = get_logger("PipelineStateAggregator")
        self.worker_states: Dict[str, WorkerState] = {}
        self.queue_states: Dict[str, int] = {}
        
        # FIXED: Separate Video-Level vs Worker-Level Metrics
        self.video_metrics = {
            'videos_started': 0,      # Videos entered pipeline
            'videos_completed': 0,    # Videos finished completely  
            'videos_failed': 0,       # Videos failed completely
            'videos_archived': 0,     # Videos archived (analysis failed)
            'start_time': None,
            'last_completion': None
        }
        
        self.worker_metrics = {
            'total_worker_completions': 0,  # All worker task completions
            'total_worker_errors': 0        # All worker task errors
        }
        
        # Thread-safe access
        self.state_mutex = QMutex()
        
        # Activity tracking
        self.last_state_change = datetime.now()
        self.activity_timeout = timedelta(seconds=30)
    
    def register_worker(self, worker_name: str) -> None:
        """Registriert neuen Worker"""
        with QMutexLocker(self.state_mutex):
            self.worker_states[worker_name] = WorkerState(name=worker_name)
            self.logger.debug(f"Worker registered: {worker_name}")
    
    def update_worker_state(self, worker_name: str, **kwargs) -> None:
        """Aktualisiert Worker-State (thread-safe)"""
        with QMutexLocker(self.state_mutex):
            if worker_name in self.worker_states:
                worker_state = self.worker_states[worker_name]
                
                # Update fields
                for key, value in kwargs.items():
                    if hasattr(worker_state, key):
                        setattr(worker_state, key, value)
                
                # Always update activity timestamp
                worker_state.update_activity()
                self.last_state_change = datetime.now()
                
                # ENHANCED: Debug-Logging für Worker-Updates
                self.logger.debug(
                    f"Worker state updated: {worker_name}",
                    extra={
                        'worker': worker_name,
                        'is_processing': worker_state.is_processing,
                        'queue_size': worker_state.queue_size,
                        'current_object': worker_state.current_object,
                        'worker_completions': worker_state.worker_completions
                    }
                )
    
    def update_queue_size(self, queue_name: str, size: int) -> None:
        """FIXED: Aktualisiert Queue-Größe mit Enhanced Logging"""
        with QMutexLocker(self.state_mutex):
            old_size = self.queue_states.get(queue_name, -1)
            self.queue_states[queue_name] = size
            self.last_state_change = datetime.now()
            
            # ENHANCED: Debug-Logging für Queue-Changes (ALWAYS LOG)
            self.logger.info(f"🔄 QUEUE UPDATE: {queue_name} ({old_size} → {size}) | ALL_QUEUES: {dict(self.queue_states)}")
    
    def worker_started_processing(self, worker_name: str, object_title: str) -> None:
        """Worker hat Processing gestartet"""
        self.update_worker_state(
            worker_name,
            is_processing=True,
            current_object=object_title
        )
    
    def worker_finished_processing(self, worker_name: str, success: bool) -> None:
        """FIXED: Worker hat Processing beendet - nur Worker-Level-Tracking"""
        with QMutexLocker(self.state_mutex):
            if worker_name in self.worker_states:
                worker_state = self.worker_states[worker_name]
                worker_state.is_processing = False
                worker_state.current_object = None
                worker_state.update_activity()
                
                # FIXED: Nur Worker-Level Metrics updaten
                if success:
                    worker_state.worker_completions += 1
                    self.worker_metrics['total_worker_completions'] += 1
                else:
                    worker_state.worker_errors += 1
                    self.worker_metrics['total_worker_errors'] += 1
                
                self.logger.debug(
                    f"Worker task completed: {worker_name}",
                    extra={
                        'worker': worker_name,
                        'success': success,
                        'worker_completions': worker_state.worker_completions,
                        'worker_errors': worker_state.worker_errors
                    }
                )
    
    # FIXED: Separate Video-Level Completion Tracking
    def video_entered_pipeline(self, video_title: str) -> None:
        """Video ist in Pipeline eingetreten"""
        with QMutexLocker(self.state_mutex):
            self.video_metrics['videos_started'] += 1
            if self.video_metrics['start_time'] is None:
                self.video_metrics['start_time'] = datetime.now()
            
            self.logger.info(
                f"Video entered pipeline: {video_title}",
                extra={
                    'video_title': video_title,
                    'total_started': self.video_metrics['videos_started']
                }
            )
    
    def video_completed_pipeline(self, video_title: str, success: bool) -> None:
        """FIXED: Video hat komplette Pipeline durchlaufen"""
        with QMutexLocker(self.state_mutex):
            if success:
                self.video_metrics['videos_completed'] += 1
                self.video_metrics['last_completion'] = datetime.now()
            else:
                self.video_metrics['videos_failed'] += 1
            
            self.logger.info(
                f"Video completed pipeline: {video_title}",
                extra={
                    'video_title': video_title,
                    'success': success,
                    'total_completed': self.video_metrics['videos_completed'],
                    'total_failed': self.video_metrics['videos_failed']
                }
            )
    
    def video_archived(self, video_title: str) -> None:
        """Video wurde archiviert (Analysis failed)"""
        with QMutexLocker(self.state_mutex):
            self.video_metrics['videos_archived'] += 1
            
            self.logger.info(
                f"Video archived (analysis failed): {video_title}",
                extra={
                    'video_title': video_title,
                    'total_archived': self.video_metrics['videos_archived']
                }
            )
    
    def get_consolidated_status(self) -> PipelineStatus:
        """FIXED: Erstellt konsolidierten Pipeline-Status mit korrekten Metriken"""
        with QMutexLocker(self.state_mutex):
            # DEBUG: Log current state before processing  
            self.logger.info(f"📊 RAW STATE: queues={self.queue_states}, videos_completed={self.video_metrics['videos_completed']}, videos_failed={self.video_metrics['videos_failed']}")
            
            # Collect active workers PER STAGE
            active_workers_by_stage = {
                "metadata": 0,
                "audio_download": 0,
                "transcription": 0,
                "analysis": 0,
                "video_download": 0,
                "upload": 0,
                "processing": 0
            }
            
            active_workers = []
            current_stage = "Idle"
            current_video = None
            
            now = datetime.now()
            
            for worker_name, worker_state in self.worker_states.items():
                # Check if worker is actually active (recent activity)
                time_since_activity = now - worker_state.last_activity
                is_recently_active = time_since_activity < self.activity_timeout
                
                if worker_state.is_running and is_recently_active:
                    if worker_state.is_processing:
                        active_workers.append(worker_name)
                        current_stage = worker_name
                        current_video = worker_state.current_object
                        
                        # FIXED: Count active workers per stage
                        stage_mapping = {
                            "Audio Download": "audio_download",
                            "Transcription": "transcription", 
                            "Analysis": "analysis",
                            "Video Download": "video_download",
                            "Upload": "upload",
                            "Processing": "processing"
                        }
                        stage_key = stage_mapping.get(worker_name, "unknown")
                        if stage_key in active_workers_by_stage:
                            active_workers_by_stage[stage_key] += 1
                            
                    elif worker_state.queue_size > 0:
                        # Worker ready with queue items
                        current_stage = f"{worker_name} (Ready)"
            
            # Determine pipeline health
            pipeline_health = "healthy"
            total_videos = (self.video_metrics['videos_completed'] + 
                          self.video_metrics['videos_failed'] + 
                          self.video_metrics['videos_archived'])
            
            if total_videos > 0:
                error_rate = (self.video_metrics['videos_failed']) / total_videos
                if error_rate > 0.3:
                    pipeline_health = "degraded"
                elif error_rate > 0.5:
                    pipeline_health = "failed"
            
            # FIXED: Calculate total queued videos (sum of all queues)
            total_queued = sum(self.queue_states.values())
            
            # DEBUG: Log calculated values
            self.logger.info(f"📊 CALCULATED: total_queued={total_queued}, active_workers_by_stage={active_workers_by_stage}, videos_completed={self.video_metrics['videos_completed']}")
            
            # Create consolidated status
            status = PipelineStatus(
                # FIXED: Queue sizes + Active Workers per Stage
                metadata_queue=self.queue_states.get("metadata", 0) + active_workers_by_stage["metadata"],
                audio_download_queue=self.queue_states.get("audio_download", 0) + active_workers_by_stage["audio_download"],
                transcription_queue=self.queue_states.get("transcription", 0) + active_workers_by_stage["transcription"], 
                analysis_queue=self.queue_states.get("analysis", 0) + active_workers_by_stage["analysis"],
                video_download_queue=self.queue_states.get("video_download", 0) + active_workers_by_stage["video_download"],
                upload_queue=self.queue_states.get("upload", 0) + active_workers_by_stage["upload"],
                processing_queue=self.queue_states.get("processing", 0) + active_workers_by_stage["processing"],
                
                # FIXED: Video-Level Metrics
                total_queued=total_queued,
                total_completed=self.video_metrics['videos_completed'],
                total_failed=self.video_metrics['videos_failed'],
                total_archived=self.video_metrics['videos_archived'],
                
                # Current state
                current_stage=current_stage,
                current_video=current_video,
                active_workers=active_workers.copy(),
                pipeline_health=pipeline_health,
                
                # DEBUG: Worker-Level Metrics
                total_worker_tasks=self.worker_metrics['total_worker_completions']
            )
            
            # ENHANCED: Debug-Logging für Final Status
            self.logger.info(f"📊 FINAL STATUS: audio_download={status.audio_download_queue}(q:{self.queue_states.get('audio_download', 0)}+w:{active_workers_by_stage['audio_download']}), transcription={status.transcription_queue}(q:{self.queue_states.get('transcription', 0)}+w:{active_workers_by_stage['transcription']}), completed={status.total_completed}")
            
            return status
    
    def is_pipeline_finished(self) -> bool:
        """FIXED: Robuste Pipeline-Finish-Detection"""
        with QMutexLocker(self.state_mutex):
            # Check if any queues have items
            has_queued_items = any(size > 0 for size in self.queue_states.values())
            
            # Check if any workers are actively processing
            now = datetime.now()
            has_active_workers = any(
                worker.is_processing and 
                (now - worker.last_activity) < self.activity_timeout
                for worker in self.worker_states.values()
            )
            
            finished = not has_queued_items and not has_active_workers
            
            if finished:
                self.logger.info(
                    "Pipeline finish detected",
                    extra={
                        'queued_items': has_queued_items,
                        'active_workers': has_active_workers,
                        'queue_states': dict(self.queue_states),
                        'worker_processing': {
                            name: w.is_processing for name, w in self.worker_states.items()
                        }
                    }
                )
            
            return finished

# =============================================================================
# FIXED BASE WORKER CLASS
# =============================================================================

class BaseWorker(QThread):
    """FIXED Basis-Klasse mit korrekter Queue-Size-Propagation"""
    
    # Enhanced Signals
    object_started = Signal(str, str)  # worker_name, object_title
    object_processed = Signal(ProcessObject, str, str)  # ProcessObject, next_stage, worker_name
    processing_error = Signal(ProcessObject, str, str)  # ProcessObject, error_message, worker_name
    queue_size_changed = Signal(str, int)  # queue_name, size
    worker_status_changed = Signal(str, dict)  # worker_name, status_dict
    
    def __init__(self, stage_name: str, input_queue: ProcessingQueue, 
                 output_queue: Optional[ProcessingQueue] = None,
                 state_aggregator: Optional[PipelineStateAggregator] = None):
        super().__init__()
        self.stage_name = stage_name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.state_aggregator = state_aggregator
        self.logger = get_logger(f"Worker-{stage_name}")
        self.should_stop = threading.Event()
        
        # Register with state aggregator
        if self.state_aggregator:
            self.state_aggregator.register_worker(stage_name)
    
    def run(self):
        """FIXED Haupt-Worker-Loop mit korrekter Queue-Size-Propagation"""
        self.logger.info(f"{self.stage_name} worker started")
        
        if self.state_aggregator:
            self.state_aggregator.update_worker_state(
                self.stage_name,
                is_running=True
            )
        
        while not self.should_stop.is_set():
            try:
                # FIXED: Update queue size BEFORE processing
                current_queue_size = self.input_queue.size()
                
                # DEBUG: Log queue size before processing
                self.logger.info(
                    f"🔍 {self.stage_name} - Queue Check",
                    extra={
                        'worker': self.stage_name,
                        'queue_name': self.input_queue.name,
                        'queue_size_before': current_queue_size,
                        'checking_for_work': True
                    }
                )
                
                if self.state_aggregator:
                    self.state_aggregator.update_queue_size(
                        self.input_queue.name, 
                        current_queue_size
                    )
                
                # Get ProcessObject from queue (with timeout)
                obj_result = self.input_queue.get(timeout=1.0)
                
                if isinstance(obj_result, Err):
                    # DEBUG: Log when queue is empty
                    if current_queue_size > 0:
                        self.logger.warning(
                            f"🚨 {self.stage_name} - Queue shows {current_queue_size} items but get() failed"
                        )
                    continue  # Queue empty, try again
                
                process_obj = unwrap_ok(obj_result)
                
                # DEBUG: Log successful object retrieval
                self.logger.info(
                    f"📦 {self.stage_name} - Got Object",
                    extra={
                        'worker': self.stage_name,
                        'object_title': process_obj.titel,
                        'queue_name': self.input_queue.name
                    }
                )
                
                # FIXED: Update queue size AFTER getting object
                new_queue_size = self.input_queue.size()
                if self.state_aggregator:
                    self.state_aggregator.update_queue_size(
                        self.input_queue.name, 
                        new_queue_size
                    )
                
                # Signal processing start
                self.object_started.emit(self.stage_name, process_obj.titel)
                if self.state_aggregator:
                    self.state_aggregator.worker_started_processing(
                        self.stage_name,
                        process_obj.titel
                    )
                
                # Process object
                with log_feature(f"{self.stage_name}_processing") as feature:
                    feature.add_metric("video_title", process_obj.titel)
                    
                    process_result = self.process_object(process_obj)
                    
                    if isinstance(process_result, Ok):
                        processed_obj = unwrap_ok(process_result)
                        
                        # Handle routing decision
                        next_stage_info = self.get_routing_decision(processed_obj)
                        
                        if next_stage_info["route_to_output"]:
                            # Send to normal output queue
                            if self.output_queue:
                                put_result = self.output_queue.put(processed_obj)
                                
                                # FIXED: Update output queue size immediately
                                if self.state_aggregator:
                                    self.state_aggregator.update_queue_size(
                                        self.output_queue.name,
                                        self.output_queue.size()
                                    )
                                
                                if isinstance(put_result, Ok):
                                    self.object_processed.emit(
                                        processed_obj, 
                                        next_stage_info["next_stage"],
                                        self.stage_name
                                    )
                                    feature.add_metric("status", "forwarded")
                                    
                                    # Worker task completed successfully
                                    if self.state_aggregator:
                                        self.state_aggregator.worker_finished_processing(
                                            self.stage_name, success=True
                                        )
                                else:
                                    error_msg = f"Queue overflow: {unwrap_err(put_result).message}"
                                    self.processing_error.emit(processed_obj, error_msg, self.stage_name)
                                    if self.state_aggregator:
                                        self.state_aggregator.worker_finished_processing(
                                            self.stage_name, success=False
                                        )
                            else:
                                # Final stage - completion
                                self.object_processed.emit(
                                    processed_obj, "completed", self.stage_name
                                )
                                feature.add_metric("status", "completed")
                                if self.state_aggregator:
                                    self.state_aggregator.worker_finished_processing(
                                        self.stage_name, success=True
                                    )
                        else:
                            # Route to archive (failed analysis)
                            self.object_processed.emit(
                                processed_obj, "archive", self.stage_name
                            )
                            feature.add_metric("status", "archived")
                            if self.state_aggregator:
                                self.state_aggregator.worker_finished_processing(
                                    self.stage_name, success=True
                                )
                    else:
                        # Processing failed
                        error = unwrap_err(process_result)
                        process_obj.add_error(f"{self.stage_name}: {error.message}")
                        self.processing_error.emit(process_obj, error.message, self.stage_name)
                        feature.add_metric("status", "failed")
                        
                        if self.state_aggregator:
                            self.state_aggregator.worker_finished_processing(
                                self.stage_name, success=False
                            )
                
                self.input_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Unexpected error in {self.stage_name} worker: {e}")
                if self.state_aggregator:
                    self.state_aggregator.worker_finished_processing(
                        self.stage_name, success=False
                    )
                time.sleep(1)  # Avoid tight error loop
        
        # Worker stopping
        if self.state_aggregator:
            self.state_aggregator.update_worker_state(
                self.stage_name,
                is_running=False,
                is_processing=False
            )
        
        self.logger.info(f"{self.stage_name} worker stopped")
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        """Override in subclasses - main processing logic"""
        raise NotImplementedError("Subclasses must implement process_object")
    
    def get_routing_decision(self, obj: ProcessObject) -> Dict[str, Any]:
        """Override für conditional routing (z.B. Analysis stage)"""
        return {
            "route_to_output": True,
            "next_stage": self.get_next_stage_name()
        }
    
    def get_next_stage_name(self) -> str:
        """Override in subclasses - return next stage name"""
        return "unknown"
    
    def stop_worker(self):
        """Graceful worker shutdown"""
        self.should_stop.set()
        if self.isRunning():
            self.wait(5000)  # Wait up to 5 seconds

# =============================================================================
# SPECIFIC WORKER IMPLEMENTATIONS (FIXED)
# =============================================================================

class AudioDownloadWorker(BaseWorker):
    """FIXED Audio-Download Worker"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, 
                 config: AppConfig, state_aggregator: PipelineStateAggregator):
        super().__init__("Audio Download", input_queue, output_queue, state_aggregator)
        self.config = config
        
        # Ensure temp directory exists
        temp_dir = Path(self.config.processing.temp_folder)
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    @log_function(log_performance=True)
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        """Downloads Audio-Spur mit yt-dlp"""
        try:
            self.logger.info(f"🎵 Starting audio download for: {obj.titel}")
            
            # Use the real audio download function
            download_result = download_audio_for_process_object(obj, self.config)
            
            if isinstance(download_result, Ok):
                downloaded_obj = unwrap_ok(download_result)
                self.logger.info(f"✅ Audio download completed for: {downloaded_obj.titel}")
                return Ok(downloaded_obj)
            else:
                return download_result
            
        except Exception as e:
            context = ErrorContext.create(
                "audio_download_worker",
                input_data={"title": obj.titel, "original_url": obj.original_url},
                suggestions=["Check yt-dlp installation", "Verify network connection"]
            )
            return Err(CoreError(f"Audio download worker failed: {e}", context))
    
    def get_next_stage_name(self) -> str:
        return "Transcription"

class TranscriptionWorker(BaseWorker):
    """FIXED Transcription Worker"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, 
                 config: AppConfig, state_aggregator: PipelineStateAggregator):
        super().__init__("Transcription", input_queue, output_queue, state_aggregator)
        self.config = config
    
    @log_function(log_performance=True)
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        """Transkribiert Audio mit faster-whisper"""
        try:
            self.logger.info(f"🎙️ Starting transcription for: {obj.titel}")
            
            # Use the real transcription function
            transcription_result = transcribe_process_object(obj, self.config)
            
            if isinstance(transcription_result, Ok):
                transcribed_obj = unwrap_ok(transcription_result)
                self.logger.info(f"✅ Transcription completed for: {transcribed_obj.titel}")
                return Ok(transcribed_obj)
            else:
                return transcription_result
            
        except Exception as e:
            context = ErrorContext.create(
                "transcription_worker",
                input_data={"title": obj.titel},
                suggestions=["Check faster-whisper installation", "Verify GPU setup"]
            )
            return Err(CoreError(f"Transcription worker failed: {e}", context))
    
    def get_next_stage_name(self) -> str:
        return "Analysis"

class AnalysisWorker(BaseWorker):
    """FIXED Analysis Worker mit Enhanced Routing Debug"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, 
                 config: AppConfig, state_aggregator: PipelineStateAggregator):
        super().__init__("Analysis", input_queue, output_queue, state_aggregator)
        self.config = config
    
    @log_function(log_performance=True)
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        """FIXED Analysiert Content mit Enhanced Debugging"""
        try:
            self.logger.info(f"🧠 Starting analysis for: {obj.titel}")
            
            # TODO: Echte Ollama + Rule System Integration
            time.sleep(1.5)  # Simulate analysis time
            
            # Mock analysis results
            import random
            obj.rule_amount = random.randint(2, 4)
            obj.rule_accuracy = random.uniform(0.6, 0.9)
            obj.relevancy = random.uniform(0.5, 0.8)
            
            # Calculate weighted score (mock)
            weighted_score = (obj.rule_accuracy * 0.7) + (obj.relevancy * 0.3)
            
            # Decision based on thresholds
            obj.passed_analysis = (
                weighted_score >= self.config.scoring.threshold and
                obj.rule_accuracy >= self.config.scoring.min_confidence
            )
            
            obj.analysis_results = {
                "weighted_score": weighted_score,
                "decision": "DOWNLOAD" if obj.passed_analysis else "SKIP",
                "rules_fulfilled": obj.rule_amount,
                "mock_analysis": True
            }
            
            obj.update_stage("analyzed")
            
            # ENHANCED: Debug-Logging für Analysis
            self.logger.info(
                f"Analysis completed: {obj.titel} -> {'DOWNLOAD' if obj.passed_analysis else 'SKIP'}",
                extra={
                    'video_title': obj.titel,
                    'weighted_score': weighted_score,
                    'threshold': self.config.scoring.threshold,
                    'passed_analysis': obj.passed_analysis,
                    'rule_accuracy': obj.rule_accuracy,
                    'relevancy': obj.relevancy
                }
            )
            
            return Ok(obj)
            
        except Exception as e:
            context = ErrorContext.create(
                "content_analysis",
                input_data={"title": obj.titel},
                suggestions=["Check Ollama connection", "Verify rule files"]
            )
            return Err(CoreError(f"Content analysis failed: {e}", context))
    
    def get_routing_decision(self, obj: ProcessObject) -> Dict[str, Any]:
        """FIXED Conditional routing mit Enhanced Debug-Logging"""
        decision = {
            "route_to_output": obj.passed_analysis,
            "next_stage": "Video Download" if obj.passed_analysis else "Archive"
        }
        
        # ENHANCED: Debug-Logging für Routing
        self.logger.info(
            f"Analysis routing decision for {obj.titel}",
            extra={
                'video_title': obj.titel,
                'passed_analysis': obj.passed_analysis,
                'route_to_output': decision["route_to_output"],
                'next_stage': decision["next_stage"],
                'weighted_score': obj.analysis_results.get('weighted_score', 'unknown') if obj.analysis_results else 'no_results'
            }
        )
        
        return decision

# Mock-Worker für verbleibende Stages (unchanged but with logging)
class VideoDownloadWorker(BaseWorker):
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, 
                 config: AppConfig, state_aggregator: PipelineStateAggregator):
        super().__init__("Video Download", input_queue, output_queue, state_aggregator)
        self.config = config
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"📹 Mock video download for: {obj.titel}")
        time.sleep(1.2)  # Mock processing
        obj.temp_video_path = Path(f"/tmp/video_{obj.get_unique_key()}.mp4")
        obj.update_stage("video_downloaded")
        return Ok(obj)
    
    def get_next_stage_name(self) -> str:
        return "Upload"

class UploadWorker(BaseWorker):
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, 
                 config: AppConfig, state_aggregator: PipelineStateAggregator):
        super().__init__("Upload", input_queue, output_queue, state_aggregator)
        self.config = config
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"☁️ Mock upload for: {obj.titel}")
        time.sleep(0.8)  # Mock upload
        obj.nextcloud_link = f"https://nextcloud.example.com/file/{obj.get_unique_key()}.mp4"
        obj.update_stage("uploaded_to_nextcloud")
        return Ok(obj)
    
    def get_next_stage_name(self) -> str:
        return "Processing"

class ProcessingWorker(BaseWorker):
    def __init__(self, input_queue: ProcessingQueue, output_queue: Optional[ProcessingQueue], 
                 config: AppConfig, state_aggregator: PipelineStateAggregator):
        super().__init__("Processing", input_queue, output_queue, state_aggregator)
        self.config = config
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"⚙️ Mock processing for: {obj.titel}")
        time.sleep(0.5)  # Mock processing
        obj.bearbeiteter_transkript = f"Bearbeitetes Transkript: {obj.transkript}"
        obj.trilium_link = f"trilium://note/{obj.get_unique_key()}"
        obj.update_stage("completed")
        return Ok(obj)
    
    def get_next_stage_name(self) -> str:
        return "completed"

# =============================================================================
# FIXED PIPELINE MANAGER
# =============================================================================

class PipelineManager(QThread):
    """FIXED Pipeline Manager mit korrekter Video-Level-Tracking"""
    
    # Signals für GUI-Updates
    status_updated = Signal(PipelineStatus)
    video_completed = Signal(str, bool)  # title, success
    pipeline_finished = Signal(int, int, list)  # total, success, error_messages
    
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger("PipelineManager")
        
        # State Management
        self.state_aggregator = PipelineStateAggregator()
        self.state = PipelineState.IDLE
        self.input_urls_text: str = ""
        self.processing_errors: List[ProcessingError] = []
        self.completed_videos: List[ProcessObject] = []
        
        # Queues für Pipeline-Stages
        self.queues = {
            "audio_download": ProcessingQueue("audio_download"),
            "transcription": ProcessingQueue("transcription"),
            "analysis": ProcessingQueue("analysis"),
            "video_download": ProcessingQueue("video_download"),
            "upload": ProcessingQueue("upload"),
            "processing": ProcessingQueue("processing")
        }
        
        # Workers
        self.workers: List[BaseWorker] = []
        
        # Event-driven Status Updates
        self.setup_event_driven_updates()
    
    def setup_event_driven_updates(self):
        """Setup Event-driven Status Updates"""
        # Update-Timer als Fallback (längere Intervalle)
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.emit_status_update)
        self.status_timer.start(1000)  # 1s für Debug-Zwecke
        
        # State-Change-Detection
        self.last_emitted_status = None
        
        # DEBUG: Add test timer for debugging
        self.debug_timer = QTimer()
        self.debug_timer.timeout.connect(self.debug_test_status)
        # Uncomment next line to enable debug test values
        # self.debug_timer.start(3000)  # Test every 3 seconds
    
    def debug_test_status(self):
        """DEBUG: Test function to inject test values"""
        if self.state == PipelineState.RUNNING:
            self.logger.info("🧪 DEBUG: Injecting test values for status testing")
            
            # Inject some test values to see if GUI updates
            self.state_aggregator.queue_states["audio_download"] = 5
            self.state_aggregator.queue_states["transcription"] = 3
            self.state_aggregator.video_metrics['videos_completed'] = 2
            self.state_aggregator.video_metrics['videos_failed'] = 1
            
            # Force an immediate update
            self.emit_immediate_status_update()
    
    def start_pipeline(self, urls_text: str) -> Result[None, CoreError]:
        """FIXED Startet Pipeline mit korrekter Video-Level-Tracking"""
        if self.state != PipelineState.IDLE:
            return Err(CoreError("Pipeline already running"))
        
        self.input_urls_text = urls_text
        self.processing_errors.clear()
        self.completed_videos.clear()
        
        # Reset state aggregator
        self.state_aggregator.video_metrics['start_time'] = datetime.now()
        
        self.logger.info(f"Starting FIXED pipeline for URL input (length: {len(urls_text)} chars)")
        
        with log_feature("pipeline_startup") as feature:
            # Step 1: Process URLs to ProcessObjects with Metadata
            objects_result = process_urls_to_objects(urls_text, self.config.processing.__dict__)
            
            if isinstance(objects_result, Err):
                errors = unwrap_err(objects_result)
                error_summary = f"Failed to process URLs: {len(errors)} errors"
                for error in errors[:3]:
                    self.logger.error(f"URL processing error: {error.message}")
                
                feature.add_metric("url_processing_errors", len(errors))
                return Err(CoreError(error_summary))
            
            process_objects = unwrap_ok(objects_result)
            
            if not process_objects:
                return Err(CoreError("No valid videos found in input"))
            
            feature.add_metric("extracted_videos", len(process_objects))
            
            # FIXED: Track videos entering pipeline
            for process_obj in process_objects:
                self.state_aggregator.video_entered_pipeline(process_obj.titel)
            
            # Step 2: Queue ProcessObjects for Audio Download
            for process_obj in process_objects:
                put_result = self.queues["audio_download"].put(process_obj)
                if isinstance(put_result, Err):
                    self.logger.error(f"Failed to queue video {process_obj.titel}")
                else:
                    self.logger.info(f"✅ Queued video: {process_obj.titel}")
            
            # FIXED: Update queue size immediately after queuing
            final_queue_size = self.queues["audio_download"].size()
            self.state_aggregator.update_queue_size("audio_download", final_queue_size)
            
            # DEBUG: Log final queue state after setup
            self.logger.info(
                f"🎯 PIPELINE SETUP COMPLETE",
                extra={
                    'videos_processed': len(process_objects),
                    'audio_download_queue_final': final_queue_size,
                    'videos_entered_pipeline': self.state_aggregator.video_metrics['videos_started']
                }
            )
            
            # Step 3: Setup and start workers
            setup_result = self.setup_workers()
            if isinstance(setup_result, Err):
                return setup_result
            
            feature.add_metric("videos_queued", len(process_objects))
            feature.add_metric("workers_started", len(self.workers))
        
        self.state = PipelineState.RUNNING
        self.start()  # Start QThread
        
        self.logger.info(f"FIXED Pipeline started successfully with {len(process_objects)} videos")
        return Ok(None)
    
    def setup_workers(self) -> Result[None, CoreError]:
        """Richtet alle Worker-Threads ein mit State-Integration"""
        try:
            # Clear existing workers
            self.cleanup_workers()
            
            # Create worker chain mit State-Aggregator
            workers_config = [
                (AudioDownloadWorker, "audio_download", "transcription"),
                (TranscriptionWorker, "transcription", "analysis"),
                (AnalysisWorker, "analysis", "video_download"),
                (VideoDownloadWorker, "video_download", "upload"),
                (UploadWorker, "upload", "processing"),
                (ProcessingWorker, "processing", None)  # Final stage
            ]
            
            for worker_class, input_queue_name, output_queue_name in workers_config:
                input_queue = self.queues[input_queue_name]
                output_queue = self.queues[output_queue_name] if output_queue_name else None
                
                worker = worker_class(
                    input_queue, output_queue, self.config, self.state_aggregator
                )
                
                # Connect enhanced signals
                worker.object_started.connect(self.on_object_started)
                worker.object_processed.connect(self.on_object_processed)
                worker.processing_error.connect(self.on_processing_error)
                worker.queue_size_changed.connect(self.on_queue_size_changed)
                worker.worker_status_changed.connect(self.on_worker_status_changed)
                
                self.workers.append(worker)
                worker.start()
            
            self.logger.info(f"Started {len(self.workers)} FIXED worker threads with state integration")
            return Ok(None)
            
        except Exception as e:
            context = ErrorContext.create(
                "worker_setup",
                suggestions=["Check thread limits", "Verify worker configuration"]
            )
            return Err(CoreError(f"Failed to setup workers: {e}", context))
    
    def on_object_started(self, worker_name: str, object_title: str):
        """Handler für Processing-Start"""
        self.logger.debug(f"Processing started: {worker_name} -> {object_title}")
        self.emit_immediate_status_update()
    
    def on_object_processed(self, obj: ProcessObject, next_stage: str, worker_name: str):
        """FIXED Handler für Object-Processing mit korrekter Video-Level-Tracking"""
        self.logger.debug(f"Object processed: {obj.titel} -> {next_stage} (by {worker_name})")
        
        if next_stage == "completed":
            # FIXED: Video hat komplette Pipeline durchlaufen
            self.completed_videos.append(obj)
            self.state_aggregator.video_completed_pipeline(obj.titel, success=True)
            self.video_completed.emit(obj.titel, True)
            
        elif next_stage == "archive":
            # FIXED: Video wurde archiviert (Analysis failed)
            obj.update_stage("analysis_failed")
            self.completed_videos.append(obj)
            self.state_aggregator.video_archived(obj.titel)
            self.video_completed.emit(obj.titel, False)
            self.logger.info(f"Video archived (failed analysis): {obj.titel}")
        
        self.emit_immediate_status_update()
    
    def on_processing_error(self, obj: ProcessObject, error_message: str, worker_name: str):
        """FIXED Handler für Processing-Fehler mit Video-Level-Tracking"""
        error = ProcessingError(
            video_title=obj.titel,
            video_url=obj.original_url or obj.titel,
            stage=worker_name,
            error_message=error_message,
            timestamp=datetime.now(),
            process_object=obj
        )
        
        self.processing_errors.append(error)
        
        # FIXED: Video failed completely
        self.state_aggregator.video_completed_pipeline(obj.titel, success=False)
        self.video_completed.emit(obj.titel, False)
        
        self.logger.error(f"Processing error: {obj.titel} in {worker_name}: {error_message}")
        self.emit_immediate_status_update()
    
    def on_queue_size_changed(self, queue_name: str, size: int):
        """Handler für Queue-Size-Änderungen"""
        self.emit_immediate_status_update()
    
    def on_worker_status_changed(self, worker_name: str, status_dict: dict):
        """Handler für Worker-Status-Änderungen"""
        self.emit_immediate_status_update()
    
    def emit_immediate_status_update(self):
        """Sofortiges Status-Update (event-driven) mit DEBUG"""
        if self.state == PipelineState.IDLE:
            self.logger.info("🚫 SKIP STATUS UPDATE - Pipeline is IDLE")
            return
        
        current_status = self.state_aggregator.get_consolidated_status()
        
        # DEBUG: Log status before emitting
        self.logger.info(f"🚀 EMITTING: total_queued={current_status.total_queued}, completed={current_status.total_completed}, current_stage='{current_status.current_stage}'")
        
        # Only emit if status actually changed
        if (current_status.total_queued != getattr(self.last_emitted_status, 'total_queued', -1) or
            current_status.total_completed != getattr(self.last_emitted_status, 'total_completed', -1) or
            current_status.current_stage != getattr(self.last_emitted_status, 'current_stage', '')):
            
            self.logger.info(f"✅ STATUS CHANGED - Emitting to GUI")
            self.status_updated.emit(current_status)
            self.last_emitted_status = current_status
            
            # Check for pipeline finish
            if self.state_aggregator.is_pipeline_finished() and self.state == PipelineState.RUNNING:
                self.logger.info("🏁 PIPELINE FINISH DETECTED")
                self.finish_pipeline()
        else:
            self.logger.info(f"⏸️ STATUS UNCHANGED - Not emitting")
    
    def emit_status_update(self):
        """Timer-basierte Status-Updates (Fallback)"""
        self.emit_immediate_status_update()
    
    def finish_pipeline(self):
        """FIXED Beendet Pipeline mit korrekter Video-Level-Summary"""
        # FIXED: Use video-level metrics for summary
        total_videos = len(self.completed_videos) + len(self.processing_errors)
        successful_videos = len([v for v in self.completed_videos if v.processing_stage == "completed"])
        failed_videos = len(self.processing_errors)
        archived_videos = len([v for v in self.completed_videos if v.processing_stage == "analysis_failed"])
        
        self.logger.info(
            f"FIXED Pipeline finished: {successful_videos} completed, {archived_videos} archived, {failed_videos} failed"
        )
        
        self.cleanup_workers()
        self.state = PipelineState.FINISHED
        
        # Error messages für GUI
        error_messages = [f"{err.stage}: {err.error_message}" for err in self.processing_errors]
        
        self.pipeline_finished.emit(
            total_videos,
            successful_videos,
            error_messages
        )
        
        self.state = PipelineState.IDLE
    
    def stop_pipeline(self):
        """Stoppt Pipeline gracefully"""
        if self.state != PipelineState.RUNNING:
            return
        
        self.logger.info("Stopping FIXED pipeline...")
        self.state = PipelineState.STOPPING
        
        # Stop all workers
        self.cleanup_workers()
        
        # Wait for this thread to finish
        if self.isRunning():
            self.wait(5000)
        
        self.state = PipelineState.IDLE
        self.logger.info("FIXED Pipeline stopped")
    
    def cleanup_workers(self):
        """Beendet alle Worker-Threads"""
        for worker in self.workers:
            worker.stop_worker()
        
        self.workers.clear()
    
    def run(self):
        """QThread main loop - minimal da Event-driven"""
        while self.state == PipelineState.RUNNING:
            self.msleep(100)  # Sleep 100ms

# =============================================================================
# INTEGRATION MIT GUI (Enhanced)
# =============================================================================

def integrate_pipeline_with_gui(main_window, config: AppConfig):
    """Integriert FIXED PipelineManager in MainWindow"""
    
    # Create FIXED PipelineManager
    main_window.pipeline_manager = PipelineManager(config)
    
    # Connect signals
    main_window.pipeline_manager.status_updated.connect(
        main_window.status_widget.update_status
    )
    
    main_window.pipeline_manager.video_completed.connect(
        lambda title, success: main_window.logger.info(
            f"Video completed: {title} ({'Success' if success else 'Failed'})"
        )
    )
    
    main_window.pipeline_manager.pipeline_finished.connect(
        lambda total, success, errors: show_pipeline_summary(main_window, total, success, errors)
    )
    
    # Override start_analysis method
    def enhanced_start_analysis():
        urls_text = main_window.url_input.toPlainText().strip()
        
        if not urls_text:
            main_window.status_bar.showMessage("Please enter YouTube URLs")
            return
        
        # Start FIXED pipeline with URLs text
        start_result = main_window.pipeline_manager.start_pipeline(urls_text)
        
        if isinstance(start_result, Ok):
            main_window.status_bar.showMessage("Started FIXED URL processing and metadata extraction...")
            main_window.url_input.clear()
        else:
            error = unwrap_err(start_result)
            main_window.status_bar.showMessage(f"Failed to start pipeline: {error.message}")
            main_window.logger.error(f"Pipeline start failed: {error.message}")
    
    main_window.start_analysis = enhanced_start_analysis

def show_pipeline_summary(main_window, total: int, success: int, errors: List[str]):
    """Zeigt FIXED Pipeline-Summary-Dialog"""
    failed = total - success
    
    if failed == 0:
        # Success case
        msg = QMessageBox(main_window)
        msg.setWindowTitle("Pipeline Complete")
        msg.setText(f"✅ Successfully processed {success} of {total} videos!")
        msg.setIcon(QMessageBox.Information)
        msg.exec()
    else:
        # Some failures
        error_text = "\n".join(errors[:10])  # Show max 10 errors
        if len(errors) > 10:
            error_text += f"\n... and {len(errors) - 10} more errors"
        
        msg = QMessageBox(main_window)
        msg.setWindowTitle("Pipeline Complete with Errors")
        msg.setText(f"Processed {total} videos:\n✅ {success} successful\n❌ {failed} failed")
        msg.setDetailedText(f"Error details:\n{error_text}")
        msg.setIcon(QMessageBox.Warning)
        msg.exec()
