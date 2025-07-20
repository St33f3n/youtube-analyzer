"""
YouTube Analyzer - Enhanced Fork-Join Pipeline Manager
Vollständige Fork-Join-Architektur mit zentraler Secret-Resolution und config_dict-Distribution
"""

from __future__ import annotations
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading

from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import QMessageBox

# Import our core libraries  
from core_types import Result, Ok, Err, is_ok, unwrap_ok, unwrap_err, CoreError
from yt_analyzer_core import ProcessObject, TranskriptObject, ArchivObject, ProcessingQueue, ArchiveDatabase
from logging_plus import get_logger
from yt_analyzer_config import AppConfig, SecureConfigManager
from yt_url_processor import process_urls_to_objects
from yt_transcription_worker import transcribe_process_object
from yt_audio_downloader import download_audio_for_process_object
from yt_rulechain import analyze_process_object
from yt_video_downloader import download_video_for_process_object


# CORRECTED IMPORTS: config_dict-Varianten für Secret-abhängige Worker
from yt_nextcloud_uploader import upload_to_nextcloud_for_process_object_dict
from yt_llm_processor import process_transcript_with_llm_dict
# from yt_trilium_uploader import upload_to_trilium_dict

# =============================================================================
# ENHANCED PIPELINE STATUS (Simple GUI-Compatible)
# =============================================================================

@dataclass
class PipelineStatus:
    """Enhanced Pipeline Status mit vollständigen LLM-Metriken"""
    # Existing queue counters (unchanged)
    audio_download_queue: int = 0
    transcription_queue: int = 0
    analysis_queue: int = 0
    video_download_queue: int = 0
    upload_queue: int = 0
    processing_queue: int = 0
    
    # Fork-join specific queues
    llm_processing_queue: int = 0
    trilium_upload_queue: int = 0
    
    # Enhanced tracking
    total_queued: int = 0
    total_completed: int = 0
    total_failed: int = 0
    
    # Current processing info
    current_stage: str = "Idle"
    current_video: Optional[str] = None
    
    # Basic features
    active_workers: List[str] = field(default_factory=list)
    pipeline_health: str = "healthy"
    estimated_completion: Optional[datetime] = None
    
    # ✅ HINZUGEFÜGT: LLM-Metriken (fehlten vorher!)
    total_llm_tokens: int = 0
    total_llm_cost: float = 0.0
    active_llm_provider: Optional[str] = None
    
    # ✅ HINZUGEFÜGT: Erweiterte LLM-Metriken
    llm_videos_processed: int = 0
    average_llm_processing_time: float = 0.0
    
    # ✅ HINZUGEFÜGT: Fork-Join spezifische Metriken
    pending_merges: int = 0
    video_stream_completed: int = 0
    transcript_stream_completed: int = 0
    final_archived: int = 0

    def is_active(self) -> bool:
        """Activity check including new fork-join queues"""
        return (self.audio_download_queue + self.transcription_queue + 
                self.analysis_queue + self.video_download_queue + 
                self.upload_queue + self.processing_queue +
                self.llm_processing_queue + self.trilium_upload_queue) > 0

@dataclass
class ProcessingError:
    """Error-Information für Pipeline-Summary"""
    video_title: str
    video_url: str
    stage: str
    error_message: str
    timestamp: datetime
    object_type: str  # "ProcessObject" oder "TranskriptObject"

class PipelineState(Enum):
    """Pipeline-Zustände"""
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    FINISHED = "finished"


# =============================================================================
# LLM METRICS COLLECTOR (NEUE KLASSE)
# =============================================================================

class LLMMetricsCollector:
    """Sammelt und verwaltet LLM-Metriken aus TranskriptObjects"""
    
    def __init__(self):
        self.logger = get_logger("LLMMetricsCollector")
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset alle Metriken für neue Pipeline"""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.videos_processed = 0
        self.processing_times = []
        self.current_provider = None
        self.cost_breakdown = {}  # {"openai": 0.15, "anthropic": 0.05}
        self.token_breakdown = {}  # {"openai": 1500, "anthropic": 800}
    
    def add_transcript_metrics(self, transcript_obj: TranskriptObject):
        """Fügt Metriken von erfolgreich verarbeitetem TranskriptObject hinzu"""
        if not transcript_obj.success:
            return
        
        # Basis-Metriken
        if transcript_obj.tokens:
            self.total_tokens += transcript_obj.tokens
        
        if transcript_obj.cost:
            self.total_cost += transcript_obj.cost
        
        if transcript_obj.processing_time:
            self.processing_times.append(transcript_obj.processing_time)
        
        if transcript_obj.model:
            # Provider aus Model extrahieren
            provider = self._extract_provider_from_model(transcript_obj.model)
            self.current_provider = provider
            
            # Breakdown-Statistiken
            if provider not in self.cost_breakdown:
                self.cost_breakdown[provider] = 0.0
                self.token_breakdown[provider] = 0
            
            if transcript_obj.cost:
                self.cost_breakdown[provider] += transcript_obj.cost
            if transcript_obj.tokens:
                self.token_breakdown[provider] += transcript_obj.tokens
        
        self.videos_processed += 1
        
        self.logger.debug(
            f"Added LLM metrics from {transcript_obj.titel}",
            extra={
                'tokens_added': transcript_obj.tokens,
                'cost_added': transcript_obj.cost,
                'total_tokens': self.total_tokens,
                'total_cost': self.total_cost,
                'videos_processed': self.videos_processed
            }
        )
    
    def _extract_provider_from_model(self, model: str) -> str:
        """Extrahiert Provider-Namen aus Model-String"""
        model_lower = model.lower()
        if 'gpt' in model_lower or 'openai' in model_lower:
            return 'openai'
        elif 'claude' in model_lower or 'anthropic' in model_lower:
            return 'anthropic'
        elif 'gemini' in model_lower or 'google' in model_lower:
            return 'google'
        else:
            return 'unknown'
    
    def get_average_processing_time(self) -> float:
        """Berechnet durchschnittliche LLM-Verarbeitungszeit"""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Gibt vollständige Metriken-Zusammenfassung zurück"""
        return {
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'videos_processed': self.videos_processed,
            'average_processing_time': self.get_average_processing_time(),
            'current_provider': self.current_provider,
            'cost_breakdown': self.cost_breakdown.copy(),
            'token_breakdown': self.token_breakdown.copy(),
            'cost_per_video': self.total_cost / self.videos_processed if self.videos_processed > 0 else 0.0,
            'tokens_per_video': self.total_tokens / self.videos_processed if self.videos_processed > 0 else 0.0
        }


# =============================================================================
# ENHANCED BASE WORKER CLASS für Fork-Join
# =============================================================================

class BaseWorker(QThread):
    """Enhanced Base Worker mit Fork-Join-Support"""
    
    # Signals für Pipeline Manager
    object_processed = Signal(object, str)  # Union[ProcessObject, TranskriptObject], next_stage
    processing_error = Signal(object, str)  # Union[ProcessObject, TranskriptObject], error_message
    stage_status_changed = Signal(str, int)  # stage_name, queue_size
    
    def __init__(self, stage_name: str, input_queue: ProcessingQueue, output_queue: Optional[ProcessingQueue] = None):
        super().__init__()
        self.stage_name = stage_name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.logger = get_logger(f"Worker-{stage_name}")
        self.should_stop = threading.Event()
        self.is_processing = False
        self.current_object: Optional[Union[ProcessObject, TranskriptObject]] = None
    
    def run(self):
        """Enhanced Worker-Loop mit Fork-Join-Object-Support"""
        self.logger.info(f"{self.stage_name} worker started")
        
        while not self.should_stop.is_set():
            try:
                # Get Object from queue (ProcessObject oder TranskriptObject)
                obj_result = self.input_queue.get(timeout=1.0)
                
                if isinstance(obj_result, Err):
                    continue  # Queue empty, try again
                
                obj = unwrap_ok(obj_result)
                self.is_processing = True
                self.current_object = obj
                
                # Update status
                self.stage_status_changed.emit(self.stage_name, self.input_queue.size())
                
                # Process object
                process_result = self.process_object(obj)
                
                if isinstance(process_result, Ok):
                    processed_obj = unwrap_ok(process_result)
                    
                    # Handle routing decision
                    routing_info = self.get_routing_decision(processed_obj)
                    
                    if routing_info["route_to_output"]:
                        if self.output_queue:
                            put_result = self.output_queue.put(processed_obj)
                            if isinstance(put_result, Ok):
                                self.object_processed.emit(processed_obj, routing_info["next_stage"])
                            else:
                                self.processing_error.emit(processed_obj, f"Queue overflow: {unwrap_err(put_result).message}")
                        else:
                            # End of stream - signal completion
                            self.object_processed.emit(processed_obj, "stream_completed")
                    else:
                        # Object rejected (e.g., analysis failed)
                        self.object_processed.emit(processed_obj, routing_info["next_stage"])
                else:
                    error = unwrap_err(process_result)
                    if hasattr(obj, 'add_error'):  # ProcessObject
                        obj.add_error(f"{self.stage_name}: {error.message}")
                    else:  # TranskriptObject
                        obj.error_message = f"{self.stage_name}: {error.message}"
                    self.processing_error.emit(obj, error.message)
                
                self.input_queue.task_done()
                self.current_object = None
                self.is_processing = False
                
            except Exception as e:
                self.logger.error(f"Unexpected error in {self.stage_name} worker: {e}")
                self.current_object = None
                self.is_processing = False
                time.sleep(1)
        
        self.logger.info(f"{self.stage_name} worker stopped")
    
    def process_object(self, obj: Union[ProcessObject, TranskriptObject]) -> Result[Union[ProcessObject, TranskriptObject], CoreError]:
        """Override in subclasses"""
        raise NotImplementedError("Subclasses must implement process_object")
    
    def get_routing_decision(self, obj: Union[ProcessObject, TranskriptObject]) -> Dict[str, Any]:
        """Override für conditional routing"""
        return {
            "route_to_output": True,
            "next_stage": self.get_next_stage_name()
        }
    
    def get_next_stage_name(self) -> str:
        """Override in subclasses"""
        return "unknown"
    
    def stop_worker(self):
        """Graceful worker shutdown"""
        self.should_stop.set()
        if self.isRunning():
            self.wait(5000)

# =============================================================================
# EXISTING WORKER IMPLEMENTATIONS (AppConfig-basiert - UNCHANGED)
# =============================================================================

class AudioDownloadWorker(BaseWorker):
    """Audio-Download Worker (unchanged - kein Secret-Dependency)"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config: AppConfig):
        super().__init__("Audio Download", input_queue, output_queue)
        self.config = config
        temp_dir = Path(self.config.processing.temp_folder)
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"Starting audio download: {obj.titel}")
        return download_audio_for_process_object(obj, self.config)
    
    def get_next_stage_name(self) -> str:
        return "Transcription"

class TranscriptionWorker(BaseWorker):
    """Transcription Worker (unchanged - kein Secret-Dependency)"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config: AppConfig):
        super().__init__("Transcription", input_queue, output_queue)
        self.config = config
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"Starting transcription: {obj.titel}")
        return transcribe_process_object(obj, self.config)
    
    def get_next_stage_name(self) -> str:
        return "Analysis"

class AnalysisWorker(BaseWorker):
    """Analysis Worker - FORK POINT Implementation (unchanged - kein Secret-Dependency)"""
    
    def __init__(self, input_queue: ProcessingQueue, config: AppConfig, pipeline_manager: 'PipelineManager'):
        super().__init__("Analysis", input_queue, None)  # No single output queue
        self.config = config
        self.pipeline_manager = pipeline_manager
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"Starting content analysis: {obj.titel}")
        return analyze_process_object(obj, self.config)
    
    def get_routing_decision(self, obj: ProcessObject) -> Dict[str, Any]:
        """FORK LOGIC: Routes based on analysis result"""
        if obj.passed_analysis:
            # SUCCESSFUL ANALYSIS -> FORK into two streams
            self.logger.info(f"Analysis passed - forking processing for: {obj.titel}")
            
            # Stream A: Video Download Queue
            video_put_result = self.pipeline_manager.queues["video_download"].put(obj)
            if isinstance(video_put_result, Err):
                self.logger.error(f"Failed to queue for video download: {unwrap_err(video_put_result).message}")
            
            # Stream B: LLM Processing Queue (clone to TranskriptObject)
            transcript_obj = TranskriptObject.from_process_object(obj)
            llm_put_result = self.pipeline_manager.queues["llm_processing"].put(transcript_obj)
            if isinstance(llm_put_result, Err):
                self.logger.error(f"Failed to queue for LLM processing: {unwrap_err(llm_put_result).message}")
            
            return {"route_to_output": False, "next_stage": "forked"}
        else:
            # FAILED ANALYSIS -> Direct to archive
            self.logger.info(f"Analysis failed - archiving directly: {obj.titel}")
            obj.video_stream_success = False
            
            # Create failed ArchivObject with no transcript processing
            failed_transcript = TranskriptObject.from_process_object(obj)
            failed_transcript.success = False
            failed_transcript.error_message = "Analysis failed - no LLM processing"
            
            archive_obj = ArchivObject.from_process_and_transcript(obj, failed_transcript)
            self.pipeline_manager.archive_final_object(archive_obj)
            
            return {"route_to_output": False, "next_stage": "archived_failed"}
    
    def get_next_stage_name(self) -> str:
        return "Fork"

class VideoDownloadWorker(BaseWorker):
    """Video Download Worker - UNVERÄNDERT (keine Secrets)"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config: AppConfig):
        super().__init__("Video Download", input_queue, output_queue)
        self.config = config
        temp_dir = Path(self.config.processing.temp_folder)
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"Starting video download: {obj.titel}")
        return download_video_for_process_object(obj, self.config)
    
    def get_next_stage_name(self) -> str:
        return "Upload"
# =============================================================================
# NEW: config_dict-BASED WORKERS (Stream A & B)
# =============================================================================

class UploadWorker(BaseWorker):
    """Upload Worker - Stream A Final (NEW: config_dict-basiert)"""
    
    def __init__(self, input_queue: ProcessingQueue, config_dict: dict):
        super().__init__("Upload", input_queue, None)  # No output queue - end of Stream A
        self.config_dict = config_dict
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"Starting Nextcloud upload: {obj.titel}")
        
        # CORRECTED: Verwende config_dict-Variante mit resolved secrets
        result = upload_to_nextcloud_for_process_object_dict(obj, self.config_dict)
        
        if isinstance(result, Ok):
            processed_obj = unwrap_ok(result)
            processed_obj.video_stream_success = True
            return Ok(processed_obj)
        else:
            obj.video_stream_success = False
            return result
    
    def get_next_stage_name(self) -> str:
        return "Stream A Completed"

class LLMProcessingWorker(BaseWorker):
    """LLM Processing Worker - Stream B (UPDATED: Real LLM integration)"""

    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config_dict: dict):
        super().__init__("LLM Processing", input_queue, output_queue)
        self.config_dict = config_dict

    def process_object(self, obj: TranskriptObject) -> Result[TranskriptObject, CoreError]:
        self.logger.info(f"Starting LLM processing: {obj.titel}")
    
        result = process_transcript_with_llm_dict(obj, self.config_dict)
    
        if isinstance(result, Ok):
            processed_obj = unwrap_ok(result)
            self.logger.info(
                f"LLM processing completed successfully",
                extra={
                    'video_title': processed_obj.titel,
                    'provider': self.config_dict['llm_processing']['provider'],
                    'model': processed_obj.model,
                    'tokens': processed_obj.tokens,
                    'cost': processed_obj.cost,
                    'processing_time': processed_obj.processing_time
                }
            )
            return Ok(processed_obj)
        else:
            error = unwrap_err(result)
            obj.success = False
            obj.error_message = f"LLM processing failed: {error.message}"
            self.logger.error(
                f"LLM processing failed",
                extra={
                    'video_title': obj.titel,
                    'error': error.message,
                    'provider': self.config_dict['llm_processing']['provider']
                }
            )
            return result
    
    def get_next_stage_name(self) -> str:
        return "Trilium Upload"

class TrilliumUploadWorker(BaseWorker):
    """Trilium Upload Worker - Stream B Final (NEW: config_dict-basiert)"""
    
    def __init__(self, input_queue: ProcessingQueue, config_dict: dict):
        super().__init__("Trilium Upload", input_queue, None)  # No output queue - end of Stream B
        self.config_dict = config_dict
    
    def process_object(self, obj: TranskriptObject) -> Result[TranskriptObject, CoreError]:
        self.logger.info(f"Starting Trilium upload: {obj.titel}")
        
        # PLACEHOLDER: Mock Trilium upload until real implementation available
        # result = upload_to_trilium_dict(obj, self.config_dict)
        
        time.sleep(1.0)  # Simulate upload
        
        resolved_secrets = self.config_dict.get('resolved_secrets', {})
        trilium_token = resolved_secrets.get('trilium_token')
        
        if not trilium_token:
            obj.success = False
            obj.error_message = "Trilium token not resolved"
            return Err(CoreError("Trilium token not available"))
        
        obj.trilium_link = f"https://trilium.example.com/note/{obj.titel.replace(' ', '_')}"
        obj.update_stage("trilium_upload_completed")
        
        self.logger.info(f"Trilium upload completed with token length: {len(trilium_token)}")
        
        return Ok(obj)
    
    def get_next_stage_name(self) -> str:
        return "Stream B Completed"

# =============================================================================
# ENHANCED FORK-JOIN PIPELINE MANAGER (Zentrale Secret-Resolution)
# =============================================================================

class PipelineManager(QThread):
    """Enhanced Pipeline Manager with Zentrale Secret-Resolution und config_dict-Distribution"""
    
    # Signals für GUI
    status_updated = Signal(PipelineStatus)
    video_completed = Signal(str, bool)  # title, success
    pipeline_finished = Signal(int, int, list)  # total, success, errors
    
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger("PipelineManager")
        
        # NEW: Zentrale Secret-Resolution
        self.config_manager = SecureConfigManager()
        self.config_manager.load_config()
        self.llm_metrics = LLMMetricsCollector()       
        self.config_dict = self._resolve_config_dict()

        self.stream_completion_stats = {
            'video_completed': 0,
            'transcript_completed': 0,
            'final_archived': 0,
            'video_failed': 0,
            'transcript_failed': 0
        }

        
        # Pipeline State
        self.state = PipelineState.IDLE
        self.input_urls_text: str = ""
        self.processing_errors: List[ProcessingError] = []
        self.completed_videos: List[ArchivObject] = []
        
        # Queues für Fork-Join-Architecture
        self.queues = {
            # Sequential phase (unchanged)
            "audio_download": ProcessingQueue("audio_download"),
            "transcription": ProcessingQueue("transcription"),
            "analysis": ProcessingQueue("analysis"),
            
            # Stream A (Video Processing)
            "video_download": ProcessingQueue("video_download"),
            "upload": ProcessingQueue("upload"),
            
            # Stream B (Transcript Processing)
            "llm_processing": ProcessingQueue("llm_processing"),
            "trilium_upload": ProcessingQueue("trilium_upload")
        }
        
        # NEW: Fork-Join State Collections (as specified in architecture)
        self.video_success_list: List[ProcessObject] = []
        self.video_failure_list: List[ProcessObject] = []
        self.transcript_success_list: List[TranskriptObject] = []
        self.transcript_failure_list: List[TranskriptObject] = []
        
        # Archive management
        self.archive_database = ArchiveDatabase()
        
        # Thread safety
        self.state_lock = threading.Lock()
        
        # Workers
        self.workers: List[BaseWorker] = []
        
        # Status Update Timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.emit_status_update)
        self.status_timer.start(2000)  # 2 seconds
    
    def _resolve_config_dict(self) -> dict:
        """NEW: Zentrale Secret-Resolution für alle Worker"""
        config_dict = self.config.dict()
        resolved_secrets = {}
        
        self.logger.info("🔐 Starting zentrale secret resolution...")
        
        # Nextcloud Secret Resolution
        nextcloud_result = self.config_manager.get_nextcloud_password()
        if isinstance(nextcloud_result, Ok):
            resolved_secrets['nextcloud_password'] = unwrap_ok(nextcloud_result)
            self.logger.info("✅ Nextcloud password resolved")
        else:
            self.logger.warning(f"⚠️ Nextcloud password resolution failed: {unwrap_err(nextcloud_result).message}")
        
        # Trilium Secret Resolution
        trilium_result = self.config_manager.get_trilium_token()
        if isinstance(trilium_result, Ok):
            resolved_secrets['trilium_token'] = unwrap_ok(trilium_result)
            self.logger.info("✅ Trilium token resolved")
        else:
            self.logger.warning(f"⚠️ Trilium token resolution failed: {unwrap_err(trilium_result).message}")
        
        # LLM API Key Resolution
        llm_provider = self.config.llm_processing.provider
        llm_key_result = self.config_manager.get_llm_api_key(llm_provider)
        if isinstance(llm_key_result, Ok):
            resolved_secrets['llm_api_key'] = unwrap_ok(llm_key_result)
            self.logger.info(f"✅ {llm_provider.title()} API key resolved")
        else:
            self.logger.warning(f"⚠️ {llm_provider.title()} API key resolution failed: {unwrap_err(llm_key_result).message}")
        
        # Add resolved secrets to config_dict
        config_dict['resolved_secrets'] = resolved_secrets
        
        self.logger.info(
            f"🔐 Secret resolution completed",
            extra={
                'resolved_secrets_count': len(resolved_secrets),
                'resolved_secrets_keys': list(resolved_secrets.keys()),
                'nextcloud_ready': 'nextcloud_password' in resolved_secrets,
                'trilium_ready': 'trilium_token' in resolved_secrets,
                'llm_ready': 'llm_api_key' in resolved_secrets
            }
        )
        
        return config_dict
    
    def start_pipeline(self, urls_text: str) -> Result[None, CoreError]:
        """Startet Pipeline mit synchroner Metadata-Extraktion"""
        if self.state != PipelineState.IDLE:
            return Err(CoreError("Pipeline already running"))
        
        self.input_urls_text = urls_text
        self.processing_errors.clear()
        self.completed_videos.clear()


        self.llm_metrics.reset_metrics()
        self.stream_completion_stats = {
            'video_completed': 0,
            'transcript_completed': 0,
            'final_archived': 0,
            'video_failed': 0,
            'transcript_failed': 0
        }

        
        # Clear state collections
        with self.state_lock:
            self.video_success_list.clear()
            self.video_failure_list.clear()
            self.transcript_success_list.clear()
            self.transcript_failure_list.clear()
        
        self.logger.info("Starting Fork-Join pipeline with zentrale secret resolution")
        
        # SYNCHRONE Metadata-Extraktion (no worker needed)
        objects_result = process_urls_to_objects(urls_text, self.config.processing.__dict__)
        
        if isinstance(objects_result, Err):
            errors = unwrap_err(objects_result)
            return Err(CoreError(f"URL processing failed: {len(errors)} errors"))
        
        process_objects = unwrap_ok(objects_result)
        if not process_objects:
            return Err(CoreError("No valid videos found"))
        
        # Queue ProcessObjects für Audio Download
        for process_obj in process_objects:
            duplicate_result = self.archive_database.check_duplicate(process_obj)
            if isinstance(duplicate_result, Ok) and unwrap_ok(duplicate_result):
                self.logger.info(f"Skipping duplicate video: {process_obj.titel}")
                continue
            
            put_result = self.queues["audio_download"].put(process_obj)
            if isinstance(put_result, Err):
                self.logger.error(f"Failed to queue audio download: {unwrap_err(put_result).message}")
        
        # Setup und start workers
        setup_result = self.setup_workers()
        if isinstance(setup_result, Err):
            return setup_result
        
        self.state = PipelineState.RUNNING
        self.start()  # Start QThread
        
        self.logger.info(f"Fork-Join pipeline started with {len(process_objects)} videos")
        return Ok(None)
    
    def setup_workers(self) -> Result[None, CoreError]:
        """Enhanced worker setup mit config_dict-Distribution"""
        try:
            self.cleanup_workers()
            
            # Create all workers mit korrekter config-Distribution
            self.workers = [
                # Sequential workers (AppConfig - kein Secret-Dependency)
                AudioDownloadWorker(self.queues["audio_download"], self.queues["transcription"], self.config),
                TranscriptionWorker(self.queues["transcription"], self.queues["analysis"], self.config),
                AnalysisWorker(self.queues["analysis"], self.config, self),  # Fork point
                
                # Stream A: Video Processing (gemischt)
                VideoDownloadWorker(self.queues["video_download"], self.queues["upload"], self.config),  # kein Secret
                UploadWorker(self.queues["upload"], self.config_dict),  # NEW: config_dict mit Secret
                
                # Stream B: Transcript Processing (NEW: config_dict mit Secrets)
                LLMProcessingWorker(self.queues["llm_processing"], self.queues["trilium_upload"], self.config_dict),
                TrilliumUploadWorker(self.queues["trilium_upload"], self.config_dict)
            ]
            
            # Connect signals für Fork-Join-Orchestration
            for worker in self.workers:
                worker.object_processed.connect(self.handle_worker_completion)
                worker.processing_error.connect(self.handle_worker_error)
            
            # Start all workers
            for worker in self.workers:
                worker.start()
            
            self.logger.info(
                f"Started {len(self.workers)} workers for Fork-Join pipeline",
                extra={
                    'appconfig_workers': 4,  # Audio, Transcription, Analysis, VideoDownload  
                    'config_dict_workers': 3,  # Upload, LLM, Trilium
                    'resolved_secrets': list(self.config_dict.get('resolved_secrets', {}).keys())
                }
            )
            return Ok(None)
            
        except Exception as e:
            return Err(CoreError(f"Fork-Join worker setup failed: {e}"))
    
    def handle_worker_completion(self, obj: Union[ProcessObject, TranskriptObject], next_stage: str):
            """Enhanced completion handler mit LLM-Metriken-Sammlung"""
            with self.state_lock:
                if isinstance(obj, ProcessObject):
                    # Stream A completion
                    if next_stage == "stream_completed" or next_stage == "Stream A Completed":
                        obj.video_stream_success = True
                        self.stream_completion_stats['video_completed'] += 1
                        self.process_video_completion(obj)
                    
                elif isinstance(obj, TranskriptObject):
                    # Stream B completion
                    if next_stage == "stream_completed" or next_stage == "Stream B Completed":
                        obj.success = True
                        self.stream_completion_stats['transcript_completed'] += 1
                    
                        # ✅ HINZUGEFÜGT: LLM-Metriken sammeln
                        self.llm_metrics.add_transcript_metrics(obj)
                    
                        self.process_transcript_completion(obj)
                            

    def handle_worker_error(self, obj: Union[ProcessObject, TranskriptObject], error_message: str):
            """Enhanced error handler mit Stream-Statistiken"""
            with self.state_lock:
                if isinstance(obj, ProcessObject):
                    # Stream A failure
                    obj.video_stream_success = False
                    self.stream_completion_stats['video_failed'] += 1
                    if hasattr(obj, 'add_error'):
                        obj.add_error(f"Video stream failed: {error_message}")
                    self.process_video_completion(obj)
                
                elif isinstance(obj, TranskriptObject):
                    # Stream B failure
                    obj.success = False
                    obj.error_message = error_message
                    self.stream_completion_stats['transcript_failed'] += 1
                    self.process_transcript_completion(obj)
    
    def process_video_completion(self, video_obj: ProcessObject):
        """Processes Stream A completion and attempts merging"""
        titel = video_obj.titel
        
        # Check if corresponding transcript is already completed
        transcript_match = self.find_and_remove_transcript(titel)
        
        if transcript_match:
            # Both streams completed - merge and archive
            archive_obj = self.merge_objects(video_obj, transcript_match)
            self.completed_videos.append(archive_obj)
            self.video_completed.emit(archive_obj.titel, archive_obj.final_success)
        else:
            # Store in appropriate list for later merging
            if video_obj.video_stream_success:
                self.video_success_list.append(video_obj)
            else:
                self.video_failure_list.append(video_obj)
    
    def process_transcript_completion(self, transcript_obj: TranskriptObject):
        """Processes Stream B completion and attempts merging"""
        titel = transcript_obj.titel
        
        # Check if corresponding video is already completed
        video_match = self.find_and_remove_video(titel)
        
        if video_match:
            # Both streams completed - merge and archive
            archive_obj = self.merge_objects(video_match, transcript_obj)
            self.completed_videos.append(archive_obj)
            self.video_completed.emit(archive_obj.titel, archive_obj.final_success)
        else:
            # Store in appropriate list for later merging
            if transcript_obj.success:
                self.transcript_success_list.append(transcript_obj)
            else:
                self.transcript_failure_list.append(transcript_obj)
    
    def find_and_remove_transcript(self, titel: str) -> Optional[TranskriptObject]:
        """Finds and removes matching transcript from state lists"""
        # Check success list first
        for i, obj in enumerate(self.transcript_success_list):
            if obj.titel == titel:
                return self.transcript_success_list.pop(i)
        
        # Check failure list
        for i, obj in enumerate(self.transcript_failure_list):
            if obj.titel == titel:
                return self.transcript_failure_list.pop(i)
        
        return None
    
    def find_and_remove_video(self, titel: str) -> Optional[ProcessObject]:
        """Finds and removes matching video from state lists"""
        # Check success list first
        for i, obj in enumerate(self.video_success_list):
            if obj.titel == titel:
                return self.video_success_list.pop(i)
        
        # Check failure list
        for i, obj in enumerate(self.video_failure_list):
            if obj.titel == titel:
                return self.video_failure_list.pop(i)
        
        return None
    
    def merge_objects(self, video_obj: ProcessObject, transcript_obj: TranskriptObject) -> ArchivObject:
        """Creates ArchivObject and archives directly (Pipeline Manager handles archive)"""
        # Create archive object with clean separation
        archive_obj = ArchivObject.from_process_and_transcript(video_obj, transcript_obj)
        
        if archive_obj.final_success:
            self.stream_completion_stats['final_archived'] += 1
        
        # Direct archive insertion by Pipeline Manager
        archive_result = self.archive_database.save_processed_video(archive_obj)
        
        if isinstance(archive_result, Ok):
                    self.logger.info(
                        f"Video archived successfully: {archive_obj.titel}",
                        extra={
                            'final_success': archive_obj.final_success,
                            'video_stream_success': archive_obj.video_stream_success,
                            'transcript_stream_success': archive_obj.transcript_stream_success,
                            'llm_model': archive_obj.llm_model,
                            'llm_cost': archive_obj.llm_cost,
                            'llm_tokens': archive_obj.llm_tokens,
                            'has_nextcloud_link': bool(archive_obj.nextcloud_link),
                            'has_trilium_link': bool(archive_obj.trilium_link)
                        }
                    )
        else:
            self.logger.error(f"Archive failed for {archive_obj.titel}: {unwrap_err(archive_result).message}")
                
        return archive_obj
    
    def archive_final_object(self, archive_obj: ArchivObject):
        """Direct archiving method called by AnalysisWorker for failed analysis"""
        archive_result = self.archive_database.save_processed_video(archive_obj)
        
        if isinstance(archive_result, Ok):
            self.logger.info(f"Failed analysis object archived: {archive_obj.titel}")
            self.completed_videos.append(archive_obj)
            self.video_completed.emit(archive_obj.titel, False)
        else:
            self.logger.error(f"Archive failed for failed analysis: {archive_obj.titel}")

    def emit_status_update(self):
            """✅ KORRIGIERTE Status-Update mit vollständigen LLM-Metriken"""
            if self.state == PipelineState.IDLE:
                return
        
            # Calculate active workers and current video
            active_workers_list = []
            current_video_title = None
        
            for worker in self.workers:
                if worker.is_processing:
                    active_workers_list.append(worker.stage_name)
                    if worker.current_object and not current_video_title:
                        current_video_title = worker.current_object.titel
        
            # Calculate pipeline health
            total_errors = len(self.processing_errors)
            total_processed = len(self.completed_videos) + total_errors
        
            if total_processed == 0:
                health = "healthy"
            elif total_errors / total_processed > 0.3:  # > 30% error rate
                health = "failed"
            elif total_errors / total_processed > 0.1:  # > 10% error rate
                health = "degraded"
            else:
                health = "healthy"
        
            # ✅ HINZUGEFÜGT: LLM-Metriken aus Collector holen
            llm_summary = self.llm_metrics.get_metrics_summary()
        
            # ✅ KORRIGIERT: Status-Berechnung mit allen LLM-Metriken
            status = PipelineStatus(
                # Queue sizes + active worker tracking
                audio_download_queue=self.queues["audio_download"].size() + (1 if any(w.stage_name == "Audio Download" and w.is_processing for w in self.workers) else 0),
                transcription_queue=self.queues["transcription"].size() + (1 if any(w.stage_name == "Transcription" and w.is_processing for w in self.workers) else 0),
                analysis_queue=self.queues["analysis"].size() + (1 if any(w.stage_name == "Analysis" and w.is_processing for w in self.workers) else 0),
                video_download_queue=self.queues["video_download"].size() + (1 if any(w.stage_name == "Video Download" and w.is_processing for w in self.workers) else 0),
                upload_queue=self.queues["upload"].size() + (1 if any(w.stage_name == "Upload" and w.is_processing for w in self.workers) else 0),
                processing_queue=0,  # Not used in Fork-Join
            
                # Fork-Join queues
                llm_processing_queue=self.queues["llm_processing"].size() + (1 if any(w.stage_name == "LLM Processing" and w.is_processing for w in self.workers) else 0),
                trilium_upload_queue=self.queues["trilium_upload"].size() + (1 if any(w.stage_name == "Trilium Upload" and w.is_processing for w in self.workers) else 0),
            
                total_queued=self.get_total_objects_count(),
                total_completed=len(self.completed_videos),
                total_failed=len(self.processing_errors),
            
                current_stage=self.get_current_stage(active_workers_list),
                current_video=current_video_title[:30] + "..." if current_video_title and len(current_video_title) > 30 else current_video_title,
            
                active_workers=active_workers_list,
                pipeline_health=health,
            
                # ✅ HINZUGEFÜGT: Vollständige LLM-Metriken
                total_llm_tokens=llm_summary['total_tokens'],
                total_llm_cost=llm_summary['total_cost'],
                active_llm_provider=llm_summary['current_provider'],
                llm_videos_processed=llm_summary['videos_processed'],
                average_llm_processing_time=llm_summary['average_processing_time'],
            
                # ✅ HINZUGEFÜGT: Fork-Join spezifische Metriken
                pending_merges=(len(self.video_success_list) + len(self.video_failure_list) + 
                               len(self.transcript_success_list) + len(self.transcript_failure_list)),
                video_stream_completed=self.stream_completion_stats['video_completed'],
                transcript_stream_completed=self.stream_completion_stats['transcript_completed'],
                final_archived=self.stream_completion_stats['final_archived']
            )
        
            self.status_updated.emit(status)
        
            # Enhanced Debug Logging mit LLM-Metriken
            self.logger.debug(
                f"GUI Status Update mit LLM-Metriken",
                extra={
                    'total_queued': status.total_queued,
                    'sequential_active': status.audio_download_queue + status.transcription_queue + status.analysis_queue,
                    'stream_a_active': status.video_download_queue + status.upload_queue,
                    'stream_b_active': status.llm_processing_queue + status.trilium_upload_queue,
                    'final_archived': status.final_archived,
                    'llm_tokens': status.total_llm_tokens,
                    'llm_cost': status.total_llm_cost,
                    'llm_provider': status.active_llm_provider,
                    'llm_videos_processed': status.llm_videos_processed,
                    'pending_merges': status.pending_merges
                }
            )
        
            # Check if pipeline finished (with pending merges check)
            all_queues_empty = not status.is_active()
            no_workers_active = len(active_workers_list) == 0
            no_pending_merges = status.pending_merges == 0
        
            if all_queues_empty and no_workers_active and no_pending_merges and self.state == PipelineState.RUNNING:
                self.finish_pipeline()
    
    def get_total_objects_count(self) -> int:
        """Gesamtzahl Objects in Pipeline"""
        queue_count = sum(queue.size() for queue in self.queues.values())
        state_count = (len(self.video_success_list) + len(self.video_failure_list) + 
                      len(self.transcript_success_list) + len(self.transcript_failure_list))
        return queue_count + state_count + len(self.completed_videos) + len(self.processing_errors)
    
    def get_current_stage(self, active_workers: List[str]) -> str:
        """Current Stage basierend auf aktiven Workern"""
        if active_workers:
            return active_workers[0]  # First active worker
        
        # Fallback: Größte Queue
        for stage_name, queue in self.queues.items():
            if queue.size() > 0:
                return stage_name.replace("_", " ").title()
        
        return "Idle"
    
    def finish_pipeline(self):
        """Pipeline beenden und Summary senden"""
        total_processed = len(self.completed_videos) + len(self.processing_errors)
        successful_videos = sum(1 for video in self.completed_videos if video.final_success)
        
        self.logger.info(
            f"Fork-Join pipeline finished",
            extra={
                'total_videos': total_processed,
                'successful': successful_videos,
                'failed': total_processed - successful_videos,
                'processing_errors': len(self.processing_errors),
                'resolved_secrets_used': list(self.config_dict.get('resolved_secrets', {}).keys())
            }
        )
        
        self.cleanup_workers()
        self.state = PipelineState.FINISHED
        
        # Error messages für GUI
        error_messages = [f"{err.stage}: {err.error_message}" for err in self.processing_errors]
        
        self.pipeline_finished.emit(
            total_processed,
            successful_videos,
            error_messages
        )
        
        self.state = PipelineState.IDLE
    
    def stop_pipeline(self):
        """Pipeline stoppen"""
        if self.state != PipelineState.RUNNING:
            return
        
        self.logger.info("Stopping Fork-Join pipeline...")
        self.state = PipelineState.STOPPING
        
        self.cleanup_workers()
        
        if self.isRunning():
            self.wait(5000)
        
        self.state = PipelineState.IDLE
        self.logger.info("Fork-Join pipeline stopped")
    
    def cleanup_workers(self):
        """Alle Worker beenden"""
        for worker in self.workers:
            worker.stop_worker()
        self.workers.clear()
    
    def run(self):
        """QThread main loop"""
        while self.state == PipelineState.RUNNING:
            self.msleep(100)

# =============================================================================
# GUI INTEGRATION (Enhanced for Fork-Join)
# =============================================================================

def integrate_pipeline_with_gui(main_window, config: AppConfig):
    """Integriert Enhanced Fork-Join Pipeline Manager in GUI"""
    
    # Create Enhanced Pipeline Manager
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
        lambda total, success, errors: show_fork_join_pipeline_summary(main_window, total, success, errors)
    )
    
    # Enhanced start_analysis method
    def enhanced_start_analysis():
        urls_text = main_window.url_input.toPlainText().strip()
        
        if not urls_text:
            main_window.status_bar.showMessage("Please enter YouTube URLs")
            return
        
        # Start Fork-Join pipeline
        start_result = main_window.pipeline_manager.start_pipeline(urls_text)
        
        if isinstance(start_result, Ok):
            main_window.status_bar.showMessage("Fork-Join Pipeline started - Processing URLs...")
            main_window.url_input.clear()
        else:
            error = unwrap_err(start_result)
            main_window.status_bar.showMessage(f"Failed to start: {error.message}")
            main_window.logger.error(f"Fork-Join Pipeline start failed: {error.message}")
    
    main_window.start_analysis = enhanced_start_analysis

def show_fork_join_pipeline_summary(main_window, total: int, success: int, errors: List[str]):
    """Enhanced Pipeline Summary Dialog für Fork-Join"""
    failed = total - success
    
    if failed == 0:
        msg = QMessageBox(main_window)
        msg.setWindowTitle("Fork-Join Pipeline Complete")
        msg.setText(f"✅ Successfully processed {success} of {total} videos!\n\nFork-Join architecture with zentrale secret resolution completed successfully.")
        msg.setIcon(QMessageBox.Information)
        msg.exec()
    else:
        error_text = "\n".join(errors[:10])
        if len(errors) > 10:
            error_text += f"\n... and {len(errors) - 10} more errors"
        
        msg = QMessageBox(main_window)
        msg.setWindowTitle("Fork-Join Pipeline Complete with Errors")
        msg.setText(f"Fork-Join Pipeline processed {total} videos:\n✅ {success} successful\n❌ {failed} failed")
        msg.setDetailedText(f"Error details:\n{error_text}")
        msg.setIcon(QMessageBox.Warning)
        msg.exec()

# =============================================================================
# EXAMPLE USAGE & TESTING
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    from yt_analyzer_config import SecureConfigManager
    
    # Setup
    setup_logging("fork_join_pipeline_test", "DEBUG")
    
    # Test configuration
    config_manager = SecureConfigManager()
    config_result = config_manager.load_config()
    
    if isinstance(config_result, Ok):
        config = unwrap_ok(config_result)
        
        # Create test pipeline manager
        pipeline_manager = PipelineManager(config)
        
        print("✅ Fork-Join Pipeline Manager created successfully")
        print(f"   Queues: {list(pipeline_manager.queues.keys())}")
        print(f"   State Collections: video_lists + transcript_lists")
        print(f"   Archive Database: {pipeline_manager.archive_database}")
        print(f"   Resolved Secrets: {list(pipeline_manager.config_dict.get('resolved_secrets', {}).keys())}")
        
        # Test queue operations
        test_urls = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        print(f"\n🧪 Testing Fork-Join Pipeline startup...")
        start_result = pipeline_manager.start_pipeline(test_urls)
        
        if isinstance(start_result, Ok):
            print("✅ Pipeline started successfully")
            
            # Let it run for a few seconds
            time.sleep(5)
            
            # Stop pipeline
            pipeline_manager.stop_pipeline()
            print("✅ Pipeline stopped successfully")
        else:
            error = unwrap_err(start_result)
            print(f"❌ Pipeline start failed: {error.message}")
    
    else:
        error = unwrap_err(config_result)
        print(f"❌ Config loading failed: {error.message}")
    
    print("\n🚀 Fork-Join Pipeline Manager Implementation with zentrale Secret-Resolution Complete!")
