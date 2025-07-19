"""
YouTube Analyzer - GUI-Compatible Pipeline Manager
Mit einmaliger Secret-Resolution beim Start für Thread-Safety
"""

from __future__ import annotations
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import threading

from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import QMessageBox

# Import our core libraries  
from core_types import Result, Ok, Err, is_ok, unwrap_ok, unwrap_err, CoreError
from yt_analyzer_core import ProcessObject, ProcessingQueue, ArchiveDatabase
from logging_plus import get_logger
from yt_analyzer_config import AppConfig, SecureConfigManager
from yt_url_processor import process_urls_to_objects
from yt_transcription_worker import transcribe_process_object
from yt_audio_downloader import download_audio_for_process_object
from yt_rulechain import analyze_process_object
from yt_video_downloader import download_video_for_process_object
from yt_nextcloud_uploader import upload_to_nextcloud_for_process_object_dict

# =============================================================================
# GUI-COMPATIBLE PIPELINE STATUS (unchanged)
# =============================================================================

@dataclass
class PipelineStatus:
    """GUI-Compatible Pipeline Status (ohne metadata_queue)"""
    audio_download_queue: int = 0
    transcription_queue: int = 0
    analysis_queue: int = 0
    video_download_queue: int = 0
    upload_queue: int = 0
    processing_queue: int = 0
    
    total_queued: int = 0
    total_completed: int = 0
    total_failed: int = 0
    
    current_stage: str = "Idle"
    current_video: Optional[str] = None
    
    # Enhanced Features für GUI
    active_workers: List[str] = field(default_factory=list)
    pipeline_health: str = "healthy"
    estimated_completion: Optional[datetime] = None
    
    def is_active(self) -> bool:
        """GUI-kompatible is_active ohne metadata_queue"""
        return (self.audio_download_queue + self.transcription_queue + 
                self.analysis_queue + self.video_download_queue + 
                self.upload_queue + self.processing_queue) > 0

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
# ENHANCED BASE WORKER CLASS
# =============================================================================

class BaseWorker(QThread):
    """Enhanced Base Worker mit Current-Object-Tracking"""
    
    # Signals
    object_processed = Signal(ProcessObject, str)
    processing_error = Signal(ProcessObject, str)
    stage_status_changed = Signal(str, int)
    
    def __init__(self, stage_name: str, input_queue: ProcessingQueue, output_queue: Optional[ProcessingQueue] = None):
        super().__init__()
        self.stage_name = stage_name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.logger = get_logger(f"Worker-{stage_name}")
        self.should_stop = threading.Event()
        self.is_processing = False
        self.current_object: Optional[ProcessObject] = None
    
    def run(self):
        """Enhanced Worker-Loop mit Current-Object-Tracking"""
        self.logger.info(f"{self.stage_name} worker started")
        
        while not self.should_stop.is_set():
            try:
                # Get ProcessObject from queue
                obj_result = self.input_queue.get(timeout=1.0)
                
                if isinstance(obj_result, Err):
                    continue  # Queue empty, try again
                
                process_obj = unwrap_ok(obj_result)
                self.is_processing = True
                self.current_object = process_obj
                
                # Update status
                self.stage_status_changed.emit(self.stage_name, self.input_queue.size())
                
                # Process object
                process_result = self.process_object(process_obj)
                
                if isinstance(process_result, Ok):
                    processed_obj = unwrap_ok(process_result)
                    
                    # Handle routing decision
                    next_stage_info = self.get_routing_decision(processed_obj)
                    
                    if next_stage_info["route_to_output"]:
                        if self.output_queue:
                            put_result = self.output_queue.put(processed_obj)
                            if isinstance(put_result, Ok):
                                self.object_processed.emit(processed_obj, next_stage_info["next_stage"])
                            else:
                                self.processing_error.emit(processed_obj, f"Queue overflow")
                        else:
                            self.object_processed.emit(processed_obj, "completed")
                    else:
                        self.object_processed.emit(processed_obj, "archive")
                else:
                    error = unwrap_err(process_result)
                    process_obj.add_error(f"{self.stage_name}: {error.message}")
                    self.processing_error.emit(process_obj, error.message)
                
                self.input_queue.task_done()
                self.current_object = None
                self.is_processing = False
                
            except Exception as e:
                self.logger.error(f"Unexpected error in {self.stage_name} worker: {e}")
                self.current_object = None
                self.is_processing = False
                time.sleep(1)
        
        self.logger.info(f"{self.stage_name} worker stopped")
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        """Override in subclasses"""
        raise NotImplementedError("Subclasses must implement process_object")
    
    def get_routing_decision(self, obj: ProcessObject) -> Dict[str, Any]:
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
# WORKER IMPLEMENTATIONS - NUR Upload Worker geändert!
# =============================================================================

class AudioDownloadWorker(BaseWorker):
    """Audio-Download Worker - UNVERÄNDERT (keine Secrets)"""
    
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
    """Transcription Worker - UNVERÄNDERT (keine Secrets)"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config: AppConfig):
        super().__init__("Transcription", input_queue, output_queue)
        self.config = config
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"Starting transcription: {obj.titel}")
        return transcribe_process_object(obj, self.config)
    
    def get_next_stage_name(self) -> str:
        return "Analysis"

class AnalysisWorker(BaseWorker):
    """Analysis Worker - UNVERÄNDERT (keine Secrets)"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config: AppConfig):
        super().__init__("Analysis", input_queue, output_queue)
        self.config = config
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"Starting content analysis: {obj.titel}")
        return analyze_process_object(obj, self.config)
    
    def get_routing_decision(self, obj: ProcessObject) -> Dict[str, Any]:
        """Conditional routing based on analysis result"""
        if obj.passed_analysis:
            return {"route_to_output": True, "next_stage": "Video Download"}
        else:
            return {"route_to_output": False, "next_stage": "Archive"}
    
    def get_next_stage_name(self) -> str:
        return "Video Download"

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

class UploadWorker(BaseWorker):
    """Upload Worker - EINZIGER mit Config-Dict Support (braucht Secrets)"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config_dict: dict):
        super().__init__("Upload", input_queue, output_queue)
        self.config_dict = config_dict  # ← Dict für Secrets
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"Starting Nextcloud upload: {obj.titel}")
        return upload_to_nextcloud_for_process_object_dict(obj, self.config_dict)
    
    def get_next_stage_name(self) -> str:
        return "Processing"

class ProcessingWorker(BaseWorker):
    """Processing Worker - Mock (könnte Trilium-Secrets brauchen)"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: Optional[ProcessingQueue], config_dict: dict):
        super().__init__("Processing", input_queue, output_queue)
        self.config_dict = config_dict  # ← Dict für potentielle Trilium-Secrets
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        time.sleep(0.5)  # Mock processing
        obj.bearbeiteter_transkript = f"Bearbeitetes Transkript: {obj.transkript}"
        obj.trilium_link = f"trilium://note/{obj.get_unique_key()}"
        obj.update_stage("completed")
        return Ok(obj)
    
    def get_next_stage_name(self) -> str:
        return "completed"

# =============================================================================
# GUI-COMPATIBLE PIPELINE MANAGER mit SECRET-RESOLUTION
# =============================================================================

class PipelineManager(QThread):
    """GUI-Compatible Pipeline Manager mit einmaliger Secret-Resolution"""
    
    # Signals für GUI
    status_updated = Signal(PipelineStatus)
    video_completed = Signal(str, bool)  # title, success
    pipeline_finished = Signal(int, int, list)  # total, success, errors
    
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger("PipelineManager")
        
        # Pipeline State
        self.state = PipelineState.IDLE
        self.input_urls_text: str = ""
        self.processing_errors: List[ProcessingError] = []
        self.completed_videos: List[ProcessObject] = []
        
        # SECRETS BEIM START EINMALIG LADEN (nur für Upload + Processing)
        self.resolved_config_dict = self._resolve_secrets_to_dict()
        
        # Queues
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
        
        # Status Update Timer - 2 seconds
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.emit_status_update)
        self.status_timer.start(2000)
    
    def _resolve_secrets_to_dict(self) -> dict:
        """Lädt alle Secrets und erstellt vollständige Config-Dict"""
        
        # Config zu Dict konvertieren
        config_dict = self.config.dict()
        
        # Secret Manager für einmalige Secret-Resolution
        secret_manager = SecureConfigManager()
        secret_manager.load_config()
        
        try:
            # Initialize resolved_secrets dict
            config_dict['resolved_secrets'] = {}
            
            # Trilium Secret laden (für Processing Worker)
            trilium_token_result = secret_manager.get_trilium_token()
            if isinstance(trilium_token_result, Ok):
                config_dict['resolved_secrets']['trilium_token'] = unwrap_ok(trilium_token_result)
                self.logger.info("✅ Trilium token resolved")
            else:
                error = unwrap_err(trilium_token_result)
                self.logger.warning(f"⚠️ Trilium token resolution failed: {error.message}")
                config_dict['resolved_secrets']['trilium_token'] = None
            
            # Nextcloud Secret laden (für Upload Worker)
            nextcloud_password_result = secret_manager.get_nextcloud_password()
            if isinstance(nextcloud_password_result, Ok):
                config_dict['resolved_secrets']['nextcloud_password'] = unwrap_ok(nextcloud_password_result)
                self.logger.info("✅ Nextcloud password resolved")
            else:
                error = unwrap_err(nextcloud_password_result)
                self.logger.warning(f"⚠️ Nextcloud password resolution failed: {error.message}")
                config_dict['resolved_secrets']['nextcloud_password'] = None
            
            self.logger.info(
                f"✅ Secret resolution completed",
                extra={
                    'trilium_resolved': config_dict['resolved_secrets']['trilium_token'] is not None,
                    'nextcloud_resolved': config_dict['resolved_secrets']['nextcloud_password'] is not None,
                    'total_secrets': len([s for s in config_dict['resolved_secrets'].values() if s is not None])
                }
            )
            
        except Exception as e:
            self.logger.error(f"Critical error during secret resolution: {e}")
            config_dict['resolved_secrets'] = {
                'trilium_token': None,
                'nextcloud_password': None
            }
        
        return config_dict
    
    def start_pipeline(self, urls_text: str) -> Result[None, CoreError]:
        """Startet Pipeline (Metadata synchron, dann Worker-Pipeline)"""
        if self.state != PipelineState.IDLE:
            return Err(CoreError("Pipeline already running"))
        
        # Validate resolved secrets (kritisch nur für Upload)
        resolved_secrets = self.resolved_config_dict.get('resolved_secrets', {})
        if not resolved_secrets.get('nextcloud_password'):
            return Err(CoreError("Nextcloud password not resolved - check keyring configuration"))
        
        self.input_urls_text = urls_text
        self.processing_errors.clear()
        self.completed_videos.clear()
        
        self.logger.info("Starting pipeline with resolved secrets")
        
        # SYNCHRONE Metadata-Extraktion
        objects_result = process_urls_to_objects(urls_text, self.config.processing.__dict__)
        
        if isinstance(objects_result, Err):
            errors = unwrap_err(objects_result)
            return Err(CoreError(f"URL processing failed: {len(errors)} errors"))
        
        process_objects = unwrap_ok(objects_result)
        if not process_objects:
            return Err(CoreError("No valid videos found"))
        
        # Queue ProcessObjects für Audio Download
        for process_obj in process_objects:
            self.queues["audio_download"].put(process_obj)
        
        # Setup und start workers
        setup_result = self.setup_workers()
        if isinstance(setup_result, Err):
            return setup_result
        
        self.state = PipelineState.RUNNING
        self.start()  # Start QThread
        
        self.logger.info(f"Pipeline started with {len(process_objects)} videos and resolved secrets")
        return Ok(None)
    
    def setup_workers(self) -> Result[None, CoreError]:
        """Worker Setup - NUR Upload + Processing bekommen Config-Dict"""
        try:
            self.cleanup_workers()
            
            # Worker Setup: AppConfig für Worker ohne Secrets, Dict für Worker mit Secrets
            self.workers = [
                AudioDownloadWorker(self.queues["audio_download"], self.queues["transcription"], self.config),       # ← AppConfig
                TranscriptionWorker(self.queues["transcription"], self.queues["analysis"], self.config),             # ← AppConfig
                AnalysisWorker(self.queues["analysis"], self.queues["video_download"], self.config),                 # ← AppConfig
                VideoDownloadWorker(self.queues["video_download"], self.queues["upload"], self.config),              # ← AppConfig
                UploadWorker(self.queues["upload"], self.queues["processing"], self.resolved_config_dict),           # ← Dict!
                ProcessingWorker(self.queues["processing"], None, self.resolved_config_dict)                         # ← Dict!
            ]
            
            # Connect signals and start
            for worker in self.workers:
                worker.object_processed.connect(self.on_object_processed)
                worker.processing_error.connect(self.on_processing_error)
                worker.stage_status_changed.connect(self.on_stage_status_changed)
                worker.start()
            
            self.logger.info(f"Started {len(self.workers)} workers (2 with resolved secrets)")
            return Ok(None)
            
        except Exception as e:
            self.logger.error(f"Worker setup failed: {e}")
            return Err(CoreError(f"Worker setup failed: {e}"))
    
    def on_object_processed(self, obj: ProcessObject, next_stage: str):
        """Handler für erfolgreich verarbeitete Objects"""
        if next_stage == "completed":
            self.completed_videos.append(obj)
            self.video_completed.emit(obj.titel, True)
        elif next_stage == "archive":
            obj.update_stage("analysis_failed")
            self.completed_videos.append(obj)
            self.video_completed.emit(obj.titel, False)
            self.logger.info(f"Video archived (failed analysis): {obj.titel}")
    
    def on_processing_error(self, obj: ProcessObject, error_message: str):
        """Handler für Processing-Fehler"""
        error = ProcessingError(
            video_title=obj.titel,
            video_url=obj.original_url or obj.titel,
            stage=obj.processing_stage,
            error_message=error_message,
            timestamp=datetime.now(),
            process_object=obj
        )
        
        self.processing_errors.append(error)
        self.video_completed.emit(obj.titel, False)
        self.logger.error(f"Processing error: {obj.titel}: {error_message} in Stage:{obj.processing_stage}")
    
    def on_stage_status_changed(self, stage_name: str, queue_size: int):
        """Handler für Stage-Status-Änderungen (handled by timer)"""
        pass
    
    def emit_status_update(self):
        """Enhanced Status-Update mit Worker-Activity + Health"""
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
        
        # Estimate completion time
        estimated_completion = None
        if active_workers_list:
            remaining_items = sum(queue.size() for queue in self.queues.values())
            if remaining_items > 0:
                avg_processing_time = 3.0  # Estimated seconds per item per stage
                eta_seconds = remaining_items * avg_processing_time
                estimated_completion = datetime.now() + timedelta(seconds=eta_seconds)
        
        # Status-Berechnung mit Worker-Activity
        status = PipelineStatus(
            # Queue size + active worker (0 oder 1)
            audio_download_queue=self.queues["audio_download"].size() + (1 if any(w.stage_name == "Audio Download" and w.is_processing for w in self.workers) else 0),
            transcription_queue=self.queues["transcription"].size() + (1 if any(w.stage_name == "Transcription" and w.is_processing for w in self.workers) else 0),
            analysis_queue=self.queues["analysis"].size() + (1 if any(w.stage_name == "Analysis" and w.is_processing for w in self.workers) else 0),
            video_download_queue=self.queues["video_download"].size() + (1 if any(w.stage_name == "Video Download" and w.is_processing for w in self.workers) else 0),
            upload_queue=self.queues["upload"].size() + (1 if any(w.stage_name == "Upload" and w.is_processing for w in self.workers) else 0),
            processing_queue=self.queues["processing"].size() + (1 if any(w.stage_name == "Processing" and w.is_processing for w in self.workers) else 0),
            
            total_queued=self.get_total_objects_count(),
            total_completed=len(self.completed_videos),
            total_failed=len(self.processing_errors),
            
            current_stage=self.get_current_stage(active_workers_list),
            current_video=current_video_title[:30] + "..." if current_video_title and len(current_video_title) > 30 else current_video_title,
            
            # Enhanced features
            active_workers=active_workers_list,
            pipeline_health=health,
            estimated_completion=estimated_completion
        )
        
        self.status_updated.emit(status)
        
        # Check if pipeline finished
        all_queues_empty = not status.is_active()
        no_workers_active = len(active_workers_list) == 0
        
        if all_queues_empty and no_workers_active and self.state == PipelineState.RUNNING:
            self.finish_pipeline()
    
    def get_total_objects_count(self) -> int:
        """Gesamtzahl Objects in Pipeline"""
        queue_count = sum(queue.size() for queue in self.queues.values())
        return queue_count + len(self.completed_videos) + len(self.processing_errors)
    
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
        
        self.logger.info(f"Pipeline finished: {len(self.completed_videos)} completed, {len(self.processing_errors)} failed")
        
        self.cleanup_workers()
        self.state = PipelineState.FINISHED
        
        # Error messages für GUI
        error_messages = [f"{err.stage}: {err.error_message}" for err in self.processing_errors]
        
        self.pipeline_finished.emit(
            total_processed,
            len(self.completed_videos),
            error_messages
        )
        
        self.state = PipelineState.IDLE
    
    def stop_pipeline(self):
        """Pipeline stoppen"""
        if self.state != PipelineState.RUNNING:
            return
        
        self.logger.info("Stopping pipeline...")
        self.state = PipelineState.STOPPING
        
        self.cleanup_workers()
        
        if self.isRunning():
            self.wait(5000)
        
        self.state = PipelineState.IDLE
        self.logger.info("Pipeline stopped")
    
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
# GUI INTEGRATION (unchanged)
# =============================================================================

def integrate_pipeline_with_gui(main_window, config: AppConfig):
    """Integriert Pipeline Manager in GUI"""
    
    # Create Pipeline Manager
    main_window.pipeline_manager = PipelineManager(config)
    
    # Connect signals
    main_window.pipeline_manager.status_updated.connect(
        main_window.status_widget.update_status
    )
    
    main_window.pipeline_manager.video_completed.connect(
        lambda title, success: main_window.logger.info(f"Video completed: {title} ({'Success' if success else 'Failed'})")
    )
    
    main_window.pipeline_manager.pipeline_finished.connect(
        lambda total, success, errors: show_pipeline_summary(main_window, total, success, errors)
    )
    
    # Enhanced start_analysis method
    def enhanced_start_analysis():
        urls_text = main_window.url_input.toPlainText().strip()
        
        if not urls_text:
            main_window.status_bar.showMessage("Please enter YouTube URLs")
            return
        
        # Start pipeline
        start_result = main_window.pipeline_manager.start_pipeline(urls_text)
        
        if isinstance(start_result, Ok):
            main_window.status_bar.showMessage("Pipeline started - Processing URLs...")
            main_window.url_input.clear()
        else:
            error = unwrap_err(start_result)
            main_window.status_bar.showMessage(f"Failed to start: {error.message}")
            main_window.logger.error(f"Pipeline start failed: {error.message}")
    
    main_window.start_analysis = enhanced_start_analysis

def show_pipeline_summary(main_window, total: int, success: int, errors: List[str]):
    """Pipeline Summary Dialog"""
    failed = total - success
    
    if failed == 0:
        msg = QMessageBox(main_window)
        msg.setWindowTitle("Pipeline Complete")
        msg.setText(f"✅ Successfully processed {success} of {total} videos!")
        msg.setIcon(QMessageBox.Information)
        msg.exec()
    else:
        error_text = "\n".join(errors[:10])
        if len(errors) > 10:
            error_text += f"\n... and {len(errors) - 10} more errors"
        
        msg = QMessageBox(main_window)
        msg.setWindowTitle("Pipeline Complete with Errors")
        msg.setText(f"Processed {total} videos:\n✅ {success} successful\n❌ {failed} failed")
        msg.setDetailedText(f"Error details:\n{error_text}")
        msg.setIcon(QMessageBox.Warning)
        msg.exec()
