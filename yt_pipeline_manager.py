"""
YouTube Analyzer - Enhanced Fork-Join Pipeline Manager
EXTENDED: Complete Trilium Integration mit TrilliumUploadWorker und trilium_note_id handling
"""

from __future__ import annotations
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import gc
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import QMessageBox
import queue
# Import our core libraries
from core_types import Result, Ok, Err, unwrap_ok, unwrap_err, CoreError, ErrorContext
from yt_analyzer_core import (
    ProcessObject,
    TranskriptObject,
    ArchivObject,
    ProcessingQueue,
    ArchiveDatabase,
)
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
from yt_trilium_uploader import (
    upload_to_trilium_dict,
)  # ✅ ACTIVATED: Real Trilium integration

# =============================================================================
# ENHANCED PIPELINE STATUS (Complete LLM + Trilium Metrics)
# =============================================================================


@dataclass
class PipelineStatus:
    """Enhanced Pipeline Status mit vollständigen LLM + Trilium-Metriken"""

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

    # LLM-Metriken (existing)
    total_llm_tokens: int = 0
    total_llm_cost: float = 0.0
    active_llm_provider: Optional[str] = None
    llm_videos_processed: int = 0
    average_llm_processing_time: float = 0.0

    # ✅ EXTENDED: Trilium-specific metrics
    trilium_notes_created: int = 0
    trilium_upload_success_rate: float = 0.0
    average_trilium_upload_time: float = 0.0
    trilium_server_status: str = "unknown"  # "connected", "disconnected", "error"

    # Fork-Join specific tracking
    pending_merges: int = 0
    video_stream_completed: int = 0
    transcript_stream_completed: int = 0
    final_archived: int = 0

    def is_active(self) -> bool:
        """Activity check including new fork-join queues"""
        return (
            self.audio_download_queue
            + self.transcription_queue
            + self.analysis_queue
            + self.video_download_queue
            + self.upload_queue
            + self.processing_queue
            + self.llm_processing_queue
            + self.trilium_upload_queue
        ) > 0


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
# ENHANCED LLM METRICS COLLECTOR (Extended for Trilium)
# =============================================================================


class LLMMetricsCollector:
    """Enhanced LLM + Trilium Metrics Collector"""

    def __init__(self):
        self.logger = get_logger("LLMMetricsCollector")
        self.reset_metrics()

    def reset_metrics(self):
        """Reset alle Metriken für neue Pipeline"""
        # LLM Metrics (existing)
        self.total_tokens = 0
        self.total_cost = 0.0
        self.videos_processed = 0
        self.processing_times = []
        self.current_provider = None
        self.cost_breakdown = {}  # {"openai": 0.15, "anthropic": 0.05}
        self.token_breakdown = {}  # {"openai": 1500, "anthropic": 800}

        # ✅ EXTENDED: Trilium Metrics
        self.trilium_notes_created = 0
        self.trilium_upload_times = []
        self.trilium_upload_successes = 0
        self.trilium_upload_failures = 0
        self.trilium_server_status = "unknown"
        self.trilium_note_ids_created = []  # Track note IDs for reference

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
                "tokens_added": transcript_obj.tokens,
                "cost_added": transcript_obj.cost,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "videos_processed": self.videos_processed,
            },
        )

    def add_trilium_metrics(
        self, transcript_obj: TranskriptObject, upload_time: float, success: bool
    ) -> None:
        """✅ NEW: Sammelt Trilium-Upload-Metriken"""
        self.trilium_upload_times.append(upload_time)

        if success:
            self.trilium_upload_successes += 1
            self.trilium_notes_created += 1

            # Track note ID if available
            if transcript_obj.trilium_note_id:
                self.trilium_note_ids_created.append(transcript_obj.trilium_note_id)

            self.trilium_server_status = "connected"
        else:
            self.trilium_upload_failures += 1
            if self.trilium_server_status != "connected":
                self.trilium_server_status = "error"

        self.logger.debug(
            f"Added Trilium metrics from {transcript_obj.titel}",
            extra={
                "upload_time": upload_time,
                "success": success,
                "note_id": transcript_obj.trilium_note_id,
                "total_notes_created": self.trilium_notes_created,
                "success_rate": self.get_trilium_success_rate(),
            },
        )

    def get_trilium_success_rate(self) -> float:
        """✅ NEW: Berechnet Trilium-Upload-Erfolgsrate"""
        total_attempts = self.trilium_upload_successes + self.trilium_upload_failures
        if total_attempts == 0:
            return 0.0
        return (self.trilium_upload_successes / total_attempts) * 100.0

    def get_average_trilium_upload_time(self) -> float:
        """✅ NEW: Berechnet durchschnittliche Trilium-Upload-Zeit"""
        if not self.trilium_upload_times:
            return 0.0
        return sum(self.trilium_upload_times) / len(self.trilium_upload_times)

    def _extract_provider_from_model(self, model: str) -> str:
        """Extrahiert Provider-Namen aus Model-String"""
        model_lower = model.lower()
        if "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        elif "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "google" in model_lower:
            return "google"
        else:
            return "unknown"

    def get_average_processing_time(self) -> float:
        """Berechnet durchschnittliche LLM-Verarbeitungszeit"""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Gibt vollständige Metriken-Zusammenfassung zurück (extended)"""
        return {
            # LLM Metrics (existing)
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "videos_processed": self.videos_processed,
            "average_processing_time": self.get_average_processing_time(),
            "current_provider": self.current_provider,
            "cost_breakdown": self.cost_breakdown.copy(),
            "token_breakdown": self.token_breakdown.copy(),
            "cost_per_video": self.total_cost / self.videos_processed
            if self.videos_processed > 0
            else 0.0,
            "tokens_per_video": self.total_tokens / self.videos_processed
            if self.videos_processed > 0
            else 0.0,
            # ✅ EXTENDED: Trilium Metrics
            "trilium_notes_created": self.trilium_notes_created,
            "trilium_success_rate": self.get_trilium_success_rate(),
            "average_trilium_upload_time": self.get_average_trilium_upload_time(),
            "trilium_server_status": self.trilium_server_status,
            "trilium_upload_attempts": self.trilium_upload_successes
            + self.trilium_upload_failures,
            "trilium_note_ids": self.trilium_note_ids_created.copy(),
        }


# =============================================================================
# ENHANCED BASE WORKER CLASS für Fork-Join
# =============================================================================


class BaseWorker(QThread):
    """Enhanced Base Worker mit Fork-Join-Support"""

    # Signals für Pipeline Manager
    object_processed = Signal(
        object, str
    )  # Union[ProcessObject, TranskriptObject], next_stage
    processing_error = Signal(
        object, str
    )  # Union[ProcessObject, TranskriptObject], error_message
    stage_status_changed = Signal(str, int)  # stage_name, queue_size

    def __init__(
        self,
        stage_name: str,
        input_queue: ProcessingQueue,
        output_queue: Optional[ProcessingQueue] = None,
    ):
        super().__init__()
        self.stage_name = stage_name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.logger = get_logger(f"Worker-{stage_name}")
        self.should_stop = threading.Event()
        self.is_processing = False
        self.current_object: Optional[Union[ProcessObject, TranskriptObject]] = None

    def run(self):
        """
        Enhanced Worker-Loop mit proper should_stop-Checks
        LÖST: Blocking Queue.get() verhindert graceful shutdown
        """
        self.logger.info(f"🏃 {self.stage_name} worker started")

        while not self.should_stop.is_set():
            try:
                # CRITICAL: Queue.get() mit Timeout, damit should_stop gecheckt werden kann
                obj_result = self.input_queue.get(timeout=1.0)  # 1 Sekunde Timeout

                if isinstance(obj_result, Err):
                    # Queue empty nach Timeout - das ist normal
                    continue

                obj = unwrap_ok(obj_result)
                self.current_object = obj
                self.is_processing = True

                # Double-check should_stop vor Processing (Race-Condition-Schutz)
                if self.should_stop.is_set():
                    self.logger.debug(f"{self.stage_name} stopping before processing {getattr(obj, 'titel', 'unknown')}")
                    break

                # Update status
                self.stage_status_changed.emit(self.stage_name, self.input_queue.size())

                # Process object
                process_result = self.process_object(obj)

                # Check should_stop nach Processing
                if self.should_stop.is_set():
                    self.logger.debug(f"{self.stage_name} stopping after processing")
                    break

                if isinstance(process_result, Ok):
                    processed_obj = unwrap_ok(process_result)

                    # Handle routing decision
                    routing_info = self.get_routing_decision(processed_obj)

                    if routing_info["route_to_output"]:
                        if self.output_queue:
                            put_result = self.output_queue.put(processed_obj)
                            if isinstance(put_result, Ok):
                                self.object_processed.emit(
                                    processed_obj, routing_info["next_stage"]
                                )
                            else:
                                self.processing_error.emit(
                                    processed_obj,
                                    f"Queue overflow: {unwrap_err(put_result).message}",
                                )
                        else:
                            # End of stream - signal completion
                            self.object_processed.emit(
                                processed_obj, "stream_completed"
                            )
                    else:
                        # Object rejected (e.g., analysis failed)
                        self.object_processed.emit(
                            processed_obj, routing_info["next_stage"]
                        )
                else:
                    error = unwrap_err(process_result)
                    if hasattr(obj, "add_error"):  # ProcessObject
                        obj.add_error(f"{self.stage_name}: {error.message}")
                    else:  # TranskriptObject
                        obj.error_message = f"{self.stage_name}: {error.message}"
                    self.processing_error.emit(obj, error.message)

                self.current_object = None
                self.is_processing = False

            except queue.Empty:
                # Timeout bei Queue.get() - das ist normal, weiter mit should_stop-Check
                continue
        
            except Exception as e:
                if not self.should_stop.is_set():  # Nur loggen wenn nicht intentional gestoppt
                    self.logger.error(
                        f"Unexpected error in {self.stage_name} worker: {e}",
                        extra={
                            'stage_name': self.stage_name,
                            'error_type': type(e).__name__,
                            'current_object': getattr(self.current_object, 'titel', None) if self.current_object else None
                        }
                    )
                self.current_object = None
                self.is_processing = False
                time.sleep(1)  # Kurze Pause bei Fehlern

        # Clean shutdown logging
        self.logger.info(
            f"✅ {self.stage_name} worker stopped cleanly",
            extra={
                'stage_name': self.stage_name,
                'stop_reason': 'should_stop_set' if self.should_stop.is_set() else 'loop_exit',
                'was_processing': self.is_processing
            }
        )


    def process_object(
        self, obj: Union[ProcessObject, TranskriptObject]
    ) -> Result[Union[ProcessObject, TranskriptObject], CoreError]:
        """Override in subclasses"""
        raise NotImplementedError("Subclasses must implement process_object")

    def get_routing_decision(
        self, obj: Union[ProcessObject, TranskriptObject]
    ) -> Dict[str, Any]:
        """Override für conditional routing"""
        return {"route_to_output": True, "next_stage": self.get_next_stage_name()}

    def get_next_stage_name(self) -> str:
        """Override in subclasses"""
        return "unknown"

    def stop_worker(self):
        """
        Graceful worker shutdown mit enhanced debugging
        VERBESSERT: Besseres Logging für QThread-Lifecycle-Debugging
        """
        self.logger.debug(f"🛑 Stopping {self.stage_name} worker...")
    
        # Set stop signal
        self.should_stop.set()
    
        # Enhanced logging für Debugging
        self.logger.debug(
            f"Stop signal set for {self.stage_name}",
            extra={
                'stage_name': self.stage_name,
                'thread_running': self.isRunning(),
                'thread_finished': self.isFinished(),
                'current_object': getattr(self.current_object, 'titel', None) if self.current_object else None,
                'is_processing': self.is_processing
            }
        )


# =============================================================================
# EXISTING WORKER IMPLEMENTATIONS (AppConfig-basiert - UNCHANGED)
# =============================================================================


class AudioDownloadWorker(BaseWorker):
    """Audio-Download Worker (unchanged - kein Secret-Dependency)"""

    def __init__(
        self,
        input_queue: ProcessingQueue,
        output_queue: ProcessingQueue,
        config: AppConfig,
    ):
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

    def __init__(
        self,
        input_queue: ProcessingQueue,
        output_queue: ProcessingQueue,
        config: AppConfig,
    ):
        super().__init__("Transcription", input_queue, output_queue)
        self.config = config

    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        self.logger.info(f"Starting transcription: {obj.titel}")
        return transcribe_process_object(obj, self.config)

    def get_next_stage_name(self) -> str:
        return "Analysis"

class AnalysisWorker(BaseWorker):
    """Analysis Worker - FORK POINT Implementation (unchanged - kein Secret-Dependency)"""

    def __init__(
        self,
        input_queue: ProcessingQueue,
        config: AppConfig,
        pipeline_manager: "PipelineManager",
    ):
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
                self.logger.error(
                    f"Failed to queue for video download: {unwrap_err(video_put_result).message}"
                )

            # Stream B: LLM Processing Queue (clone to TranskriptObject)
            transcript_obj = TranskriptObject.from_process_object(obj)
            llm_put_result = self.pipeline_manager.queues["llm_processing"].put(
                transcript_obj
            )
            if isinstance(llm_put_result, Err):
                self.logger.error(
                    f"Failed to queue for LLM processing: {unwrap_err(llm_put_result).message}"
                )

            return {"route_to_output": False, "next_stage": "forked"}
        else:
            # FAILED ANALYSIS -> Direct to archive
            self.logger.info(f"Analysis failed - archiving directly: {obj.titel}")
            obj.video_stream_success = False

            # Create failed ArchivObject with no transcript processing
            failed_transcript = TranskriptObject.from_process_object(obj)
            failed_transcript.success = False
            failed_transcript.error_message = "Analysis failed - no LLM processing"

            archive_obj = ArchivObject.from_process_and_transcript(
                obj, failed_transcript
            )
            self.pipeline_manager.archive_final_object(archive_obj)

            return {"route_to_output": False, "next_stage": "archived_failed"}

    def get_next_stage_name(self) -> str:
        return "Fork"


class VideoDownloadWorker(BaseWorker):
    """Video Download Worker - UNVERÄNDERT (keine Secrets)"""

    def __init__(
        self,
        input_queue: ProcessingQueue,
        output_queue: ProcessingQueue,
        config: AppConfig,
    ):
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
        super().__init__(
            "Upload", input_queue, None
        )  # No output queue - end of Stream A
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

    def __init__(
        self,
        input_queue: ProcessingQueue,
        output_queue: ProcessingQueue,
        config_dict: dict,
    ):
        super().__init__("LLM Processing", input_queue, output_queue)
        self.config_dict = config_dict

    def process_object(
        self, obj: TranskriptObject
    ) -> Result[TranskriptObject, CoreError]:
        self.logger.info(f"Starting LLM processing: {obj.titel}")

        result = process_transcript_with_llm_dict(obj, self.config_dict)

        if isinstance(result, Ok):
            processed_obj = unwrap_ok(result)
            self.logger.info(
                "LLM processing completed successfully",
                extra={
                    "video_title": processed_obj.titel,
                    "provider": self.config_dict["llm_processing"]["provider"],
                    "model": processed_obj.model,
                    "tokens": processed_obj.tokens,
                    "cost": processed_obj.cost,
                    "processing_time": processed_obj.processing_time,
                },
            )
            return Ok(processed_obj)
        else:
            error = unwrap_err(result)
            obj.success = False
            obj.error_message = f"LLM processing failed: {error.message}"
            self.logger.error(
                "LLM processing failed",
                extra={
                    "video_title": obj.titel,
                    "error": error.message,
                    "provider": self.config_dict["llm_processing"]["provider"],
                },
            )
            return result

    def get_next_stage_name(self) -> str:
        return "Trilium Upload"


class TrilliumUploadWorker(BaseWorker):
    """✅ EXTENDED: Trilium Upload Worker - Stream B Final (Real Implementation)"""

    def __init__(
        self,
        input_queue: ProcessingQueue,
        config_dict: dict,
        metrics_collector: LLMMetricsCollector,
    ):
        super().__init__(
            "Trilium Upload", input_queue, None
        )  # No output queue - end of Stream B
        self.config_dict = config_dict
        self.metrics_collector = metrics_collector

    def process_object(
        self, obj: TranskriptObject
    ) -> Result[TranskriptObject, CoreError]:
        self.logger.info(f"Starting Trilium upload: {obj.titel}")

        upload_start_time = time.time()

        # ✅ EXTENDED: Real Trilium upload with metrics collection
        result = upload_to_trilium_dict(obj, self.config_dict)

        upload_time = time.time() - upload_start_time

        if isinstance(result, Ok):
            processed_obj = unwrap_ok(result)

            # ✅ EXTENDED: Collect Trilium metrics
            self.metrics_collector.add_trilium_metrics(processed_obj, upload_time, True)

            self.logger.info(
                "Trilium upload completed successfully",
                extra={
                    "video_title": processed_obj.titel,
                    "trilium_note_id": processed_obj.trilium_note_id,
                    "trilium_link": processed_obj.trilium_link,
                    "upload_time": upload_time,
                },
            )

            return Ok(processed_obj)
        else:
            error = unwrap_err(result)

            # ✅ EXTENDED: Collect failure metrics
            self.metrics_collector.add_trilium_metrics(obj, upload_time, False)

            obj.success = False
            obj.error_message = f"Trilium upload failed: {error.message}"

            self.logger.error(
                "Trilium upload failed",
                extra={
                    "video_title": obj.titel,
                    "error": error.message,
                    "upload_time": upload_time,
                },
            )

            return result

    def get_next_stage_name(self) -> str:
        return "Stream B Completed"


# =============================================================================
# ENHANCED FORK-JOIN PIPELINE MANAGER (Zentrale Secret-Resolution)
# =============================================================================


class PipelineManager(QThread):
    """Enhanced Pipeline Manager with complete Trilium integration"""

    # Signals für GUI
    status_updated = Signal(PipelineStatus)
    video_completed = Signal(str, bool)  # title, success
    transcript_completed = Signal(str, bool)  # ✅ NEW: Trilium completion signal
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
            "video_completed": 0,
            "transcript_completed": 0,
            "trilium_uploads": 0,  # ✅ NEW
            "final_archived": 0,
            "video_failed": 0,
            "transcript_failed": 0,
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
            "trilium_upload": ProcessingQueue("trilium_upload"),
        }

        # NEW: Fork-Join State Collections (as specified in architecture)
        self.video_success_list: List[ProcessObject] = []
        self.video_failure_list: List[ProcessObject] = []
        self.transcript_success_list: List[TranskriptObject] = []
        self.transcript_failure_list: List[TranskriptObject] = []

        # Archive management
        self.archive_database = ArchiveDatabase(Path(self.config.storage.sqlite_path))

        # Thread safety
        self.state_lock = threading.Lock()

        # Workers
        self.workers: List[BaseWorker] = []

        # Status Update Timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.emit_status_update)
        self.status_timer.start(2000)  # 2 seconds

    def _resolve_config_dict(self) -> dict:
        """✅ EXTENDED: Zentrale Secret-Resolution für alle Worker inkl. Trilium"""
        config_dict = self.config.dict()
        resolved_secrets = {}

        self.logger.info("🔐 Starting zentrale secret resolution...")

        # Nextcloud Secret Resolution
        nextcloud_result = self.config_manager.get_nextcloud_password()
        if isinstance(nextcloud_result, Ok):
            resolved_secrets["nextcloud_password"] = unwrap_ok(nextcloud_result)
            self.logger.info("✅ Nextcloud password resolved")
        else:
            self.logger.warning(
                f"⚠️ Nextcloud password resolution failed: {unwrap_err(nextcloud_result).message}"
            )

        # ✅ EXTENDED: Trilium Secret Resolution
        trilium_result = self.config_manager.get_trilium_token()
        if isinstance(trilium_result, Ok):
            resolved_secrets["trilium_token"] = unwrap_ok(trilium_result)
            self.logger.info("✅ Trilium token resolved")
        else:
            self.logger.warning(
                f"⚠️ Trilium token resolution failed: {unwrap_err(trilium_result).message}"
            )

        # LLM API Key Resolution
        llm_provider = self.config.llm_processing.provider
        llm_key_result = self.config_manager.get_llm_api_key(llm_provider)
        if isinstance(llm_key_result, Ok):
            resolved_secrets["llm_api_key"] = unwrap_ok(llm_key_result)
            self.logger.info(f"✅ {llm_provider.title()} API key resolved")
        else:
            self.logger.warning(
                f"⚠️ {llm_provider.title()} API key resolution failed: {unwrap_err(llm_key_result).message}"
            )

        # Add resolved secrets to config_dict
        config_dict["resolved_secrets"] = resolved_secrets

        self.logger.info(
            "🔐 Secret resolution completed",
            extra={
                "resolved_secrets_count": len(resolved_secrets),
                "resolved_secrets_keys": list(resolved_secrets.keys()),
                "nextcloud_ready": "nextcloud_password" in resolved_secrets,
                "trilium_ready": "trilium_token" in resolved_secrets,  # ✅ EXTENDED
                "llm_ready": "llm_api_key" in resolved_secrets,
            },
        )

        return config_dict

    def start_pipeline(self, urls_text: str) -> Result[None, CoreError]:
        """Startet Pipeline mit explizitem Worker-Cleanup-Check"""
        if self.state != PipelineState.IDLE:
            return Err(CoreError("Pipeline already running"))

        self.input_urls_text = urls_text
        self.processing_errors.clear()
        self.completed_videos.clear()

        # CRITICAL: Expliziter Worker-Cleanup vor Start
        if self.workers:
            self.logger.info("🔄 Cleaning up previous workers before starting new pipeline...")
            self.cleanup_workers()
        
            # Sicherheitscheck: Sind wirklich alle Worker weg?
            running_workers = [w.stage_name for w in self.workers if w.isRunning()]
            if running_workers:
                context = ErrorContext.create(
                    "start_pipeline_worker_check",
                    input_data={"running_workers": running_workers},
                    suggestions=[
                        "Wait longer for workers to stop",
                        "Check for deadlocked threads",
                        "Restart application if problem persists"
                    ]
                )
                return Err(CoreError(f"Previous workers still running: {running_workers}", context))

        self.llm_metrics.reset_metrics()
        self.stream_completion_stats = {
            "video_completed": 0,
            "transcript_completed": 0,
            "trilium_uploads": 0,
            "final_archived": 0,
            "video_failed": 0,
            "transcript_failed": 0,
        }

        # Clear state collections
        with self.state_lock:
            self.video_success_list.clear()
            self.video_failure_list.clear()
            self.transcript_success_list.clear()
            self.transcript_failure_list.clear()

        self.logger.info("🚀 Starting clean Fork-Join pipeline")

        # SYNCHRONE Metadata-Extraktion (rest bleibt unverändert)
        objects_result = process_urls_to_objects(
            urls_text, self.config.processing.__dict__
        )

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
                self.logger.error(
                    f"Failed to queue audio download: {unwrap_err(put_result).message}"
                )

        # Setup und start workers
        setup_result = self.setup_workers()
        if isinstance(setup_result, Err):
            return setup_result

        self.state = PipelineState.RUNNING
        self.start()  # Start QThread

        self.logger.info(
            f"✅ Clean Fork-Join pipeline started with {len(process_objects)} videos"
        )
        return Ok(None)

    def setup_workers(self) -> Result[None, CoreError]:
        """✅ EXTENDED: Enhanced worker setup mit Trilium integration"""
        try:
            self.cleanup_workers()

            # Create all workers mit korrekter config-Distribution
            self.workers = [
                # Sequential workers (AppConfig - kein Secret-Dependency)
                AudioDownloadWorker(
                    self.queues["audio_download"],
                    self.queues["transcription"],
                    self.config,
                ),
                TranscriptionWorker(
                    self.queues["transcription"], self.queues["analysis"], self.config
                ),
                AnalysisWorker(
                    self.queues["analysis"], self.config, self
                ),  # Fork point
                # Stream A: Video Processing (gemischt)
                VideoDownloadWorker(
                    self.queues["video_download"], self.queues["upload"], self.config
                ),  # kein Secret
                UploadWorker(
                    self.queues["upload"], self.config_dict
                ),  # NEW: config_dict mit Secret
                # ✅ EXTENDED: Stream B: Transcript Processing (config_dict mit Trilium)
                LLMProcessingWorker(
                    self.queues["llm_processing"],
                    self.queues["trilium_upload"],
                    self.config_dict,
                ),
                TrilliumUploadWorker(
                    self.queues["trilium_upload"], self.config_dict, self.llm_metrics
                ),  # ✅ NEW
            ]

            # Connect signals für Fork-Join-Orchestration
            for worker in self.workers:
                worker.object_processed.connect(self.handle_worker_completion)
                worker.processing_error.connect(self.handle_worker_error)

            # Start all workers
            for worker in self.workers:
                worker.start()

            self.logger.info(
                "Enhanced workers started successfully",
                extra={
                    "total_workers": len(self.workers),
                    "config_workers": 4,  # Audio, Transcription, Analysis, VideoDownload
                    "config_dict_workers": 3,  # Upload, LLM, Trilium
                    "trilium_integration": True,  # ✅ EXTENDED
                    "resolved_secrets": list(
                        self.config_dict.get("resolved_secrets", {}).keys()
                    ),
                },
            )

            return Ok(None)

        except Exception as e:
            return Err(CoreError(f"Enhanced worker setup failed: {e}"))

    def handle_worker_completion(
        self, obj: Union[ProcessObject, TranskriptObject], next_stage: str
    ):
        """✅ EXTENDED: Enhanced completion handler mit Trilium support"""
        with self.state_lock:
            if isinstance(obj, ProcessObject):
                # Stream A completion
                if (
                    next_stage == "stream_completed"
                    or next_stage == "Stream A Completed"
                ):
                    obj.video_stream_success = True
                    self.stream_completion_stats["video_completed"] += 1
                    self.process_video_completion(obj)

            elif isinstance(obj, TranskriptObject):
                # ✅ EXTENDED: Stream B completion with Trilium handling
                if (
                    next_stage == "stream_completed"
                    or next_stage == "Stream B Completed"
                ):
                    obj.success = True
                    self.stream_completion_stats["transcript_completed"] += 1

                    # ✅ EXTENDED: LLM-Metriken sammeln
                    self.llm_metrics.add_transcript_metrics(obj)

                    # ✅ EXTENDED: Track Trilium uploads
                    if obj.trilium_note_id:
                        self.stream_completion_stats["trilium_uploads"] += 1

                    self.process_transcript_completion(obj)

                    # ✅ NEW: Emit Trilium completion signal
                    self.transcript_completed.emit(obj.titel, obj.success)

    def handle_worker_error(
        self, obj: Union[ProcessObject, TranskriptObject], error_message: str
    ):
        """✅ EXTENDED: Enhanced error handler mit Trilium error tracking"""
        with self.state_lock:
            if isinstance(obj, ProcessObject):
                # Stream A failure
                obj.video_stream_success = False
                self.stream_completion_stats["video_failed"] += 1
                if hasattr(obj, "add_error"):
                    obj.add_error(f"Video stream failed: {error_message}")
                self.process_video_completion(obj)

            elif isinstance(obj, TranskriptObject):
                # Stream B failure
                obj.success = False
                obj.error_message = error_message
                self.stream_completion_stats["transcript_failed"] += 1

                # ✅ EXTENDED: Track Trilium upload failures
                if "trilium" in error_message.lower():
                    self.logger.error(
                        "Trilium upload error detected",
                        extra={
                            "video_title": obj.titel,
                            "error_message": error_message,
                            "trilium_note_id": getattr(obj, "trilium_note_id", None),
                        },
                    )

                self.process_transcript_completion(obj)

            # Add to processing errors for summary
            error_obj = ProcessingError(
                video_title=getattr(obj, "titel", "unknown"),
                video_url=getattr(obj, "original_url", "unknown"),
                stage=error_message.split(":")[0]
                if ":" in error_message
                else "unknown",
                error_message=error_message,
                timestamp=datetime.now(),
                object_type="ProcessObject"
                if isinstance(obj, ProcessObject)
                else "TranskriptObject",
            )
            self.processing_errors.append(error_obj)

    def process_video_completion(self, video_obj: ProcessObject):
        """Processes Stream A completion and attempts merging"""
        titel = video_obj.titel

        # Check if corresponding transcript is already completed
        transcript_match = self.find_and_remove_transcript(titel)

        if transcript_match:
            # Both streams completed - merge and archive
            archive_obj = self.merge_objects(video_obj, transcript_match)
            self.archive_final_object(archive_obj)
        else:
            # Store in appropriate list for later merging
            if video_obj.video_stream_success:
                self.video_success_list.append(video_obj)
            else:
                self.video_failure_list.append(video_obj)

    def process_transcript_completion(self, transcript_obj: TranskriptObject):
        """✅ EXTENDED: Process Stream B completion with Trilium note tracking"""
        titel = transcript_obj.titel

        # Check if corresponding video is already completed
        video_match = self.find_and_remove_video(titel)

        if video_match:
            # Both streams completed - merge and archive
            merged_obj = self.merge_objects(video_match, transcript_obj)
            self.archive_final_object(merged_obj)
        else:
            # Store in appropriate list for later merging
            if transcript_obj.success:
                self.transcript_success_list.append(transcript_obj)
                self.logger.debug(
                    f"Transcript success stored for merging: {transcript_obj.titel}",
                    extra={
                        "trilium_note_id": transcript_obj.trilium_note_id,  # ✅ EXTENDED
                        "trilium_link": bool(transcript_obj.trilium_link),
                    },
                )
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

    def merge_objects(
        self, video_obj: ProcessObject, transcript_obj: TranskriptObject
    ) -> ArchivObject:
        """✅ EXTENDED: Merge objects with Trilium note ID preservation"""
        archive_obj = ArchivObject.from_process_and_transcript(
            video_obj, transcript_obj
        )

        self.logger.info(
            f"Merged fork-join objects: {archive_obj.titel}",
            extra={
                "final_success": archive_obj.final_success,
                "video_stream_success": archive_obj.video_stream_success,
                "transcript_stream_success": archive_obj.transcript_stream_success,
                "trilium_note_id": archive_obj.trilium_note_id,  # ✅ EXTENDED
                "nextcloud_link": bool(archive_obj.nextcloud_link),
                "trilium_link": bool(archive_obj.trilium_link),
            },
        )

        return archive_obj

    def archive_final_object(self, archive_obj: ArchivObject):
        """✅ EXTENDED: Archive with Trilium note ID persistence"""
        archive_result = self.archive_database.save_processed_video(archive_obj)

        from yt_trilium_uploader import TriliumUploader

        trilium = TriliumUploader(self.config_dict)

        etapi = trilium._init_trilium_client().value
        try:
            result = etapi.create_attribute(
                noteId=archive_obj.trilium_note_id,
                type="label",
                name="nextcloud_link",
                value=str(archive_obj.nextcloud_link),
                isInheritable=False,
            )
            if result:
                self.logger.debug(f"Created nextcloud link for: {archive_obj.titel[-10]}")
        except Exception as e:
            self.logger.warning(f"Failed to add nextcloud link: {e}")

        
        if isinstance(archive_result, Ok):
            self.stream_completion_stats["final_archived"] += 1
            self.completed_videos.append(archive_obj)

            self.logger.info(
                f"Final object archived successfully: {archive_obj.titel}",
                extra={
                    "final_success": archive_obj.final_success,
                    "trilium_note_id": archive_obj.trilium_note_id,  # ✅ EXTENDED
                    "archive_id": len(self.completed_videos),
                },
            )

            self.video_completed.emit(archive_obj.titel, archive_obj.final_success)

        else:
            error = unwrap_err(archive_result)
            self.logger.error(f"Archive failed: {error.message}")
            self.processing_errors.append(
                f"Archive failed for {archive_obj.titel}: {error.message}"
            )

    def emit_status_update(self):
        """✅ EXTENDED: Enhanced status update with complete Trilium metrics"""
        if self.state == PipelineState.IDLE:
            return

        # Calculate active workers and current video
        active_workers_list = []
        current_video_title = None

        for worker in self.workers:
            if worker.is_processing:
                active_workers_list.append(worker.stage_name)
                if worker.current_object and not current_video_title:
                    current_video_title = getattr(worker.current_object, "titel", None)

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

        # ✅ EXTENDED: Get enhanced metrics including Trilium
        llm_summary = self.llm_metrics.get_metrics_summary()

        # ✅ EXTENDED: Status-Berechnung mit allen LLM + Trilium-Metriken
        status = PipelineStatus(
            # Queue sizes + active worker tracking
            audio_download_queue=self.queues["audio_download"].size()
            + (
                1
                if any(
                    w.stage_name == "Audio Download" and w.is_processing
                    for w in self.workers
                )
                else 0
            ),
            transcription_queue=self.queues["transcription"].size()
            + (
                1
                if any(
                    w.stage_name == "Transcription" and w.is_processing
                    for w in self.workers
                )
                else 0
            ),
            analysis_queue=self.queues["analysis"].size()
            + (
                1
                if any(
                    w.stage_name == "Analysis" and w.is_processing for w in self.workers
                )
                else 0
            ),
            video_download_queue=self.queues["video_download"].size()
            + (
                1
                if any(
                    w.stage_name == "Video Download" and w.is_processing
                    for w in self.workers
                )
                else 0
            ),
            upload_queue=self.queues["upload"].size()
            + (
                1
                if any(
                    w.stage_name == "Upload" and w.is_processing for w in self.workers
                )
                else 0
            ),
            processing_queue=0,  # Not used in Fork-Join
            # ✅ EXTENDED: Fork-Join queues including Trilium
            llm_processing_queue=self.queues["llm_processing"].size()
            + (
                1
                if any(
                    w.stage_name == "LLM Processing" and w.is_processing
                    for w in self.workers
                )
                else 0
            ),
            trilium_upload_queue=self.queues["trilium_upload"].size()
            + (
                1
                if any(
                    w.stage_name == "Trilium Upload" and w.is_processing
                    for w in self.workers
                )
                else 0
            ),
            total_queued=self.get_total_objects_count(),
            total_completed=len(self.completed_videos),
            total_failed=len(self.processing_errors),
            current_stage=self.get_current_stage(active_workers_list),
            current_video=current_video_title[:30] + "..."
            if current_video_title and len(current_video_title) > 30
            else current_video_title,
            active_workers=active_workers_list,
            pipeline_health=health,
            # LLM metrics
            total_llm_tokens=llm_summary["total_tokens"],
            total_llm_cost=llm_summary["total_cost"],
            active_llm_provider=llm_summary["current_provider"],
            llm_videos_processed=llm_summary["videos_processed"],
            average_llm_processing_time=llm_summary["average_processing_time"],
            # ✅ EXTENDED: Trilium metrics
            trilium_notes_created=llm_summary["trilium_notes_created"],
            trilium_upload_success_rate=llm_summary["trilium_success_rate"],
            average_trilium_upload_time=llm_summary["average_trilium_upload_time"],
            trilium_server_status=llm_summary["trilium_server_status"],
            # Fork-Join metrics
            pending_merges=(
                len(self.video_success_list)
                + len(self.video_failure_list)
                + len(self.transcript_success_list)
                + len(self.transcript_failure_list)
            ),
            video_stream_completed=self.stream_completion_stats["video_completed"],
            transcript_stream_completed=self.stream_completion_stats[
                "transcript_completed"
            ],
            final_archived=self.stream_completion_stats["final_archived"],
        )

        self.status_updated.emit(status)

        # ✅ EXTENDED: Enhanced debug logging with Trilium metrics
        self.logger.debug(
            "Enhanced status update with Trilium metrics",
            extra={
                "total_queued": status.total_queued,
                "llm_tokens": status.total_llm_tokens,
                "llm_cost": status.total_llm_cost,
                "trilium_notes_created": status.trilium_notes_created,
                "trilium_success_rate": status.trilium_upload_success_rate,
                "trilium_server_status": status.trilium_server_status,
                "pending_merges": status.pending_merges,
                "final_archived": status.final_archived,
            },
        )

        # Check if pipeline finished (with pending merges check)
        all_queues_empty = not status.is_active()
        no_workers_active = len(active_workers_list) == 0
        no_pending_merges = status.pending_merges == 0

        if (
            all_queues_empty
            and no_workers_active
            and no_pending_merges
            and self.state == PipelineState.RUNNING
        ):
            self.finish_pipeline()

    def get_total_objects_count(self) -> int:
        """Gesamtzahl Objects in Pipeline"""
        queue_count = sum(queue.size() for queue in self.queues.values())
        state_count = (
            len(self.video_success_list)
            + len(self.video_failure_list)
            + len(self.transcript_success_list)
            + len(self.transcript_failure_list)
        )
        return (
            queue_count
            + state_count
            + len(self.completed_videos)
            + len(self.processing_errors)
        )

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
        successful_videos = sum(
            1 for video in self.completed_videos if video.final_success
        )

        self.logger.info(
            "Enhanced Fork-Join pipeline finished",
            extra={
                "total_videos": total_processed,
                "successful": successful_videos,
                "failed": total_processed - successful_videos,
                "processing_errors": len(self.processing_errors),
                "trilium_uploads": self.stream_completion_stats[
                    "trilium_uploads"
                ],  # ✅ EXTENDED
                "resolved_secrets_used": list(
                    self.config_dict.get("resolved_secrets", {}).keys()
                ),
            },
        )

        self.cleanup_workers()
        self.state = PipelineState.FINISHED

        # Error messages für GUI
        error_messages = [
            f"{err.stage}: {err.error_message}" for err in self.processing_errors
        ]

        self.pipeline_finished.emit(total_processed, successful_videos, error_messages)

        self.state = PipelineState.IDLE

    def stop_pipeline(self):
        """
        Pipeline stoppen mit verbesserter Worker-Shutdown-Reihenfolge
        VERBESSERT: Bessere State-Management und Worker-Status-Logging
        """
        if self.state not in [PipelineState.RUNNING, PipelineState.STOPPING]:
            self.logger.debug(f"Pipeline not running (state: {self.state.value}), nothing to stop")
            return

        self.logger.info("🛑 Stopping Enhanced Fork-Join pipeline...")
    
        # Phase 1: State auf STOPPING setzen
        previous_state = self.state
        self.state = PipelineState.STOPPING

        # Phase 2: Worker-Status vor Shutdown loggen (für Debugging)
        if self.workers:
            running_workers = []
            processing_workers = []
        
            for worker in self.workers:
                if worker.isRunning():
                    running_workers.append(worker.stage_name)
                    if getattr(worker, 'is_processing', False):
                        processing_workers.append(worker.stage_name)
        
            self.logger.info(
                f"Pipeline stop initiated: {len(running_workers)} running workers",
                extra={
                    'previous_state': previous_state.value,
                    'running_workers': running_workers,
                    'processing_workers': processing_workers,
                    'total_workers': len(self.workers)
                }
            )

        # Phase 3: Workers cleanup (enhanced version)
        cleanup_start_time = time.time()
        self.cleanup_workers()
        cleanup_duration = time.time() - cleanup_start_time

        # Phase 4: QThread selbst stoppen falls läuft
        if self.isRunning():
            self.logger.debug("Stopping pipeline manager QThread...")
            if not self.wait(5000):  # 5 Sekunden Timeout
                self.logger.warning("Pipeline manager QThread did not stop gracefully")

        # Phase 5: Final state
        self.state = PipelineState.IDLE
    
        self.logger.info(
            f"✅ Enhanced Fork-Join pipeline stopped",
            extra={
                'cleanup_duration_seconds': round(cleanup_duration, 2),
                'final_state': self.state.value,
                'successful_stop': True
            }
        )


    def cleanup_workers(self):
        """
        Alle Worker sauber beenden mit explizitem QThread.wait()
        LÖST: Race-Condition bei Pipeline-Restart
        """
        if not self.workers:
            self.logger.debug("No workers to cleanup")
            return

        total_workers = len(self.workers)
        self.logger.info(f"🛑 Stopping {total_workers} workers gracefully...")

        # Phase 1: Allen Workern Signal zum Stoppen geben
        for worker in self.workers:
            try:
                worker.stop_worker()  # Setzt should_stop Event
                self.logger.debug(f"Stop signal sent to {worker.stage_name}")
            except Exception as e:
                self.logger.warning(f"Failed to send stop signal to {worker.stage_name}: {e}")

        # Phase 2: Explizit auf Thread-Beendigung warten
        successful_stops = 0
        forced_terminations = 0
    
        for worker in self.workers:
            if not worker.isRunning():
                self.logger.debug(f"✅ {worker.stage_name} already stopped")
                successful_stops += 1
                continue

            self.logger.debug(f"⏳ Waiting for {worker.stage_name} to finish...")
        
            # CRITICAL: Explizites wait() mit Timeout
            if worker.wait(5000):  # 5 Sekunden Timeout
                self.logger.debug(f"✅ {worker.stage_name} stopped gracefully")
                successful_stops += 1
            else:
                # Graceful shutdown failed → Terminate
                self.logger.warning(f"⚠️ {worker.stage_name} did not stop gracefully, terminating...")
                try:
                    worker.terminate()
                    if worker.wait(2000):  # 2 Sekunden nach terminate
                        self.logger.warning(f"🔥 {worker.stage_name} terminated forcefully")
                        forced_terminations += 1
                    else:
                        self.logger.error(f"❌ {worker.stage_name} could not be terminated!")
                except Exception as e:
                    self.logger.error(f"❌ Failed to terminate {worker.stage_name}: {e}")

        # Phase 3: Worker-Liste leeren und Statistics
        self.workers.clear()

        self.logger.info(
            f"✅ Worker cleanup completed: {successful_stops} graceful, {forced_terminations} terminated",
            extra={
                'total_workers': total_workers,
                'successful_stops': successful_stops,
                'forced_terminations': forced_terminations,
                'cleanup_success': successful_stops + forced_terminations == total_workers
            }
        )

        # Kurze Pause für Qt-Cleanup
        time.sleep(0.1)
    
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

    # ✅ NEW: Trilium completion signal
    main_window.pipeline_manager.transcript_completed.connect(
        lambda title, success: main_window.logger.info(
            f"Trilium upload completed: {title} ({'Success' if success else 'Failed'})"
        )
    )

    main_window.pipeline_manager.pipeline_finished.connect(
        lambda total, success, errors: show_enhanced_pipeline_summary(
            main_window, total, success, errors
        )
    )

    # Enhanced start_analysis method
    def enhanced_start_analysis():
        urls_text = main_window.url_input.toPlainText().strip()

        if not urls_text:
            main_window.status_bar.showMessage("Please enter YouTube URLs")
            return

        # Start Enhanced Fork-Join pipeline
        start_result = main_window.pipeline_manager.start_pipeline(urls_text)

        if isinstance(start_result, Ok):
            main_window.status_bar.showMessage(
                "Enhanced Fork-Join Pipeline started - Processing URLs..."
            )
            main_window.url_input.clear()
        else:
            error = unwrap_err(start_result)
            main_window.status_bar.showMessage(f"Failed to start: {error.message}")
            main_window.logger.error(
                f"Enhanced Fork-Join Pipeline start failed: {error.message}"
            )

    main_window.start_analysis = enhanced_start_analysis


def show_enhanced_pipeline_summary(
    main_window, total: int, success: int, errors: List[str]
):
    """Enhanced Pipeline Summary Dialog für Fork-Join mit Trilium"""
    failed = total - success

    if failed == 0:
        msg = QMessageBox(main_window)
        msg.setWindowTitle("Enhanced Fork-Join Pipeline Complete")
        msg.setText(
            f"✅ Successfully processed {success} of {total} videos!\n\nEnhanced Fork-Join architecture with Trilium integration completed successfully."
        )
        msg.setIcon(QMessageBox.Information)
        msg.exec()
    else:
        error_text = "\n".join(errors[:10])
        if len(errors) > 10:
            error_text += f"\n... and {len(errors) - 10} more errors"

        msg = QMessageBox(main_window)
        msg.setWindowTitle("Enhanced Fork-Join Pipeline Complete with Errors")
        msg.setText(
            f"Enhanced Fork-Join Pipeline processed {total} videos:\n✅ {success} successful\n❌ {failed} failed"
        )
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
    setup_logging("enhanced_fork_join_pipeline_test", "DEBUG")

    # Test configuration
    config_manager = SecureConfigManager()
    config_result = config_manager.load_config()

    if isinstance(config_result, Ok):
        config = unwrap_ok(config_result)

        # Create enhanced pipeline manager
        pipeline_manager = PipelineManager(config)

        print("✅ Enhanced Fork-Join Pipeline Manager created successfully")
        print(f"   Queues: {list(pipeline_manager.queues.keys())}")
        print("   Trilium Integration: Active")  # ✅ EXTENDED
        print(f"   Archive Database: {pipeline_manager.archive_database}")
        print(
            f"   Resolved Secrets: {list(pipeline_manager.config_dict.get('resolved_secrets', {}).keys())}"
        )

        # Test enhanced queue operations
        test_urls = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        print("\n🧪 Testing Enhanced Fork-Join Pipeline startup...")
        start_result = pipeline_manager.start_pipeline(test_urls)

        if isinstance(start_result, Ok):
            print("✅ Enhanced pipeline started successfully")

            # Let it run for a few seconds
            time.sleep(5)

            # Stop pipeline
            pipeline_manager.stop_pipeline()
            print("✅ Enhanced pipeline stopped successfully")
        else:
            error = unwrap_err(start_result)
            print(f"❌ Enhanced pipeline start failed: {error.message}")

    else:
        error = unwrap_err(config_result)
        print(f"❌ Config loading failed: {error.message}")

    print(
        "\n🚀 Enhanced Fork-Join Pipeline Manager with complete Trilium Integration Complete!"
    )
