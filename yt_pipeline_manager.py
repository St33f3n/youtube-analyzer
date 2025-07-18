"""
YouTube Analyzer - Pipeline Manager with QThreads
Orchestriert Video-Processing-Pipeline mit parallelen Worker-Threads
"""

from __future__ import annotations
import time
import re
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import threading

from PySide6.QtCore import QThread, Signal, QObject, QTimer
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
# PIPELINE STATUS & ERROR TRACKING
# =============================================================================

@dataclass
class PipelineStatus:
    """Vollständiger Pipeline-Status für GUI-Updates"""
    audio_download_queue: int = 0  # Renamed from download_queue for clarity
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
    
    def is_active(self) -> bool:
        """Prüft ob Pipeline aktiv ist"""
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
# BASE WORKER CLASS
# =============================================================================

class BaseWorker(QThread):
    """Basis-Klasse für alle Pipeline-Worker"""
    
    # Signals
    object_processed = Signal(ProcessObject, str)  # ProcessObject, next_stage
    processing_error = Signal(ProcessObject, str)  # ProcessObject, error_message
    stage_status_changed = Signal(str, int)  # stage_name, queue_size
    
    def __init__(self, stage_name: str, input_queue: ProcessingQueue, output_queue: Optional[ProcessingQueue] = None):
        super().__init__()
        self.stage_name = stage_name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.logger = get_logger(f"Worker-{stage_name}")
        self.should_stop = threading.Event()
        self.is_processing = False
        self.processed_count = 0
    
    def run(self):
        """Haupt-Worker-Loop"""
        self.logger.info(f"🚀 {self.stage_name} worker started and ready")
        
        while not self.should_stop.is_set():
            try:
                # Get ProcessObject from queue (with timeout)
                obj_result = self.input_queue.get(timeout=1.0)
                
                if isinstance(obj_result, Err):
                    continue  # Queue empty, try again
                
                process_obj = unwrap_ok(obj_result)
                self.is_processing = True
                
                # Enhanced processing start logging
                self.logger.info(
                    f"🎬 {self.stage_name} processing started: {process_obj.titel}",
                    extra={
                        'worker_stage': self.stage_name,
                        'video_title': process_obj.titel,
                        'video_channel': process_obj.kanal,
                        'processing_stage': process_obj.processing_stage,
                        'queue_size_before': self.input_queue.size(),
                        'worker_processed_count': self.processed_count
                    }
                )
                
                # Update status
                self.stage_status_changed.emit(self.stage_name, self.input_queue.size())
                
                # Process object
                with log_feature(f"{self.stage_name}_processing") as feature:
                    feature.add_metric("video_title", process_obj.titel)
                    feature.add_metric("worker_stage", self.stage_name)
                    
                    process_result = self.process_object(process_obj)
                    
                    if isinstance(process_result, Ok):
                        processed_obj = unwrap_ok(process_result)
                        self.processed_count += 1
                        
                        # Handle routing decision (especially for Analysis stage)
                        next_stage_info = self.get_routing_decision(processed_obj)
                        
                        # Enhanced routing logging
                        self.logger.info(
                            f"✅ {self.stage_name} completed: {processed_obj.titel} → {next_stage_info['next_stage']}",
                            extra={
                                'worker_stage': self.stage_name,
                                'video_title': processed_obj.titel,
                                'next_stage': next_stage_info['next_stage'],
                                'route_to_output': next_stage_info['route_to_output'],
                                'worker_processed_count': self.processed_count,
                                'processing_duration': feature.metrics.get('duration_ms', 0)
                            }
                        )
                        
                        if next_stage_info["route_to_output"]:
                            # Send to normal output queue
                            if self.output_queue:
                                put_result = self.output_queue.put(processed_obj)
                                if isinstance(put_result, Ok):
                                    self.object_processed.emit(processed_obj, next_stage_info["next_stage"])
                                    feature.add_metric("status", "forwarded")
                                    
                                    self.logger.debug(
                                        f"📤 Forwarded to {next_stage_info['next_stage']}: {processed_obj.titel}",
                                        extra={
                                            'output_queue': next_stage_info['next_stage'],
                                            'output_queue_size': self.output_queue.size()
                                        }
                                    )
                                else:
                                    error_msg = f"Queue overflow: {unwrap_err(put_result).message}"
                                    self.processing_error.emit(processed_obj, error_msg)
                                    self.logger.error(f"❌ Queue overflow for {processed_obj.titel}: {error_msg}")
                            else:
                                # Final stage - completion
                                self.object_processed.emit(processed_obj, "completed")
                                feature.add_metric("status", "completed")
                                self.logger.info(f"🏁 Final completion: {processed_obj.titel}")
                        else:
                            # Route to archive (failed analysis)
                            self.object_processed.emit(processed_obj, "archive")
                            feature.add_metric("status", "archived")
                            self.logger.info(f"📁 Archived (failed rules): {processed_obj.titel}")
                    else:
                        # Processing failed
                        error = unwrap_err(process_result)
                        process_obj.add_error(f"{self.stage_name}: {error.message}")
                        self.processing_error.emit(process_obj, error.message)
                        feature.add_metric("status", "failed")
                        
                        self.logger.error(
                            f"❌ {self.stage_name} failed: {process_obj.titel}",
                            extra={
                                'worker_stage': self.stage_name,
                                'video_title': process_obj.titel,
                                'error_message': error.message,
                                'error_context': error.context.to_dict() if hasattr(error, 'context') else {}
                            }
                        )
                
                self.input_queue.task_done()
                self.is_processing = False
                
            except Exception as e:
                self.logger.error(f"💥 Unexpected error in {self.stage_name} worker: {e}")
                self.is_processing = False
                time.sleep(1)  # Avoid tight error loop
        
        self.logger.info(f"🛑 {self.stage_name} worker stopped (processed {self.processed_count} videos)")
    
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
# SPECIFIC WORKER IMPLEMENTATIONS 
# =============================================================================

class AudioDownloadWorker(BaseWorker):
    """Worker für Audio-Download via yt-dlp (ProcessObjects mit Metadata als Input)"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config: AppConfig):
        super().__init__("Audio Download", input_queue, output_queue)
        self.config = config
        
        # Ensure temp directory exists
        temp_dir = Path(self.config.processing.temp_folder)
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    @log_function(log_performance=True)
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        """Downloaded Audio-Spur mit yt-dlp"""
        try:
            self.logger.info(
                f"🎵 Starting audio download for: {obj.titel}",
                extra={
                    'video_title': obj.titel,
                    'channel': obj.kanal,
                    'original_url': obj.original_url,
                    'audio_format': self.config.processing.audio_format,
                    'temp_folder': str(self.config.processing.temp_folder)
                }
            )
            
            # Use the real audio download function
            download_result = download_audio_for_process_object(obj, self.config)
            
            if isinstance(download_result, Ok):
                downloaded_obj = unwrap_ok(download_result)
                
                self.logger.info(
                    f"✅ Audio download completed for: {downloaded_obj.titel}",
                    extra={
                        'audio_path': str(downloaded_obj.temp_audio_path),
                        'file_size_mb': round(downloaded_obj.temp_audio_path.stat().st_size / (1024 * 1024), 2) if downloaded_obj.temp_audio_path and downloaded_obj.temp_audio_path.exists() else 0,
                        'audio_format': self.config.processing.audio_format
                    }
                )
                
                return Ok(downloaded_obj)
            else:
                return download_result
            
        except Exception as e:
            context = ErrorContext.create(
                "audio_download_worker",
                input_data={"title": obj.titel, "original_url": obj.original_url},
                suggestions=["Check yt-dlp installation", "Verify network connection", "Check disk space"]
            )
            return Err(CoreError(f"Audio download worker failed: {e}", context))
    
    def get_next_stage_name(self) -> str:
        return "Transcription"

class TranscriptionWorker(BaseWorker):
    """Worker für Audio-Transkription via faster-whisper"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config: AppConfig):
        super().__init__("Transcription", input_queue, output_queue)
        self.config = config
    
    @log_function(log_performance=True)
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        """Transkribiert Audio mit faster-whisper"""
        try:
            self.logger.info(
                f"🎙️ Starting transcription for: {obj.titel}",
                extra={
                    'video_title': obj.titel,
                    'audio_path': str(obj.temp_audio_path) if obj.temp_audio_path else 'None',
                    'whisper_model': self.config.whisper.model,
                    'whisper_device': self.config.whisper.device,
                    'whisper_enabled': self.config.whisper.enabled
                }
            )
            
            # Use the real transcription function
            transcription_result = transcribe_process_object(obj, self.config)
            
            if isinstance(transcription_result, Ok):
                transcribed_obj = unwrap_ok(transcription_result)
                
                self.logger.info(
                    f"✅ Transcription completed for: {transcribed_obj.titel}",
                    extra={
                        'language': transcribed_obj.sprache,
                        'transcript_length': len(transcribed_obj.transkript),
                        'transcript_preview': transcribed_obj.transkript[:100] + "..." if len(transcribed_obj.transkript) > 100 else transcribed_obj.transkript
                    }
                )
                
                return Ok(transcribed_obj)
            else:
                return transcription_result
            
        except Exception as e:
            context = ErrorContext.create(
                "transcription_worker",
                input_data={"title": obj.titel, "audio_path": str(obj.temp_audio_path)},
                suggestions=["Check faster-whisper installation", "Verify GPU/CUDA setup", "Check audio file"]
            )
            return Err(CoreError(f"Transcription worker failed: {e}", context))
    
    def get_next_stage_name(self) -> str:
        return "Analysis"

class AnalysisWorker(BaseWorker):
    """Worker für Content-Analyse via Ollama + Rule System"""
    
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config: AppConfig):
        super().__init__("Analysis", input_queue, output_queue)
        self.config = config
        self.archive_queue = ProcessingQueue("archive")  # Separate queue für failed analysis
    
    @log_function(log_performance=True)
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        """Analysiert Content mit Ollama + Rule System"""
        try:
            self.logger.info(
                f"🧠 Starting content analysis for: {obj.titel}",
                extra={
                    'video_title': obj.titel,
                    'transcript_length': len(obj.transkript or ""),
                    'ollama_model': self.config.ollama.model,
                    'rules_enabled': list(self.config.rules.get_enabled_rules().keys()),
                    'scoring_threshold': self.config.scoring.threshold,
                    'min_confidence': self.config.scoring.min_confidence,
                    'total_rules_weight': self.config.rules.get_total_weight()
                }
            )
            
            # Use the real analysis function from yt_rulechain
            analysis_result = analyze_process_object(obj, self.config)
            
            if isinstance(analysis_result, Ok):
                analyzed_obj = unwrap_ok(analysis_result)
                
                # Enhanced analysis completion logging with detailed rule results
                decision = "DOWNLOAD" if analyzed_obj.passed_analysis else "SKIP"
                
                # Extract detailed rule analysis if available
                rule_details = {}
                if analyzed_obj.analysis_results and isinstance(analyzed_obj.analysis_results, dict):
                    rules_analysis = analyzed_obj.analysis_results.get('rules_analysis', [])
                    for rule in rules_analysis:
                        if isinstance(rule, dict):
                            rule_name = rule.get('rule_name', 'unknown')
                            rule_details[rule_name] = {
                                'fulfilled': rule.get('fulfilled', False),
                                'score': rule.get('score', 0.0),
                                'confidence': rule.get('confidence', 0.0),
                                'reasoning': rule.get('reasoning', '')[:100]  # First 100 chars
                            }
                
                # Comprehensive analysis logging
                analysis_info = (
                    f"✅ Content analysis completed: {analyzed_obj.titel} → {decision}\n"
                    f"  📊 Overall Score: {analyzed_obj.relevancy:.3f} (threshold: {self.config.scoring.threshold})\n"
                    f"  🎯 Confidence: {analyzed_obj.rule_accuracy:.3f} (min: {self.config.scoring.min_confidence})\n"
                    f"  📋 Rules fulfilled: {analyzed_obj.rule_amount}/{len(self.config.rules.get_enabled_rules())}\n"
                    f"  🤖 Model: {self.config.ollama.model}\n"
                    f"  📝 Transcript: {len(obj.transkript or '')} chars\n"
                    f"  ⚖️ Decision factors: score_ok={analyzed_obj.relevancy >= self.config.scoring.threshold}, confidence_ok={analyzed_obj.rule_accuracy >= self.config.scoring.min_confidence}"
                )
                
                self.logger.info(analysis_info)
                
                # Log individual rule results for debugging
                if rule_details:
                    self.logger.info(
                        f"📋 Rule-by-rule analysis for: {analyzed_obj.titel}",
                        extra={
                            'video_title': analyzed_obj.titel,
                            'rule_details': rule_details,
                            'analysis_summary': analyzed_obj.analysis_results.get('summary', 'No summary') if analyzed_obj.analysis_results else 'No summary'
                        }
                    )
                
                return Ok(analyzed_obj)
            else:
                # Analysis failed - log detailed error
                error = unwrap_err(analysis_result)
                self.logger.error(
                    f"❌ Content analysis failed for: {obj.titel}",
                    extra={
                        'video_title': obj.titel,
                        'error_message': error.message,
                        'error_context': error.context.to_dict() if hasattr(error, 'context') else {},
                        'transcript_length': len(obj.transkript or ""),
                        'ollama_model': self.config.ollama.model,
                        'ollama_host': self.config.ollama.host,
                        'enabled_rules': list(self.config.rules.get_enabled_rules().keys())
                    }
                )
                return analysis_result
            
        except Exception as e:
            context = ErrorContext.create(
                "content_analysis_worker",
                input_data={
                    "title": obj.titel, 
                    "transcript_length": len(obj.transkript or ""),
                    "ollama_model": self.config.ollama.model,
                    "ollama_host": self.config.ollama.host
                },
                suggestions=[
                    "Check Ollama connectivity", 
                    "Verify rule files exist", 
                    "Check GPU/CUDA setup",
                    "Verify rule configuration",
                    f"Test Ollama: curl {self.config.ollama.host}/api/tags"
                ]
            )
            return Err(CoreError(f"Analysis worker failed: {e}", context))
    
    def get_routing_decision(self, obj: ProcessObject) -> Dict[str, Any]:
        """Conditional routing basierend auf Analysis-Ergebnis"""
        if obj.passed_analysis:
            return {
                "route_to_output": True,
                "next_stage": "Video Download"
            }
        else:
            return {
                "route_to_output": False,  # Route to archive instead
                "next_stage": "Archive"
            }
    
    def get_next_stage_name(self) -> str:
        return "Video Download"  # Default, wird durch get_routing_decision() überschrieben

# Mock-Worker für verbleibende Stages (Video Download, Upload, Processing)
class VideoDownloadWorker(BaseWorker):
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config: AppConfig):
        super().__init__("Video Download", input_queue, output_queue)
        self.config = config
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        time.sleep(1.2)  # Mock processing
        obj.temp_video_path = Path(f"/tmp/video_{obj.get_unique_key()}.mp4")
        obj.update_stage("video_downloaded")
        return Ok(obj)
    
    def get_next_stage_name(self) -> str:
        return "Upload"

class UploadWorker(BaseWorker):
    def __init__(self, input_queue: ProcessingQueue, output_queue: ProcessingQueue, config: AppConfig):
        super().__init__("Upload", input_queue, output_queue)
        self.config = config
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        time.sleep(0.8)  # Mock upload
        obj.nextcloud_link = f"https://nextcloud.example.com/file/{obj.get_unique_key()}.mp4"
        obj.update_stage("uploaded_to_nextcloud")
        return Ok(obj)
    
    def get_next_stage_name(self) -> str:
        return "Processing"

class ProcessingWorker(BaseWorker):
    def __init__(self, input_queue: ProcessingQueue, output_queue: Optional[ProcessingQueue], config: AppConfig):
        super().__init__("Processing", input_queue, output_queue)
        self.config = config
    
    def process_object(self, obj: ProcessObject) -> Result[ProcessObject, CoreError]:
        time.sleep(0.5)  # Mock processing
        obj.bearbeiteter_transkript = f"Bearbeitetes Transkript: {obj.transkript}"
        obj.trilium_link = f"trilium://note/{obj.get_unique_key()}"
        obj.update_stage("completed")
        return Ok(obj)
    
    def get_next_stage_name(self) -> str:
        return "completed"

# =============================================================================
# PIPELINE MANAGER
# =============================================================================

class PipelineManager(QThread):
    """Hauptkoordinator für YouTube-Processing-Pipeline"""
    
    # Signals für GUI-Updates
    status_updated = Signal(PipelineStatus)
    video_completed = Signal(str, bool)  # title, success
    pipeline_finished = Signal(int, int, list)  # total, success, error_messages
    
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger("PipelineManager")
        
        # Pipeline State
        self.state = PipelineState.IDLE
        self.input_urls_text: str = ""
        self.processing_errors: List[ProcessingError] = []
        self.completed_videos: List[ProcessObject] = []
        
        # FIXED: Static video count for accurate GUI display
        self.initial_video_count: int = 0
        
        # Queues für Pipeline-Stages (Metadata-Queue entfernt)
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
        
        # Status Update Timer - IMPROVED: Much faster updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.emit_status_update)
        self.status_timer.start(100)  # Update every 100ms (was 200ms)
    
    def start_pipeline(self, urls_text: str) -> Result[None, CoreError]:
        """Startet Pipeline für Multiline-URL-Text"""
        if self.state != PipelineState.IDLE:
            return Err(CoreError("Pipeline already running"))
        
        self.input_urls_text = urls_text
        self.processing_errors.clear()
        self.completed_videos.clear()
        self.initial_video_count = 0  # Reset
        
        self.logger.info(f"🚀 Starting pipeline for URL input (length: {len(urls_text)} chars)")
        
        with log_feature("pipeline_startup") as feature:
            # Step 1: Process URLs to ProcessObjects with Metadata - ENHANCED LOGGING
            self.logger.info("📝 Step 1: Processing URLs to ProcessObjects...")
            objects_result = process_urls_to_objects(urls_text, self.config.processing.__dict__)
            
            if isinstance(objects_result, Err):
                errors = unwrap_err(objects_result)
                error_summary = f"Failed to process URLs: {len(errors)} errors"
                
                # Enhanced error logging
                self.logger.error(
                    f"❌ URL processing failed with {len(errors)} errors",
                    extra={
                        'input_length': len(urls_text),
                        'error_count': len(errors),
                        'first_3_errors': [str(e.message) for e in errors[:3]]
                    }
                )
                
                for i, error in enumerate(errors[:3]):  # Log first 3 errors
                    self.logger.error(f"URL processing error {i+1}: {error.message}")
                
                feature.add_metric("url_processing_errors", len(errors))
                return Err(CoreError(error_summary))
            
            process_objects = unwrap_ok(objects_result)
            
            if not process_objects:
                self.logger.error("❌ No valid videos found in input")
                return Err(CoreError("No valid videos found in input"))
            
            # ENHANCED: Log extracted video details
            self.logger.info(
                f"✅ Successfully extracted {len(process_objects)} videos",
                extra={
                    'extracted_count': len(process_objects),
                    'video_titles': [obj.titel[:50] for obj in process_objects[:3]],  # First 3 titles
                    'video_channels': list(set([obj.kanal for obj in process_objects])),
                    'total_duration_minutes': sum(
                        (obj.länge.hour * 60 + obj.länge.minute + obj.länge.second / 60) 
                        for obj in process_objects if obj.länge
                    )
                }
            )
            
            feature.add_metric("extracted_videos", len(process_objects))
            
            # Step 2: Check for duplicates in Archive Database - ENHANCED LOGGING
            self.logger.info("🔍 Step 2: Checking for duplicates...")
            # TODO: Implement duplicate checking
            unique_objects = process_objects  # For now, process all
            
            self.logger.info(
                f"✅ Duplicate check completed: {len(unique_objects)} unique videos to process",
                extra={
                    'total_extracted': len(process_objects),
                    'unique_videos': len(unique_objects),
                    'duplicates_found': len(process_objects) - len(unique_objects)
                }
            )
            
            # FIXED: Set static video count for GUI
            self.initial_video_count = len(unique_objects)
            
            # Step 3: Queue ProcessObjects for Audio Download - ENHANCED LOGGING
            self.logger.info("📤 Step 3: Queuing videos for audio download...")
            queued_count = 0
            failed_queue_count = 0
            
            for i, process_obj in enumerate(unique_objects):
                put_result = self.queues["audio_download"].put(process_obj)
                if isinstance(put_result, Ok):
                    queued_count += 1
                    self.logger.debug(
                        f"📤 Queued video {i+1}/{len(unique_objects)}: {process_obj.titel}",
                        extra={
                            'video_number': i+1,
                            'total_videos': len(unique_objects),
                            'video_title': process_obj.titel,
                            'queue_size': self.queues["audio_download"].size()
                        }
                    )
                else:
                    failed_queue_count += 1
                    error_msg = unwrap_err(put_result).message
                    self.logger.error(f"❌ Failed to queue video {process_obj.titel}: {error_msg}")
            
            self.logger.info(
                f"✅ Queueing completed: {queued_count} queued, {failed_queue_count} failed",
                extra={
                    'queued_successfully': queued_count,
                    'queue_failures': failed_queue_count,
                    'audio_download_queue_size': self.queues["audio_download"].size()
                }
            )
            
            # Step 4: Setup and start workers - ENHANCED LOGGING
            self.logger.info("🔧 Step 4: Setting up worker threads...")
            setup_result = self.setup_workers()
            if isinstance(setup_result, Err):
                return setup_result
            
            feature.add_metric("videos_queued", queued_count)
            feature.add_metric("queue_failures", failed_queue_count)
            feature.add_metric("workers_started", len(self.workers))
            feature.add_metric("initial_video_count", self.initial_video_count)
        
        self.state = PipelineState.RUNNING
        self.start()  # Start QThread
        
        # Enhanced pipeline start completion logging
        pipeline_info = (
            f"🚀 Pipeline started successfully:\n"
            f"  📹 Videos to process: {self.initial_video_count}\n"
            f"  📤 Successfully queued: {queued_count}\n"
            f"  🔧 Workers started: {len(self.workers)}\n"
            f"  🎯 Audio download queue: {self.queues['audio_download'].size()}\n"
            f"  📊 Status updates: every 100ms"
        )
        
        self.logger.info(pipeline_info)
        return Ok(None)
    
    def setup_workers(self) -> Result[None, CoreError]:
        """Richtet alle Worker-Threads ein (ohne MetadataWorker)"""
        try:
            # Clear existing workers
            self.cleanup_workers()
            
            # Create worker chain (Metadata-Worker entfernt)
            workers_config = [
                (AudioDownloadWorker, "audio_download", "transcription"),
                (TranscriptionWorker, "transcription", "analysis"),
                (AnalysisWorker, "analysis", "video_download"),
                (VideoDownloadWorker, "video_download", "upload"),
                (UploadWorker, "upload", "processing"),
                (ProcessingWorker, "processing", None)  # Final stage
            ]
            
            for i, (worker_class, input_queue_name, output_queue_name) in enumerate(workers_config):
                input_queue = self.queues[input_queue_name]
                output_queue = self.queues[output_queue_name] if output_queue_name else None
                
                worker = worker_class(input_queue, output_queue, self.config)
                
                # Connect signals
                worker.object_processed.connect(self.on_object_processed)
                worker.processing_error.connect(self.on_processing_error)
                worker.stage_status_changed.connect(self.on_stage_status_changed)
                
                self.workers.append(worker)
                worker.start()
                
                # Enhanced worker start logging
                self.logger.info(
                    f"✅ Worker {i+1}/{len(workers_config)} started: {worker.stage_name}",
                    extra={
                        'worker_number': i+1,
                        'total_workers': len(workers_config),
                        'worker_stage': worker.stage_name,
                        'input_queue': input_queue_name,
                        'output_queue': output_queue_name or 'final',
                        'input_queue_size': input_queue.size()
                    }
                )
            
            self.logger.info(f"🎉 All {len(self.workers)} worker threads started and ready")
            return Ok(None)
            
        except Exception as e:
            context = ErrorContext.create(
                "worker_setup",
                suggestions=["Check thread limits", "Verify worker configuration"]
            )
            return Err(CoreError(f"Failed to setup workers: {e}", context))
    
    def on_object_processed(self, obj: ProcessObject, next_stage: str):
        """Handler für erfolgreich verarbeitete Objects"""
        self.logger.debug(
            f"📋 Object processed: {obj.titel} → {next_stage}",
            extra={
                'video_title': obj.titel,
                'next_stage': next_stage,
                'processing_stage': obj.processing_stage,
                'current_completed': len(self.completed_videos),
                'current_failed': len(self.processing_errors)
            }
        )
        
        if next_stage == "completed":
            self.completed_videos.append(obj)
            self.video_completed.emit(obj.titel, True)
            self.logger.info(f"🏁 Video completed successfully: {obj.titel}")
            # TODO: Save to ArchiveDatabase
        elif next_stage == "archive":
            # Failed analysis - archive only (no video download)
            obj.update_stage("analysis_failed")
            self.completed_videos.append(obj)
            self.video_completed.emit(obj.titel, False)
            # TODO: Save to ArchiveDatabase as failed
            self.logger.info(f"📁 Video archived (failed analysis): {obj.titel}")
        # Note: Normal stage transitions don't emit video_completed
    
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
        
        self.logger.error(
            f"💥 Processing error: {obj.titel} in {obj.processing_stage}",
            extra={
                'video_title': obj.titel,
                'processing_stage': obj.processing_stage,
                'error_message': error_message,
                'total_errors': len(self.processing_errors),
                'total_completed': len(self.completed_videos)
            }
        )
    
    def on_stage_status_changed(self, stage_name: str, queue_size: int):
        """Handler für Stage-Status-Änderungen"""
        # Status wird über Timer-basierte emit_status_update gehandelt
        pass
    
    def emit_status_update(self):
        """Sendet aktuellen Pipeline-Status an GUI"""
        if self.state == PipelineState.IDLE:
            return
        
        # Check if any workers are actively processing
        workers_active = any(worker.is_processing for worker in self.workers if worker.isRunning())
        active_worker_names = [worker.stage_name for worker in self.workers if worker.is_processing]
        
        # FIXED: Use static initial_video_count instead of dynamic calculation
        status = PipelineStatus(
            audio_download_queue=self.queues["audio_download"].size(),
            transcription_queue=self.queues["transcription"].size(),
            analysis_queue=self.queues["analysis"].size(),
            video_download_queue=self.queues["video_download"].size(),
            upload_queue=self.queues["upload"].size(),
            processing_queue=self.queues["processing"].size(),
            
            total_queued=self.initial_video_count,  # FIXED: Static count
            total_completed=len(self.completed_videos),
            total_failed=len(self.processing_errors),
            
            current_stage=self.get_current_stage(),
            current_video=self.get_current_video()
        )
        
        self.status_updated.emit(status)
        
        # Enhanced debug logging every 20th update (every 2 seconds at 100ms)
        if hasattr(self, '_status_update_counter'):
            self._status_update_counter += 1
        else:
            self._status_update_counter = 1
        
        if self._status_update_counter % 20 == 0:  # Every 2 seconds at 100ms intervals
            self.logger.debug(
                f"📊 Pipeline status update #{self._status_update_counter}",
                extra={
                    'total_videos': self.initial_video_count,
                    'completed': len(self.completed_videos),
                    'failed': len(self.processing_errors),
                    'remaining': self.initial_video_count - len(self.completed_videos) - len(self.processing_errors),
                    'queue_sizes': {name: queue.size() for name, queue in self.queues.items()},
                    'active_workers': active_worker_names,
                    'workers_running': len([w for w in self.workers if w.isRunning()]),
                    'current_stage': status.current_stage
                }
            )
        
        # Check if pipeline finished (IMPROVED - consider active workers!)
        if not status.is_active() and not workers_active and self.state == PipelineState.RUNNING:
            self.logger.debug(
                f"🏁 Pipeline finish check: queues_empty={not status.is_active()}, workers_active={workers_active}",
                extra={
                    'queue_sizes': {name: queue.size() for name, queue in self.queues.items()},
                    'active_workers': active_worker_names,
                    'running_workers': [worker.stage_name for worker in self.workers if worker.isRunning()],
                    'total_processed': len(self.completed_videos) + len(self.processing_errors),
                    'initial_count': self.initial_video_count
                }
            )
            self.finish_pipeline()
        elif workers_active:
            # Only log this occasionally to avoid spam
            if self._status_update_counter % 50 == 0:  # Every 5 seconds at 100ms intervals
                self.logger.debug(
                    f"⚡ Pipeline still active - workers processing",
                    extra={
                        'active_workers': active_worker_names,
                        'queue_sizes': {name: queue.size() for name, queue in self.queues.items()}
                    }
                )
    
    def get_current_stage(self) -> str:
        """Ermittelt aktuelle Haupt-Processing-Stage"""
        for stage_name, queue in self.queues.items():
            if queue.size() > 0:
                return stage_name.replace("_", " ").title()
        return "Idle"
    
    def get_current_video(self) -> Optional[str]:
        """Ermittelt aktuell verarbeitetes Video (falls möglich)"""
        # Simplified: Return None - würde echte Worker-Status erfordern
        return None
    
    def finish_pipeline(self):
        """Beendet Pipeline und sendet Final-Summary"""
        total_processed = len(self.completed_videos) + len(self.processing_errors)
        
        # Enhanced pipeline completion logging
        completion_info = (
            f"🏁 Pipeline finished:\n"
            f"  📹 Total videos: {self.initial_video_count}\n"
            f"  ✅ Completed: {len(self.completed_videos)}\n"
            f"  ❌ Failed: {len(self.processing_errors)}\n"
            f"  📊 Success rate: {len(self.completed_videos) / self.initial_video_count * 100:.1f}%\n"
            f"  ⏱️ Total processed: {total_processed}/{self.initial_video_count}"
        )
        
        self.logger.info(completion_info)
        
        self.cleanup_workers()
        self.state = PipelineState.FINISHED
        
        # Error messages für GUI
        error_messages = [f"{err.stage}: {err.error_message}" for err in self.processing_errors]
        
        self.pipeline_finished.emit(
            self.initial_video_count,  # FIXED: Use initial count
            len(self.completed_videos),
            error_messages
        )
        
        self.state = PipelineState.IDLE
    
    def stop_pipeline(self):
        """Stoppt Pipeline gracefully"""
        if self.state != PipelineState.RUNNING:
            return
        
        self.logger.info("🛑 Stopping pipeline...")
        self.state = PipelineState.STOPPING
        
        # Stop all workers
        self.cleanup_workers()
        
        # Wait for this thread to finish
        if self.isRunning():
            self.wait(5000)
        
        self.state = PipelineState.IDLE
        self.logger.info("✅ Pipeline stopped")
    
    def cleanup_workers(self):
        """Beendet alle Worker-Threads"""
        for worker in self.workers:
            worker.stop_worker()
        
        self.workers.clear()
        self.logger.debug(f"🧹 Cleaned up {len(self.workers)} workers")
    
    def run(self):
        """QThread main loop - minimal da Timer-basiert"""
        while self.state == PipelineState.RUNNING:
            self.msleep(100)  # Sleep 100ms, status updates via timer

# =============================================================================
# INTEGRATION MIT GUI
# =============================================================================

def integrate_pipeline_with_gui(main_window, config: AppConfig):
    """Integriert PipelineManager in MainWindow"""
    
    # Create PipelineManager
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
    
    # Override start_analysis method
    original_start_analysis = main_window.start_analysis
    
    def enhanced_start_analysis():
        urls_text = main_window.url_input.toPlainText().strip()
        
        if not urls_text:
            main_window.status_bar.showMessage("Please enter YouTube URLs")
            return
        
        # Start pipeline with URLs text (not parsed list)
        start_result = main_window.pipeline_manager.start_pipeline(urls_text)
        
        if isinstance(start_result, Ok):
            main_window.status_bar.showMessage("Started URL processing and metadata extraction...")
            main_window.url_input.clear()  # Clear input after successful start
        else:
            error = unwrap_err(start_result)
            main_window.status_bar.showMessage(f"Failed to start pipeline: {error.message}")
            main_window.logger.error(f"Pipeline start failed: {error.message}")
    
    main_window.start_analysis = enhanced_start_analysis

def show_pipeline_summary(main_window, total: int, success: int, errors: List[str]):
    """Zeigt Pipeline-Summary-Dialog"""
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

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    from yt_analyzer_config import SecureConfigManager
    
    # Setup
    setup_logging("pipeline_test", "DEBUG")
    
    # Mock config for testing
    config_manager = SecureConfigManager()
    config_result = config_manager.load_config()
    
    if isinstance(config_result, Ok):
        config = unwrap_ok(config_result)
        
        # Test pipeline with mock URLs
        pipeline = PipelineManager(config)
        
        urls = [
            "https://youtube.com/watch?v=test1",
            "https://youtube.com/watch?v=test2",
            "https://youtu.be/test3"
        ]
        
        start_result = pipeline.start_pipeline(urls)
        
        if isinstance(start_result, Ok):
            print("Pipeline started successfully!")
            # In real app, GUI would handle the rest
        else:
            print(f"Pipeline start failed: {unwrap_err(start_result).message}")
    else:
        print(f"Config loading failed: {unwrap_err(config_result).message}")
