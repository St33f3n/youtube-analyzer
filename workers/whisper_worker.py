"""
Whisper Worker - QThread für Audio-Transkription mit Result-Types
Vollständig überarbeitet nach Quality-Gate-Standards
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread
from PySide6.QtCore import Signal

from services.whisper import WhisperService
from services.whisper import get_whisper_service
from yt_types import AudioMetadata
from yt_types import Err
from yt_types import Ok
from yt_types import TranscriptionResult
from utils.logging import ComponentLogger
from utils.logging import log_function_calls
from utils.logging import log_performance


class WhisperWorker(QThread):
    """QThread Worker für Whisper Audio-Transkription mit Result-Types"""
    
    # Signals für UI Communication
    progress_updated = Signal(str)           # status_message
    model_loading = Signal()                 # Model wird geladen
    model_ready = Signal(dict)               # Model-Info für UI
    transcription_started = Signal()         # Transkription gestartet
    transcript_ready = Signal(str)           # Fertiges Transkript (string)
    error_occurred = Signal(str)             # Fehler-Nachricht (string)
    finished = Signal()                      # Worker beendet
    
    def __init__(self, parent: Optional[QThread] = None) -> None:
        super().__init__(parent)
        self.audio_metadata: Optional[AudioMetadata] = None
        self._should_stop = False
        self.whisper_service = get_whisper_service()
        self.logger = ComponentLogger("WhisperWorker")
        
        self.logger.debug("WhisperWorker initialized")
        
    @log_function_calls
    def set_audio_metadata(self, audio_metadata: AudioMetadata) -> None:
        """Audio-Metadaten für Transkription setzen"""
        self.audio_metadata = audio_metadata
        self._should_stop = False
        
        self.logger.debug(
            "Audio metadata set",
            audio_file=str(audio_metadata.file_path),
            file_size=audio_metadata.file_size,
            duration=audio_metadata.duration,
        )
        
    @log_function_calls
    def stop_transcription(self) -> None:
        """Transkription abbrechen"""
        self._should_stop = True
        self.logger.info("Transcription stop requested")
        
    @log_performance
    def run(self) -> None:
        """Haupt-Transkriptionsprozess mit vollständigem Error-Handling"""
        try:
            self.logger.info("Whisper worker started")
            
            # 1. Audio-Metadaten validieren
            if not self.audio_metadata:
                self._emit_error("No audio metadata provided")
                return
                
            if not self.audio_metadata.file_path.exists():
                self._emit_error(f"Audio file does not exist: {self.audio_metadata.file_path}")
                return
                
            if self.audio_metadata.file_size < 1000:
                self._emit_error("Audio file too small (< 1000 bytes)")
                return
                
            if self._should_stop:
                return
                
            # 2. Model-Status prüfen und ggf. laden
            self._handle_model_loading()
            
            if self._should_stop:
                return
                
            # 3. Transkription ausführen
            self._perform_transcription()
            
        except Exception as e:
            error_msg = f"Unexpected error in Whisper worker: {str(e)}"
            self.logger.error(
                "Whisper worker failed with exception",
                error=e,
                audio_file=str(self.audio_metadata.file_path) if self.audio_metadata else "None",
            )
            self._emit_error(error_msg)
            
        finally:
            self._cleanup()
            self.finished.emit()
            self.logger.info("Whisper worker finished")
    
    @log_function_calls
    def _handle_model_loading(self) -> None:
        """Model-Loading mit UI-Feedback"""
        model_info = self.whisper_service.get_model_info()
        
        if not model_info['loaded']:
            self.progress_updated.emit("🎤 Loading Whisper Large-v3 Model...")
            self.model_loading.emit()
            
            self.logger.info(
                "Whisper model loading required",
                model_name=model_info['model_name'],
                device=model_info['device'],
            )
        
        # Model-Info an UI senden
        self.model_ready.emit(model_info)
        
        self.logger.debug(
            "Model info sent to UI",
            model_loaded=model_info['loaded'],
            device=model_info['device'],
            gpu_available=model_info.get('cuda_available', False),
        )
    
    @log_performance
    def _perform_transcription(self) -> None:
        """Führe Transkription mit Result-Type-Handling durch"""
        if not self.audio_metadata:
            self._emit_error("Audio metadata lost during transcription")
            return
            
        self.progress_updated.emit("🎤 Transcribing audio...")
        self.transcription_started.emit()
        
        self.logger.info(
            "Starting transcription",
            audio_file=str(self.audio_metadata.file_path),
            file_size=self.audio_metadata.file_size,
            duration=self.audio_metadata.duration,
        )
        
        # Whisper-Service aufrufen mit Result-Type
        transcription_result = self.whisper_service.transcribe_audio(self.audio_metadata)
        
        if self._should_stop:
            return
            
        # Result-Type-Handling
        if isinstance(transcription_result, Ok):
            self._handle_transcription_success(transcription_result.value)
        elif isinstance(transcription_result, Err):
            self._handle_transcription_error(transcription_result.error)
        else:
            # Fallback für unerwartete Return-Types
            self._emit_error("Transcription returned unexpected result type")
    
    @log_function_calls
    def _handle_transcription_success(self, transcription: TranscriptionResult) -> None:
        """Erfolgreiche Transkription verarbeiten"""
        
        # Ergebnis validieren
        if not transcription.text or len(transcription.text.strip()) < 10:
            self._emit_error("Transcription too short - possibly no speech content")
            return
        
        # Success-Feedback
        self.progress_updated.emit("✅ Transcription completed")
        
        self.logger.info(
            "Transcription completed successfully",
            transcript_length=len(transcription.text),
            language=transcription.language,
            confidence=transcription.confidence,
            processing_time=transcription.processing_time,
            device=transcription.device,
        )
        
        # Transkript an UI senden (nur Text für Kompatibilität)
        self.transcript_ready.emit(transcription.text)
    
    @log_function_calls
    def _handle_transcription_error(self, error) -> None:
        """Transkriptionsfehler behandeln"""
        error_message = f"Transcription failed: {error.message}"
        
        self.logger.error(
            "Transcription failed",
            error_message=error.message,
            error_context=getattr(error, 'context', {}),
            audio_file=str(self.audio_metadata.file_path) if self.audio_metadata else "None",
        )
        
        self._emit_error(error_message)
    
    @log_function_calls
    def _emit_error(self, error_message: str) -> None:
        """Strukturierte Fehler-Emission"""
        formatted_error = f"❌ {error_message}"
        
        self.logger.error(
            "Emitting error to UI",
            error_message=error_message,
            audio_file=str(self.audio_metadata.file_path) if self.audio_metadata else "None",
        )
        
        self.error_occurred.emit(formatted_error)
    
    @log_function_calls
    def _cleanup(self) -> None:
        """Cleanup nach Transkription"""
        try:
            # Audio-Datei löschen falls vorhanden
            if (self.audio_metadata and 
                self.audio_metadata.file_path.exists() and
                self.audio_metadata.file_path.is_file()):
                
                self.audio_metadata.file_path.unlink()
                self.logger.debug(
                    "Audio file deleted after transcription",
                    audio_file=str(self.audio_metadata.file_path),
                )
        
        except Exception as e:
            self.logger.warning(
                "Failed to delete audio file",
                error=e,
                audio_file=str(self.audio_metadata.file_path) if self.audio_metadata else "None",
            )
        
        try:
            # Model-Cleanup nach einzelnem Video (wie gewünscht)
            self.whisper_service.cleanup()
            self.logger.debug("Whisper model cleanup completed")
            
        except Exception as e:
            self.logger.warning(
                "Model cleanup failed",
                error=e,
            )


class WhisperManager:
    """Manager für Whisper-Operationen mit Result-Type-Integration"""
    
    def __init__(self, parent: Optional[QThread] = None) -> None:
        self.current_worker: Optional[WhisperWorker] = None
        self.whisper_service = get_whisper_service()
        self.logger = ComponentLogger("WhisperManager")
        
        self.logger.debug("WhisperManager initialized")
        
    @log_function_calls
    def start_transcription(self, audio_metadata: AudioMetadata) -> WhisperWorker:
        """Transkription starten mit Audio-Metadaten"""
        
        # Vorherigen Worker stoppen
        if self.current_worker and self.current_worker.isRunning():
            self.logger.info("Stopping previous transcription worker")
            self.current_worker.stop_transcription()
            self.current_worker.wait(5000)  # 5 Sekunden warten
            
        # Neuen Worker erstellen
        self.current_worker = WhisperWorker()
        self.current_worker.set_audio_metadata(audio_metadata)
        
        self.logger.info(
            "Starting new transcription",
            audio_file=str(audio_metadata.file_path),
            file_size=audio_metadata.file_size,
        )
        
        return self.current_worker
        
    @log_function_calls
    def stop_current_transcription(self) -> None:
        """Aktuelle Transkription stoppen"""
        if self.current_worker and self.current_worker.isRunning():
            self.logger.info("Stopping current transcription")
            self.current_worker.stop_transcription()
        else:
            self.logger.debug("No active transcription to stop")
            
    def get_service_info(self) -> dict:
        """Whisper-Service Informationen"""
        return self.whisper_service.get_model_info()
        
    def is_ready(self) -> bool:
        """Service bereit für Transkription?"""
        return self.whisper_service.is_ready()
        
    @log_function_calls
    def cleanup(self) -> None:
        """Cleanup bei Beendigung"""
        self.stop_current_transcription()
        
        try:
            self.whisper_service.cleanup()
            self.logger.info("WhisperManager cleanup completed")
        except Exception as e:
            self.logger.error(
                "WhisperManager cleanup failed",
                error=e,
            )


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def test_whisper_worker() -> None:
    """Test-Funktion für Whisper Worker (nur Service-Check)"""
    from youtube_analyzer.utils.logging import get_development_config
    from youtube_analyzer.utils.logging import setup_logging
    
    # Setup logging für Test
    setup_logging(get_development_config())
    
    logger = ComponentLogger("WhisperWorkerTest")
    logger.info("Starting Whisper worker test")
    
    # Manager erstellen
    manager = WhisperManager()
    
    # Service-Info prüfen
    service_info = manager.get_service_info()
    logger.info(
        "Whisper service info",
        model_name=service_info['model_name'],
        device=service_info['device'],
        cuda_available=service_info['cuda_available'],
        loaded=service_info['loaded'],
    )
    
    # Ready-Status prüfen
    ready = manager.is_ready()
    logger.info(
        "Whisper service ready",
        ready=ready,
    )
    
    if service_info['cuda_available']:
        logger.info("✅ Whisper worker ready for GPU transcription")
    else:
        logger.info("⚠️ Whisper worker will run on CPU (slower)")
        
    # Cleanup
    manager.cleanup()
    logger.info("✅ Whisper worker test completed")


if __name__ == "__main__":
    test_whisper_worker()
