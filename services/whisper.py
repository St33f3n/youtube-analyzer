"""
Whisper Service - faster-whisper Integration mit Result-Types und vollständigen Type-Hints
Vollständig überarbeitet nach Quality-Gate-Standards
REPARIERTE IMPORTS - Konsistente direkte Imports
GC-SAFE VERSION - Defensive __del__ Implementierung
"""

from __future__ import annotations

import locale
import os
import tempfile
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import torch
from faster_whisper import WhisperModel

# REPARIERTE IMPORTS - Direkte Imports aus Hauptverzeichnis
from yt_types import AudioMetadata
from yt_types import Err
from yt_types import GPUInfo
from yt_types import Ok
from yt_types import Result
from yt_types import ServiceStatus
from yt_types import TranscriptionError
from yt_types import TranscriptionResult
from yt_types import validate_path
from utils.logging import ComponentLogger
from utils.logging import log_function_calls
from utils.logging import log_performance


class WhisperService:
    """Service für Audio-Transkription mit faster-whisper Large-v3"""
    
    def __init__(self, model_name: str = "large-v3") -> None:
        self.model_name = model_name
        self._model: Optional[WhisperModel] = None
        self._model_loaded = False
        self._device = self._detect_device()
        self._temp_dir = Path(tempfile.mkdtemp(prefix="whisper_analyzer_"))
        self.logger = ComponentLogger("WhisperService")
        
        self.logger.info(
            "Whisper service initialized",
            model_name=model_name,
            device=self._device,
            temp_dir=str(self._temp_dir),
        )
    
    def __del__(self) -> None:
        """GC-Safe Cleanup beim Beenden"""
        try:
            # Defensive Prüfungen für GC-Safety
            if hasattr(self, 'cleanup') and callable(self.cleanup):
                # Cleanup ohne Decorator-Calls um Logger-Probleme zu vermeiden
                self._safe_cleanup()
        except Exception:
            # Ignore alle Errors in __del__ - GC-Safety
            pass
    
    def _safe_cleanup(self) -> None:
        """Cleanup ohne Decorator-Abhängigkeiten für __del__"""
        try:
            # Model freigeben
            if hasattr(self, '_model') and self._model is not None:
                if hasattr(self, '_device') and self._device == "cuda":
                    torch.cuda.empty_cache()
                
                self._model = None
                self._model_loaded = False
            
            # Temporäre Dateien aufräumen
            if hasattr(self, '_temp_dir') and self._temp_dir and self._temp_dir.exists():
                import shutil
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            # Vollständig defensiv - keine Exceptions in __del__
            pass
    
    @log_function_calls
    def _detect_device(self) -> str:
        """CUDA-Verfügbarkeit erkennen"""
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                # Logger existiert hier noch nicht - defensiv prüfen
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info(
                        "CUDA GPU detected",
                        gpu_name=device_name,
                        cuda_version=torch.version.cuda,
                    )
                return "cuda"
            else:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning(
                        "CUDA not available, falling back to CPU",
                        torch_version=torch.__version__,
                    )
                return "cpu"
        
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(
                    "Device detection failed",
                    error=e,
                    fallback_device="cpu",
                )
            return "cpu"
    
    @log_function_calls
    def _load_model(self) -> Result[WhisperModel, TranscriptionError]:
        """Model lazy loading mit Error-Handling"""
        if self._model_loaded and self._model is not None:
            return Ok(self._model)
        
        try:
            self.logger.info(
                "Loading Whisper model",
                model_name=self.model_name,
                device=self._device,
            )
            
            # Compute-Type basierend auf Device
            compute_type = "float16" if self._device == "cuda" else "int8"
            
            # Model laden
            self._model = WhisperModel(
                self.model_name,
                device=self._device,
                compute_type=compute_type,
                download_root=None,
                local_files_only=False,
            )
            
            self._model_loaded = True
            
            self.logger.info(
                "Whisper model loaded successfully",
                model_name=self.model_name,
                device=self._device,
                compute_type=compute_type,
            )
            
            return Ok(self._model)
        
        except Exception as e:
            error_msg = f"Failed to load Whisper model: {str(e)}"
            self.logger.error(
                "Model loading failed",
                error=e,
                model_name=self.model_name,
                device=self._device,
            )
            
            self._model = None
            self._model_loaded = False
            
            return Err(TranscriptionError(
                error_msg,
                {
                    'model_name': self.model_name,
                    'device': self._device,
                    'error_type': type(e).__name__,
                }
            ))
    
    @log_performance
    def transcribe_audio(self, audio_metadata: AudioMetadata) -> Result[TranscriptionResult, TranscriptionError]:
        """Audio-Datei transkribieren mit Locale-Fix"""
        
        # Validate audio file
        path_validation = validate_path(audio_metadata.file_path)
        if isinstance(path_validation, Err):
            return Err(TranscriptionError(f"Invalid audio file: {path_validation.error.message}"))
        
        # Load model if needed
        model_result = self._load_model()
        if isinstance(model_result, Err):
            return Err(model_result.error)
        
        model = model_result.value
        
        try:
            self.logger.info(
                "Starting audio transcription",
                audio_file=str(audio_metadata.file_path),
                file_size=audio_metadata.file_size,
                model_name=self.model_name,
                device=self._device,
            )
            
            # Locale-Fix für faster-whisper
            transcription_result = self._transcribe_with_locale_fix(
                model, 
                audio_metadata.file_path
            )
            
            if isinstance(transcription_result, Err):
                return transcription_result
            
            result = transcription_result.value
            
            self.logger.info(
                "Transcription completed successfully",
                transcript_length=len(result.text),
                detected_language=result.language,
                confidence=result.confidence,
                processing_time=result.processing_time,
            )
            
            return Ok(result)
        
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            self.logger.error(
                "Transcription failed",
                error=e,
                audio_file=str(audio_metadata.file_path),
                model_name=self.model_name,
            )
            
            return Err(TranscriptionError(
                error_msg,
                {
                    'audio_file': str(audio_metadata.file_path),
                    'model_name': self.model_name,
                    'error_type': type(e).__name__,
                }
            ))
    
    @log_function_calls
    def _transcribe_with_locale_fix(
        self, 
        model: WhisperModel, 
        audio_path: Path
    ) -> Result[TranscriptionResult, TranscriptionError]:
        """Transkription mit ASCII-Locale-Fix"""
        
        # Aktuelle Locale sichern
        current_locale = locale.getlocale()
        current_env = os.environ.copy()
        
        try:
            # ASCII-Locale für faster-whisper setzen
            self.logger.debug(
                "Setting ASCII locale for faster-whisper",
                original_locale=current_locale,
            )
            
            os.environ['LC_ALL'] = 'C'
            os.environ['LANG'] = 'C'
            os.environ['LC_CTYPE'] = 'C'
            locale.setlocale(locale.LC_ALL, 'C')
            
            # Transkription mit ASCII-Locale
            import time
            start_time = time.time()
            
            segments, info = model.transcribe(
                str(audio_path),
                language=None,  # Auto-Detection
                beam_size=5,
                best_of=5,
                word_timestamps=False,
                vad_filter=True,
                vad_parameters={
                    'min_silence_duration_ms': 500,
                    'threshold': 0.5,
                }
            )
            
            processing_time = time.time() - start_time
            
            self.logger.debug(
                "Transcription completed with ASCII locale",
                processing_time=processing_time,
                detected_language=info.language,
                language_probability=info.language_probability,
            )
            
            # Segmente zu Text verarbeiten
            transcript_parts = []
            for segment in segments:
                if segment.text.strip():
                    transcript_parts.append(segment.text.strip())
            
            full_transcript = " ".join(transcript_parts).strip()
            
            # Validierung
            if not full_transcript or len(full_transcript) < 10:
                raise TranscriptionError(
                    "Transcription too short or empty",
                    {'transcript_length': len(full_transcript)}
                )
            
            # TranscriptionResult erstellen
            result = TranscriptionResult(
                text=full_transcript,
                language=info.language,
                confidence=info.language_probability,
                processing_time=processing_time,
                model_name=self.model_name,
                device=self._device,
            )
            
            return Ok(result)
        
        except Exception as e:
            error_msg = f"Transcription with locale fix failed: {str(e)}"
            self.logger.error(
                "Locale-fixed transcription failed",
                error=e,
                audio_path=str(audio_path),
            )
            
            return Err(TranscriptionError(
                error_msg,
                {
                    'audio_path': str(audio_path),
                    'locale_fix': True,
                    'error_type': type(e).__name__,
                }
            ))
        
        finally:
            # Locale und Umgebung wiederherstellen
            try:
                os.environ.clear()
                os.environ.update(current_env)
                locale.setlocale(locale.LC_ALL, current_locale)
                
                self.logger.debug(
                    "Locale restored",
                    restored_locale=current_locale,
                )
            
            except Exception as restore_error:
                self.logger.warning(
                    "Locale restoration failed",
                    error=restore_error,
                    original_locale=current_locale,
                )
    
    @log_function_calls
    def cleanup(self) -> None:
        """Model und temporäre Dateien aufräumen - Public API Version"""
        try:
            # Model freigeben
            if self._model is not None:
                if self._device == "cuda":
                    torch.cuda.empty_cache()
                
                self._model = None
                self._model_loaded = False
                
                self.logger.info(
                    "Whisper model cleanup completed",
                    device=self._device,
                )
            
            # Temporäre Dateien aufräumen
            if self._temp_dir.exists():
                import shutil
                shutil.rmtree(self._temp_dir)
                
                self.logger.info(
                    "Temporary files cleanup completed",
                    temp_dir=str(self._temp_dir),
                )
        
        except Exception as e:
            self.logger.error(
                "Cleanup failed",
                error=e,
                temp_dir=str(self._temp_dir),
            )
    
    def get_gpu_info(self) -> GPUInfo:
        """GPU-Informationen für Monitoring"""
        try:
            if self._device == "cuda" and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) // (1024 * 1024)
                
                return GPUInfo(
                    device="cuda",
                    name=device_name,
                    memory_total=memory_total,
                    memory_free=memory_free,
                    available=True,
                )
            else:
                return GPUInfo(
                    device="cpu",
                    name="CPU",
                    memory_total=0,
                    memory_free=0,
                    available=False,
                )
        
        except Exception as e:
            self.logger.error(
                "GPU info retrieval failed",
                error=e,
            )
            
            return GPUInfo(
                device="unknown",
                name="Unknown",
                memory_total=0,
                memory_free=0,
                available=False,
            )
    
    def get_service_status(self) -> ServiceStatus:
        """Service-Status für Monitoring"""
        try:
            if self._model_loaded:
                status = "ready"
                message = f"Model {self.model_name} loaded on {self._device}"
            else:
                status = "loading"
                message = f"Model {self.model_name} not loaded"
            
            gpu_info = self.get_gpu_info()
            
            return ServiceStatus(
                service_name="WhisperService",
                status=status,
                message=message,
                details={
                    'model_name': self.model_name,
                    'device': self._device,
                    'model_loaded': self._model_loaded,
                    'gpu_info': gpu_info.dict(),
                    'temp_dir': str(self._temp_dir),
                },
            )
        
        except Exception as e:
            self.logger.error(
                "Service status check failed",
                error=e,
            )
            
            return ServiceStatus(
                service_name="WhisperService",
                status="error",
                message=f"Status check failed: {str(e)}",
                details={'error_type': type(e).__name__},
            )
    
    def is_ready(self) -> bool:
        """Service bereit für Transkription?"""
        return self._model_loaded and self._model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Model-Informationen für UI"""
        gpu_info = self.get_gpu_info()
        
        return {
            'model_name': self.model_name,
            'device': self._device,
            'loaded': self._model_loaded,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': gpu_info.name,
            'gpu_memory_total': gpu_info.memory_total,
            'gpu_memory_free': gpu_info.memory_free,
            'temp_dir': str(self._temp_dir),
        }


# =============================================================================
# SERVICE FACTORY
# =============================================================================

_whisper_service_instance: Optional[WhisperService] = None


def get_whisper_service() -> WhisperService:
    """Singleton Factory für Whisper-Service"""
    global _whisper_service_instance
    
    if _whisper_service_instance is None:
        _whisper_service_instance = WhisperService()
    
    return _whisper_service_instance


def create_whisper_service(model_name: str = "large-v3") -> WhisperService:
    """Factory für neuen Whisper-Service mit spezifischem Model"""
    return WhisperService(model_name)


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def test_whisper_service() -> None:
    """Test-Funktion für Whisper-Service"""
    from utils.logging import get_development_config
    from utils.logging import setup_logging
    
    # Setup logging für Test
    setup_logging(get_development_config())
    
    service = get_whisper_service()
    logger = ComponentLogger("WhisperServiceTest")
    
    logger.info("Starting Whisper service test")
    
    # Test Service Status
    status = service.get_service_status()
    logger.info(
        "Service status",
        status=status.status,
        message=status.message,
        details=status.details,
    )
    
    # Test GPU Info
    gpu_info = service.get_gpu_info()
    logger.info(
        "GPU info",
        device=gpu_info.device,
        name=gpu_info.name,
        available=gpu_info.available,
    )
    
    # Test Model Info
    model_info = service.get_model_info()
    logger.info(
        "Model info",
        model_name=model_info['model_name'],
        device=model_info['device'],
        loaded=model_info['loaded'],
        cuda_available=model_info['cuda_available'],
    )
    
    # Test Cleanup
    service.cleanup()
    logger.info("✅ Whisper service test completed")
    
    if gpu_info.available:
        logger.info("✅ Whisper service ready for GPU transcription")
    else:
        logger.info("⚠️ Whisper service will run on CPU (slower)")


if __name__ == "__main__":
    test_whisper_service()
