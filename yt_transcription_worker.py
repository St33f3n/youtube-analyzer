"""
YouTube Analyzer - Transcription Worker
Audio-zu-Text mit faster-whisper und korrektem Locale-Setup
"""

from __future__ import annotations
import os
import locale
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import threading

# Locale-Setup für faster-whisper (CRITICAL!)
try:
    # Set UTF-8 locale to avoid faster-whisper complaints
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = 'C.UTF-8'
    locale.setlocale(locale.LC_ALL, 'C.UTF-8')
except locale.Error:
    try:
        # Fallback für verschiedene Systeme
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        # Last resort
        os.environ['LC_ALL'] = 'C'
        os.environ['LANG'] = 'C'

# faster-whisper import (nach Locale-Setup!)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_err, unwrap_ok, unwrap_or
from yt_analyzer_core import ProcessObject
from logging_plus import get_logger, log_feature, log_function
from yt_analyzer_config import AppConfig

# =============================================================================
# TRANSCRIPTION ENGINE
# =============================================================================

class WhisperTranscriptionEngine:
    """faster-whisper Engine mit GPU/CPU Support und Locale-Handling"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger("WhisperEngine")
        self.model: Optional[WhisperModel] = None
        self.model_lock = threading.Lock()
        
        # Validate faster-whisper availability
        if not FASTER_WHISPER_AVAILABLE:
            self.logger.error("faster-whisper not available - install with: pip install faster-whisper")
            return
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self) -> Result[None, CoreError]:
        """Lädt Whisper-Model mit GPU/CPU-Konfiguration"""
        if not FASTER_WHISPER_AVAILABLE:
            context = ErrorContext.create(
                "load_whisper_model",
                suggestions=[
                    "Install faster-whisper: pip install faster-whisper",
                    "Check CUDA installation if using GPU"
                ]
            )
            return Err(CoreError("faster-whisper not available", context))
        
        try:
            with log_feature("whisper_model_loading") as feature:
                model_size = self.config.whisper.model
                device = self.config.whisper.device if self.config.whisper.enabled else "cpu"
                compute_type = self.config.whisper.compute_type
                
                self.logger.info(
                    f"Loading Whisper model: {model_size}",
                    extra={
                        'model_size': model_size,
                        'device': device,
                        'compute_type': compute_type,
                        'gpu_enabled': self.config.whisper.enabled and device == "cuda"
                    }
                )
                
                # Load model with proper device configuration
                self.model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type if device == "cuda" else "int8"
                )
                
                feature.add_metric("model_loaded", True)
                feature.add_metric("device", device)
                feature.add_metric("model_size", model_size)
                
                self.logger.info(
                    f"✅ Whisper model loaded successfully",
                    extra={
                        'model': model_size,
                        'device': device,
                        'compute_type': compute_type,
                        'memory_usage': 'GPU' if device == 'cuda' else 'CPU'
                    }
                )
                
                return Ok(None)
        
        except Exception as e:
            context = ErrorContext.create(
                "load_whisper_model",
                input_data={
                    'model_size': self.config.whisper.model,
                    'device': self.config.whisper.device,
                    'error': str(e)
                },
                suggestions=[
                    "Check CUDA installation for GPU mode",
                    "Try CPU mode if GPU fails",
                    "Verify faster-whisper installation",
                    "Check available disk space for model download"
                ]
            )
            return Err(CoreError(f"Failed to load Whisper model: {e}", context))
    
    def _format_timestamp(self, seconds: float) -> str:
            """Formatiert Timestamp in HH:MM:SS Format"""
            total_seconds = int(round(seconds))
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @log_function(log_performance=True)
    def transcribe_audio(self, audio_path: Path, language: Optional[str] = None) -> Result[Tuple[str, str], CoreError]:
        """
        Transkribiert Audio-Datei zu Text
        
        Args:
            audio_path: Pfad zur Audio-Datei
            language: Optional: Sprache forcieren (None = auto-detect)
            
        Returns:
            Ok((transcript, detected_language)): Transcript und erkannte Sprache
            Err: Transcription-Fehler
        """
        if not self.model:
            load_result = self._load_model()
            if isinstance(load_result, Err):
                return load_result
        
        try:
            with log_feature("audio_transcription") as feature:
                feature.add_metric("audio_file", str(audio_path))
                feature.add_metric("file_size_mb", round(audio_path.stat().st_size / (1024 * 1024), 2))
                feature.add_metric("forced_language", language or "auto-detect")
                
                self.logger.info(
                    f"🎙️ Starting transcription",
                    extra={
                        'audio_file': audio_path.name,
                        'file_size_mb': round(audio_path.stat().st_size / (1024 * 1024), 2),
                        'language': language or 'auto-detect',
                        'model': self.config.whisper.model,
                        'device': self.config.whisper.device
                    }
                )
                
                # Thread-safe model access
                with self.model_lock:
                    # Transcribe with faster-whisper (simplified punctuation)
                    segments, info = self.model.transcribe(
                        str(audio_path),
    
                        # === QUALITÄTS-PARAMETER (hoher Einfluss) ===
                        language=language,
                        beam_size=8,                           # ↑ Erhöht von 5 → 8 (bessere Qualität)
                        best_of=5,                             # Beibehalten (wichtig für Qualität)
                        temperature=0.0,                       # Beibehalten (deterministische Ausgabe)
                        compression_ratio_threshold=2.4,       # Beibehalten (Anti-Halluzination)
                        log_prob_threshold=-1.0,               # Beibehalten (Qualitätskontrolle)
    
                        # === SPEED-OPTIMIERUNGEN ===
                        no_speech_threshold=0.7,               # ↑ Erhöht von 0.6 → 0.7 (striktere Stille)
                        condition_on_previous_text=False,      # Beibehalten (robuster + schneller)
                        word_timestamps=False,                 # Beibehalten (Speed)
    
                        # === NEUE SPEED-FEATURES ===
                        vad_filter=True,                       # ✅ NEU: Entfernt Stille vor Transkription
                        vad_parameters={                       # ✅ NEU: Optimierte VAD-Konfiguration
                            "threshold": 0.5,
                            "min_speech_duration_ms": 200,
                            "min_silence_duration_ms": 1500,
                            "speech_pad_ms": 300
                        },
                        initial_prompt=None                    # Kann später optimiert werden
                    )                
                
                # Collect all segments mit Timestamps
                transcript_parts = []
                total_duration = 0.0
                
                for segment in segments:
                    # Timestamp formatieren
                    timestamp = self._format_timestamp(segment.start)
                    segment_text = segment.text.strip()
                    
                    # Nur nicht-leere Segmente hinzufügen
                    if segment_text:
                        formatted_segment = f"[{timestamp}] {segment_text}"
                        transcript_parts.append(formatted_segment)
                    
                    total_duration = max(total_duration, segment.end)
                
                # Combine transcript mit Zeilenumbrüchen
                full_transcript = "\n".join(transcript_parts)
                detected_language = info.language
                
                feature.add_metric("transcript_length", len(full_transcript))
                feature.add_metric("detected_language", detected_language)
                feature.add_metric("language_probability", round(info.language_probability, 3))
                feature.add_metric("duration_seconds", round(total_duration, 1))
                
                # Enhanced transcription logging
                transcription_info = (
                    f"✅ Transcription completed successfully:\n"
                    f"  🎯 Language: {detected_language} (confidence: {info.language_probability:.3f})\n"
                    f"  📝 Transcript length: {len(full_transcript)} characters\n"
                    f"  ⏱️ Audio duration: {total_duration:.1f} seconds\n"
                    f"  🔧 Model: {self.config.whisper.model} on {self.config.whisper.device}\n"
                    f"  📊 Processing rate: {len(full_transcript) / total_duration:.1f} chars/sec\n"
                    f"  🎙️ First 200 chars: {full_transcript[:200]}..."
                )
                
                self.logger.info(transcription_info)
                
                if not full_transcript.strip():
                    context = ErrorContext.create(
                        "transcribe_audio",
                        input_data={
                            'audio_file': str(audio_path),
                            'detected_language': detected_language,
                            'duration': total_duration
                        },
                        suggestions=[
                            "Check if audio contains speech",
                            "Verify audio quality",
                            "Try different model size",
                            "Check audio format compatibility"
                        ]
                    )
                    return Err(CoreError("No transcript generated (silent audio?)", context))
                
                return Ok((full_transcript, detected_language))        
        except FileNotFoundError:
            context = ErrorContext.create(
                "transcribe_audio",
                input_data={'audio_path': str(audio_path)},
                suggestions=["Check if audio file exists", "Verify file path"]
            )
            return Err(CoreError(f"Audio file not found: {audio_path}", context))
        
        except Exception as e:
            context = ErrorContext.create(
                "transcribe_audio",
                input_data={
                    'audio_path': str(audio_path),
                    'error_type': type(e).__name__,
                    'error': str(e)
                },
                suggestions=[
                    "Check audio file format (mp3, wav, etc.)",
                    "Verify CUDA/GPU availability",
                    "Try CPU mode if GPU fails",
                    "Check available memory",
                    "Verify faster-whisper installation"
                ]
            )
            return Err(CoreError(f"Transcription failed: {e}", context))

# =============================================================================
# MOCK TRANSCRIPTION ENGINE (für Testing ohne GPU)
# =============================================================================

class MockTranscriptionEngine:
    """Mock-Engine für Testing ohne faster-whisper/GPU"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger("MockWhisperEngine")
        
        self.logger.warning("Using MOCK transcription engine - install faster-whisper for real transcription")
    
    @log_function(log_performance=True)
    def transcribe_audio(self, audio_path: Path, language: Optional[str] = None) -> Result[Tuple[str, str], CoreError]:
        """Mock-Transkription für Testing"""
        try:
            # Simulate processing time
            file_size_mb = audio_path.stat().st_size / (1024 * 1024) if audio_path.exists() else 1.0
            processing_time = max(0.5, file_size_mb * 0.1)  # ~0.1s per MB
            time.sleep(processing_time)
            
            # Generate mock transcript
            mock_transcript = (
                f"Mock-Transkript für Audio-Datei '{audio_path.name}'. "
                f"Dies ist ein Beispiel-Transkript für Testing der Pipeline-Funktionalität. "
                f"In einem echten System würde hier das Ergebnis von faster-whisper stehen. "
                f"Das Video behandelt technische Themen und enthält relevante Informationen "
                f"für die nachgelagerte Analyse. Die Qualität ist gut und der Inhalt ist verständlich."
            )
            
            detected_language = language or "deutsch"
            
            self.logger.info(
                f"🎭 Mock transcription completed",
                extra={
                    'audio_file': audio_path.name,
                    'transcript_length': len(mock_transcript),
                    'detected_language': detected_language,
                    'processing_time': round(processing_time, 2),
                    'mock_mode': True
                }
            )
            
            return Ok((mock_transcript, detected_language))
            
        except Exception as e:
            context = ErrorContext.create(
                "mock_transcribe_audio",
                input_data={'audio_path': str(audio_path)},
                suggestions=["Check file path", "Install faster-whisper for real transcription"]
            )
            return Err(CoreError(f"Mock transcription failed: {e}", context))

# =============================================================================
# TRANSCRIPTION WORKER
# =============================================================================

def create_transcription_engine(config: AppConfig) -> WhisperTranscriptionEngine:
    """Factory für Transcription-Engine (real oder mock)"""
    if FASTER_WHISPER_AVAILABLE and config.whisper.enabled:
        return WhisperTranscriptionEngine(config)
    else:
        return MockTranscriptionEngine(config)

# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def transcribe_process_object(process_obj: ProcessObject, config: AppConfig) -> Result[ProcessObject, CoreError]:
    """
    Standalone-Funktion für ProcessObject-Transkription
    
    Args:
        process_obj: ProcessObject mit temp_audio_path
        config: App-Konfiguration
        
    Returns:
        Ok(ProcessObject): ProcessObject mit transcript + sprache
        Err: Transcription-Fehler
    """
    if not process_obj.temp_audio_path:
        context = ErrorContext.create(
            "transcribe_process_object",
            input_data={'video_title': process_obj.titel},
            suggestions=["Ensure audio download completed", "Check pipeline order"]
        )
        return Err(CoreError("No audio file path in ProcessObject", context))
    
    if not process_obj.temp_audio_path.exists():
        context = ErrorContext.create(
            "transcribe_process_object",
            input_data={
                'video_title': process_obj.titel,
                'audio_path': str(process_obj.temp_audio_path)
            },
            suggestions=["Check audio download success", "Verify file permissions"]
        )
        return Err(CoreError(f"Audio file not found: {process_obj.temp_audio_path}", context))
    
    # Create transcription engine
    engine = create_transcription_engine(config)
    
    # Transcribe audio
    forced_language = config.whisper.language if hasattr(config.whisper, 'language') else None
    transcription_result = engine.transcribe_audio(process_obj.temp_audio_path, forced_language)
    
    if isinstance(transcription_result, Err):
        return transcription_result
    
    transcript, detected_language = unwrap_ok(transcription_result)
    
    # Update ProcessObject
    process_obj.transkript = transcript
    print(transcript)
    process_obj.sprache = detected_language
    process_obj.update_stage("transcribed")
    
    return Ok(process_obj)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    from yt_analyzer_config import SecureConfigManager
    from datetime import datetime, time as dt_time
    
    # Setup
    setup_logging("transcription_test", "DEBUG")
    
    # Test configuration
    config_manager = SecureConfigManager()
    config_result = config_manager.load_config()
    
    if isinstance(config_result, Ok):
        config = unwrap_ok(config_result)
        
        # Create test ProcessObject
        test_obj = ProcessObject(
            titel="Test Video",
            kanal="Test Channel", 
            länge=dt_time(0, 5, 30),
            upload_date=datetime.now()
        )
        
        # Mock audio file path (for testing)
        test_obj.temp_audio_path = Path("/tmp/test_audio.mp3")
        
        # Test transcription (will use mock if faster-whisper not available)
        result = transcribe_process_object(test_obj, config)
        
        if isinstance(result, Ok):
            transcribed_obj = unwrap_ok(result)
            print(f"✅ Transcription successful:")
            print(f"  Language: {transcribed_obj.sprache}")
            print(f"  Transcript: {transcribed_obj.transkript[:200]}...")
        else:
            error = unwrap_err(result)
            print(f"❌ Transcription failed: {error.message}")
    else:
        print("❌ Config loading failed")
