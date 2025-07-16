
"""
Whisper Service - faster-whisper Integration für Audio-Transkription
"""

import torch
from faster_whisper import WhisperModel
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import uuid
from loguru import logger


class WhisperService:
    """Service für Audio-Transkription mit faster-whisper Large-v3"""
    
    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._device = self._detect_device()
        self._temp_dir = tempfile.mkdtemp(prefix="whisper_analyzer_")
        
        logger.info(f"Whisper Service initialisiert (Device: {self._device})")
        
    def _detect_device(self) -> str:
        """CUDA-Verfügbarkeit prüfen"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA verfügbar: {gpu_name}")
            return "cuda"
        else:
            logger.warning("CUDA nicht verfügbar - Fallback auf CPU")
            return "cpu"
            
    def _load_model(self) -> bool:
        """Model lazy loading (nur beim ersten Aufruf)"""
        if self._model_loaded:
            return True
            
        try:
            logger.info("Lade Whisper Large-v3 Model...")
            
            compute_type = "float16" if self._device == "cuda" else "int8"
            
            self._model = WhisperModel(
                "large-v3",
                device=self._device,
                compute_type=compute_type,
                download_root=None,
                local_files_only=False
            )
            
            self._model_loaded = True
            logger.info(f"Whisper Large-v3 geladen ({self._device}, {compute_type})")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Whisper Models: {e}")
            self._model = None
            self._model_loaded = False
            return False
            
    def transcribe_audio_file(self, audio_file_path: Path) -> Optional[str]:
        """Audio-Datei transkribieren mit Locale-Fix"""
        try:
            # Model lazy loading
            if not self._load_model():
                raise Exception("Whisper Model konnte nicht geladen werden")
            
            # Datei validieren
            if not audio_file_path.exists():
                raise Exception(f"Audio-Datei existiert nicht: {audio_file_path}")
            
            if audio_file_path.stat().st_size < 1000:
                raise Exception(f"Audio-Datei zu klein: {audio_file_path.stat().st_size} bytes")
            
            logger.info(f"Starte Whisper-Transkription: {audio_file_path}")
        
            # LÖSUNG: Locale temporär auf ASCII setzen
            import locale
            import os
        
            # Aktuelle Locale sichern
            current_locale = locale.getlocale()
            current_env = os.environ.copy()
        
            try:
                # ASCII-Locale für faster-whisper erzwingen
                logger.info("DEBUG: Setze ASCII-Locale für faster-whisper...")
            
                os.environ['LC_ALL'] = 'C'
                os.environ['LANG'] = 'C'
                os.environ['LC_CTYPE'] = 'C'
            
                # Locale neu setzen
                locale.setlocale(locale.LC_ALL, 'C')
            
                logger.info(f"DEBUG: Neue Locale: {locale.getlocale()}")
            
                # Transkription mit ASCII-Locale
                segments, info = self._model.transcribe(
                    str(audio_file_path),
                    language=None,  # Auto-Detection
                    beam_size=5,
                    best_of=5,
                    word_timestamps=False,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        threshold=0.5
                    )
                )
            
                logger.info("DEBUG: Transkription mit ASCII-Locale erfolgreich!")
            
            finally:
                # Locale und Umgebung wiederherstellen
                try:
                    os.environ.clear()
                    os.environ.update(current_env)
                    locale.setlocale(locale.LC_ALL, current_locale)
                    logger.info("DEBUG: Originale Locale wiederhergestellt")
                except Exception as restore_error:
                    logger.warning(f"DEBUG: Locale-Wiederherstellung fehlgeschlagen: {restore_error}")
        
            # Segmente verarbeiten
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text.strip())
            
            full_transcript = " ".join(transcript_parts).strip()
        
            # Validierung
            if not full_transcript or len(full_transcript) < 10:
                raise Exception("Transkription zu kurz oder leer")
            
            logger.info(f"Transkription erfolgreich: {len(full_transcript)} Zeichen")
            logger.info(f"Erkannte Sprache: {info.language} (Confidence: {info.language_probability:.2f})")
        
            return full_transcript
        
        except Exception as e:
            logger.error(f"Fehler bei Whisper-Transkription: {e}")
            return None
                
    def cleanup_model(self):
        """Model aus Memory freigeben"""
        if self._model:
            try:
                if self._device == "cuda":
                    torch.cuda.empty_cache()
                    
                self._model = None
                self._model_loaded = False
                logger.info("Whisper Model aus Memory freigegeben")
                
            except Exception as e:
                logger.warning(f"Fehler beim Model-Cleanup: {e}")
                
    def get_model_info(self) -> Dict[str, Any]:
        """Model-Informationen für UI"""
        return {
            "model_name": "large-v3",
            "device": self._device,
            "loaded": self._model_loaded,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
    def is_ready(self) -> bool:
        """Service bereit für Transkription?"""
        return torch.cuda.is_available() or True
        
    def __del__(self):
        """Cleanup beim Beenden"""
        self.cleanup_model()
        
        try:
            import shutil
            if Path(self._temp_dir).exists():
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Whisper Temp-Ordner gelöscht: {self._temp_dir}")
        except Exception as e:
            logger.warning(f"Fehler beim Temp-Cleanup: {e}")


# Globale Service-Instanz
whisper_service = WhisperService()


def get_whisper_service() -> WhisperService:
    """Convenience-Funktion für Whisper-Service Zugriff"""
    return whisper_service
