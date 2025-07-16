"""
Whisper Service - faster-whisper Integration für Audio-Transkription
"""

import torch
from faster_whisper import WhisperModel
from typing import Optional, Dict, Any, BinaryIO
from io import BytesIO
import tempfile
from pathlib import Path
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
            
            # Model-Parameter (optimiert für RTX 3070Ti)
            compute_type = "float16" if self._device == "cuda" else "int8"
            
            self._model = WhisperModel(
                "large-v3",
                device=self._device,
                compute_type=compute_type,
                download_root=None,  # Default download location
                local_files_only=False  # Allow download if not present
            )
            
            self._model_loaded = True
            logger.info(f"Whisper Large-v3 geladen ({self._device}, {compute_type})")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Whisper Models: {e}")
            self._model = None
            self._model_loaded = False
            return False
            
    def transcribe_audio(self, audio_buffer: BytesIO) -> Optional[str]:
        """Audio-Buffer zu Text transkribieren"""
        try:
            # Model lazy loading
            if not self._load_model():
                raise Exception("Whisper Model konnte nicht geladen werden")
                
            # Audio-Buffer zu temporärer Datei (faster-whisper benötigt Datei-Pfad)
            temp_audio_file = self._save_buffer_to_temp_file(audio_buffer)
            
            if not temp_audio_file:
                raise Exception("Audio-Buffer konnte nicht als Datei gespeichert werden")
                
            # Transkription ausführen
            logger.info("Starte Whisper-Transkription...")
            
            segments, info = self._model.transcribe(
                str(temp_audio_file),
                language=None,  # Auto-Detection
                beam_size=5,
                best_of=5,
                word_timestamps=False,  # Für Performance
                vad_filter=True,  # Voice Activity Detection
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=0.5
                )
            )
            
            # Segmente zu vollständigem Text zusammenfügen
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text.strip())
                
            full_transcript = " ".join(transcript_parts).strip()
            
            # Cleanup
            temp_audio_file.unlink()
            
            # Validierung
            if not full_transcript or len(full_transcript) < 10:
                raise Exception("Transkription zu kurz oder leer")
                
            logger.info(f"Transkription erfolgreich: {len(full_transcript)} Zeichen")
            logger.info(f"Erkannte Sprache: {info.language} (Confidence: {info.language_probability:.2f})")
            
            return full_transcript
            
        except Exception as e:
            logger.error(f"Fehler bei Whisper-Transkription: {e}")
            return None
            
    def _save_buffer_to_temp_file(self, audio_buffer: BytesIO) -> Optional[Path]:
        """BytesIO zu temporärer Datei speichern"""
        try:
            # Reset buffer position
            audio_buffer.seek(0)
            
            # Temporäre Datei erstellen
            temp_file = Path(self._temp_dir) / f"audio_{id(audio_buffer)}.wav"
            
            with open(temp_file, 'wb') as f:
                f.write(audio_buffer.read())
                
            # Buffer position für weitere Verwendung zurücksetzen
            audio_buffer.seek(0)
            
            if temp_file.exists() and temp_file.stat().st_size > 0:
                logger.debug(f"Audio-Buffer als Datei gespeichert: {temp_file}")
                return temp_file
            else:
                raise Exception("Datei ist leer oder wurde nicht erstellt")
                
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Audio-Buffers: {e}")
            return None
            
    def cleanup_model(self):
        """Model aus Memory freigeben (nach einzelnem Video)"""
        if self._model:
            try:
                # GPU Memory freigeben
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
        return torch.cuda.is_available() or True  # CPU fallback immer verfügbar
        
    # API für zukünftige Batch-Verarbeitung
    def start_batch_mode(self):
        """Batch-Modus: Model im Memory behalten"""
        logger.info("Batch-Modus aktiviert - Model wird im Memory behalten")
        self._batch_mode = True
        self._load_model()  # Model vorladen
        
    def end_batch_mode(self):
        """Batch-Modus beenden: Model freigeben"""
        logger.info("Batch-Modus beendet")
        self._batch_mode = False
        self.cleanup_model()
        
    def transcribe_batch(self, audio_buffers: list) -> list:
        """Batch-Transkription (zukünftige Erweiterung)"""
        self.start_batch_mode()
        
        results = []
        for i, buffer in enumerate(audio_buffers):
            logger.info(f"Transkribiere Audio {i+1}/{len(audio_buffers)}")
            result = self.transcribe_audio(buffer)
            results.append(result)
            
        self.end_batch_mode()
        return results
        
    def __del__(self):
        """Cleanup beim Beenden"""
        self.cleanup_model()
        
        # Temporäre Dateien aufräumen
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


def test_whisper_service():
    """Test des Whisper-Services"""
    service = get_whisper_service()
    
    print("🔍 Testing Whisper Service...")
    print(f"Model Info: {service.get_model_info()}")
    print(f"Ready: {service.is_ready()}")
    
    # Test nur wenn CUDA verfügbar (sonst zu langsam für Test)
    if torch.cuda.is_available():
        print("✅ CUDA verfügbar - Service bereit für Transkription")
    else:
        print("⚠️ Nur CPU verfügbar - Service funktional aber langsam")
        
    print("✅ Whisper Service Test abgeschlossen")


if __name__ == "__main__":
    test_whisper_service()
