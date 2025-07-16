"""
Whisper Worker - QThread für Audio-Transkription
"""

from PySide6.QtCore import QThread, Signal
from typing import Optional
from io import BytesIO
from loguru import logger

from services.whisper_service import get_whisper_service


class WhisperWorker(QThread):
    """Worker Thread für Whisper Audio-Transkription"""
    
    # Signals für UI Communication
    progress_updated = Signal(str)           # status_message
    model_loading = Signal()                 # Model wird geladen (einmalig)
    model_ready = Signal(dict)               # Model-Info für UI
    transcription_started = Signal()         # Transkription gestartet
    transcript_ready = Signal(str)           # Fertiges Transkript
    error_occurred = Signal(str)             # Fehler-Nachricht
    finished = Signal()                      # Worker beendet
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_buffer = None
        self._should_stop = False
        self.whisper_service = get_whisper_service()
        
    def set_audio_buffer(self, audio_buffer: BytesIO):
        """Audio-Buffer für Transkription setzen"""
        self.audio_buffer = audio_buffer
        self._should_stop = False
        
    def stop_transcription(self):
        """Transkription abbrechen"""
        self._should_stop = True
        
    def run(self):
        """Haupt-Transkriptionsprozess"""
        try:
            logger.info("Whisper Worker gestartet")
            
            if not self.audio_buffer:
                self.error_occurred.emit("❌ Kein Audio-Buffer vorhanden")
                return
                
            if self._should_stop:
                return
                
            # 1. Model-Status prüfen und ggf. laden
            model_info = self.whisper_service.get_model_info()
            
            if not model_info['loaded']:
                self.progress_updated.emit("🎤 Lade Whisper Large-v3 Model...")
                self.model_loading.emit()
                
                # Model wird beim ersten transcribe_audio() automatisch geladen
                # Hier nur UI-Feedback
                
            # 2. Model-Info an UI senden
            self.model_ready.emit(model_info)
            
            if self._should_stop:
                return
                
            # 3. Transkription starten
            self.progress_updated.emit("🎤 Transkribiere Audio...")
            self.transcription_started.emit()
            
            # Transkription ausführen (kann mehrere Minuten dauern)
            transcript = self.whisper_service.transcribe_audio(self.audio_buffer)
            
            if self._should_stop:
                return
                
            # 4. Ergebnis validieren
            if not transcript:
                self.error_occurred.emit("❌ Transkription fehlgeschlagen - kein Text erkannt")
                return
                
            if len(transcript.strip()) < 10:
                self.error_occurred.emit("❌ Transkription zu kurz - möglicherweise kein Sprachinhalt")
                return
                
            # 5. Erfolgreiche Transkription
            self.progress_updated.emit("✅ Transkription abgeschlossen")
            self.transcript_ready.emit(transcript)
            
            logger.info(f"Whisper-Transkription erfolgreich: {len(transcript)} Zeichen")
            
        except Exception as e:
            error_msg = f"❌ Whisper-Fehler: {str(e)}"
            logger.error(f"Fehler im Whisper Worker: {e}")
            self.error_occurred.emit(error_msg)
            
        finally:
            # Model nach einzelnem Video freigeben (wie gewünscht)
            try:
                self.whisper_service.cleanup_model()
                logger.debug("Whisper Model nach Transkription freigegeben")
            except Exception as e:
                logger.warning(f"Fehler beim Model-Cleanup: {e}")
                
            self.finished.emit()
            logger.info("Whisper Worker beendet")


class WhisperManager:
    """Manager für Whisper-Operationen (zukünftige Erweiterungen)"""
    
    def __init__(self, parent=None):
        self.current_worker = None
        self.whisper_service = get_whisper_service()
        
    def start_transcription(self, audio_buffer: BytesIO) -> WhisperWorker:
        """Transkription starten"""
        # Vorherigen Worker stoppen
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop_transcription()
            self.current_worker.wait(5000)  # 5 Sekunden warten
            
        # Neuen Worker erstellen
        self.current_worker = WhisperWorker()
        self.current_worker.set_audio_buffer(audio_buffer)
        
        return self.current_worker
        
    def stop_current_transcription(self):
        """Aktuelle Transkription stoppen"""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop_transcription()
            
    def get_service_info(self) -> dict:
        """Whisper-Service Informationen"""
        return self.whisper_service.get_model_info()
        
    def is_ready(self) -> bool:
        """Service bereit für Transkription?"""
        return self.whisper_service.is_ready()
        
    def cleanup(self):
        """Cleanup bei Beendigung"""
        self.stop_current_transcription()
        self.whisper_service.cleanup_model()


# Test-Funktion
def test_whisper_worker():
    """Test des Whisper Workers (nur CUDA-Check)"""
    from services.whisper_service import get_whisper_service
    
    service = get_whisper_service()
    info = service.get_model_info()
    
    print("🎤 Whisper Worker Test:")
    print(f"   Device: {info['device']}")
    print(f"   CUDA Available: {info['cuda_available']}")
    print(f"   GPU: {info['gpu_name']}")
    print(f"   Ready: {service.is_ready()}")
    
    if info['cuda_available']:
        print("✅ Whisper Worker bereit für GPU-Transkription")
    else:
        print("⚠️ Whisper Worker läuft auf CPU (langsamer)")
        
    print("✅ Whisper Worker Test abgeschlossen")


if __name__ == "__main__":
    test_whisper_worker()
