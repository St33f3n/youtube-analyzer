"""
Workers - QThread Workers für Download und Whisper
Vollständig überarbeitet mit Result-Types und vollständigen Type-Hints
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread
from PySide6.QtCore import Signal

from services.download import get_download_service
from services.whisper import get_whisper_service
from yt_types import AudioMetadata
from yt_types import Err
from yt_types import Ok
from yt_types import TranscriptionResult
from yt_types import VideoMetadata
from utils.logging import ComponentLogger
from utils.logging import log_function_calls


class DownloadWorker(QThread):
    """Worker Thread für YouTube Downloads mit Result-Types"""
    
    # Signals für UI Communication
    progress_updated = Signal(int, str)    # progress_percent, status_message
    video_info_ready = Signal(object)      # VideoMetadata
    audio_ready = Signal(object)           # AudioMetadata
    video_ready = Signal(str)              # video_file_path
    error_occurred = Signal(str)           # error_message
    finished = Signal()                    # download_complete
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        
        self.logger = ComponentLogger("DownloadWorker")
        self.download_service = get_download_service()
        
        # Parameters
        self.url = ""
        self.download_audio = False
        self.download_video = False
        self._should_stop = False
        
        self.logger.debug("Download worker initialized")
    
    @log_function_calls
    def set_download_params(
        self,
        url: str,
        download_audio: bool = False,
        download_video: bool = False,
    ) -> None:
        """Download-Parameter setzen"""
        self.url = url
        self.download_audio = download_audio
        self.download_video = download_video
        self._should_stop = False
        
        self.logger.info(
            "Download parameters set",
            url=url,
            download_audio=download_audio,
            download_video=download_video,
        )
    
    def stop_download(self) -> None:
        """Download abbrechen"""
        self._should_stop = True
        self.logger.info("Download stop requested")
    
    def run(self) -> None:
        """Haupt-Download-Prozess"""
        try:
            self.logger.info(
                "Download worker started",
                url=self.url,
                audio=self.download_audio,
                video=self.download_video,
            )
            
            # 1. Video-Informationen abrufen (5%)
            self.progress_updated.emit(5, "📋 Video-Informationen abrufen...")
            
            info_result = self.download_service.get_video_info(self.url)
            
            if isinstance(info_result, Err):
                self.error_occurred.emit(f"❌ Video-Info Fehler: {info_result.error.message}")
                return
            
            if self._should_stop:
                return
            
            video_info = info_result.value
            self.video_info_ready.emit(video_info)
            
            self.logger.info(
                "Video info retrieved",
                video_id=video_info.id,
                title=video_info.title,
                duration=video_info.duration,
            )
            
            # 2. Audio-Download (optional, 10% - 60%)
            if self.download_audio:
                self.progress_updated.emit(10, "🎵 Audio wird heruntergeladen...")
                
                def audio_progress(progress: int, message: str) -> None:
                    if self._should_stop:
                        return
                    
                    # Mappe auf 10-60% Bereich
                    mapped_progress = 10 + int((progress / 100) * 50)
                    self.progress_updated.emit(mapped_progress, f"🎵 {message}")
                
                audio_result = self.download_service.download_audio(
                    self.url,
                    progress_callback=audio_progress,
                )
                
                if isinstance(audio_result, Err):
                    self.error_occurred.emit(f"❌ Audio-Download Fehler: {audio_result.error.message}")
                    return
                
                if self._should_stop:
                    return
                
                audio_metadata = audio_result.value
                self.progress_updated.emit(60, "✅ Audio-Download abgeschlossen")
                self.audio_ready.emit(audio_metadata)
                
                self.logger.info(
                    "Audio download completed",
                    file_path=str(audio_metadata.file_path),
                    file_size=audio_metadata.file_size,
                )
            
            # 3. Video-Download (optional, 70% - 95%)
            if self.download_video:
                self.progress_updated.emit(70, "📹 Video wird heruntergeladen...")
                
                def video_progress(progress: int, message: str) -> None:
                    if self._should_stop:
                        return
                    
                    # Mappe auf 70-95% Bereich
                    mapped_progress = 70 + int((progress / 100) * 25)
                    self.progress_updated.emit(mapped_progress, f"📹 {message}")
                
                video_result = self.download_service.download_video(
                    self.url,
                    progress_callback=video_progress,
                )
                
                if isinstance(video_result, Err):
                    self.error_occurred.emit(f"❌ Video-Download Fehler: {video_result.error.message}")
                    return
                
                if self._should_stop:
                    return
                
                video_path = video_result.value
                self.progress_updated.emit(95, "✅ Video-Download abgeschlossen")
                self.video_ready.emit(str(video_path))
                
                self.logger.info(
                    "Video download completed",
                    video_path=str(video_path),
                    file_size=video_path.stat().st_size,
                )
            
            # 4. Abschluss (100%)
            if not self._should_stop:
                self.progress_updated.emit(100, "🎉 Download vollständig!")
                
                self.logger.info(
                    "Download worker completed successfully",
                    url=self.url,
                    audio_downloaded=self.download_audio,
                    video_downloaded=self.download_video,
                )
        
        except Exception as e:
            error_msg = f"❌ Unerwarteter Download-Fehler: {str(e)}"
            self.logger.error(
                "Download worker failed with exception",
                error=e,
                url=self.url,
            )
            self.error_occurred.emit(error_msg)
        
        finally:
            self.finished.emit()
            self.logger.info("Download worker finished")


# =============================================================================
# WORKER MANAGER (für zukünftige Erweiterungen)
# =============================================================================

class WorkerManager:
    """Manager für Worker-Koordination und Cleanup"""
    
    def __init__(self) -> None:
        self.logger = ComponentLogger("WorkerManager")
        self.active_workers: list[QThread] = []
    
    def register_worker(self, worker: QThread) -> None:
        """Worker registrieren für Cleanup"""
        self.active_workers.append(worker)
        
        # Auto-cleanup wenn Worker fertig ist
        worker.finished.connect(lambda: self.unregister_worker(worker))
        
        self.logger.debug(
            "Worker registered",
            worker_type=type(worker).__name__,
            active_workers=len(self.active_workers),
        )
    
    def unregister_worker(self, worker: QThread) -> None:
        """Worker aus Registry entfernen"""
        if worker in self.active_workers:
            self.active_workers.remove(worker)
            
            self.logger.debug(
                "Worker unregistered",
                worker_type=type(worker).__name__,
                active_workers=len(self.active_workers),
            )
    
    def stop_all_workers(self, timeout: int = 5000) -> None:
        """Alle aktiven Worker stoppen"""
        for worker in self.active_workers.copy():
            try:
                # Worker-spezifische Stop-Methoden
                if hasattr(worker, 'stop_download'):
                    worker.stop_download()
                elif hasattr(worker, 'stop_transcription'):
                    worker.stop_transcription()
                
                # Auf Beendigung warten
                if worker.isRunning():
                    worker.wait(timeout)
                
                self.logger.info(
                    "Worker stopped",
                    worker_type=type(worker).__name__,
                )
            
            except Exception as e:
                self.logger.error(
                    "Failed to stop worker",
                    error=e,
                    worker_type=type(worker).__name__,
                )
        
        self.active_workers.clear()
        self.logger.info("All workers stopped")
    
    def get_active_worker_count(self) -> int:
        """Anzahl aktiver Worker"""
        return len(self.active_workers)
    
    def get_worker_info(self) -> dict:
        """Informationen über aktive Worker"""
        return {
            'total_workers': len(self.active_workers),
            'worker_types': [type(w).__name__ for w in self.active_workers],
            'running_workers': sum(1 for w in self.active_workers if w.isRunning()),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_download_worker() -> DownloadWorker:
    """Factory für Download-Worker"""
    return DownloadWorker()


def create_whisper_worker() -> WhisperWorker:
    """Factory für Whisper-Worker"""
    return WhisperWorker()


def create_worker_manager() -> WorkerManager:
    """Factory für Worker-Manager"""
    return WorkerManager()


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def test_download_worker() -> None:
    """Test-Funktion für Download-Worker"""
    from youtube_analyzer.utils.logging import get_development_config
    from youtube_analyzer.utils.logging import setup_logging
    
    # Setup logging für Test
    setup_logging(get_development_config())
    
    logger = ComponentLogger("DownloadWorkerTest")
    logger.info("Starting download worker test")
    
    # Test-URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # Worker erstellen
    worker = create_download_worker()
    
    # Test-Callbacks
    def on_progress(progress: int, message: str) -> None:
        logger.info(f"Progress: {progress}% - {message}")
    
    def on_video_info(video_info: VideoMetadata) -> None:
        logger.info(f"Video Info: {video_info.title}")
    
    def on_audio_ready(audio_metadata: AudioMetadata) -> None:
        logger.info(f"Audio Ready: {audio_metadata.file_path}")
    
    def on_error(error: str) -> None:
        logger.error(f"Error: {error}")
    
    def on_finished() -> None:
        logger.info("Worker finished")
    
    # Signals verbinden
    worker.progress_updated.connect(on_progress)
    worker.video_info_ready.connect(on_video_info)
    worker.audio_ready.connect(on_audio_ready)
    worker.error_occurred.connect(on_error)
    worker.finished.connect(on_finished)
    
    # Parameter setzen und starten
    worker.set_download_params(test_url, download_audio=True, download_video=False)
    
    logger.info("✅ Download worker test setup completed")
    # Note: In echtem Test würde worker.start() aufgerufen und auf finished gewartet


def test_whisper_worker() -> None:
    """Test-Funktion für Whisper-Worker"""
    from youtube_analyzer.utils.logging import get_development_config
    from youtube_analyzer.utils.logging import setup_logging
    
    # Setup logging für Test
    setup_logging(get_development_config())
    
    logger = ComponentLogger("WhisperWorkerTest")
    logger.info("Starting whisper worker test")
    
    # Mock AudioMetadata
    mock_audio = AudioMetadata(
        file_path=Path("/tmp/test_audio.wav"),
        file_size=1000000,
        format="wav",
        duration=60.0,
        sample_rate=44100,
        channels=1,
    )
    
    # Worker erstellen
    worker = create_whisper_worker()
    
    # Test-Callbacks
    def on_progress(message: str) -> None:
        logger.info(f"Progress: {message}")
    
    def on_model_ready(model_info: dict) -> None:
        logger.info(f"Model Ready: {model_info}")
    
    def on_transcript(transcription: TranscriptionResult) -> None:
        logger.info(f"Transcript: {len(transcription.text)} chars")
    
    def on_error(error: str) -> None:
        logger.error(f"Error: {error}")
    
    def on_finished() -> None:
        logger.info("Worker finished")
    
    # Signals verbinden
    worker.progress_updated.connect(on_progress)
    worker.model_ready.connect(on_model_ready)
    worker.transcript_ready.connect(on_transcript)
    worker.error_occurred.connect(on_error)
    worker.finished.connect(on_finished)
    
    # Parameter setzen
    worker.set_audio_metadata(mock_audio)
    
    logger.info("✅ Whisper worker test setup completed")
    # Note: In echtem Test würde worker.start() aufgerufen und auf finished gewartet


if __name__ == "__main__":
    test_download_worker()
    test_whisper_worker()
