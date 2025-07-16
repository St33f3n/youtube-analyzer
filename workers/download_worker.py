"""
Download Worker - QThread für YouTube Downloads
"""

from PySide6.QtCore import QThread, Signal, QObject
from typing import Optional, Dict, Any
from pathlib import Path
from io import BytesIO
from loguru import logger

from services.download_service import get_download_service


class DownloadWorker(QThread):
    """Worker Thread für YouTube Downloads"""
    
    # Signals für UI Communication
    progress_updated = Signal(int, str)  # progress_percent, status_message
    video_info_ready = Signal(dict)      # video_metadata
    audio_ready = Signal(object)         # BytesIO audio_buffer
    video_ready = Signal(object)         # Path to video_file
    error_occurred = Signal(str)         # error_message
    finished = Signal()                  # download_complete
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.url = ""
        self.download_audio = False
        self.download_video = False
        self.download_service = get_download_service()
        self._should_stop = False
        
    def set_download_params(self, url: str,download_audio: bool = True, download_video: bool = False):
        """Download-Parameter setzen"""
        self.url = url
        self.download_audio = download_audio
        self.download_video = download_video
        self._should_stop = False
        
    def stop_download(self):
        """Download abbrechen"""
        self._should_stop = True
        
    def run(self):
        """Haupt-Download-Prozess"""
        try:
            logger.info(f"Download Worker gestartet für: {self.url}")
            
            # 1. Video-Informationen abrufen (5%)
            self.progress_updated.emit(5, "📋 Video-Informationen abrufen...")
            video_info = self.download_service.get_video_info(self.url)
            
            if not video_info:
                self.error_occurred.emit("❌ Video-Informationen konnten nicht abgerufen werden")
                return
                
            if self._should_stop:
                return
                
            self.video_info_ready.emit(video_info)
            logger.info(f"Video-Info: {video_info['title']}")

            if self.download_audio:
                # 2. Audio-Download (10% - 60%)
                self.progress_updated.emit(10, "🎵 Audio wird heruntergeladen...")
            
                def audio_progress_callback(progress_data):
                    """Progress-Callback für Audio-Download"""
                    if self._should_stop:
                        return
                    
                    # Audio-Download: 10% - 60% der Gesamt-Progress
                    base_progress = 10
                    max_progress = 60
                
                    if progress_data.get('total_bytes'):
                        downloaded = progress_data.get('downloaded_bytes', 0)
                        total = progress_data['total_bytes']
                        audio_percent = (downloaded / total) * 100
                    else:
                        # Fallback wenn total_bytes unbekannt
                        audio_percent = 50  # Schätzung
                    
                    overall_progress = base_progress + int((audio_percent / 100) * (max_progress - base_progress))
                
                    speed_text = ""
                    if progress_data.get('speed'):
                        speed_mb = progress_data['speed'] / (1024 * 1024)
                        speed_text = f" ({speed_mb:.1f} MB/s)"
                    
                    self.progress_updated.emit(
                        overall_progress, 
                        f"🎵 Audio-Download läuft...{speed_text}"
                    )
            
                audio_buffer = self.download_service.download_audio_to_memory(
                    self.url, 
                    progress_callback=audio_progress_callback
                )
            
                if not audio_buffer:
                    self.error_occurred.emit("❌ Audio-Download fehlgeschlagen")
                    return
                
                if self._should_stop:
                    audio_buffer.close()
                    return
                
                self.progress_updated.emit(60, "✅ Audio-Download abgeschlossen")
                self.audio_ready.emit(audio_buffer)
                logger.info("Audio-Download erfolgreich abgeschlossen")
            
            # 3. Video-Download (optional, 70% - 95%)
            if self.download_video:
                self.progress_updated.emit(70, "📹 Video wird heruntergeladen...")
                
                def video_progress_callback(progress_data):
                    """Progress-Callback für Video-Download"""
                    if self._should_stop:
                        return
                        
                    # Video-Download: 70% - 95% der Gesamt-Progress
                    base_progress = 70
                    max_progress = 95
                    
                    if progress_data.get('total_bytes'):
                        downloaded = progress_data.get('downloaded_bytes', 0)
                        total = progress_data['total_bytes']
                        video_percent = (downloaded / total) * 100
                    else:
                        video_percent = 50  # Schätzung
                        
                    overall_progress = base_progress + int((video_percent / 100) * (max_progress - base_progress))
                    
                    speed_text = ""
                    if progress_data.get('speed'):
                        speed_mb = progress_data['speed'] / (1024 * 1024)
                        speed_text = f" ({speed_mb:.1f} MB/s)"
                        
                    self.progress_updated.emit(
                        overall_progress,
                        f"📹 Video-Download läuft...{speed_text}"
                    )
                
                video_file = self.download_service.download_video_to_file(
                    self.url,
                    progress_callback=video_progress_callback
                )
                
                if not video_file:
                    self.error_occurred.emit("❌ Video-Download fehlgeschlagen")
                    return
                    
                if self._should_stop:
                    return
                    
                self.progress_updated.emit(95, "✅ Video-Download abgeschlossen")
                self.video_ready.emit(video_file)
                logger.info(f"Video-Download erfolgreich: {video_file}")
            
            # 4. Abschluss (100%)
            self.progress_updated.emit(100, "🎉 Download vollständig abgeschlossen!")
            
        except Exception as e:
            logger.error(f"Fehler im Download Worker: {e}")
            self.error_occurred.emit(f"❌ Unerwarteter Fehler: {str(e)}")
            
        finally:
            self.finished.emit()
            logger.info("Download Worker beendet")


class DownloadManager(QObject):
    """Manager für Download-Operationen mit Worker-Koordination"""
    
    # Manager-Level Signals
    download_completed = Signal(dict)    # Komplette Download-Metadaten
    analysis_ready = Signal(object)      # Audio-Buffer für Analyse
    storage_ready = Signal(object, dict) # Video-File + Metadaten für Storage
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_worker = None
        self.video_metadata = {}
        self.audio_buffer = None
        self.video_file = None
        
    def start_audio_download(self, url: str):
        """Nur Audio-Download starten"""
        self._start_download(url,download_video=False)
        
    def start_full_download(self, url: str):
        """Audio + Video Download starten"""
        self._start_download(url, download_video=True)
        
    def request_video_download(self):
        """Video-Download nachträglich anfordern (nach Audio-Analyse)"""
        if self.current_worker and self.current_worker.isRunning():
            logger.warning("Download bereits aktiv")
            return
            
        if not self.video_metadata:
            logger.error("Keine Video-Metadaten für nachträglichen Download")
            return
            
        # Neuer Worker nur für Video
        self._start_download(self.video_metadata.get('webpage_url', ''),download_audio=False, download_video=True)
        
    def _start_download(self, url: str,download_audio: bool , download_video: bool):
        """Internen Download starten"""
        # Vorherigen Worker stoppen
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop_download()
            self.current_worker.wait(3000)  # 3 Sekunden warten
            
        # Neuen Worker erstellen
        self.current_worker = DownloadWorker()
        
        # Signals verbinden
        self.current_worker.video_info_ready.connect(self._on_video_info_ready)
        self.current_worker.audio_ready.connect(self._on_audio_ready)
        self.current_worker.video_ready.connect(self._on_video_ready)
        self.current_worker.finished.connect(self._on_download_finished)
        
        # Download starten
        self.current_worker.set_download_params(url ,download_audio, download_video)
        self.current_worker.start()
        
    def _on_video_info_ready(self, video_info: Dict[str, Any]):
        """Video-Metadaten empfangen"""
        self.video_metadata = video_info
        logger.info(f"Video-Metadaten empfangen: {video_info['title']}")
        
    def _on_audio_ready(self, audio_buffer: BytesIO):
        """Audio-Buffer empfangen"""
        self.audio_buffer = audio_buffer
        self.analysis_ready.emit(audio_buffer)
        logger.info("Audio für Analyse bereit")
        
    def _on_video_ready(self, video_file: Path):
        """Video-Datei empfangen"""
        self.video_file = video_file
        self.storage_ready.emit(video_file, self.video_metadata)
        logger.info("Video für Storage bereit")
        
    def _on_download_finished(self):
        """Download abgeschlossen"""
        complete_data = {
            'metadata': self.video_metadata,
            'audio_buffer': self.audio_buffer,
            'video_file': self.video_file
        }
        self.download_completed.emit(complete_data)
        logger.info("Download-Prozess vollständig abgeschlossen")
        
    def stop_current_download(self):
        """Aktuellen Download stoppen"""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop_download()
            
    def cleanup(self):
        """Cleanup bei Beendigung"""
        self.stop_current_download()
        if self.audio_buffer:
            self.audio_buffer.close()
            
        # Download Service cleanup wird automatisch gemacht
