"""
Download Service - yt-dlp Integration für Audio und Video (Vereinfacht)
"""

import tempfile
import yt_dlp
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from io import BytesIO
from loguru import logger


class DownloadProgressHook:
    """Progress Hook für yt-dlp Downloads"""
    
    def __init__(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.callback = callback
        
    def __call__(self, d: Dict[str, Any]):
        """yt-dlp Progress Hook"""
        if self.callback:
            # Normalisiere Progress-Daten
            progress_data = {
                'status': d.get('status', 'unknown'),
                'filename': d.get('filename', ''),
                'total_bytes': d.get('total_bytes', 0),
                'downloaded_bytes': d.get('downloaded_bytes', 0),
                'speed': d.get('speed', 0),
                'eta': d.get('eta', 0)
            }
            self.callback(progress_data)


class DownloadService:
    """Service für YouTube Downloads mit yt-dlp (vereinfacht)"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="youtube_analyzer_")
        logger.info(f"Download Service initialisiert: {self.temp_dir}")
        
    def __del__(self):
        """Cleanup beim Beenden"""
        self.cleanup_temp_files()
        
    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Hole Video-Informationen ohne Download"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
            # Wichtige Metadaten extrahieren
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', ''),
                'description': info.get('description', ''),
                'webpage_url': info.get('webpage_url', url),
                'id': info.get('id', ''),
                'formats_available': len(info.get('formats', []))
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Video-Info: {e}")
            return None
            
    def download_audio_to_temp_file(self, url: str, progress_callback: Optional[Callable] = None) -> Optional[Path]:
        """Download Audio mit DEBUG für UTF-8 Metadaten"""
        try:
            logger.info(f"Starte Audio-Download (Temp File): {url}")
        
            # DEBUG: URL auf UTF-8 prüfen
            logger.info(f"DEBUG: URL ASCII-safe: {url.isascii()}")
        
            progress_hook = DownloadProgressHook(progress_callback) if progress_callback else None
        
            # Einfacher ASCII Dateiname
            temp_audio_file = Path(self.temp_dir) / f"debug_{hash(url) % 10000}.wav"
        
            # ERWEITERTE yt-dlp Optionen mit Metadaten-Bereinigung
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(temp_audio_file).replace('.wav', '.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '0',
                }],
                # DEBUG: Metadaten explizit entfernen
                'postprocessor_args': [
                    '-map_metadata', '-1',  # Alle Metadaten löschen
                    '-fflags', '+bitexact',  # Deterministische Ausgabe
                    '-avoid_negative_ts', 'make_zero'  # Saubere Timestamps
                ],
                'writethumbnail': False,
                'writeinfojson': False,
                'writedescription': False,
                'writesubtitles': False,
            }
        
            if progress_hook:
                ydl_opts['progress_hooks'] = [progress_hook]
            
            # Download mit yt-dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # WAV-Datei finden
            wav_files = list(Path(self.temp_dir).glob(f"debug_{hash(url) % 10000}.wav"))
        
            if not wav_files:
                raise Exception("WAV-Datei nach Download nicht gefunden")
            
            wav_file = wav_files[0]
        
            if not wav_file.exists() or wav_file.stat().st_size < 1000:
                raise Exception(f"WAV-Datei ungültig: {wav_file}")
            
            # DEBUG: WAV-Datei analysieren
            logger.info(f"DEBUG: WAV-Datei erstellt: {wav_file}")
            logger.info(f"DEBUG: WAV-Datei Größe: {wav_file.stat().st_size} bytes")
        
            # Erste Bytes der WAV-Datei prüfen
            with open(wav_file, 'rb') as f:
                first_bytes = f.read(100)
                logger.info(f"DEBUG: WAV erste 12 bytes: {first_bytes[:12]}")
            
                # KORRIGIERT: Backslash-Problem vermeiden
                utf8_byte = b'\xc3'
                has_utf8 = utf8_byte in first_bytes
                logger.info(f"DEBUG: Enthält 0xC3: {has_utf8}")
            
            logger.info(f"Audio-Download erfolgreich: {wav_file} ({wav_file.stat().st_size} bytes)")
            return wav_file
        
        except Exception as e:
            logger.error(f"Fehler beim Audio-Download: {e}")
            return None
                    
    def download_video_to_file(self, url: str, progress_callback: Optional[Callable] = None) -> Optional[Path]:
        """Download Video zu temporärer Datei"""
        try:
            logger.info(f"Starte Video-Download (File): {url}")
            
            # Temporäre Datei erstellen
            temp_file = Path(self.temp_dir) / f"video_{hash(url) % 10000}.%(ext)s"
            
            # Progress Hook
            progress_hook = DownloadProgressHook(progress_callback) if progress_callback else None
            
            # yt-dlp Optionen für Video
            ydl_opts = {
                'format': 'best[height<=1080]/best',  # Max 1080p für Größe
                'outtmpl': str(temp_file),
                'quiet': True,
                'no_warnings': True,
            }
            
            if progress_hook:
                ydl_opts['progress_hooks'] = [progress_hook]
                
            # Download ausführen
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            # Tatsächliche Datei finden (Extension wird von yt-dlp gesetzt)
            actual_files = list(Path(self.temp_dir).glob(f"video_{hash(url) % 10000}.*"))
            
            if not actual_files:
                raise Exception("Video-Datei nach Download nicht gefunden")
                
            video_file = actual_files[0]
            
            if not video_file.exists():
                raise Exception(f"Video-Datei existiert nicht: {video_file}")
                
            logger.info(f"Video-Download erfolgreich: {video_file} ({video_file.stat().st_size} bytes)")
            return video_file
            
        except Exception as e:
            logger.error(f"Fehler beim Video-Download: {e}")
            return None
            
    # ========================================
    # DEPRECATED: BytesIO-Methoden (für Kompatibilität)
    # ========================================
    
    def download_audio_to_memory(self, url: str, progress_callback: Optional[Callable] = None) -> Optional[BytesIO]:
        """Download Audio zu BytesIO (DEPRECATED - verwende download_audio_to_temp_file())"""
        logger.warning("download_audio_to_memory() ist deprecated. Verwende download_audio_to_temp_file().")
        
        # Fallback: Temp-Datei erstellen und in BytesIO laden
        temp_file = self.download_audio_to_temp_file(url, progress_callback)
        
        if not temp_file:
            return None
            
        try:
            # Datei in BytesIO laden
            audio_buffer = BytesIO()
            with open(temp_file, 'rb') as f:
                audio_buffer.write(f.read())
                
            # Temp-Datei löschen
            temp_file.unlink()
            
            # Buffer zum Lesen vorbereiten
            audio_buffer.seek(0)
            
            logger.info(f"Audio in BytesIO geladen: {audio_buffer.tell()} bytes")
            return audio_buffer
            
        except Exception as e:
            logger.error(f"Fehler beim Laden in BytesIO: {e}")
            # Cleanup bei Fehler
            if temp_file.exists():
                temp_file.unlink()
            return None
            
    def cleanup_temp_files(self):
        """Temporäre Dateien aufräumen"""
        try:
            import shutil
            if Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Temporäre Dateien gelöscht: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Fehler beim Cleanup: {e}")
            
    def get_audio_formats(self, url: str) -> Optional[Dict[str, Any]]:
        """Verfügbare Audio-Formate abrufen"""
        try:
            ydl_opts = {'quiet': True, 'no_warnings': True}
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
            audio_formats = []
            for fmt in info.get('formats', []):
                if fmt.get('acodec') != 'none':  # Hat Audio
                    audio_formats.append({
                        'format_id': fmt.get('format_id'),
                        'ext': fmt.get('ext'),
                        'acodec': fmt.get('acodec'),
                        'abr': fmt.get('abr'),
                        'filesize': fmt.get('filesize')
                    })
                    
            return {
                'available_formats': audio_formats,
                'best_audio': info.get('formats', [{}])[-1] if info.get('formats') else None
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Audio-Formate: {e}")
            return None


# Globale Service-Instanz
download_service = DownloadService()


def get_download_service() -> DownloadService:
    """Convenience-Funktion für Download-Service Zugriff"""
    return download_service


# Test-Funktionen
def test_download_service():
    """Test des Download-Services (vereinfacht)"""
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll für Test
    
    service = get_download_service()
    
    print("🔍 Testing Video Info...")
    info = service.get_video_info(test_url)
    if info:
        print(f"✅ Title: {info['title']}")
        print(f"✅ Duration: {info['duration']}s")
        print(f"✅ Uploader: {info['uploader']}")
    else:
        print("❌ Video Info failed")
        return
        
    print("\n🎵 Testing Audio Download (Temp File)...")
    audio_file = service.download_audio_to_temp_file(test_url)
    if audio_file:
        print(f"✅ Audio downloaded: {audio_file}")
        print(f"✅ Size: {audio_file.stat().st_size} bytes")
        print(f"✅ Format: WAV für Whisper bereit")
        
        # Cleanup Test-Datei
        audio_file.unlink()
        print("✅ Test-Datei gelöscht")
    else:
        print("❌ Audio download failed")
        
    print("\n📹 Testing Video Download...")
    video_file = service.download_video_to_file(test_url)
    if video_file:
        print(f"✅ Video downloaded: {video_file}")
        print(f"✅ Size: {video_file.stat().st_size} bytes")
    else:
        print("❌ Video download failed")
        
    print("\n🔄 Testing Deprecated BytesIO Method...")
    audio_buffer = service.download_audio_to_memory(test_url)
    if audio_buffer:
        print(f"✅ BytesIO Audio: {audio_buffer.tell()} bytes (deprecated)")
        audio_buffer.close()
    else:
        print("❌ BytesIO download failed")
        
    print("\n🧹 Cleanup...")
    service.cleanup_temp_files()
    print("✅ Test complete")


if __name__ == "__main__":
    test_download_service()
