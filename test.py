import subprocess
import json
import os
from core_types import *
from yt_analyzer_core import *
from typing import Dict, Any, Optional
from datetime import datetime, time as dt_time

def extract_metadata_with_os_ytdlp(url: str, cookie_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Extrahiert Metadata mit OS-Version von yt-dlp
    
    Args:
        url: YouTube-URL
        cookie_file: Pfad zur cookies.txt Datei (optional)
    
    Returns:
        Dict mit Metadata oder wirft Exception bei Fehlern
    """
    # yt-dlp Kommando zusammenbauen
    cmd = [
        "yt-dlp",
        "--dump-json",  # JSON-Output für Metadata
        "--no-download",  # Nur Metadata, kein Download
        "--no-warnings",
        "--quiet",  # Weniger Output
    ]
    
    # Cookie-Optionen
    if cookie_file and os.path.exists(cookie_file):
        cmd.extend(["--cookies", cookie_file])
    else:
        cmd.extend(["--cookies-from-browser", "firefox"])
    
    # URL hinzufügen
    cmd.append(url)
    
    try:
        # yt-dlp ausführen
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60s Timeout
            check=True
        )
        
        # JSON parsen
        metadata = json.loads(result.stdout.strip())
        return metadata
        
    except subprocess.TimeoutExpired:
        raise Exception(f"yt-dlp timeout after 60s for URL: {url}")
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "Unknown error"
        raise Exception(f"yt-dlp failed: {error_msg}")
        
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse yt-dlp JSON output: {e}")
        
    except FileNotFoundError:
        raise Exception("yt-dlp not found. Install with: pip install yt-dlp")


def extract_single_metadata_os(url: str, cookie_file: Optional[str] = None) -> ProcessObject:
    """
    OS-Version der Metadata-Extraktion - erstellt ProcessObject
    
    Args:
        url: YouTube-URL  
        cookie_file: Pfad zur cookies.txt
        
    Returns:
        ProcessObject mit extrahierter Metadata
    """
    try:
        # Metadata mit OS yt-dlp extrahieren
        info = extract_metadata_with_os_ytdlp(url, cookie_file)
        
        # ProcessObject erstellen (gleiche Logik wie vorher)
        title = info.get("title", "Unknown Title")
        uploader = info.get("uploader", info.get("channel", "Unknown Channel"))
        duration_seconds = info.get("duration", 0)
        upload_date_str = info.get("upload_date", "")
        
        # Duration parsen
        if duration_seconds and duration_seconds > 0:
            hours = duration_seconds // 3600
            minutes = (duration_seconds % 3600) // 60
            seconds = duration_seconds % 60
            länge = dt_time(hours, minutes, seconds)
        else:
            länge = dt_time(0, 0, 0)
        
        # Upload date parsen
        upload_date = datetime.now()
        if upload_date_str:
            try:
                upload_date = datetime.strptime(upload_date_str, "%Y%m%d")
            except ValueError:
                pass
        
        # ProcessObject erstellen
        process_obj = ProcessObject(
            titel=title,
            kanal=uploader,
            länge=länge,
            upload_date=upload_date,
            original_url=url,
        )
        
        process_obj.update_stage("metadata_extracted_os")
        
        print(f"✅ OS yt-dlp extraction successful:")
        print(f"  📹 {title}")
        print(f"  📺 {uploader}")
        print(f"  ⏱️ {länge}")
        print(f"  👀 Views: {info.get('view_count', 'unknown'):,}")
        
        return process_obj
        
    except Exception as e:
        print(f"❌ OS yt-dlp extraction failed: {e}")
        raise


class YouTubeMetadataExtractorOS:
    """OS-Version des Metadata Extractors"""
    
    def __init__(self, cookie_file: Optional[str] = None):
        self.logger = get_logger("YouTubeMetadataExtractorOS")
        self.cookie_file = cookie_file
        
        # Cookie-Datei finden falls nicht angegeben
        if not self.cookie_file:
            cookie_paths = [
                "cookies.txt",
                os.path.expanduser("~/cookies.txt"),
                os.path.expanduser("~/Downloads/cookies.txt"),
            ]
            
            for path in cookie_paths:
                if os.path.exists(path):
                    self.cookie_file = path
                    self.logger.info(f"Found cookie file: {path}")
                    break
    
    def extract_single_metadata(self, url: str) -> Result[ProcessObject, CoreError]:
        """
        OS-Version der Single-Metadata-Extraktion
        """
        try:
            process_obj = extract_single_metadata_os(url, self.cookie_file)
            return Ok(process_obj)
            
        except Exception as e:
            context = ErrorContext.create(
                "extract_single_metadata_os",
                input_data={"url": url, "cookie_file": self.cookie_file},
                suggestions=[
                    "Check yt-dlp installation",
                    "Verify cookie file exists",
                    "Test URL manually with yt-dlp",
                ]
            )
            return Err(CoreError(f"OS yt-dlp extraction failed: {e}", context))
    
    def extract_batch_metadata(self, urls: List[str]) -> Result[List[ProcessObject], List[CoreError]]:
        """
        OS-Version der Batch-Extraktion
        """
        all_objects = []
        all_errors = []
        
        for url in urls:
            result = self.extract_single_metadata(url)
            
            if isinstance(result, Ok):
                all_objects.append(unwrap_ok(result))
            else:
                all_errors.append(unwrap_err(result))
        
        if all_objects:
            self.logger.info(
                f"🎯 OS batch extraction completed: {len(all_objects)} videos, {len(all_errors)} errors"
            )
            return Ok(all_objects)
        else:
            return Err(all_errors)


# Integration in den Hauptcode
def process_urls_to_objects_os(
    urls_text: str, 
    cookie_file: Optional[str] = None
) -> Result[List[ProcessObject], List[CoreError]]:
    """
    OS-Version der kompletten Pipeline
    """
    # URLs parsen (gleiche Logik)
    url_processor = YouTubeURLProcessor()
    urls_result = url_processor.parse_multiline_input(urls_text)
    
    if isinstance(urls_result, Err):
        return Err([unwrap_err(urls_result)])
    
    urls = unwrap_ok(urls_result)
    
    # OS-Extractor verwenden
    extractor = YouTubeMetadataExtractorOS(cookie_file)
    return extractor.extract_batch_metadata(urls)


# Test-Funktion
def test_os_extraction():
    """Test der OS-Version"""
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        # Test mit cookies.txt falls vorhanden
        cookie_file = "data/yt_analyzer_cookies.txt" if os.path.exists("data/yt_analyzer_cookies.txt") else None
        
        print(f"Testing OS yt-dlp with URL: {test_url}")
        print(f"Cookie file: {cookie_file}")
        
        process_obj = extract_single_metadata_os(test_url, cookie_file)
        print(f"✅ Success: {process_obj.titel}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_os_extraction()
