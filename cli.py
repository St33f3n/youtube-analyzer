"""
CLI Interface - Command Line Interface für YouTube Analyzer
Vollständig entwickelt mit Result-Types und vollständigen Type-Hints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
from typing import NoReturn
from typing import Optional

from youtube_analyzer.config.secrets import setup_secrets_interactive
from youtube_analyzer.config.settings import get_config
from youtube_analyzer.config.settings import test_config_system
from youtube_analyzer.services.analysis import get_analysis_service
from youtube_analyzer.services.analysis import test_analysis_service
from youtube_analyzer.services.download import get_download_service
from youtube_analyzer.services.download import test_download_service
from youtube_analyzer.services.ollama import get_ollama_service
from youtube_analyzer.services.ollama import test_ollama_service
from youtube_analyzer.services.whisper import get_whisper_service
from youtube_analyzer.services.whisper import test_whisper_service
from youtube_analyzer.yt_types import AnalysisDecision
from youtube_analyzer.yt_types import Err
from youtube_analyzer.yt_types import Ok
from youtube_analyzer.yt_types import Result
from youtube_analyzer.yt_types import ValidationError
from youtube_analyzer.yt_types import validate_youtube_url
from youtube_analyzer.utils.logging import ComponentLogger
from youtube_analyzer.utils.logging import get_development_config
from youtube_analyzer.utils.logging import get_production_config
from youtube_analyzer.utils.logging import setup_logging


class YouTubeAnalyzerCLI:
    """Command Line Interface für YouTube Analyzer"""
    
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.logger = ComponentLogger("YouTubeAnalyzerCLI")
        
        # Logging Setup
        if verbose:
            setup_logging(get_development_config())
        else:
            setup_logging(get_production_config())
        
        self.logger.info(
            "YouTube Analyzer CLI initialized",
            verbose=verbose,
        )
    
    def analyze_url(self, url: str) -> Result[dict, ValidationError]:
        """Einzelne URL analysieren"""
        # URL validieren
        url_validation = validate_youtube_url(url)
        if isinstance(url_validation, Err):
            return Err(url_validation.error)
        
        try:
            print(f"🔍 Analysiere: {url}")
            
            # Services laden
            download_service = get_download_service()
            whisper_service = get_whisper_service()
            analysis_service = get_analysis_service()
            
            # 1. Video-Informationen
            print("📋 Lade Video-Informationen...")
            info_result = download_service.get_video_info(url)
            
            if isinstance(info_result, Err):
                return Err(ValidationError(f"Video-Info Fehler: {info_result.error.message}"))
            
            video_info = info_result.value
            print(f"📺 Video: {video_info.title}")
            print(f"📺 Kanal: {video_info.uploader}")
            print(f"⏱️ Dauer: {video_info.duration//60}:{video_info.duration%60:02d}")
            
            # 2. Audio-Download
            print("🎵 Lade Audio herunter...")
            audio_result = download_service.download_audio(url)
            
            if isinstance(audio_result, Err):
                return Err(ValidationError(f"Audio-Download Fehler: {audio_result.error.message}"))
            
            audio_metadata = audio_result.value
            print(f"🎵 Audio: {audio_metadata.file_size} bytes")
            
            # 3. Transkription
            print("🎤 Transkribiere Audio...")
            transcription_result = whisper_service.transcribe_audio(audio_metadata)
            
            if isinstance(transcription_result, Err):
                return Err(ValidationError(f"Transkription Fehler: {transcription_result.error.message}"))
            
            transcription = transcription_result.value
            print(f"🎤 Transkript: {len(transcription.text)} Zeichen ({transcription.language})")
            
            # 4. KI-Analyse
            print("🤖 Führe KI-Analyse durch...")
            analysis_result = analysis_service.analyze_transcript(transcription)
            
            if isinstance(analysis_result, Err):
                return Err(ValidationError(f"Analyse Fehler: {analysis_result.error.message}"))
            
            analysis = analysis_result.value
            
            # Ergebnisse anzeigen
            print("\n" + "="*50)
            print("📊 ANALYSE-ERGEBNISSE")
            print("="*50)
            print(f"🎯 Finale Bewertung: {analysis.final_score:.3f}")
            print(f"🎯 Entscheidung: {analysis.decision.value}")
            
            print(f"\n📋 Regel-Bewertungen:")
            for rule_score in analysis.rule_scores:
                print(f"  • {rule_score.rule_name}: {rule_score.score:.3f} (Konfidenz: {rule_score.confidence:.3f})")
                if self.verbose:
                    print(f"    Grund: {rule_score.reason}")
                    print(f"    Keywords: {', '.join(rule_score.keywords[:5])}")
            
            # Empfehlung
            if analysis.decision == AnalysisDecision.APPROVE:
                print(f"\n✅ EMPFEHLUNG: Video für Archivierung genehmigt")
            elif analysis.decision == AnalysisDecision.REJECT:
                print(f"\n❌ EMPFEHLUNG: Video nicht für Archivierung geeignet")
            else:
                print(f"\n⚠️ EMPFEHLUNG: Manuelle Überprüfung erforderlich")
            
            # Cleanup
            download_service.cleanup()
            whisper_service.cleanup()
            
            return Ok({
                'url': url,
                'video_info': video_info.dict(),
                'transcription': transcription.dict(),
                'analysis': analysis.dict(),
            })
        
        except Exception as e:
            return Err(ValidationError(f"Unerwarteter Fehler: {str(e)}"))
    
    def analyze_urls_batch(self, urls: List[str]) -> dict:
        """Mehrere URLs in Batch analysieren"""
        print(f"🔄 Starte Batch-Analyse für {len(urls)} URLs")
        
        results = {
            'successful': [],
            'failed': [],
            'approved': [],
            'rejected': [],
            'manual_review': [],
        }
        
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] " + "="*40)
            
            result = self.analyze_url(url)
            
            if isinstance(result, Ok):
                data = result.value
                analysis = data['analysis']
                decision = analysis['decision']
                
                results['successful'].append(data)
                
                if decision == 'approve':
                    results['approved'].append(data)
                elif decision == 'reject':
                    results['rejected'].append(data)
                else:
                    results['manual_review'].append(data)
            
            else:
                error_data = {
                    'url': url,
                    'error': result.error.message,
                }
                results['failed'].append(error_data)
                print(f"❌ Fehler: {result.error.message}")
        
        # Batch-Zusammenfassung
        print(f"\n" + "="*60)
        print("📊 BATCH-ZUSAMMENFASSUNG")
        print("="*60)
        print(f"📊 Verarbeitet: {len(urls)} URLs")
        print(f"✅ Erfolgreich: {len(results['successful'])}")
        print(f"❌ Fehlgeschlagen: {len(results['failed'])}")
        print(f"✅ Genehmigt: {len(results['approved'])}")
        print(f"❌ Abgelehnt: {len(results['rejected'])}")
        print(f"⚠️ Manuelle Überprüfung: {len(results['manual_review'])}")
        
        return results
    
    def run_system_tests(self) -> bool:
        """System-Tests ausführen"""
        print("🧪 YouTube Analyzer - System Tests")
        print("="*40)
        
        all_passed = True
        
        # Config Test
        print("\n1. 📋 Config System Test...")
        try:
            test_config_system()
            print("   ✅ Config System OK")
        except Exception as e:
            print(f"   ❌ Config System FEHLER: {e}")
            all_passed = False
        
        # Download Service Test
        print("\n2. 📥 Download Service Test...")
        try:
            test_download_service()
            print("   ✅ Download Service OK")
        except Exception as e:
            print(f"   ❌ Download Service FEHLER: {e}")
            all_passed = False
        
        # Whisper Service Test
        print("\n3. 🎤 Whisper Service Test...")
        try:
            test_whisper_service()
            print("   ✅ Whisper Service OK")
        except Exception as e:
            print(f"   ❌ Whisper Service FEHLER: {e}")
            all_passed = False
        
        # Ollama Service Test
        print("\n4. 🤖 Ollama Service Test...")
        try:
            test_ollama_service()
            print("   ✅ Ollama Service OK")
        except Exception as e:
            print(f"   ❌ Ollama Service FEHLER: {e}")
            all_passed = False
        
        # Analysis Service Test
        print("\n5. 📊 Analysis Service Test...")
        try:
            test_analysis_service()
            print("   ✅ Analysis Service OK")
        except Exception as e:
            print(f"   ❌ Analysis Service FEHLER: {e}")
            all_passed = False
        
        # Zusammenfassung
        print(f"\n" + "="*40)
        if all_passed:
            print("🎉 ALLE TESTS BESTANDEN!")
            print("✅ System ist bereit für den Einsatz")
        else:
            print("⚠️ EINIGE TESTS FEHLGESCHLAGEN!")
            print("❌ Prüfe Konfiguration und Dependencies")
        
        return all_passed
    
    def show_config_info(self) -> None:
        """Konfigurationsinformationen anzeigen"""
        try:
            config = get_config()
            
            print("⚙️ YouTube Analyzer - Konfiguration")
            print("="*40)
            
            print(f"\n📋 Analyse-Regeln:")
            for rule_name, rule_config in config.rules.items():
                status = "✅" if rule_config.enabled else "❌"
                print(f"  {status} {rule_name}: {rule_config.weight:.2f} ({rule_config.file})")
            
            print(f"\n🎯 Bewertung:")
            print(f"  Threshold: {config.scoring.threshold:.2f}")
            print(f"  Min. Confidence: {config.scoring.min_confidence:.2f}")
            print(f"  Gesamt-Gewichtung: {config.get_total_weight():.3f}")
            
            print(f"\n🎤 Whisper:")
            print(f"  Aktiviert: {config.whisper.enabled}")
            print(f"  Model: {config.whisper.model_name}")
            
            print(f"\n🤖 Ollama:")
            print(f"  URL: {config.ollama.base_url}")
            print(f"  Model: {config.ollama.model_name}")
            print(f"  Timeout: {config.ollama.timeout}s")
            
            print(f"\n💾 Storage:")
            print(f"  NextCloud: {config.storage.nextcloud_path}")
            print(f"  Trilium: {config.storage.trilium_parent_note}")
            print(f"  SQLite: {config.storage.sqlite_path}")
            
        except Exception as e:
            print(f"❌ Config-Fehler: {e}")
    
    def setup_secrets(self) -> None:
        """Secrets interaktiv einrichten"""
        setup_secrets_interactive()


def create_argument_parser() -> argparse.ArgumentParser:
    """Argument Parser erstellen"""
    parser = argparse.ArgumentParser(
        description="YouTube Analyzer - Intelligente Video-Analyse für Wissensmanagement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  youtube-analyzer analyze "https://youtube.com/watch?v=..."
  youtube-analyzer batch urls.txt
  youtube-analyzer test
  youtube-analyzer config
  youtube-analyzer setup-secrets
        """
    )
    
    # Global Options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Ausführliche Ausgabe und Debug-Logging'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Verfügbare Befehle')
    
    # Analyze Command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Einzelne YouTube-URL analysieren'
    )
    analyze_parser.add_argument(
        'url',
        help='YouTube-URL zur Analyse'
    )
    
    # Batch Command
    batch_parser = subparsers.add_parser(
        'batch',
        help='Mehrere URLs aus Datei analysieren'
    )
    batch_parser.add_argument(
        'file',
        help='Datei mit URLs (eine pro Zeile)'
    )
    
    # Test Command
    subparsers.add_parser(
        'test',
        help='System-Tests ausführen'
    )
    
    # Config Command
    subparsers.add_parser(
        'config',
        help='Konfigurationsinformationen anzeigen'
    )
    
    # Setup Secrets Command
    subparsers.add_parser(
        'setup-secrets',
        help='Secrets interaktiv einrichten'
    )
    
    return parser


def load_urls_from_file(file_path: str) -> List[str]:
    """URLs aus Datei laden"""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        urls = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # URL validieren
                    url_validation = validate_youtube_url(line)
                    if isinstance(url_validation, Ok):
                        urls.append(line)
                    else:
                        print(f"⚠️ Zeile {line_num}: Ungültige URL: {line}")
        
        return urls
    
    except Exception as e:
        print(f"❌ Fehler beim Laden der Datei: {e}")
        return []


def main() -> NoReturn:
    """CLI Haupteinstiegspunkt"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Keine Argumente -> GUI starten
    if not args.command:
        print("🌊 Starte YouTube Analyzer GUI...")
        from youtube_analyzer.main import gui_main
        gui_main()
    
    # CLI-Interface erstellen
    cli = YouTubeAnalyzerCLI(verbose=args.verbose)
    
    try:
        if args.command == 'analyze':
            result = cli.analyze_url(args.url)
            if isinstance(result, Err):
                print(f"❌ Analyse fehlgeschlagen: {result.error.message}")
                sys.exit(1)
            else:
                print(f"\n✅ Analyse erfolgreich abgeschlossen")
                sys.exit(0)
        
        elif args.command == 'batch':
            urls = load_urls_from_file(args.file)
            if not urls:
                print(f"❌ Keine gültigen URLs in Datei gefunden")
                sys.exit(1)
            
            results = cli.analyze_urls_batch(urls)
            
            # Exit-Code basierend auf Erfolgsrate
            success_rate = len(results['successful']) / len(urls)
            if success_rate >= 0.8:
                sys.exit(0)  # Erfolg
            elif success_rate >= 0.5:
                sys.exit(2)  # Teilweise erfolgreich
            else:
                sys.exit(1)  # Fehlgeschlagen
        
        elif args.command == 'test':
            success = cli.run_system_tests()
            sys.exit(0 if success else 1)
        
        elif args.command == 'config':
            cli.show_config_info()
            sys.exit(0)
        
        elif args.command == 'setup-secrets':
            cli.setup_secrets()
            sys.exit(0)
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Benutzer-Abbruch")
        sys.exit(130)
    
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
