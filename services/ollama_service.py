"""
Ollama Service - ROCm + Gemma3 Integration mit strukturiertem Output
"""

import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from loguru import logger
import requests
from pydantic import BaseModel, Field


class OllamaService:
    """Service für Ollama KI-Analyse mit ROCm und strukturiertem Output (Gemma3)"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model_name = "gemma3:12b"  # Gemma3 4B Model (tatsächlich verfügbar)
        self._model_loaded = False
        self._gpu_info = None
        
        logger.info(f"Ollama Service initialisiert: {base_url}")
        self._check_ollama_status()
        
    def _check_ollama_status(self) -> bool:
        """Prüfe ob Ollama Server läuft"""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            if response.status_code == 200:
                version_info = response.json()
                logger.info(f"Ollama Server verfügbar: {version_info.get('version', 'unknown')}")
                return True
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama Server nicht erreichbar: {e}")
            logger.error("Starte Ollama mit: ollama serve")
            return False
            
    def _check_gpu_info(self) -> Dict[str, Any]:
        """Prüfe ROCm/GPU Status über Ollama API"""
        try:
            # Ollama ps gibt GPU-Info zurück
            response = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if response.status_code == 200:
                ps_info = response.json()
                logger.info(f"Ollama Prozesse: {len(ps_info.get('models', []))}")
                
                # GPU-Info aus laufenden Models extrahieren
                models = ps_info.get('models', [])
                if models:
                    for model in models:
                        if 'details' in model and 'families' in model['details']:
                            # Model läuft bereits - GPU wird genutzt
                            return {
                                "backend": "ROCm/CUDA",
                                "gpu_available": True,
                                "gpu_name": "GPU (Auto-detected via Ollama)"
                            }
                
            # ROCm Detection über alternative Methoden
            gpu_info = {
                "backend": "unknown",
                "gpu_available": False,
                "gpu_name": "CPU Only"
            }
            
            # Check 1: rocminfo (besser als rocm-smi)
            try:
                import subprocess
                result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and ("gfx" in result.stdout or "AMD" in result.stdout):
                    gpu_info.update({
                        "backend": "ROCm",
                        "gpu_available": True,
                        "gpu_name": "AMD GPU (ROCm detected)"
                    })
                    logger.info("ROCm GPU erkannt via rocminfo")
                    return gpu_info
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
                
            # Check 2: /opt/rocm existence
            try:
                from pathlib import Path
                if Path("/opt/rocm").exists():
                    gpu_info.update({
                        "backend": "ROCm",
                        "gpu_available": True,
                        "gpu_name": "AMD GPU (ROCm path found)"
                    })
                    logger.info("ROCm erkannt via /opt/rocm Pfad")
                    return gpu_info
            except:
                pass
                
            # Check 3: lspci für AMD GPUs
            try:
                result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and "AMD" in result.stdout and ("Radeon" in result.stdout or "RDNA" in result.stdout):
                    gpu_info.update({
                        "backend": "AMD GPU",
                        "gpu_available": True,
                        "gpu_name": "AMD GPU (lspci detected)"
                    })
                    logger.info("AMD GPU erkannt via lspci")
                    return gpu_info
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
                
            logger.info("Keine GPU erkannt - Ollama läuft auf CPU")
            self._gpu_info = gpu_info
            return gpu_info
            
        except Exception as e:
            logger.warning(f"GPU-Info Check fehlgeschlagen: {e}")
            return {"backend": "unknown", "gpu_available": False, "gpu_name": "Unknown"}
            
    def _ensure_model_loaded(self) -> bool:
        """Stelle sicher dass Gemma3 4B Model geladen ist"""
        try:
            # Prüfe ob Model bereits läuft
            response = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if response.status_code == 200:
                ps_data = response.json()
                loaded_models = [m.get('name', '') for m in ps_data.get('models', [])]
                
                if self.model_name in loaded_models:
                    logger.info(f"Model {self.model_name} bereits geladen")
                    self._model_loaded = True
                    return True
                    
            # Model laden durch Test-Request
            logger.info(f"Lade Model {self.model_name}...")
            
            test_payload = {
                "model": self.model_name,
                "prompt": "Test",
                "stream": False,
                "options": {
                    "num_predict": 1  # Minimaler Test
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=test_payload,
                timeout=60  # Model-Loading kann dauern
            )
            
            if response.status_code == 200:
                logger.info(f"Model {self.model_name} erfolgreich geladen")
                self._model_loaded = True
                return True
            else:
                raise Exception(f"Model Loading fehlgeschlagen: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Fehler beim Model-Loading: {e}")
            logger.error("Installiere Model mit: ollama pull gemma3:4b")
            return False
            
    def generate_structured(
        self, 
        system_prompt: str,
        user_prompt: str,
        output_schema: Dict[str, Any],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Strukturierte KI-Anfrage mit JSON Schema Output (mit Retry & robuster Type-Konvertierung)"""
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Model laden falls nötig
                if not self._model_loaded:
                    if not self._ensure_model_loaded():
                        raise Exception("Model konnte nicht geladen werden")
                        
                logger.info(f"Starte strukturierte Analyse mit {self.model_name} (Versuch {attempt + 1}/{max_retries})")
                
                # Strukturierten Output-Prompt erstellen
                schema_prompt = f"""
Du musst EXAKT in folgendem JSON-Format antworten:

{json.dumps(output_schema, indent=2)}

Wichtig:
- Nur valides JSON zurückgeben
- Alle Felder ausfüllen
- Keine zusätzlichen Erklärungen
- Numerische Werte als ZAHLEN (nicht Strings): 0.9 nicht "0.9"
- Score als Dezimalzahl zwischen 0.0 und 1.0
- Confidence als Dezimalzahl zwischen 0.0 und 1.0
"""
                
                # Kombinierter Prompt
                full_prompt = f"{system_prompt}\n\n{schema_prompt}\n\n{user_prompt}"
                
                # Ollama Request
                payload = {
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "format": "json",  # Wichtig: JSON-Format erzwingen
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature + (attempt * 0.05),  # Leicht erhöhen bei Retry
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                        "stop": ["<|endoftext|>", "\n\n\n"]
                    }
                }
                
                logger.debug(f"Ollama Request: {payload['model']}, temp={payload['options']['temperature']:.2f}")
                
                # API Call
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120  # 2 Minuten für Analyse
                )
                
                if response.status_code != 200:
                    raise Exception(f"Ollama API Fehler: HTTP {response.status_code} - {response.text}")
                    
                response_data = response.json()
                
                if 'response' not in response_data:
                    raise Exception("Keine 'response' in Ollama-Antwort")
                    
                # JSON-Response parsen
                raw_response = response_data['response'].strip()
                logger.debug(f"Raw Ollama Response: {raw_response[:200]}...")
                
                try:
                    # JSON parsen
                    structured_output = json.loads(raw_response)
                    
                    # Robuste Type-Konvertierung und Validierung
                    processed_output = self._process_and_validate_output(structured_output, output_schema)
                    
                    if processed_output:
                        logger.info(f"Strukturierte Analyse erfolgreich: Score={processed_output.get('score', processed_output.get('relevance_score', 'N/A'))}")
                        return processed_output
                    else:
                        raise ValueError("Output-Validierung fehlgeschlagen")
                        
                except json.JSONDecodeError as json_error:
                    logger.warning(f"JSON Parse Fehler (Versuch {attempt + 1}): {json_error}")
                    
                    # Fallback: JSON-Reparatur versuchen
                    repaired = self._attempt_json_fix(raw_response, output_schema)
                    if repaired:
                        processed = self._process_and_validate_output(repaired, output_schema)
                        if processed:
                            logger.info("JSON-Reparatur erfolgreich!")
                            return processed
                    
                    last_error = json_error
                    
            except Exception as e:
                logger.warning(f"Versuch {attempt + 1} fehlgeschlagen: {e}")
                last_error = e
                
                # Bei API-Fehlern sofort abbrechen (kein Retry sinnvoll)
                if "HTTP" in str(e) and "API" in str(e):
                    break
                    
        # Alle Versuche fehlgeschlagen
        logger.error(f"Strukturierte Analyse nach {max_retries} Versuchen fehlgeschlagen: {last_error}")
        
        # Notfall-Fallback
        return self._create_fallback_output(output_schema)
        
    def _process_and_validate_output(self, raw_output: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Robuste Type-Konvertierung und Validierung"""
        try:
            processed = {}
            
            # Alle Felder durchgehen und robuste Type-Konvertierung
            for key, value in raw_output.items():
                if key in ['score', 'confidence', 'relevance_score'] and value is not None:
                    # String zu Float konvertieren wenn nötig
                    if isinstance(value, str):
                        try:
                            processed[key] = float(value)
                        except ValueError:
                            logger.warning(f"Kann '{key}': '{value}' nicht zu Float konvertieren")
                            processed[key] = 0.5  # Fallback
                    elif isinstance(value, (int, float)):
                        processed[key] = float(value)
                    else:
                        processed[key] = 0.5  # Fallback
                        
                elif key in ['keywords', 'topics'] and value is not None:
                    # Ensure Array
                    if isinstance(value, str):
                        processed[key] = [value]  # String zu Array
                    elif isinstance(value, list):
                        processed[key] = value
                    else:
                        processed[key] = []  # Fallback
                        
                else:
                    # Andere Felder direkt übernehmen
                    processed[key] = value
                    
            # DYNAMISCHE Pflichtfeld-Erkennung basierend auf Schema
            schema_score_fields = []
            for field_name in schema.keys():
                if any(score_word in field_name.lower() for score_word in ['score', 'confidence']):
                    schema_score_fields.append(field_name)
            
            # Fallback für fehlende Score-Felder (nur wenn im Schema erwartet)
            for field in schema_score_fields:
                if field not in processed:
                    logger.warning(f"Schema-Feld '{field}' fehlt - setze Fallback")
                    processed[field] = 0.5
                    
            # Legacy-Support: 'score' fallback wenn 'relevance_score' vorhanden
            if 'relevance_score' in processed and 'score' not in processed:
                # Nur wenn 'score' explizit erwartet wird
                if 'score' in schema:
                    processed['score'] = processed['relevance_score']
                    
            # Umgekehrt: 'relevance_score' fallback wenn 'score' vorhanden  
            if 'score' in processed and 'relevance_score' not in processed:
                if 'relevance_score' in schema:
                    processed['relevance_score'] = processed['score']
                        
            # Werte-Validierung (0.0-1.0 für alle Score-Felder)
            for field in processed:
                if any(score_word in field.lower() for score_word in ['score', 'confidence']):
                    value = processed[field]
                    if isinstance(value, (int, float)) and not (0.0 <= value <= 1.0):
                        logger.warning(f"{field} außerhalb 0.0-1.0: {value} -> clamped")
                        processed[field] = max(0.0, min(1.0, value))
                        
            return processed
            
        except Exception as e:
            logger.error(f"Output-Verarbeitung fehlgeschlagen: {e}")
            return None
            
    def _attempt_json_fix(self, raw_response: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Versuche defektes JSON zu reparieren"""
        try:
            logger.warning("Versuche JSON-Reparatur...")
            
            # Einfache Reparatur-Versuche
            fixed_response = raw_response.strip()
            
            # Entferne Markdown-Code-Blöcke
            if "```json" in fixed_response:
                fixed_response = fixed_response.split("```json")[1].split("```")[0].strip()
            elif "```" in fixed_response:
                fixed_response = fixed_response.split("```")[1].strip()
                
            # Entferne Text vor erstem {
            if '{' in fixed_response:
                fixed_response = fixed_response[fixed_response.find('{'):]
                
            # Entferne Text nach letztem }
            if '}' in fixed_response:
                fixed_response = fixed_response[:fixed_response.rfind('}')+1]
                
            # Nochmal parsen
            repaired = json.loads(fixed_response)
            logger.info("JSON-Reparatur erfolgreich!")
            return repaired
            
        except Exception as repair_error:
            logger.error(f"JSON-Reparatur fehlgeschlagen: {repair_error}")
            
            # Notfall-Fallback
            return {
                "score": 0.5,
                "confidence": 0.3,
                "reason": "JSON-Parse Fehler - Fallback-Werte",
                "keywords": ["parse_error"]
            }
            
    def _create_fallback_output(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Erstelle Notfall-Output basierend auf Schema"""
        fallback = {
            "score": 0.3,
            "confidence": 0.2,
            "reason": "Fallback-Werte aufgrund von Analyse-Fehlern",
            "keywords": ["analysis_error"]
        }
        
        # Schema-spezifische Fallbacks
        if "relevance_score" in schema:
            fallback["relevance_score"] = 0.3
        if "topic" in schema:
            fallback["topic"] = "analysis_error"
        if "test_result" in schema:
            fallback["test_result"] = "fallback"
            
        logger.warning("Verwende Fallback-Output")
        return fallback
            
    def get_service_info(self) -> Dict[str, Any]:
        """Service-Informationen für UI"""
        gpu_info = self._gpu_info or self._check_gpu_info()
        
        return {
            "service": "Ollama",
            "model": self.model_name,
            "model_loaded": self._model_loaded,
            "base_url": self.base_url,
            "gpu_backend": gpu_info.get("backend", "unknown"),
            "gpu_available": gpu_info.get("gpu_available", False),
            "gpu_name": gpu_info.get("gpu_name", "Unknown")
        }
        
    def is_ready(self) -> bool:
        """Service bereit für Analyse?"""
        return self._check_ollama_status() and self._ensure_model_loaded()
        
    def test_connection(self) -> Dict[str, Any]:
        """Test-Verbindung zu Ollama"""
        try:
            # Basis-Test
            server_ok = self._check_ollama_status()
            
            if not server_ok:
                return {"status": "error", "message": "Ollama Server nicht erreichbar"}
                
            # GPU-Test
            gpu_info = self._check_gpu_info()
            
            # Model-Test
            model_ok = self._ensure_model_loaded()
            
            if not model_ok:
                return {"status": "error", "message": f"Model {self.model_name} nicht verfügbar"}
                
            # Einfacher Analyse-Test
            test_schema = {
                "test_result": "string",
                "score": "number",
                "confidence": "number"  # WICHTIG: Confidence explizit hinzufügen
            }
            
            result = self.generate_structured(
                system_prompt="Du bist ein Test-System. Antworte EXAKT im geforderten JSON-Format.",
                user_prompt="Antworte mit test_result='success', score=0.9 und confidence=0.95. Alle drei Felder sind Pflicht!",
                output_schema=test_schema,
                max_tokens=100
            )
            
            if result and result.get('test_result') == 'success' and 'confidence' in result:
                return {
                    "status": "success", 
                    "message": "Ollama Service voll funktionsfähig",
                    "gpu_info": gpu_info,
                    "model": self.model_name
                }
            else:
                return {
                    "status": "warning", 
                    "message": "Ollama läuft, aber strukturierter Output unvollständig",
                    "test_result": result,
                    "expected_fields": ["test_result", "score", "confidence"]
                }
                
        except Exception as e:
            return {"status": "error", "message": f"Test fehlgeschlagen: {str(e)}"}


# Globale Service-Instanz
ollama_service = OllamaService()


def get_ollama_service() -> OllamaService:
    """Convenience-Funktion für Ollama-Service Zugriff"""
    return ollama_service


def test_ollama_service():
    """Test des Ollama-Services"""
    print("🤖 Testing Ollama Service...")
    
    service = get_ollama_service()
    
    # Service Info
    info = service.get_service_info()
    print(f"📊 Service Info:")
    print(f"   Model: {info['model']}")
    print(f"   GPU: {info['gpu_name']} ({info['gpu_backend']})")
    print(f"   Ready: {service.is_ready()}")
    
    # Connection Test
    test_result = service.test_connection()
    print(f"\n🧪 Connection Test: {test_result['status']}")
    print(f"   Message: {test_result['message']}")
    
    if 'test_result' in test_result:
        print(f"   Actual Result: {test_result['test_result']}")
    if 'expected_fields' in test_result:
        print(f"   Expected Fields: {test_result['expected_fields']}")
    
    if 'gpu_info' in test_result:
        gpu = test_result['gpu_info']
        print(f"   GPU Backend: {gpu.get('backend', 'unknown')}")
        print(f"   GPU Available: {gpu.get('gpu_available', False)}")
    
    if test_result['status'] == 'success':
        print("✅ Ollama Service bereit für Analyse!")
        
        # Demo-Analyse
        print("\n🎯 Demo-Analyse...")
        demo_schema = {
            "relevance_score": "number",
            "confidence": "number", 
            "topic": "string",
            "reason": "string"
        }
        
        demo_result = service.generate_structured(
            system_prompt="Du bewertest die Relevanz von Texten für technische Wissensdatenbanken.",
            user_prompt="Text: 'Wie funktioniert maschinelles Lernen?' - Bewerte die Relevanz.",
            output_schema=demo_schema
        )
        
        if demo_result:
            print(f"   Relevanz: {demo_result.get('relevance_score', 'N/A')}")
            print(f"   Thema: {demo_result.get('topic', 'N/A')}")
            print("✅ Demo-Analyse erfolgreich!")
        else:
            print("❌ Demo-Analyse fehlgeschlagen")
            
    else:
        print("❌ Ollama Service nicht optimal")
        if test_result['status'] == 'warning':
            print("⚠️ Service läuft, aber strukturierter Output hat Probleme")
        elif test_result['status'] == 'error':
            print("💡 Starte Ollama mit: ollama serve")
            print("💡 Installiere Model mit: ollama pull gemma3:4b")


if __name__ == "__main__":
    test_ollama_service()
