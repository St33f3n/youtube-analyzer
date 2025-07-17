"""
Ollama Service - ROCm + Gemma3 Integration mit strukturiertem Output
Vollständig überarbeitet mit Result-Types und vollständigen Type-Hints
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import requests
from pydantic import BaseModel
from pydantic import Field

from yt_types import AnalysisError
from yt_types import Err
from yt_types import GPUInfo
from yt_types import Ok
from yt_types import Result
from yt_types import ServiceStatus
from yt_types import ServiceUnavailableError
from utils.logging import ComponentLogger
from utils.logging import log_function_calls
from utils.logging import log_performance


class OllamaResponse(BaseModel):
    """Ollama API Response Schema"""
    model: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaService:
    """Service für Ollama KI-Analyse mit ROCm und strukturiertem Output"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "gemma3:12b",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self._model_loaded = False
        self._gpu_info: Optional[GPUInfo] = None
        self.logger = ComponentLogger("OllamaService")
        
        self.logger.info(
            "Ollama service initialized",
            base_url=base_url,
            model_name=model_name,
            timeout=timeout,
        )
        
        # Initial server check
        self._check_server_status()
        
        # Initial GPU detection
        self._detect_gpu_info()
    
    @log_function_calls
    def _check_server_status(self) -> Result[Dict[str, Any], ServiceUnavailableError]:
        """Prüfe Ollama Server Status"""
        try:
            response = requests.get(
                f"{self.base_url}/api/version",
                timeout=5,
            )
            
            if response.status_code == 200:
                version_info = response.json()
                
                self.logger.info(
                    "Ollama server is available",
                    version=version_info.get('version', 'unknown'),
                    base_url=self.base_url,
                )
                
                return Ok(version_info)
            else:
                raise ServiceUnavailableError(f"HTTP {response.status_code}: {response.text}")
        
        except requests.RequestException as e:
            error_msg = f"Ollama server not reachable: {str(e)}"
            self.logger.error(
                "Ollama server check failed",
                error=e,
                base_url=self.base_url,
                suggestion="Start Ollama with: ollama serve",
            )
            
            return Err(ServiceUnavailableError(
                error_msg,
                {
                    'base_url': self.base_url,
                    'error_type': type(e).__name__,
                    'suggestion': 'ollama serve',
                }
            ))
    
    @log_function_calls
    def _detect_gpu_info(self) -> GPUInfo:
        """Erkenne GPU-Informationen für ROCm/CUDA"""
        try:
            self.logger.debug("Detecting GPU information")
            
            # Prüfe laufende Models für GPU-Info
            ps_result = self._get_running_processes()
            if isinstance(ps_result, Ok) and ps_result.value.get('models'):
                # GPU wird bereits verwendet
                gpu_info = GPUInfo(
                    device="gpu",
                    name="GPU (Active via Ollama)",
                    memory_total=0,
                    memory_free=0,
                    available=True,
                )
                
                self.logger.info(
                    "GPU detected via running models",
                    device=gpu_info.device,
                    name=gpu_info.name,
                )
                
                self._gpu_info = gpu_info
                return gpu_info
            
            # ROCm Detection
            rocm_info = self._detect_rocm()
            if rocm_info.available:
                self._gpu_info = rocm_info
                return rocm_info
            
            # CUDA Detection (fallback)
            cuda_info = self._detect_cuda()
            if cuda_info.available:
                self._gpu_info = cuda_info
                return cuda_info
            
            # CPU fallback
            cpu_info = GPUInfo(
                device="cpu",
                name="CPU Only",
                memory_total=0,
                memory_free=0,
                available=False,
            )
            
            self.logger.warning(
                "No GPU detected, falling back to CPU",
                device=cpu_info.device,
            )
            
            self._gpu_info = cpu_info
            return cpu_info
        
        except Exception as e:
            self.logger.error(
                "GPU detection failed",
                error=e,
            )
            
            fallback_info = GPUInfo(
                device="unknown",
                name="Unknown",
                memory_total=0,
                memory_free=0,
                available=False,
            )
            
            self._gpu_info = fallback_info
            return fallback_info
    
    def _detect_rocm(self) -> GPUInfo:
        """Erkenne ROCm-verfügbare AMD GPUs"""
        try:
            # ROCm-Info über rocminfo
            result = subprocess.run(
                ['rocminfo'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0 and ("gfx" in result.stdout or "AMD" in result.stdout):
                # Parse GPU name from rocminfo
                lines = result.stdout.split('\n')
                gpu_name = "AMD GPU (ROCm)"
                
                for line in lines:
                    if "Marketing Name:" in line:
                        gpu_name = line.split("Marketing Name:")[-1].strip()
                        break
                    elif "Name:" in line and "gfx" in line:
                        gpu_name = f"AMD GPU {line.split('Name:')[-1].strip()}"
                        break
                
                self.logger.info(
                    "ROCm GPU detected",
                    gpu_name=gpu_name,
                    detection_method="rocminfo",
                )
                
                return GPUInfo(
                    device="rocm",
                    name=gpu_name,
                    memory_total=0,  # ROCm memory info ist kompliziert
                    memory_free=0,
                    available=True,
                )
        
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Alternative: ROCm-Path-Check
        try:
            if Path("/opt/rocm").exists():
                self.logger.info(
                    "ROCm detected via path",
                    rocm_path="/opt/rocm",
                    detection_method="path",
                )
                
                return GPUInfo(
                    device="rocm",
                    name="AMD GPU (ROCm path found)",
                    memory_total=0,
                    memory_free=0,
                    available=True,
                )
        except Exception:
            pass
        
        return GPUInfo(
            device="cpu",
            name="No ROCm GPU",
            memory_total=0,
            memory_free=0,
            available=False,
        )
    
    def _detect_cuda(self) -> GPUInfo:
        """Erkenne CUDA-verfügbare NVIDIA GPUs"""
        try:
            # nvidia-smi check
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split(', ')
                    if len(parts) >= 3:
                        gpu_name = parts[0]
                        memory_total = int(parts[1])
                        memory_free = int(parts[2])
                        
                        self.logger.info(
                            "CUDA GPU detected",
                            gpu_name=gpu_name,
                            memory_total=memory_total,
                            memory_free=memory_free,
                            detection_method="nvidia-smi",
                        )
                        
                        return GPUInfo(
                            device="cuda",
                            name=gpu_name,
                            memory_total=memory_total,
                            memory_free=memory_free,
                            available=True,
                        )
        
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        
        return GPUInfo(
            device="cpu",
            name="No CUDA GPU",
            memory_total=0,
            memory_free=0,
            available=False,
        )
    
    @log_function_calls
    def _get_running_processes(self) -> Result[Dict[str, Any], ServiceUnavailableError]:
        """Hole laufende Ollama-Prozesse"""
        try:
            response = requests.get(
                f"{self.base_url}/api/ps",
                timeout=5,
            )
            
            if response.status_code == 200:
                ps_data = response.json()
                
                self.logger.debug(
                    "Running processes retrieved",
                    models_count=len(ps_data.get('models', [])),
                )
                
                return Ok(ps_data)
            else:
                raise ServiceUnavailableError(f"HTTP {response.status_code}: {response.text}")
        
        except requests.RequestException as e:
            return Err(ServiceUnavailableError(f"Failed to get running processes: {str(e)}"))
    
    @log_function_calls
    def _ensure_model_loaded(self) -> Result[bool, ServiceUnavailableError]:
        """Stelle sicher dass Model geladen ist"""
        try:
            # Prüfe laufende Models
            ps_result = self._get_running_processes()
            if isinstance(ps_result, Ok):
                loaded_models = [
                    m.get('name', '') for m in ps_result.value.get('models', [])
                ]
                
                if self.model_name in loaded_models:
                    self.logger.debug(
                        "Model already loaded",
                        model_name=self.model_name,
                    )
                    self._model_loaded = True
                    return Ok(True)
            
            # Model laden durch minimalen Test-Request
            self.logger.info(
                "Loading model",
                model_name=self.model_name,
            )
            
            test_payload = {
                "model": self.model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {
                    "num_predict": 1,
                    "temperature": 0.1,
                },
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=test_payload,
                timeout=60,  # Model loading kann dauern
            )
            
            if response.status_code == 200:
                self.logger.info(
                    "Model loaded successfully",
                    model_name=self.model_name,
                )
                self._model_loaded = True
                return Ok(True)
            else:
                raise ServiceUnavailableError(f"Model loading failed: HTTP {response.status_code}")
        
        except requests.RequestException as e:
            error_msg = f"Model loading failed: {str(e)}"
            self.logger.error(
                "Model loading failed",
                error=e,
                model_name=self.model_name,
                suggestion=f"Install model with: ollama pull {self.model_name}",
            )
            
            return Err(ServiceUnavailableError(
                error_msg,
                {
                    'model_name': self.model_name,
                    'error_type': type(e).__name__,
                    'suggestion': f'ollama pull {self.model_name}',
                }
            ))
    
    @log_performance
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Dict[str, Any],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """Strukturierte KI-Anfrage mit JSON Schema Output"""
        
        # Model laden falls nötig
        model_result = self._ensure_model_loaded()
        if isinstance(model_result, Err):
            self.logger.error(
                "Model not available for generation",
                error=model_result.error.message,
            )
            return None
        
        for attempt in range(max_retries):
            try:
                self.logger.info(
                    "Starting structured generation",
                    model_name=self.model_name,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    temperature=temperature,
                )
                
                # Strukturierten Output-Prompt erstellen
                schema_prompt = self._create_schema_prompt(output_schema)
                
                # Kombinierter Prompt
                full_prompt = f"{system_prompt}\n\n{schema_prompt}\n\n{user_prompt}"
                
                # Ollama Request
                payload = {
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature + (attempt * 0.05),
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                        "stop": ["<|endoftext|>", "\n\n\n"],
                    },
                }
                
                # API Call
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                
                if response.status_code != 200:
                    raise AnalysisError(f"Ollama API error: HTTP {response.status_code}")
                
                response_data = response.json()
                
                if 'response' not in response_data:
                    raise AnalysisError("No 'response' field in Ollama response")
                
                # JSON Response parsen
                raw_response = response_data['response'].strip()
                
                self.logger.debug(
                    "Raw Ollama response",
                    response_length=len(raw_response),
                    response_preview=raw_response[:200],
                )
                
                try:
                    # JSON parsen
                    structured_output = json.loads(raw_response)
                    
                    # Validieren und verarbeiten
                    processed_output = self._process_structured_output(
                        structured_output,
                        output_schema,
                    )
                    
                    if processed_output:
                        self.logger.info(
                            "Structured generation successful",
                            attempt=attempt + 1,
                            output_keys=list(processed_output.keys()),
                        )
                        return processed_output
                    else:
                        raise ValueError("Output validation failed")
                
                except json.JSONDecodeError as json_error:
                    self.logger.warning(
                        "JSON parse error",
                        error=json_error,
                        attempt=attempt + 1,
                        raw_response_preview=raw_response[:100],
                    )
                    
                    # JSON-Reparatur versuchen
                    repaired = self._attempt_json_repair(raw_response)
                    if repaired:
                        processed = self._process_structured_output(repaired, output_schema)
                        if processed:
                            self.logger.info(
                                "JSON repair successful",
                                attempt=attempt + 1,
                            )
                            return processed
                    
                    # Bei letztem Versuch: Continue to next attempt
                    if attempt == max_retries - 1:
                        break
            
            except Exception as e:
                self.logger.warning(
                    "Generation attempt failed",
                    error=e,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                
                # Bei API-Fehlern sofort abbrechen
                if "HTTP" in str(e) and "API" in str(e):
                    break
        
        # Alle Versuche fehlgeschlagen - Fallback
        self.logger.error(
            "All generation attempts failed",
            max_retries=max_retries,
            model_name=self.model_name,
        )
        
        return self._create_fallback_output(output_schema)
    
    def _create_schema_prompt(self, output_schema: Dict[str, Any]) -> str:
        """Erstelle Schema-Prompt für strukturierte Ausgabe"""
        schema_json = json.dumps(output_schema, indent=2)
        
        return f"""
Du musst EXAKT in folgendem JSON-Format antworten:

{schema_json}

Wichtige Regeln:
- Nur valides JSON zurückgeben
- Alle Felder ausfüllen
- Keine zusätzlichen Erklärungen
- Numerische Werte als ZAHLEN (nicht Strings): 0.9 nicht "0.9"
- Score als Dezimalzahl zwischen 0.0 und 1.0
- Confidence als Dezimalzahl zwischen 0.0 und 1.0
- Arrays als JSON-Arrays: ["item1", "item2"]
"""
    
    def _process_structured_output(
        self,
        raw_output: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Verarbeite und validiere strukturierte Ausgabe"""
        try:
            processed = {}
            
            # Alle Schema-Felder verarbeiten
            for key, expected_type in schema.items():
                if key in raw_output:
                    value = raw_output[key]
                    
                    # Type-Konvertierung
                    if expected_type == "number" and isinstance(value, str):
                        try:
                            processed[key] = float(value)
                        except ValueError:
                            processed[key] = 0.5  # Fallback
                    elif expected_type == "array" and isinstance(value, str):
                        processed[key] = [value]  # String zu Array
                    elif expected_type == "array" and not isinstance(value, list):
                        processed[key] = []  # Fallback
                    else:
                        processed[key] = value
                else:
                    # Fehlende Felder mit Fallback füllen
                    if expected_type == "number":
                        processed[key] = 0.5
                    elif expected_type == "array":
                        processed[key] = []
                    elif expected_type == "string":
                        processed[key] = "unknown"
                    else:
                        processed[key] = None
            
            # Score-Validierung
            for key in processed:
                if "score" in key.lower() or "confidence" in key.lower():
                    value = processed[key]
                    if isinstance(value, (int, float)):
                        processed[key] = max(0.0, min(1.0, float(value)))
            
            return processed
        
        except Exception as e:
            self.logger.error(
                "Output processing failed",
                error=e,
                raw_output=raw_output,
            )
            return None
    
    def _attempt_json_repair(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """Versuche JSON-Reparatur"""
        try:
            # Entferne Markdown-Code-Blöcke
            cleaned = raw_response.strip()
            
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].strip()
            
            # Extrahiere JSON zwischen { und }
            if '{' in cleaned and '}' in cleaned:
                start = cleaned.find('{')
                end = cleaned.rfind('}') + 1
                cleaned = cleaned[start:end]
            
            # Versuche zu parsen
            return json.loads(cleaned)
        
        except Exception as e:
            self.logger.debug(
                "JSON repair failed",
                error=e,
                raw_response_preview=raw_response[:100],
            )
            return None
    
    def _create_fallback_output(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Erstelle Fallback-Output bei kompletten Fehlern"""
        fallback = {}
        
        for key, expected_type in schema.items():
            if expected_type == "number":
                fallback[key] = 0.3
            elif expected_type == "array":
                fallback[key] = ["analysis_error"]
            elif expected_type == "string":
                fallback[key] = "Fallback due to generation error"
            else:
                fallback[key] = None
        
        self.logger.warning(
            "Using fallback output",
            fallback_keys=list(fallback.keys()),
        )
        
        return fallback
    
    def get_service_status(self) -> ServiceStatus:
        """Service-Status für Monitoring"""
        try:
            # Server-Check
            server_result = self._check_server_status()
            if isinstance(server_result, Err):
                return ServiceStatus(
                    service_name="OllamaService",
                    status="unavailable",
                    message=server_result.error.message,
                    details={
                        'base_url': self.base_url,
                        'model_name': self.model_name,
                        'server_reachable': False,
                    },
                )
            
            # Model-Check
            model_result = self._ensure_model_loaded()
            if isinstance(model_result, Err):
                return ServiceStatus(
                    service_name="OllamaService",
                    status="error",
                    message=model_result.error.message,
                    details={
                        'base_url': self.base_url,
                        'model_name': self.model_name,
                        'server_reachable': True,
                        'model_loaded': False,
                    },
                )
            
            # Erfolgreicher Status
            return ServiceStatus(
                service_name="OllamaService",
                status="ready",
                message=f"Model {self.model_name} ready",
                details={
                    'base_url': self.base_url,
                    'model_name': self.model_name,
                    'server_reachable': True,
                    'model_loaded': True,
                    'gpu_info': self._gpu_info.dict() if self._gpu_info else None,
                },
            )
        
        except Exception as e:
            return ServiceStatus(
                service_name="OllamaService",
                status="error",
                message=f"Status check failed: {str(e)}",
                details={'error_type': type(e).__name__},
            )
    
    def is_ready(self) -> bool:
        """Service bereit für Analyse?"""
        server_result = self._check_server_status()
        if isinstance(server_result, Err):
            return False
        
        model_result = self._ensure_model_loaded()
        return isinstance(model_result, Ok)
    
    def get_gpu_info(self) -> Optional[GPUInfo]:
        """GPU-Informationen für Monitoring"""
        return self._gpu_info


# =============================================================================
# SERVICE FACTORY
# =============================================================================

_ollama_service_instance: Optional[OllamaService] = None


def get_ollama_service() -> OllamaService:
    """Singleton Factory für Ollama-Service"""
    global _ollama_service_instance
    
    if _ollama_service_instance is None:
        _ollama_service_instance = OllamaService()
    
    return _ollama_service_instance


def create_ollama_service(
    base_url: str = "http://localhost:11434",
    model_name: str = "gemma3:12b",
    timeout: int = 120,
) -> OllamaService:
    """Factory für neuen Ollama-Service mit spezifischen Parametern"""
    return OllamaService(base_url, model_name, timeout)


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def test_ollama_service() -> None:
    """Test-Funktion für Ollama-Service"""
    from youtube_analyzer.utils.logging import get_development_config
    from youtube_analyzer.utils.logging import setup_logging
    
    # Setup logging für Test
    setup_logging(get_development_config())
    
    service = get_ollama_service()
    logger = ComponentLogger("OllamaServiceTest")
    
    logger.info("Starting Ollama service test")
    
    # Test Service Status
    status = service.get_service_status()
    logger.info(
        "Service status",
        status=status.status,
        message=status.message,
        details=status.details,
    )
    
    # Test GPU Info
    gpu_info = service.get_gpu_info()
    if gpu_info:
        logger.info(
            "GPU info",
            device=gpu_info.device,
            name=gpu_info.name,
            available=gpu_info.available,
        )
    
    # Test Ready Status
    ready = service.is_ready()
    logger.info(
        "Service ready",
        ready=ready,
    )
    
    if ready:
        logger.info("Testing structured generation...")
        
        # Test Schema
        test_schema = {
            "score": "number",
            "confidence": "number",
            "reason": "string",
            "keywords": "array",
        }
        
        # Test Generation
        result = service.generate_structured(
            system_prompt="Du bewertest Text-Relevanz für technische Wissensdatenbanken.",
            user_prompt="Bewerte folgenden Text: 'Machine Learning ist ein wichtiger Bereich der KI.'",
            output_schema=test_schema,
            max_tokens=200,
        )
        
        if result:
            logger.info(
                "✅ Structured generation test passed",
                score=result.get('score'),
                confidence=result.get('confidence'),
                keywords_count=len(result.get('keywords', [])),
            )
        else:
            logger.error("❌ Structured generation test failed")
    else:
        logger.warning("⚠️ Service not ready, skipping generation test")
        
        if status.status == "unavailable":
            logger.info("💡 Start Ollama with: ollama serve")
        elif status.status == "error":
            logger.info(f"💡 Install model with: ollama pull {service.model_name}")
    
    logger.info("✅ Ollama service test completed")


if __name__ == "__main__":
    test_ollama_service()
