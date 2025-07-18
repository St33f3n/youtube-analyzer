"""
YouTube Analyzer - Langchain Analysis Engine
Content-Analyse mit Ollama, Langchain und strukturierten Regel-Prompts
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError

# Langchain imports with new langchain-ollama package
try:
    from langchain_ollama import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    OllamaLLM = None
    PromptTemplate = None

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_err, unwrap_ok
from yt_analyzer_core import ProcessObject
from logging_plus import get_logger, log_feature, log_function
from yt_analyzer_config import AppConfig, EnhancedRuleSystem

# =============================================================================
# PYDANTIC RESPONSE SCHEMAS
# =============================================================================

class RuleAnalysis(BaseModel):
    """Analyse-Ergebnis für eine einzelne Regel"""
    rule_name: str = Field(description="Name der analysierten Regel")
    fulfilled: bool = Field(description="Ob die Regel erfüllt ist")
    confidence: float = Field(ge=0.0, le=1.0, description="Konfidenz der Bewertung (0.0-1.0)")
    score: float = Field(ge=0.0, le=1.0, description="Qualitäts-Score der Erfüllung (0.0-1.0)")
    reasoning: str = Field(description="Kurze Begründung der Bewertung")

class AnalysisResult(BaseModel):
    """Vollständiges Analyse-Ergebnis für ein Video"""
    rules_analysis: List[RuleAnalysis] = Field(description="Bewertung aller Regeln")
    overall_score: float = Field(ge=0.0, le=1.0, description="Gewichteter Gesamtscore")
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Durchschnittliche Konfidenz")
    recommendation: bool = Field(description="True = DOWNLOAD, False = SKIP")
    summary: str = Field(description="Kurze Zusammenfassung der Analyse")

# =============================================================================
# RULE PROMPT BUILDER (mit Caching)
# =============================================================================

class RulePromptBuilder:
    """Baut kombinierten Regel-Prompt aus Markdown-Files mit Caching"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger("RulePromptBuilder")
        self._cached_rules_prompt: Optional[str] = None
        self._cache_timestamp: Optional[float] = None
        self.cache_ttl = 300  # 5 Minuten Cache-Lebensdauer
    
    def get_combined_rules_prompt(self) -> Result[str, CoreError]:
        """Gibt kombinierten Regel-Prompt zurück (gecacht)"""
        current_time = time.time()
        
        # Check cache validity
        if (self._cached_rules_prompt and 
            self._cache_timestamp and 
            current_time - self._cache_timestamp < self.cache_ttl):
            
            self.logger.debug("Using cached rules prompt")
            return Ok(self._cached_rules_prompt)
        
        # Build new prompt
        try:
            with log_feature("build_rules_prompt") as feature:
                rule_system = EnhancedRuleSystem(self.config, Path("."))
                
                # Load rule prompts
                rules_result = rule_system.load_rule_prompts()
                if isinstance(rules_result, Err):
                    return rules_result
                
                loaded_rules = unwrap_ok(rules_result)
                enabled_rules = self.config.rules.get_enabled_rules()
                
                feature.add_metric("enabled_rules_count", len(enabled_rules))
                
                # Build combined prompt
                rules_sections = []
                total_weight = 0.0
                
                for rule_name, rule_content in loaded_rules.items():
                    if rule_name in enabled_rules:
                        rule_config = enabled_rules[rule_name]
                        total_weight += rule_config.weight
                        
                        rule_section = f"""
## Regel: {rule_name.upper()}
**Gewichtung:** {rule_config.weight} ({rule_config.weight * 100:.1f}%)
**Status:** ✅ Aktiviert

{rule_content}
"""
                        rules_sections.append(rule_section)
                
                combined_prompt = f"""# ANALYSE-REGELN (Gesamt-Gewichtung: {total_weight:.2f})
{''.join(rules_sections)}

# BEWERTUNGS-KRITERIEN
- **Threshold für DOWNLOAD:** {self.config.scoring.threshold}
- **Minimum Confidence:** {self.config.scoring.min_confidence}
- **Gewichtete Berechnung:** Σ(regel_score × regel_weight)

Analysiere den folgenden Transkript systematisch anhand jeder aktivierten Regel."""
                
                # Cache the result
                self._cached_rules_prompt = combined_prompt
                self._cache_timestamp = current_time
                
                feature.add_metric("total_weight", total_weight)
                feature.add_metric("prompt_length", len(combined_prompt))
                
                self.logger.info(
                    f"✅ Rules prompt built and cached",
                    extra={
                        'enabled_rules': list(enabled_rules.keys()),
                        'total_weight': total_weight,
                        'prompt_length': len(combined_prompt),
                        'cache_ttl': self.cache_ttl
                    }
                )
                
                return Ok(combined_prompt)
                
        except Exception as e:
            context = ErrorContext.create(
                "build_rules_prompt",
                suggestions=["Check rule files exist", "Verify config format", "Check file permissions"]
            )
            return Err(CoreError(f"Failed to build rules prompt: {e}", context))

# =============================================================================
# OLLAMA ANALYSIS ENGINE
# =============================================================================

class OllamaAnalysisEngine:
    """Langchain-basierte Analyse-Engine mit Ollama"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger("OllamaAnalysisEngine")
        self.rule_builder = RulePromptBuilder(config)
        self.ollama_llm: Optional[OllamaLLM] = None
        
        # Validate Langchain availability
        if not LANGCHAIN_AVAILABLE:
            self.logger.error("Langchain not available - install with: pip install langchain-ollama langchain")
            return
        
        # Initialize Ollama connection
        self._init_ollama()
    
    def _init_ollama(self) -> Result[None, CoreError]:
        """Initialisiert Ollama-Verbindung"""
        if not LANGCHAIN_AVAILABLE:
            context = ErrorContext.create(
                "init_ollama",
                suggestions=["Install langchain: pip install langchain-ollama langchain"]
            )
            return Err(CoreError("Langchain not available", context))
        
        try:
            self.ollama_llm = OllamaLLM(
                model=self.config.ollama.model,
                base_url=self.config.ollama.host,
                temperature=self.config.ollama.temperature,
                num_ctx=self.config.ollama.max_tokens,
                timeout=self.config.ollama.timeout
            )
            
            # Test connection
            test_result = self.ollama_llm.invoke("Test connection")
            
            self.logger.info(
                f"✅ Ollama connection established",
                extra={
                    'model': self.config.ollama.model,
                    'host': self.config.ollama.host,
                    'temperature': self.config.ollama.temperature,
                    'max_tokens': self.config.ollama.max_tokens
                }
            )
            
            return Ok(None)
            
        except Exception as e:
            context = ErrorContext.create(
                "init_ollama",
                input_data={
                    'model': self.config.ollama.model,
                    'host': self.config.ollama.host
                },
                suggestions=[
                    "Check if Ollama is running",
                    "Verify model is available",
                    "Check network connectivity to Ollama host",
                    "Install model: ollama pull " + self.config.ollama.model
                ]
            )
            return Err(CoreError(f"Failed to connect to Ollama: {e}", context))
    
    @log_function(log_performance=True)
    def analyze_transcript(self, transcript: str, video_title: str) -> Result[AnalysisResult, CoreError]:
        """
        Analysiert Transkript mit Regel-System
        
        Args:
            transcript: Video-Transkript (wird auf 30k chars truncated)
            video_title: Video-Titel für Debugging
            
        Returns:
            Ok(AnalysisResult): Strukturierte Analyse-Ergebnisse
            Err: Analyse-Fehler
        """
        if not self.ollama_llm:
            init_result = self._init_ollama()
            if isinstance(init_result, Err):
                return init_result
        
        try:
            with log_feature("transcript_analysis") as feature:
                # Truncate transcript to 30k characters
                original_length = len(transcript)
                truncated_transcript = transcript[:30000]
                
                if original_length > 30000:
                    truncated_transcript += "\n\n[TRANSCRIPT TRUNCATED - ORIGINAL LENGTH: {}]".format(original_length)
                
                feature.add_metric("original_length", original_length)
                feature.add_metric("truncated_length", len(truncated_transcript))
                feature.add_metric("was_truncated", original_length > 30000)
                
                self.logger.info(
                    f"🧠 Starting content analysis",
                    extra={
                        'video_title': video_title,
                        'transcript_length': original_length,
                        'analysis_length': len(truncated_transcript),
                        'truncated': original_length > 30000,
                        'model': self.config.ollama.model
                    }
                )
                
                # Build analysis prompt
                prompt_result = self._build_analysis_prompt(truncated_transcript)
                if isinstance(prompt_result, Err):
                    return prompt_result
                
                full_prompt = unwrap_ok(prompt_result)
                feature.add_metric("prompt_length", len(full_prompt))
                
                # Perform analysis with retry logic
                analysis_result = self._analyze_with_retries(full_prompt, max_retries=3)
                if isinstance(analysis_result, Err):
                    return analysis_result
                
                parsed_result = unwrap_ok(analysis_result)
                
                # Calculate final recommendation
                final_result = self._finalize_analysis(parsed_result)
                
                feature.add_metric("analysis_success", True)
                feature.add_metric("overall_score", final_result.overall_score)
                feature.add_metric("recommendation", final_result.recommendation)
                
                # Enhanced analysis completion logging
                analysis_info = (
                    f"✅ Content analysis completed:\n"
                    f"  🎯 Video: {video_title}\n"
                    f"  📊 Overall Score: {final_result.overall_score:.3f}\n"
                    f"  🎪 Confidence: {final_result.overall_confidence:.3f}\n"
                    f"  🎬 Recommendation: {'DOWNLOAD' if final_result.recommendation else 'SKIP'}\n"
                    f"  📝 Rules Analyzed: {len(final_result.rules_analysis)}\n"
                    f"  📄 Transcript Length: {original_length} chars\n"
                    f"  🤖 Model: {self.config.ollama.model}"
                )
                
                self.logger.info(analysis_info)
                
                return Ok(final_result)
                
        except Exception as e:
            context = ErrorContext.create(
                "analyze_transcript",
                input_data={'video_title': video_title, 'transcript_length': len(transcript)},
                suggestions=[
                    "Check Ollama connectivity",
                    "Verify model availability",
                    "Check rule configuration"
                ]
            )
            return Err(CoreError(f"Transcript analysis failed: {e}", context))
    
    def _build_analysis_prompt(self, transcript: str) -> Result[str, CoreError]:
        """Baut vollständigen Analysis-Prompt"""
        # Get cached rules prompt
        rules_prompt_result = self.rule_builder.get_combined_rules_prompt()
        if isinstance(rules_prompt_result, Err):
            return rules_prompt_result
        
        rules_prompt = unwrap_ok(rules_prompt_result)
        
        # Build complete prompt
        full_prompt = f"""{rules_prompt}

# TRANSKRIPT ZU ANALYSIEREN
{transcript}

# AUFGABE
Analysiere den Transkript systematisch anhand jeder aktivierten Regel:

1. **Regel-Bewertung** (pro Regel):
   - fulfilled: true/false (Regel erfüllt?)
   - confidence: 0.0-1.0 (Wie sicher bist du?)
   - score: 0.0-1.0 (Qualität der Erfüllung)
   - reasoning: Kurze Begründung

2. **Gesamt-Bewertung**:
   - overall_score: Gewichteter Gesamtscore
   - overall_confidence: Durchschnitt aller Regel-Konfidenzen
   - recommendation: true (DOWNLOAD) oder false (SKIP)

# OUTPUT-FORMAT
Antworte ausschließlich mit folgendem JSON-Format:

{{
    "rules_analysis": [
        {{
            "rule_name": "fachinhalt",
            "fulfilled": true,
            "confidence": 0.85,
            "score": 0.90,
            "reasoning": "Begründung..."
        }}
    ],
    "overall_score": 0.82,
    "overall_confidence": 0.88,
    "recommendation": true,
    "summary": "Kurze Zusammenfassung der Analyse"
}}"""
        
        return Ok(full_prompt)
    
    def _analyze_with_retries(self, prompt: str, max_retries: int = 3) -> Result[AnalysisResult, CoreError]:
        """Führt Analyse mit JSON-Parse-Retry-Logic durch"""
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Analysis attempt {attempt + 1}/{max_retries}")
                
                # Call Ollama
                response = self.ollama_llm.invoke(prompt)
                
                self.logger.debug(f"Ollama response received (length: {len(response)})")
                
                # Parse JSON response
                try:
                    # Clean response (remove potential markdown formatting)
                    cleaned_response = response.strip()
                    if cleaned_response.startswith("```json"):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3]
                    cleaned_response = cleaned_response.strip()
                    
                    # Parse JSON
                    json_data = json.loads(cleaned_response)
                    
                    # Validate with Pydantic
                    analysis_result = AnalysisResult(**json_data)
                    
                    self.logger.debug(f"JSON parsing successful on attempt {attempt + 1}")
                    return Ok(analysis_result)
                    
                except (json.JSONDecodeError, ValidationError) as e:
                    self.logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                    self.logger.debug(f"Raw response: {response[:500]}...")
                    
                    if attempt == max_retries - 1:  # Last attempt
                        context = ErrorContext.create(
                            "parse_analysis_response",
                            input_data={'attempt': attempt + 1, 'response_preview': response[:200]},
                            suggestions=[
                                "Check Ollama model capability",
                                "Verify prompt format",
                                "Try different model"
                            ]
                        )
                        return Err(CoreError(f"JSON parsing failed after {max_retries} attempts: {e}", context))
                    
                    continue  # Retry
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    context = ErrorContext.create(
                        "ollama_analysis",
                        input_data={'attempt': attempt + 1},
                        suggestions=[
                            "Check Ollama connectivity",
                            "Verify model is running",
                            "Check network connectivity"
                        ]
                    )
                    return Err(CoreError(f"Ollama analysis failed: {e}", context))
                
                self.logger.warning(f"Ollama error on attempt {attempt + 1}, retrying: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        return Err(CoreError("Analysis failed after all retry attempts"))
    
    def _finalize_analysis(self, result: AnalysisResult) -> AnalysisResult:
        """Finalisiert Analyse mit Threshold-Check und Weighted-Scoring"""
        
        # Recalculate overall_score using weights
        enabled_rules = self.config.rules.get_enabled_rules()
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for rule_analysis in result.rules_analysis:
            rule_name = rule_analysis.rule_name
            if rule_name in enabled_rules and rule_analysis.fulfilled:
                rule_weight = enabled_rules[rule_name].weight
                total_weighted_score += rule_analysis.score * rule_weight
                total_weight += rule_weight
        
        # Update overall score (weighted)
        if total_weight > 0:
            result.overall_score = total_weighted_score / total_weight
        
        # Apply decision logic
        result.recommendation = (
            result.overall_score >= self.config.scoring.threshold and
            result.overall_confidence >= self.config.scoring.min_confidence
        )
        
        return result

# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def analyze_process_object(process_obj: ProcessObject, config: AppConfig) -> Result[ProcessObject, CoreError]:
    """
    Standalone-Funktion für ProcessObject-Analyse
    
    Args:
        process_obj: ProcessObject mit transkript
        config: App-Konfiguration
        
    Returns:
        Ok(ProcessObject): ProcessObject mit analysis_results + passed_analysis
        Err: Analyse-Fehler
    """
    if not process_obj.transkript:
        context = ErrorContext.create(
            "analyze_process_object",
            input_data={'video_title': process_obj.titel},
            suggestions=["Ensure transcription completed", "Check pipeline order"]
        )
        return Err(CoreError("No transcript in ProcessObject", context))
    
    # Create analysis engine
    engine = OllamaAnalysisEngine(config)
    
    # Analyze transcript
    analysis_result = engine.analyze_transcript(process_obj.transkript, process_obj.titel)
    
    if isinstance(analysis_result, Err):
        return analysis_result
    
    analysis = unwrap_ok(analysis_result)
    
    # Update ProcessObject
    process_obj.rule_amount = sum(1 for r in analysis.rules_analysis if r.fulfilled)
    process_obj.rule_accuracy = analysis.overall_confidence
    process_obj.relevancy = analysis.overall_score
    process_obj.passed_analysis = analysis.recommendation
    process_obj.analysis_results = analysis.dict()
    process_obj.update_stage("analyzed")
    
    return Ok(process_obj)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    from yt_analyzer_config import SecureConfigManager
    from datetime import datetime, time as dt_time
    
    # Setup
    setup_logging("analysis_test", "DEBUG")
    
    # Test configuration
    config_manager = SecureConfigManager()
    config_result = config_manager.load_config()
    
    if isinstance(config_result, Ok):
        config = unwrap_ok(config_result)
        
        # Create test ProcessObject
        test_obj = ProcessObject(
            titel="Test Video",
            kanal="Test Channel",
            länge=dt_time(0, 5, 30),
            upload_date=datetime.now()
        )
        
        # Mock transcript
        test_obj.transkript = """
        Dies ist ein Test-Transkript für die Content-Analyse. 
        Das Video behandelt technische Themen mit guter Struktur und hoher Qualität.
        Der Inhalt ist relevant und bietet praktischen Nutzen für Zuschauer.
        Die Informationen sind aktuell und fachlich korrekt dargestellt.
        """
        
        # Test analysis
        result = analyze_process_object(test_obj, config)
        
        if isinstance(result, Ok):
            analyzed_obj = unwrap_ok(result)
            print(f"✅ Analysis successful:")
            print(f"  Score: {analyzed_obj.relevancy}")
            print(f"  Recommendation: {'DOWNLOAD' if analyzed_obj.passed_analysis else 'SKIP'}")
            print(f"  Rules fulfilled: {analyzed_obj.rule_amount}")
        else:
            error = unwrap_err(result)
            print(f"❌ Analysis failed: {error.message}")
    else:
        print("❌ Config loading failed")
