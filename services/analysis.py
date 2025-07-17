"""
Analysis Service - Regel-Engine für KI-gestützte Transkript-Analyse
Vollständig überarbeitet mit Result-Types und vollständigen Type-Hints
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel

from config.settings import get_config
from services.ollama import OllamaService
from services.ollama import get_ollama_service
from yt_types import AnalysisDecision
from yt_types  import AnalysisError
from yt_types  import AnalysisResult
from yt_types  import AnalysisScore
from yt_types  import ConfigurationError
from yt_types  import Err
from yt_types  import Ok
from yt_types  import Result
from yt_types  import ServiceStatus
from yt_types  import TranscriptionResult
from yt_types  import ValidationError
from utils.logging import ComponentLogger
from utils.logging import FeatureLogger
from utils.logging import log_feature_execution
from utils.logging import log_function_calls
from utils.logging import log_performance


class RuleSchema(BaseModel):
    """Schema für Analyse-Regel Output"""
    score: float
    confidence: float
    reason: str
    keywords: List[str]
    categories: List[str]
    
    class Config:
        extra = "allow"  # Erlaubt zusätzliche Felder


class AnalysisRule:
    """Einzelne Analyse-Regel mit Prompt und Gewichtung"""
    
    def __init__(
        self,
        name: str,
        prompt_file: Path,
        weight: float,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.prompt_file = prompt_file
        self.weight = weight
        self.enabled = enabled
        self.logger = ComponentLogger(f"AnalysisRule.{name}")
        self._prompt_content: Optional[str] = None
    
    @log_function_calls
    def load_prompt(self) -> Result[str, AnalysisError]:
        """Lade Prompt-Datei"""
        if self._prompt_content is not None:
            return Ok(self._prompt_content)
        
        try:
            if not self.prompt_file.exists():
                return Err(AnalysisError(
                    f"Prompt file not found: {self.prompt_file}",
                    {'rule_name': self.name, 'prompt_file': str(self.prompt_file)}
                ))
            
            self._prompt_content = self.prompt_file.read_text(encoding='utf-8')
            
            self.logger.debug(
                "Prompt loaded successfully",
                prompt_file=str(self.prompt_file),
                prompt_length=len(self._prompt_content),
            )
            
            return Ok(self._prompt_content)
        
        except Exception as e:
            error_msg = f"Failed to load prompt file: {str(e)}"
            self.logger.error(
                "Prompt loading failed",
                error=e,
                prompt_file=str(self.prompt_file),
            )
            
            return Err(AnalysisError(
                error_msg,
                {
                    'rule_name': self.name,
                    'prompt_file': str(self.prompt_file),
                    'error_type': type(e).__name__,
                }
            ))
    
    @log_performance
    def analyze(
        self,
        transcript: str,
        ollama_service: OllamaService,
    ) -> Result[AnalysisScore, AnalysisError]:
        """Analysiere Transkript mit dieser Regel"""
        
        # Prompt laden
        prompt_result = self.load_prompt()
        if isinstance(prompt_result, Err):
            return Err(prompt_result.error)
        
        prompt_template = prompt_result.value
        
        try:
            self.logger.info(
                "Starting rule analysis",
                rule_name=self.name,
                transcript_length=len(transcript),
            )
            
            # Prompt mit Transkript füllen
            full_prompt = prompt_template.format(transcript=transcript)
            
            # Schema für strukturierte Ausgabe
            output_schema = {
                "score": "number",
                "confidence": "number",
                "reason": "string",
                "keywords": "array",
                "categories": "array",
            }
            
            # Ollama-Analyse
            analysis_result = ollama_service.generate_structured(
                system_prompt=f"Du analysierst YouTube-Video-Transkripte nach der Regel '{self.name}'.",
                user_prompt=full_prompt,
                output_schema=output_schema,
                max_tokens=1000,
                temperature=0.1,
            )
            
            if not analysis_result:
                return Err(AnalysisError(
                    f"Ollama analysis failed for rule '{self.name}'",
                    {'rule_name': self.name, 'transcript_length': len(transcript)}
                ))
            
            # Ergebnis validieren und konvertieren
            rule_result = self._validate_rule_result(analysis_result)
            if isinstance(rule_result, Err):
                return Err(rule_result.error)
            
            rule_schema = rule_result.value
            
            # AnalysisScore erstellen
            analysis_score = AnalysisScore(
                rule_name=self.name,
                score=rule_schema.score,
                confidence=rule_schema.confidence,
                reason=rule_schema.reason,
                keywords=rule_schema.keywords,
                categories=rule_schema.categories,
            )
            
            self.logger.info(
                "Rule analysis completed",
                rule_name=self.name,
                score=analysis_score.score,
                confidence=analysis_score.confidence,
                keywords_count=len(analysis_score.keywords),
            )
            
            return Ok(analysis_score)
        
        except Exception as e:
            error_msg = f"Rule analysis failed: {str(e)}"
            self.logger.error(
                "Rule analysis failed",
                error=e,
                rule_name=self.name,
                transcript_length=len(transcript),
            )
            
            return Err(AnalysisError(
                error_msg,
                {
                    'rule_name': self.name,
                    'transcript_length': len(transcript),
                    'error_type': type(e).__name__,
                }
            ))
    
    def _validate_rule_result(self, result: Dict[str, Any]) -> Result[RuleSchema, AnalysisError]:
        """Validiere und konvertiere Regel-Ergebnis"""
        try:
            # Pydantic-Validierung
            rule_schema = RuleSchema(**result)
            
            # Zusätzliche Validierungen
            if not 0.0 <= rule_schema.score <= 1.0:
                return Err(AnalysisError(
                    f"Score outside valid range: {rule_schema.score}",
                    {'rule_name': self.name, 'score': rule_schema.score}
                ))
            
            if not 0.0 <= rule_schema.confidence <= 1.0:
                return Err(AnalysisError(
                    f"Confidence outside valid range: {rule_schema.confidence}",
                    {'rule_name': self.name, 'confidence': rule_schema.confidence}
                ))
            
            if not rule_schema.reason or len(rule_schema.reason.strip()) < 10:
                return Err(AnalysisError(
                    f"Reason too short or empty: {rule_schema.reason}",
                    {'rule_name': self.name, 'reason_length': len(rule_schema.reason)}
                ))
            
            return Ok(rule_schema)
        
        except Exception as e:
            error_msg = f"Result validation failed: {str(e)}"
            self.logger.error(
                "Result validation failed",
                error=e,
                rule_name=self.name,
                result=result,
            )
            
            return Err(AnalysisError(
                error_msg,
                {
                    'rule_name': self.name,
                    'result': result,
                    'error_type': type(e).__name__,
                }
            ))


class AnalysisService:
    """Service für komplette Transkript-Analyse mit Regel-Engine"""
    
    def __init__(self) -> None:
        self.logger = ComponentLogger("AnalysisService")
        self.rules: Dict[str, AnalysisRule] = {}
        self.ollama_service = get_ollama_service()
        self._load_rules()
    
    @log_function_calls
    def _load_rules(self) -> None:
        """Lade Analyse-Regeln aus Konfiguration"""
        try:
            config = get_config()
            
            for rule_name, rule_config in config.rules.items():
                if rule_config.enabled:
                    rule = AnalysisRule(
                        name=rule_name,
                        prompt_file=Path(rule_config.file),
                        weight=rule_config.weight,
                        enabled=rule_config.enabled,
                    )
                    self.rules[rule_name] = rule
            
            self.logger.info(
                "Analysis rules loaded",
                total_rules=len(config.rules),
                enabled_rules=len(self.rules),
                rules=list(self.rules.keys()),
            )
        
        except Exception as e:
            self.logger.error(
                "Failed to load analysis rules",
                error=e,
            )
            raise ConfigurationError(f"Failed to load analysis rules: {str(e)}")
    
    @log_performance
    def analyze_transcript(self, transcription: TranscriptionResult) -> Result[AnalysisResult, AnalysisError]:
        """Vollständige Transkript-Analyse mit allen Regeln"""
        
        if not self.rules:
            return Err(AnalysisError(
                "No analysis rules configured",
                {'configured_rules': 0}
            ))
        
        with log_feature_execution(
            self.logger,
            "transcript_analysis",
            transcript_length=len(transcription.text),
            rules_count=len(self.rules),
        ) as feature_logger:
            
            try:
                # Alle Regeln durchlaufen
                rule_scores: List[AnalysisScore] = []
                total_weighted_score = 0.0
                total_weight = 0.0
                
                for rule_name, rule in self.rules.items():
                    feature_logger.progress(
                        f"Analyzing with rule: {rule_name}",
                        int((len(rule_scores) / len(self.rules)) * 100),
                        current_rule=rule_name,
                    )
                    
                    # Regel-Analyse
                    rule_result = rule.analyze(transcription.text, self.ollama_service)
                    
                    if isinstance(rule_result, Err):
                        self.logger.error(
                            "Rule analysis failed",
                            rule_name=rule_name,
                            error=rule_result.error.message,
                        )
                        
                        # Fallback-Score für fehlgeschlagene Regel
                        fallback_score = AnalysisScore(
                            rule_name=rule_name,
                            score=0.0,
                            confidence=0.0,
                            reason=f"Rule analysis failed: {rule_result.error.message}",
                            keywords=["analysis_error"],
                            categories=["error"],
                        )
                        rule_scores.append(fallback_score)
                        
                        # Gewichtung für Fallback
                        total_weighted_score += 0.0 * rule.weight
                        total_weight += rule.weight
                    else:
                        score = rule_result.value
                        rule_scores.append(score)
                        
                        # Gewichtete Summe
                        total_weighted_score += score.score * rule.weight
                        total_weight += rule.weight
                        
                        feature_logger.add_metric(f"rule_{rule_name}_score", score.score)
                        feature_logger.add_metric(f"rule_{rule_name}_confidence", score.confidence)
                
                # Finale Bewertung
                final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
                
                # Entscheidung treffen
                decision = self._make_decision(final_score, rule_scores)
                
                # AnalysisResult erstellen
                analysis_result = AnalysisResult(
                    video_id=transcription.text[:50],  # Pseudo-ID from transcript
                    rule_scores=rule_scores,
                    final_score=final_score,
                    decision=decision,
                    processing_time=time.time(),  # Placeholder
                    created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                )
                
                feature_logger.add_metric("final_score", final_score)
                feature_logger.add_metric("decision", decision.value)
                feature_logger.add_metric("successful_rules", len([s for s in rule_scores if s.score > 0]))
                
                self.logger.info(
                    "Transcript analysis completed",
                    final_score=final_score,
                    decision=decision.value,
                    successful_rules=len([s for s in rule_scores if s.score > 0]),
                    failed_rules=len([s for s in rule_scores if s.score == 0]),
                )
                
                return Ok(analysis_result)
            
            except Exception as e:
                error_msg = f"Transcript analysis failed: {str(e)}"
                self.logger.error(
                    "Transcript analysis failed",
                    error=e,
                    transcript_length=len(transcription.text),
                )
                
                return Err(AnalysisError(
                    error_msg,
                    {
                        'transcript_length': len(transcription.text),
                        'rules_count': len(self.rules),
                        'error_type': type(e).__name__,
                    }
                ))
    
    @log_function_calls
    def analyze_with_rule(
        self,
        transcript: str,
        rule_name: str,
    ) -> Result[AnalysisScore, AnalysisError]:
        """Analysiere Transkript mit spezifischer Regel"""
        
        if rule_name not in self.rules:
            return Err(AnalysisError(
                f"Rule not found: {rule_name}",
                {'rule_name': rule_name, 'available_rules': list(self.rules.keys())}
            ))
        
        rule = self.rules[rule_name]
        
        self.logger.info(
            "Starting single rule analysis",
            rule_name=rule_name,
            transcript_length=len(transcript),
        )
        
        return rule.analyze(transcript, self.ollama_service)
    
    def _make_decision(self, final_score: float, rule_scores: List[AnalysisScore]) -> AnalysisDecision:
        """Entscheidung basierend auf Score und Confidence"""
        try:
            config = get_config()
            threshold = config.scoring.threshold
            min_confidence = config.scoring.min_confidence
            
            # Durchschnittliche Confidence berechnen
            total_confidence = sum(score.confidence for score in rule_scores)
            avg_confidence = total_confidence / len(rule_scores) if rule_scores else 0.0
            
            # Entscheidungslogik
            if final_score >= threshold and avg_confidence >= min_confidence:
                return AnalysisDecision.APPROVE
            elif final_score < threshold * 0.5 or avg_confidence < min_confidence * 0.5:
                return AnalysisDecision.REJECT
            else:
                return AnalysisDecision.MANUAL_REVIEW
        
        except Exception as e:
            self.logger.error(
                "Decision making failed",
                error=e,
                final_score=final_score,
                rule_scores_count=len(rule_scores),
            )
            
            # Fallback-Entscheidung
            return AnalysisDecision.MANUAL_REVIEW
    
    def get_service_status(self) -> ServiceStatus:
        """Service-Status für Monitoring"""
        try:
            # Prüfe Ollama-Service
            if not self.ollama_service.is_ready():
                return ServiceStatus(
                    service_name="AnalysisService",
                    status="error",
                    message="Ollama service not ready",
                    details={
                        'rules_loaded': len(self.rules),
                        'ollama_ready': False,
                    },
                )
            
            # Prüfe Regeln
            if not self.rules:
                return ServiceStatus(
                    service_name="AnalysisService",
                    status="error",
                    message="No analysis rules loaded",
                    details={
                        'rules_loaded': 0,
                        'ollama_ready': True,
                    },
                )
            
            return ServiceStatus(
                service_name="AnalysisService",
                status="ready",
                message=f"Ready with {len(self.rules)} rules",
                details={
                    'rules_loaded': len(self.rules),
                    'rule_names': list(self.rules.keys()),
                    'ollama_ready': True,
                },
            )
        
        except Exception as e:
            self.logger.error(
                "Service status check failed",
                error=e,
            )
            
            return ServiceStatus(
                service_name="AnalysisService",
                status="error",
                message=f"Status check failed: {str(e)}",
                details={'error_type': type(e).__name__},
            )
    
    def is_ready(self) -> bool:
        """Service bereit für Analyse?"""
        return bool(self.rules) and self.ollama_service.is_ready()
    
    def get_rules_info(self) -> Dict[str, Any]:
        """Regel-Informationen für Monitoring"""
        return {
            'total_rules': len(self.rules),
            'rule_names': list(self.rules.keys()),
            'rule_details': {
                name: {
                    'weight': rule.weight,
                    'enabled': rule.enabled,
                    'prompt_file': str(rule.prompt_file),
                    'prompt_loaded': rule._prompt_content is not None,
                }
                for name, rule in self.rules.items()
            },
        }


# =============================================================================
# SERVICE FACTORY
# =============================================================================

_analysis_service_instance: Optional[AnalysisService] = None


def get_analysis_service() -> AnalysisService:
    """Singleton Factory für Analysis-Service"""
    global _analysis_service_instance
    
    if _analysis_service_instance is None:
        _analysis_service_instance = AnalysisService()
    
    return _analysis_service_instance


def create_analysis_service() -> AnalysisService:
    """Factory für neuen Analysis-Service"""
    return AnalysisService()


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def test_analysis_service() -> None:
    """Test-Funktion für Analysis-Service"""
    from youtube_analyzer.utils.logging import get_development_config
    from youtube_analyzer.utils.logging import setup_logging
    
    # Setup logging für Test
    setup_logging(get_development_config())
    
    service = get_analysis_service()
    logger = ComponentLogger("AnalysisServiceTest")
    
    logger.info("Starting analysis service test")
    
    # Test Service Status
    status = service.get_service_status()
    logger.info(
        "Service status",
        status=status.status,
        message=status.message,
        details=status.details,
    )
    
    # Test Rules Info
    rules_info = service.get_rules_info()
    logger.info(
        "Rules info",
        total_rules=rules_info['total_rules'],
        rule_names=rules_info['rule_names'],
    )
    
    # Test mit Mock-Transkript
    mock_transcription = TranscriptionResult(
        text="This is a test transcription about machine learning and AI development.",
        language="en",
        confidence=0.95,
        processing_time=10.0,
        model_name="large-v3",
        device="cuda",
    )
    
    if service.is_ready():
        logger.info("Testing transcript analysis...")
        
        # Test komplette Analyse
        analysis_result = service.analyze_transcript(mock_transcription)
        
        if isinstance(analysis_result, Ok):
            result = analysis_result.value
            logger.info(
                "✅ Analysis test passed",
                final_score=result.final_score,
                decision=result.decision.value,
                rules_count=len(result.rule_scores),
            )
        else:
            logger.error(
                "❌ Analysis test failed",
                error=analysis_result.error.message,
            )
    else:
        logger.warning("⚠️ Service not ready, skipping analysis test")
    
    logger.info("✅ Analysis service test completed")


if __name__ == "__main__":
    test_analysis_service()
