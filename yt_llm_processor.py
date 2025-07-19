"""
YouTube Analyzer - LLM Processing Integration
Drei separate Provider-Funktionen + Dispatcher für Fork-Join Architecture
config_dict-basiert mit resolved secrets (keine AppConfig dependency)
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

# LLM Provider Imports (drei separate packages)
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.callbacks.manager import get_openai_callback
    OPENAI_AVAILABLE = True
except ImportError as e:
    OPENAI_AVAILABLE = False
    import sys
    print(f"OpenAI not available: {e}", file=sys.stderr)

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError as e:
    ANTHROPIC_AVAILABLE = False
    import sys
    print(f"Anthropic not available: {e}", file=sys.stderr)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError as e:
    GOOGLE_AVAILABLE = False
    import sys
    print(f"Google not available: {e}", file=sys.stderr)

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_err, unwrap_ok
from yt_analyzer_core import TranskriptObject
from logging_plus import get_logger, log_feature, log_function

# =============================================================================
# SYSTEM PROMPT LOADING (User-provided, no Default)
# =============================================================================

def load_system_prompt_from_config(config_dict: dict) -> Result[str, CoreError]:
    """
    Lädt System-Prompt aus User-definierter Datei
    
    Args:
        config_dict: Config mit llm_processing.system_prompt_file
        
    Returns:
        Ok(str): System-Prompt-Content
        Err: File-Loading-Fehler (KEIN Default-Fallback)
    """
    try:
        system_prompt_file = config_dict['llm_processing']['system_prompt_file']
        prompt_path = Path(system_prompt_file)
        
        if not prompt_path.exists():
            context = ErrorContext.create(
                "load_system_prompt",
                input_data={'prompt_file': system_prompt_file},
                suggestions=[
                    f"Create system prompt file: {prompt_path}",
                    "Add your LLM processing instructions",
                    "No default prompt provided - user must create file"
                ]
            )
            return Err(CoreError(f"System prompt file not found: {prompt_path}", context))
        
        prompt_content = prompt_path.read_text(encoding='utf-8')
        
        if not prompt_content.strip():
            context = ErrorContext.create(
                "load_system_prompt",
                input_data={'prompt_file': system_prompt_file},
                suggestions=["Add content to system prompt file", "File exists but is empty"]
            )
            return Err(CoreError(f"System prompt file is empty: {prompt_path}", context))
        
        return Ok(prompt_content.strip())
        
    except Exception as e:
        context = ErrorContext.create(
            "load_system_prompt",
            input_data={'config_dict_keys': list(config_dict.get('llm_processing', {}).keys())},
            suggestions=["Check file permissions", "Verify file encoding", "Check config_dict structure"]
        )
        return Err(CoreError(f"Failed to load system prompt: {e}", context))

# =============================================================================
# PROVIDER-SPECIFIC FUNCTIONS (drei isolierte Implementierungen)
# =============================================================================

@log_function(log_performance=True)
def process_with_openai(transcript_obj: TranskriptObject, config_dict: dict) -> Result[TranskriptObject, CoreError]:
    """OpenAI-spezifische LLM-Verarbeitung"""
    
    logger = get_logger("OpenAI_LLM")
    
    if not OPENAI_AVAILABLE:
        context = ErrorContext.create(
            "openai_processing",
            suggestions=["Install langchain-openai: pip install langchain-openai"]
        )
        return Err(CoreError("OpenAI provider not available", context))
    
    # API-Key aus resolved secrets
    resolved_secrets = config_dict.get('resolved_secrets', {})
    api_key = resolved_secrets.get('llm_api_key')
    
    if not api_key:
        context = ErrorContext.create(
            "openai_api_key",
            input_data={'resolved_secrets_keys': list(resolved_secrets.keys())},
            suggestions=[
                "Check Pipeline Manager secret resolution",
                "Ensure OpenAI API key is configured in keyring",
                "Verify config_dict contains resolved_secrets"
            ]
        )
        return Err(CoreError("OpenAI API key not resolved", context))
    
    try:
        with log_feature("openai_llm_processing") as feature:
            feature.add_metric("transcript_title", transcript_obj.titel)
            feature.add_metric("transcript_length", len(transcript_obj.transkript))
            
            # LLM Settings aus config_dict
            llm_config = config_dict['llm_processing']
            model = llm_config['model']
            temperature = llm_config['temperature']
            max_tokens = llm_config['max_tokens']
            timeout = llm_config['timeout']
            retry_attempts = llm_config['retry_attempts']
            retry_delay = llm_config['retry_delay']
            
            feature.add_metric("model", model)
            feature.add_metric("max_tokens", max_tokens)
            
            logger.info(
                f"Starting OpenAI LLM processing: {transcript_obj.titel}",
                extra={
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'api_key_length': len(api_key),
                    'transcript_length': len(transcript_obj.transkript)
                }
            )
            
            # System-Prompt laden
            system_prompt_result = load_system_prompt_from_config(config_dict)
            if isinstance(system_prompt_result, Err):
                return system_prompt_result
            
            system_prompt = unwrap_ok(system_prompt_result)
            feature.add_metric("system_prompt_length", len(system_prompt))
            
            # OpenAI LLM initialisieren
            llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            
            # Messages erstellen (Full-Transcript, keine Truncation)
            messages = [
                ("system", system_prompt),
                ("user", f"Please process the following YouTube transcript:\n\n{transcript_obj.transkript}")
            ]
            
            feature.add_metric("total_input_length", len(system_prompt) + len(transcript_obj.transkript))
            
            # Retry-Logic (einfach, keine exponential backoff)
            start_time = time.time()
            
            for attempt in range(retry_attempts):
                try:
                    feature.checkpoint(f"attempt_{attempt + 1}")
                    
                    logger.debug(f"OpenAI attempt {attempt + 1}/{retry_attempts}")
                    
                    # API-Call mit Callback für Metrics
                    with get_openai_callback() as callback:
                        response = llm.invoke(messages)
                    
                    # Success → TranskriptObject aktualisieren
                    processing_time = time.time() - start_time
                    
                    transcript_obj.bearbeiteter_transkript = response.content
                    transcript_obj.model = model
                    transcript_obj.tokens = callback.total_tokens
                    transcript_obj.cost = callback.total_cost
                    transcript_obj.processing_time = processing_time
                    transcript_obj.success = True
                    transcript_obj.update_stage("llm_processing_completed")
                    
                    feature.add_metric("success", True)
                    feature.add_metric("tokens_used", callback.total_tokens)
                    feature.add_metric("cost_usd", callback.total_cost)
                    feature.add_metric("processing_time", processing_time)
                    feature.add_metric("attempts_needed", attempt + 1)
                    
                    logger.info(
                        f"OpenAI LLM processing completed successfully: {transcript_obj.titel}",
                        extra={
                            'model': model,
                            'tokens': callback.total_tokens,
                            'cost': callback.total_cost,
                            'processing_time': processing_time,
                            'attempt': attempt + 1,
                            'response_length': len(response.content)
                        }
                    )
                    
                    return Ok(transcript_obj)
                
                except Exception as e:
                    logger.warning(f"OpenAI attempt {attempt + 1} failed: {e}")
                    
                    if attempt < retry_attempts - 1:
                        logger.debug(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Final failure
                        processing_time = time.time() - start_time
                        error_message = f"OpenAI processing failed after {retry_attempts} attempts: {e}"
                        
                        transcript_obj.success = False
                        transcript_obj.error_message = error_message
                        transcript_obj.processing_time = processing_time
                        transcript_obj.update_stage("llm_processing_failed")
                        
                        feature.add_metric("success", False)
                        feature.add_metric("final_error", str(e))
                        feature.add_metric("attempts_used", retry_attempts)
                        
                        context = ErrorContext.create(
                            "openai_processing",
                            input_data={
                                'model': model,
                                'attempts': retry_attempts,
                                'error': str(e)
                            },
                            suggestions=[
                                "Check OpenAI API key validity",
                                "Verify network connectivity",
                                "Check OpenAI service status",
                                "Try different model if rate limited"
                            ]
                        )
                        return Err(CoreError(error_message, context))
            
            # Should never reach here
            return Err(CoreError("OpenAI processing failed unexpectedly"))
            
    except Exception as e:
        # Exception außerhalb Retry-Loop
        transcript_obj.success = False
        transcript_obj.error_message = f"OpenAI processing exception: {e}"
        
        context = ErrorContext.create(
            "openai_processing_exception",
            input_data={'error_type': type(e).__name__},
            suggestions=["Check OpenAI library installation", "Verify config_dict structure"]
        )
        return Err(CoreError(f"OpenAI processing failed: {e}", context))

@log_function(log_performance=True)
def process_with_anthropic(transcript_obj: TranskriptObject, config_dict: dict) -> Result[TranskriptObject, CoreError]:
    """Anthropic-spezifische LLM-Verarbeitung"""
    
    logger = get_logger("Anthropic_LLM")
    
    if not ANTHROPIC_AVAILABLE:
        context = ErrorContext.create(
            "anthropic_processing",
            suggestions=["Install langchain-anthropic: pip install langchain-anthropic"]
        )
        return Err(CoreError("Anthropic provider not available", context))
    
    # API-Key aus resolved secrets
    resolved_secrets = config_dict.get('resolved_secrets', {})
    api_key = resolved_secrets.get('llm_api_key')
    
    if not api_key:
        context = ErrorContext.create(
            "anthropic_api_key",
            input_data={'resolved_secrets_keys': list(resolved_secrets.keys())},
            suggestions=[
                "Check Pipeline Manager secret resolution",
                "Ensure Anthropic API key is configured in keyring",
                "Verify config_dict contains resolved_secrets"
            ]
        )
        return Err(CoreError("Anthropic API key not resolved", context))
    
    try:
        with log_feature("anthropic_llm_processing") as feature:
            feature.add_metric("transcript_title", transcript_obj.titel)
            feature.add_metric("transcript_length", len(transcript_obj.transkript))
            
            # LLM Settings aus config_dict
            llm_config = config_dict['llm_processing']
            model = llm_config['model']
            temperature = llm_config['temperature']
            max_tokens = llm_config['max_tokens']
            timeout = llm_config['timeout']
            retry_attempts = llm_config['retry_attempts']
            retry_delay = llm_config['retry_delay']
            
            feature.add_metric("model", model)
            feature.add_metric("max_tokens", max_tokens)
            
            logger.info(
                f"Starting Anthropic LLM processing: {transcript_obj.titel}",
                extra={
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'api_key_length': len(api_key),
                    'transcript_length': len(transcript_obj.transkript)
                }
            )
            
            # System-Prompt laden
            system_prompt_result = load_system_prompt_from_config(config_dict)
            if isinstance(system_prompt_result, Err):
                return system_prompt_result
            
            system_prompt = unwrap_ok(system_prompt_result)
            feature.add_metric("system_prompt_length", len(system_prompt))
            
            # Anthropic LLM initialisieren
            llm = ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            
            # Messages erstellen (Full-Transcript, keine Truncation)
            messages = [
                ("system", system_prompt),
                ("user", f"Please process the following YouTube transcript:\n\n{transcript_obj.transkript}")
            ]
            
            feature.add_metric("total_input_length", len(system_prompt) + len(transcript_obj.transkript))
            
            # Retry-Logic (einfach, keine exponential backoff)
            start_time = time.time()
            
            for attempt in range(retry_attempts):
                try:
                    feature.checkpoint(f"attempt_{attempt + 1}")
                    
                    logger.debug(f"Anthropic attempt {attempt + 1}/{retry_attempts}")
                    
                    # API-Call (Anthropic hat keinen Standard-Callback wie OpenAI)
                    # Usage info wird direkt aus response.response_metadata extrahiert
                    response = llm.invoke(messages)
                    
                    # Extract usage info from response if available
                    tokens_used = 0
                    cost_estimate = 0.0
                    
                    if hasattr(response, 'response_metadata') and response.response_metadata:
                        usage = response.response_metadata.get('usage', {})
                        tokens_used = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
                        
                        # Rough cost estimation for Claude (approximate rates)
                        input_tokens = usage.get('input_tokens', 0)
                        output_tokens = usage.get('output_tokens', 0)
                        
                        # Claude-3 Sonnet approximate pricing (per 1M tokens)
                        input_cost_per_1m = 3.0   # $3 per 1M input tokens
                        output_cost_per_1m = 15.0  # $15 per 1M output tokens
                        
                        cost_estimate = (input_tokens * input_cost_per_1m / 1_000_000) + (output_tokens * output_cost_per_1m / 1_000_000)
                    
                    # Success → TranskriptObject aktualisieren
                    processing_time = time.time() - start_time
                    
                    transcript_obj.bearbeiteter_transkript = response.content
                    transcript_obj.model = model
                    transcript_obj.tokens = tokens_used
                    transcript_obj.cost = cost_estimate
                    transcript_obj.processing_time = processing_time
                    transcript_obj.success = True
                    transcript_obj.update_stage("llm_processing_completed")
                    
                    feature.add_metric("success", True)
                    feature.add_metric("tokens_used", tokens_used)
                    feature.add_metric("cost_usd", cost_estimate)
                    feature.add_metric("processing_time", processing_time)
                    feature.add_metric("attempts_needed", attempt + 1)
                    
                    logger.info(
                        f"Anthropic LLM processing completed successfully: {transcript_obj.titel}",
                        extra={
                            'model': model,
                            'tokens': tokens_used,
                            'cost': cost_estimate,
                            'processing_time': processing_time,
                            'attempt': attempt + 1,
                            'response_length': len(response.content)
                        }
                    )
                    
                    return Ok(transcript_obj)
                
                except Exception as e:
                    logger.warning(f"Anthropic attempt {attempt + 1} failed: {e}")
                    
                    if attempt < retry_attempts - 1:
                        logger.debug(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Final failure
                        processing_time = time.time() - start_time
                        error_message = f"Anthropic processing failed after {retry_attempts} attempts: {e}"
                        
                        transcript_obj.success = False
                        transcript_obj.error_message = error_message
                        transcript_obj.processing_time = processing_time
                        transcript_obj.update_stage("llm_processing_failed")
                        
                        feature.add_metric("success", False)
                        feature.add_metric("final_error", str(e))
                        feature.add_metric("attempts_used", retry_attempts)
                        
                        context = ErrorContext.create(
                            "anthropic_processing",
                            input_data={
                                'model': model,
                                'attempts': retry_attempts,
                                'error': str(e)
                            },
                            suggestions=[
                                "Check Anthropic API key validity",
                                "Verify network connectivity",
                                "Check Anthropic service status",
                                "Try different model if rate limited"
                            ]
                        )
                        return Err(CoreError(error_message, context))
            
            # Should never reach here
            return Err(CoreError("Anthropic processing failed unexpectedly"))
            
    except Exception as e:
        # Exception außerhalb Retry-Loop
        transcript_obj.success = False
        transcript_obj.error_message = f"Anthropic processing exception: {e}"
        
        context = ErrorContext.create(
            "anthropic_processing_exception",
            input_data={'error_type': type(e).__name__},
            suggestions=["Check Anthropic library installation", "Verify config_dict structure"]
        )
        return Err(CoreError(f"Anthropic processing failed: {e}", context))

@log_function(log_performance=True)
def process_with_google(transcript_obj: TranskriptObject, config_dict: dict) -> Result[TranskriptObject, CoreError]:
    """Google-spezifische LLM-Verarbeitung"""
    
    logger = get_logger("Google_LLM")
    
    if not GOOGLE_AVAILABLE:
        context = ErrorContext.create(
            "google_processing",
            suggestions=["Install langchain-google-genai: pip install langchain-google-genai"]
        )
        return Err(CoreError("Google provider not available", context))
    
    # API-Key aus resolved secrets
    resolved_secrets = config_dict.get('resolved_secrets', {})
    api_key = resolved_secrets.get('llm_api_key')
    
    if not api_key:
        context = ErrorContext.create(
            "google_api_key",
            input_data={'resolved_secrets_keys': list(resolved_secrets.keys())},
            suggestions=[
                "Check Pipeline Manager secret resolution",
                "Ensure Google API key is configured in keyring",
                "Verify config_dict contains resolved_secrets"
            ]
        )
        return Err(CoreError("Google API key not resolved", context))
    
    try:
        with log_feature("google_llm_processing") as feature:
            feature.add_metric("transcript_title", transcript_obj.titel)
            feature.add_metric("transcript_length", len(transcript_obj.transkript))
            
            # LLM Settings aus config_dict
            llm_config = config_dict['llm_processing']
            model = llm_config['model']
            temperature = llm_config['temperature']
            max_tokens = llm_config['max_tokens']
            retry_attempts = llm_config['retry_attempts']
            retry_delay = llm_config['retry_delay']
            
            feature.add_metric("model", model)
            feature.add_metric("max_tokens", max_tokens)
            
            logger.info(
                f"Starting Google LLM processing: {transcript_obj.titel}",
                extra={
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'api_key_length': len(api_key),
                    'transcript_length': len(transcript_obj.transkript)
                }
            )
            
            # System-Prompt laden
            system_prompt_result = load_system_prompt_from_config(config_dict)
            if isinstance(system_prompt_result, Err):
                return system_prompt_result
            
            system_prompt = unwrap_ok(system_prompt_result)
            feature.add_metric("system_prompt_length", len(system_prompt))
            
            # Google LLM initialisieren
            llm = ChatGoogleGenerativeAI(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            # Messages erstellen (Full-Transcript, keine Truncation)
            messages = [
                ("system", system_prompt),
                ("user", f"Please process the following YouTube transcript:\n\n{transcript_obj.transkript}")
            ]
            
            feature.add_metric("total_input_length", len(system_prompt) + len(transcript_obj.transkript))
            
            # Retry-Logic (einfach, keine exponential backoff)
            start_time = time.time()
            
            for attempt in range(retry_attempts):
                try:
                    feature.checkpoint(f"attempt_{attempt + 1}")
                    
                    logger.debug(f"Google attempt {attempt + 1}/{retry_attempts}")
                    
                    # API-Call (Google hat keinen Standard-Callback verfügbar)
                    # Token/Cost-Tracking fehlt für Google - minimale Metrics
                    response = llm.invoke(messages)
                    
                    # Success → TranskriptObject aktualisieren
                    processing_time = time.time() - start_time
                    
                    transcript_obj.bearbeiteter_transkript = response.content
                    transcript_obj.model = model
                    transcript_obj.tokens = 0  # Google callback nicht standardisiert
                    transcript_obj.cost = 0.0  # Google pricing berechnung fehlt
                    transcript_obj.processing_time = processing_time
                    transcript_obj.success = True
                    transcript_obj.update_stage("llm_processing_completed")
                    
                    feature.add_metric("success", True)
                    feature.add_metric("tokens_used", 0)  # Not available for Google
                    feature.add_metric("cost_usd", 0.0)  # Not available for Google
                    feature.add_metric("processing_time", processing_time)
                    feature.add_metric("attempts_needed", attempt + 1)
                    
                    logger.info(
                        f"Google LLM processing completed successfully: {transcript_obj.titel}",
                        extra={
                            'model': model,
                            'tokens': 0,  # Not available
                            'cost': 0.0,  # Not available
                            'processing_time': processing_time,
                            'attempt': attempt + 1,
                            'response_length': len(response.content)
                        }
                    )
                    
                    return Ok(transcript_obj)
                
                except Exception as e:
                    logger.warning(f"Google attempt {attempt + 1} failed: {e}")
                    
                    if attempt < retry_attempts - 1:
                        logger.debug(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Final failure
                        processing_time = time.time() - start_time
                        error_message = f"Google processing failed after {retry_attempts} attempts: {e}"
                        
                        transcript_obj.success = False
                        transcript_obj.error_message = error_message
                        transcript_obj.processing_time = processing_time
                        transcript_obj.update_stage("llm_processing_failed")
                        
                        feature.add_metric("success", False)
                        feature.add_metric("final_error", str(e))
                        feature.add_metric("attempts_used", retry_attempts)
                        
                        context = ErrorContext.create(
                            "google_processing",
                            input_data={
                                'model': model,
                                'attempts': retry_attempts,
                                'error': str(e)
                            },
                            suggestions=[
                                "Check Google API key validity",
                                "Verify network connectivity",
                                "Check Google AI service status",
                                "Try different model if quota exceeded"
                            ]
                        )
                        return Err(CoreError(error_message, context))
            
            # Should never reach here
            return Err(CoreError("Google processing failed unexpectedly"))
            
    except Exception as e:
        # Exception außerhalb Retry-Loop
        transcript_obj.success = False
        transcript_obj.error_message = f"Google processing exception: {e}"
        
        context = ErrorContext.create(
            "google_processing_exception",
            input_data={'error_type': type(e).__name__},
            suggestions=["Check Google GenAI library installation", "Verify config_dict structure"]
        )
        return Err(CoreError(f"Google processing failed: {e}", context))

# =============================================================================
# DISPATCHER FUNCTION (Provider-Selection)
# =============================================================================

@log_function(log_performance=True)
def process_transcript_with_llm_dict(transcript_obj: TranskriptObject, config_dict: dict) -> Result[TranskriptObject, CoreError]:
    """
    Haupt-Dispatcher für LLM-Verarbeitung mit Provider-Selection
    
    Args:
        transcript_obj: TranskriptObject mit transkript
        config_dict: Config-Dict mit resolved_secrets + llm_processing
        
    Returns:
        Ok(TranskriptObject): TranskriptObject mit bearbeiteter_transkript + metrics
        Err: LLM-Processing-Fehler
    """
    logger = get_logger("LLM_Dispatcher")
    
    # Validate input
    if not transcript_obj.transkript:
        context = ErrorContext.create(
            "llm_dispatcher",
            input_data={'video_title': transcript_obj.titel},
            suggestions=["Ensure transcription completed", "Check pipeline order"]
        )
        return Err(CoreError("No transcript in TranskriptObject", context))
    
    # Provider aus config_dict extrahieren
    try:
        provider = config_dict['llm_processing']['provider']
    except KeyError:
        context = ErrorContext.create(
            "llm_dispatcher",
            input_data={'config_dict_keys': list(config_dict.keys())},
            suggestions=["Check config_dict structure", "Ensure llm_processing section exists"]
        )
        return Err(CoreError("LLM provider not configured in config_dict", context))
    
    logger.info(
        f"Dispatching LLM processing to {provider}",
        extra={
            'provider': provider,
            'video_title': transcript_obj.titel,
            'transcript_length': len(transcript_obj.transkript),
            'model': config_dict['llm_processing'].get('model', 'unknown')
        }
    )
    
    # Provider-spezifische Verarbeitung
    provider_functions = {
        "openai": process_with_openai,
        "anthropic": process_with_anthropic,
        "google": process_with_google
    }
    
    if provider not in provider_functions:
        context = ErrorContext.create(
            "llm_dispatcher",
            input_data={'provider': provider, 'available_providers': list(provider_functions.keys())},
            suggestions=[
                f"Use one of: {', '.join(provider_functions.keys())}",
                "Update provider in config.yaml",
                "Check provider spelling"
            ]
        )
        return Err(CoreError(f"Unknown LLM provider: {provider}", context))
    
    # Delegate to provider-specific function
    process_function = provider_functions[provider]
    result = process_function(transcript_obj, config_dict)
    
    if isinstance(result, Ok):
        processed_obj = unwrap_ok(result)
        logger.info(
            f"LLM processing successful via {provider}",
            extra={
                'provider': provider,
                'video_title': processed_obj.titel,
                'tokens': processed_obj.tokens,
                'cost': processed_obj.cost,
                'processing_time': processed_obj.processing_time,
                'response_length': len(processed_obj.bearbeiteter_transkript) if processed_obj.bearbeiteter_transkript else 0
            }
        )
    else:
        error = unwrap_err(result)
        logger.error(
            f"LLM processing failed via {provider}",
            extra={
                'provider': provider,
                'video_title': transcript_obj.titel,
                'error': error.message
            }
        )
    
    return result

# =============================================================================
# PROVIDER AVAILABILITY CHECK
# =============================================================================

def get_available_providers() -> Dict[str, bool]:
    """Gibt verfügbare LLM-Provider zurück"""
    return {
        "openai": OPENAI_AVAILABLE,
        "anthropic": ANTHROPIC_AVAILABLE,
        "google": GOOGLE_AVAILABLE
    }

def check_provider_availability(provider: str) -> Result[None, CoreError]:
    """Prüft ob Provider verfügbar ist"""
    availability = get_available_providers()
    
    if provider not in availability:
        return Err(CoreError(f"Unknown provider: {provider}"))
    
    if not availability[provider]:
        context = ErrorContext.create(
            "check_provider_availability",
            input_data={'provider': provider},
            suggestions=[
                f"Install langchain-{provider}: pip install langchain-{provider}",
                "Check provider library installation"
            ]
        )
        return Err(CoreError(f"Provider {provider} not available", context))
    
    return Ok(None)

# =============================================================================
# EXAMPLE USAGE & TESTING
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    from datetime import datetime
    
    # Setup
    setup_logging("llm_processor_test", "DEBUG")
    
    # Test config_dict (wie vom Pipeline Manager)
    test_config_dict = {
        'llm_processing': {
            'provider': 'openai',
            'model': 'gpt-4',
            'system_prompt_file': 'prompts/transcript_processing.md',
            'temperature': 0.1,
            'max_tokens': 4000,
            'retry_attempts': 3,
            'retry_delay': 2,
            'timeout': 300
        },
        'resolved_secrets': {
            'llm_api_key': 'test-api-key-12345'  # Mock API key
        }
    }
    
    # Create test TranskriptObject
    test_obj = TranskriptObject(
        titel="Test Video",
        transkript="This is a test transcript for LLM processing. The video discusses technical topics."
    )
    
    print("🧪 Testing LLM Processor Implementation")
    print("=" * 50)
    
    # Test provider availability mit Debug-Info
    available_providers = get_available_providers()
    print(f"Available providers: {available_providers}")
    
    # Debug: Warum sind Provider nicht verfügbar?
    if not available_providers['openai']:
        print("❌ OpenAI nicht verfügbar - Install: pip install langchain-openai langchain-community")
    if not available_providers['anthropic']:
        print("❌ Anthropic nicht verfügbar - Install: pip install langchain-anthropic")
    if not available_providers['google']:
        print("❌ Google nicht verfügbar - Install: pip install langchain-google-genai")
    
    print(f"✅ Available: {[p for p, available in available_providers.items() if available]}")
    print(f"❌ Missing: {[p for p, available in available_providers.items() if not available]}")
    
    # Test system prompt loading
    prompt_result = load_system_prompt_from_config(test_config_dict)
    if isinstance(prompt_result, Ok):
        prompt = unwrap_ok(prompt_result)
        print(f"✅ System prompt loaded: {len(prompt)} characters")
    else:
        error = unwrap_err(prompt_result)
        print(f"❌ System prompt loading failed: {error.message}")
    
    # Test dispatcher
    if available_providers.get('openai', False):
        result = process_transcript_with_llm_dict(test_obj, test_config_dict)
        
        if isinstance(result, Ok):
            processed_obj = unwrap_ok(result)
            print(f"✅ LLM processing successful:")
            print(f"  Model: {processed_obj.model}")
            print(f"  Tokens: {processed_obj.tokens}")
            print(f"  Cost: ${processed_obj.cost:.4f}")
            print(f"  Processing time: {processed_obj.processing_time:.2f}s")
            print(f"  Success: {processed_obj.success}")
        else:
            error = unwrap_err(result)
            print(f"❌ LLM processing failed: {error.message}")
    else:
        print("⚠️ No LLM providers available for testing")
    
    print("\n🚀 LLM Processor Implementation Complete!")
    print("Features:")
    print("- Drei isolierte Provider-Funktionen (OpenAI, Anthropic, Google)")
    print("- config_dict-basierte Configuration")
    print("- System-Prompt aus User-File (kein Default)")
    print("- Token/Cost-Tracking mit Provider-Callbacks")
    print("- Robuste Retry-Logic ohne exponential backoff")
    print("- Fork-Join-Architecture-kompatibles Error-Handling")
