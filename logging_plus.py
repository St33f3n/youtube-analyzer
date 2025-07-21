"""
logging_plus.py – Universelle Logging-Bibliothek für den Entwicklungsalltag

Features:
  - Multi-Level-Logging: Function(DEBUG), Feature(INFO), Error(ERROR)
  - Result-Types-Integration: Direkte Unterstützung für Ok/Err
  - Structured Error-Logging mit System-Context
  - Performance-Tracking mit Metrics
  - Type-Safe API mit vollständiger Annotation
"""

import sys
import time
import traceback
import psutil
import threading
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from loguru import logger

# Import our core types
from core_types import Result, is_ok, unwrap_ok, unwrap_err, CoreError

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

T = TypeVar("T")
LoggerType = logger.__class__

# =============================================================================
# SETUP LOGGING
# =============================================================================


def setup_logging(
    name: str = "app",
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_file: Optional[str] = None,
    enable_console: bool = True,
    enable_structured: bool = True,
) -> None:
    """
    Konfiguriert mehrstufiges Logging-System

    Args:
        name: Default-Komponenten-Name
        level: Minimum Log-Level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional: Pfad für Text-Log-Datei
        json_file: Optional: Pfad für JSON-Log-Datei
        enable_console: Console-Output aktivieren
        enable_structured: Structured JSON-Logging aktivieren
    """
    logger.remove()

    # Console-Handler für Entwicklung
    if enable_console:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[component]}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            format=console_format,
            level=level.upper(),
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # File-Handler für Persistence
    if log_file:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{extra[component]} | {message}"
        )
        logger.add(
            log_file,
            format=file_format,
            level=level.upper(),
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )

    # JSON-Handler für Structured Logging
    if json_file and enable_structured:
        logger.add(
            json_file,
            level=level.upper(),
            format="{message}",
            serialize=True,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )

    # Default-Kontext setzen
    logger.configure(extra={"component": name})


# =============================================================================
# LOGGER INSTANCES
# =============================================================================


def get_logger(name: str) -> LoggerType:
    """
    Erstellt Logger-Instanz mit Komponenten-Kontext

    Args:
        name: Komponenten-Name für Logging-Context

    Returns:
        Konfigurierte Logger-Instanz
    """
    return logger.bind(component=name)


# =============================================================================
# RESULT-TYPES LOGGING INTEGRATION
# =============================================================================


def log_result(
    result: Result[T, Exception],
    operation: str,
    log: Optional[LoggerType] = None,
    success_level: str = "info",
    error_level: str = "error",
) -> Result[T, Exception]:
    """
    Loggt Result-Types automatisch basierend auf Ok/Err

    Args:
        result: Result-Objekt zum Loggen
        operation: Beschreibung der Operation
        log: Optional: Logger-Instanz (default: get_logger)
        success_level: Log-Level für Ok-Results
        error_level: Log-Level für Err-Results

    Returns:
        Unveränderte Result (pass-through)
    """
    if log is None:
        log = get_logger("result_logger")

    if is_ok(result):
        value = unwrap_ok(result)
        getattr(log, success_level)(
            f"✓ {operation} succeeded",
            extra={
                "operation": operation,
                "result_type": "success",
                "value_type": type(value).__name__,
                "has_value": True,
            },
        )
    else:
        error = unwrap_err(result)
        error_context = {}

        # Erweiterte Kontext-Extraktion für CoreError
        if isinstance(error, CoreError):
            error_context = error.to_dict()
        else:
            error_context = {"error_type": type(error).__name__, "message": str(error)}

        getattr(log, error_level)(
            f"✗ {operation} failed",
            extra={
                "operation": operation,
                "result_type": "error",
                **error_context,
                "system_state": get_system_state(),
            },
        )

    return result


def log_and_unwrap(
    result: Result[T, Exception],
    operation: str,
    default: Optional[T] = None,
    log: Optional[LoggerType] = None,
) -> Optional[T]:
    """
    Loggt Result und entpackt Wert sicher

    Args:
        result: Result zum Loggen und Entpacken
        operation: Operation-Beschreibung
        default: Default-Wert bei Fehler
        log: Optional Logger-Instanz

    Returns:
        Entpackter Wert oder Default
    """
    logged_result = log_result(result, operation, log)

    if is_ok(logged_result):
        return unwrap_ok(logged_result)
    else:
        return default


# =============================================================================
# SYSTEM STATE COLLECTION (für ERROR-Level)
# =============================================================================


def get_system_state() -> Dict[str, Any]:
    """
    Sammelt System-Zustandsinformationen für Error-Logging

    Returns:
        Dictionary mit System-Metriken
    """
    try:
        return {
            "memory_usage_percent": round(psutil.virtual_memory().percent, 2),
            "cpu_usage_percent": round(psutil.cpu_percent(interval=0.1), 2),
            "disk_usage_percent": round(psutil.disk_usage("/").percent, 2),
            "active_threads": threading.active_count(),
            "timestamp": datetime.now().isoformat(),
            "process_id": psutil.Process().pid,
        }
    except Exception as e:
        return {
            "error": f"Failed to collect system state: {e}",
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# FUNCTION-LEVEL LOGGING (DEBUG)
# =============================================================================


def log_function(
    log_args: bool = False, log_result: bool = False, log_performance: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator für Function-Level-Logging (DEBUG)

    Args:
        log_args: Funktions-Argumente loggen
        log_result: Return-Value loggen
        log_performance: Ausführungszeit loggen

    Returns:
        Decorated function mit Logging
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            log = get_logger(func.__module__ or "unknown")
            start_time = time.time()

            # Function-Entry-Logging
            entry_context = {
                "function": func.__name__,
                "module": func.__module__,
                "stage": "entry",
            }

            if log_args:
                entry_context.update(
                    {
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                        "has_args": len(args) > 0,
                        "has_kwargs": len(kwargs) > 0,
                    }
                )

            log.debug(f"→ {func.__name__}() called", extra=entry_context)

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Success-Logging
                success_context = {
                    "function": func.__name__,
                    "stage": "completion",
                    "success": True,
                }

                if log_performance:
                    success_context.update(
                        {
                            "duration_ms": round(duration * 1000, 2),
                            "performance_category": get_performance_category(duration),
                        }
                    )

                if log_result:
                    success_context.update(
                        {
                            "result_type": type(result).__name__,
                            "has_result": result is not None,
                        }
                    )

                log.debug(f"✓ {func.__name__}() completed", extra=success_context)
                return result

            except Exception as e:
                duration = time.time() - start_time

                # Error-Logging mit erweiterten Kontext
                error_context = {
                    "function": func.__name__,
                    "stage": "error",
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "duration_ms": round(duration * 1000, 2),
                    "system_state": get_system_state(),
                }

                log.error(f"✗ {func.__name__}() failed", extra=error_context)
                raise

        return wrapper

    return decorator


def get_performance_category(duration: float) -> str:
    """Kategorisiert Performance basierend auf Ausführungszeit"""
    if duration < 0.01:  # < 10ms
        return "fast"
    elif duration < 0.1:  # < 100ms
        return "normal"
    elif duration < 1.0:  # < 1s
        return "slow"
    else:
        return "very_slow"


# =============================================================================
# FEATURE-LEVEL LOGGING (INFO)
# =============================================================================


class FeatureLogger:
    """Context-Manager für Feature-Level-Logging (INFO)"""

    def __init__(self, name: str, expected_duration: Optional[float] = None):
        self.name = name
        self.expected_duration = expected_duration
        self.start_time = time.time()
        self.metrics: Dict[str, Any] = {}
        self.log = get_logger(name)
        self.warnings: list[str] = []
        self.steps_completed = 0
        self.total_steps: Optional[int] = None

    def add_metric(self, key: str, value: Any) -> None:
        """Fügt Business-Metrik hinzu"""
        self.metrics[key] = value

    def add_warning(self, warning: str) -> None:
        """Fügt Warnung für Feature-Completion hinzu"""
        self.warnings.append(warning)
        self.log.warning(f"⚠ {self.name}: {warning}")

    def set_progress(self, completed: int, total: Optional[int] = None) -> None:
        """Setzt Progress-Information"""
        self.steps_completed = completed
        if total is not None:
            self.total_steps = total

    def checkpoint(
        self, step_name: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Loggt Zwischenschritt"""
        self.log.debug(
            f"📍 {self.name} checkpoint: {step_name}",
            extra={
                "feature": self.name,
                "checkpoint": step_name,
                "context": context or {},
                "elapsed_ms": round((time.time() - self.start_time) * 1000, 2),
            },
        )

    def __enter__(self) -> "FeatureLogger":
        # Feature-Start-Logging
        start_context = {
            "feature": self.name,
            "stage": "start",
            "expected_duration": self.expected_duration,
            "start_time": datetime.now().isoformat(),
        }
        self.log.info(f"▶ Feature '{self.name}' started", extra=start_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration = time.time() - self.start_time

        # Feature-Completion-Context
        completion_context = {
            "feature": self.name,
            "stage": "completion",
            "duration_ms": round(duration * 1000, 2),
            "metrics": self.metrics,
            "warnings_count": len(self.warnings),
            "warnings": self.warnings,
            "end_time": datetime.now().isoformat(),
        }

        # Progress-Information
        if self.total_steps is not None:
            completion_context.update(
                {
                    "steps_completed": self.steps_completed,
                    "total_steps": self.total_steps,
                    "completion_rate": round(
                        self.steps_completed / self.total_steps * 100, 2
                    ),
                }
            )

        # Performance-Bewertung
        if self.expected_duration:
            performance_ratio = duration / self.expected_duration
            completion_context["performance_ratio"] = round(performance_ratio, 2)
            completion_context["performance_category"] = (
                "faster_than_expected"
                if performance_ratio < 0.8
                else "as_expected"
                if performance_ratio < 1.2
                else "slower_than_expected"
            )

        if exc_type:
            # Feature-Error mit vollständigem Kontext
            completion_context.update(
                {
                    "success": False,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                    "system_state": get_system_state(),
                }
            )
            self.log.error(f"✗ Feature '{self.name}' failed", extra=completion_context)
        else:
            # Feature-Success
            completion_context["success"] = True
            self.log.info(
                f"✓ Feature '{self.name}' completed", extra=completion_context
            )


def log_feature(name: str, expected_duration: Optional[float] = None) -> FeatureLogger:
    """
    Factory für Feature-Logger

    Args:
        name: Feature-Name
        expected_duration: Erwartete Ausführungszeit in Sekunden

    Returns:
        FeatureLogger-Context-Manager
    """
    return FeatureLogger(name, expected_duration)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def log_critical_error(
    error: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    recovery_suggestions: Optional[list[str]] = None,
    log: Optional[LoggerType] = None,
) -> None:
    """
    Loggt kritische Fehler mit umfassendem Kontext (ERROR-Level)

    Args:
        error: Exception-Objekt
        operation: Beschreibung der fehlgeschlagenen Operation
        context: Zusätzlicher Kontext
        recovery_suggestions: Vorschläge zur Fehlerbehebung
        log: Optional Logger-Instanz
    """
    if log is None:
        log = get_logger("critical_error")

    error_context = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {},
        "recovery_suggestions": recovery_suggestions or [],
        "system_state": get_system_state(),
        "stack_trace": traceback.format_exc(),
    }

    # Erweiterte Kontext-Extraktion für CoreError
    if isinstance(error, CoreError):
        error_context.update(error.to_dict())

    log.error(f"🚨 Critical error in {operation}", extra=error_context)


# =============================================================================
# RESULT-AWARE DECORATORS
# =============================================================================


def log_result_function(
    operation_name: Optional[str] = None,
    log_success: bool = True,
    log_errors: bool = True,
) -> Callable[
    [Callable[..., Result[T, Exception]]], Callable[..., Result[T, Exception]]
]:
    """
    Decorator für Funktionen die Result-Types zurückgeben

    Args:
        operation_name: Name der Operation (default: Funktionsname)
        log_success: Ok-Results loggen
        log_errors: Err-Results loggen

    Returns:
        Decorated function mit Result-Logging
    """

    def decorator(
        func: Callable[..., Result[T, Exception]],
    ) -> Callable[..., Result[T, Exception]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Result[T, Exception]:
            operation = operation_name or func.__name__
            log = get_logger(func.__module__ or "result_function")

            try:
                result = func(*args, **kwargs)

                if log_success or log_errors:
                    log_result(result, operation, log)

                return result

            except Exception as e:
                # Exception außerhalb Result-System
                log_critical_error(e, operation, log=log)
                raise

        return wrapper

    return decorator


# =============================================================================
# BEISPIEL-USAGE
# =============================================================================

if __name__ == "__main__":
    # Setup
    setup_logging("demo", "DEBUG")

    # Function-Level-Demo
    @log_function(log_args=True, log_performance=True)
    def process_data(data: str) -> str:
        time.sleep(0.1)  # Simulate work
        return f"processed_{data}"

    # Feature-Level-Demo
    with log_feature("data_processing", expected_duration=0.5) as feature:
        feature.checkpoint("validation")
        result = process_data("test_data")
        feature.add_metric("items_processed", 1)
        feature.checkpoint("completion")

    # Result-Logging-Demo
    from core_types import validate_score

    score_result = validate_score(0.8)
    logged_result = log_result(score_result, "score_validation")

    print("Demo completed - check logs!")
