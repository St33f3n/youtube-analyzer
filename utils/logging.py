"""
Mehrstufiges Logging-System für YouTube Analyzer
Feature-Level | Function-Level | Error-Level Logging mit strukturierten Daten
GC-SAFE VERSION - Defensive Decorator-Implementierung für Garbage Collection
"""

from __future__ import annotations

import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Optional
from typing import TypeVar

try:
    import psutil
except ImportError:
    psutil = None

from loguru import logger

# Import der bestehenden Types
from yt_types import LogEntry, LogLevel, ProcessingStage

# Type Variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class LoggingConfig:
    """Configuration for logging system"""
    
    def __init__(
        self,
        log_level: LogLevel = LogLevel.INFO,
        log_file: Optional[Path] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_structured: bool = True,
        max_file_size: str = "10 MB",
        retention_days: int = 7,
        component_name: str = "youtube_analyzer",
    ) -> None:
        self.log_level = log_level
        self.log_file = log_file or Path("youtube_analyzer.log")
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_structured = enable_structured
        self.max_file_size = max_file_size
        self.retention_days = retention_days
        self.component_name = component_name


def setup_logging(config: LoggingConfig) -> None:
    """Setup mehrstufiges Logging-System"""
    
    # Remove default handler
    logger.remove()
    
    # Console Handler (Development)
    if config.enable_console:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[component]}</cyan> | "
            "<level>{message}</level>"
        )
        
        logger.add(
            sys.stdout,
            format=console_format,
            level=config.log_level.value,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    
    # File Handler (Production)
    if config.enable_file:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{extra[component]} | "
            "{extra[function]}:{extra[line]} | "
            "{message}"
        )
        
        logger.add(
            config.log_file,
            format=file_format,
            level=config.log_level.value,
            rotation=config.max_file_size,
            retention=f"{config.retention_days} days",
            compression="gz",
            backtrace=True,
            diagnose=True,
        )
    
    # Structured JSON Handler (Monitoring)
    if config.enable_structured:
        logger.add(
            config.log_file.with_suffix(".json"),
            format="{message}",
            level=LogLevel.INFO.value,
            serialize=True,
            rotation=config.max_file_size,
            retention=f"{config.retention_days} days",
            compression="gz",
        )
    
    # Bind default component
    logger.configure(
        extra={
            "component": config.component_name,
            "function": "",
            "line": "",
        }
    )


# =============================================================================
# STRUCTURED LOGGERS
# =============================================================================

class ComponentLogger:
    """Logger für spezifische Komponenten mit Feature-Level Logging"""
    
    def __init__(self, component_name: str) -> None:
        self.component_name = component_name
        self.logger = logger.bind(component=component_name)
    
    def info(self, message: str, **context: Any) -> None:
        """Feature-Level INFO Logging"""
        try:
            self.logger.info(
                message,
                extra={
                    "log_type": "feature",
                    "context": context,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
    
    def debug(self, message: str, **context: Any) -> None:
        """Function-Level DEBUG Logging"""
        try:
            self.logger.debug(
                message,
                extra={
                    "log_type": "function", 
                    "context": context,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
    
    def error(self, message: str, error: Optional[Exception] = None, **context: Any) -> None:
        """Error-Level ERROR Logging"""
        try:
            error_context = {
                "log_type": "error",
                "context": context,
                "timestamp": datetime.now().isoformat(),
            }
            
            if error:
                error_context.update({
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "stack_trace": traceback.format_exc(),
                })
            
            self.logger.error(message, extra=error_context)
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
    
    def warning(self, message: str, **context: Any) -> None:
        """Warning-Level Logging"""
        try:
            self.logger.warning(
                message,
                extra={
                    "log_type": "warning",
                    "context": context,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass


class FeatureLogger:
    """Feature-Level Logging für größere Operationen"""
    
    def __init__(self, component_logger: ComponentLogger, feature_name: str) -> None:
        self.component_logger = component_logger
        self.feature_name = feature_name
        self.start_time = time.time()
        self.metrics: Dict[str, Any] = {}
    
    def started(self, **context: Any) -> None:
        """Feature-Start Logging"""
        try:
            self.component_logger.info(
                f"Feature '{self.feature_name}' started",
                feature_name=self.feature_name,
                feature_stage="started",
                **context
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
    
    def progress(self, message: str, progress_percent: int, **context: Any) -> None:
        """Feature-Progress Logging"""
        try:
            self.component_logger.info(
                f"Feature '{self.feature_name}' progress: {message}",
                feature_name=self.feature_name,
                feature_stage="progress",
                progress_percent=progress_percent,
                **context
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
    
    def completed(self, **context: Any) -> None:
        """Feature-Completion Logging"""
        try:
            duration = time.time() - self.start_time
            
            self.component_logger.info(
                f"Feature '{self.feature_name}' completed successfully",
                feature_name=self.feature_name,
                feature_stage="completed",
                duration_seconds=duration,
                metrics=self.metrics,
                **context
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
    
    def failed(self, error: Exception, **context: Any) -> None:
        """Feature-Failure Logging"""
        try:
            duration = time.time() - self.start_time
            
            self.component_logger.error(
                f"Feature '{self.feature_name}' failed",
                error=error,
                feature_name=self.feature_name,
                feature_stage="failed",
                duration_seconds=duration,
                metrics=self.metrics,
                **context
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add metrics to feature logging"""
        try:
            self.metrics[name] = value
        except Exception:
            # GC-Safety: Ignore metric storage errors
            pass


class ProcessingLogger:
    """Processing-Pipeline Logging für mehrstufige Operationen"""
    
    def __init__(self, component_logger: ComponentLogger, process_id: str) -> None:
        self.component_logger = component_logger
        self.process_id = process_id
        self.current_stage: Optional[ProcessingStage] = None
        self.stage_start_time = time.time()
        self.total_start_time = time.time()
        self.stage_metrics: Dict[ProcessingStage, Dict[str, Any]] = {}
    
    def stage_started(self, stage: ProcessingStage, **context: Any) -> None:
        """Processing-Stage-Start Logging"""
        try:
            self.current_stage = stage
            self.stage_start_time = time.time()
            
            self.component_logger.info(
                f"Processing stage '{stage.value}' started",
                process_id=self.process_id,
                processing_stage=stage.value,
                stage_status="started",
                **context
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
    
    def stage_progress(self, message: str, progress_percent: int, **context: Any) -> None:
        """Processing-Stage-Progress Logging"""
        try:
            if not self.current_stage:
                return
            
            self.component_logger.info(
                f"Processing stage '{self.current_stage.value}' progress: {message}",
                process_id=self.process_id,
                processing_stage=self.current_stage.value,
                stage_status="progress",
                progress_percent=progress_percent,
                **context
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
    
    def stage_completed(self, **context: Any) -> None:
        """Processing-Stage-Completion Logging"""
        try:
            if not self.current_stage:
                return
            
            duration = time.time() - self.stage_start_time
            
            # Store stage metrics
            self.stage_metrics[self.current_stage] = {
                "duration_seconds": duration,
                "context": context,
                "completed_at": datetime.now().isoformat(),
            }
            
            self.component_logger.info(
                f"Processing stage '{self.current_stage.value}' completed",
                process_id=self.process_id,
                processing_stage=self.current_stage.value,
                stage_status="completed",
                duration_seconds=duration,
                **context
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
    
    def stage_failed(self, error: Exception, **context: Any) -> None:
        """Processing-Stage-Failure Logging"""
        try:
            if not self.current_stage:
                return
            
            duration = time.time() - self.stage_start_time
            
            self.component_logger.error(
                f"Processing stage '{self.current_stage.value}' failed",
                error=error,
                process_id=self.process_id,
                processing_stage=self.current_stage.value,
                stage_status="failed",
                duration_seconds=duration,
                **context
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
    
    def process_completed(self, **context: Any) -> None:
        """Complete Processing-Pipeline Logging"""
        try:
            total_duration = time.time() - self.total_start_time
            
            self.component_logger.info(
                f"Processing pipeline completed",
                process_id=self.process_id,
                processing_status="completed",
                total_duration_seconds=total_duration,
                stages_completed=len(self.stage_metrics),
                stage_metrics=self.stage_metrics,
                **context
            )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass


# =============================================================================
# GC-SAFE DECORATORS
# =============================================================================

def log_performance(func: F) -> F:
    """
    GC-Safe Decorator für automatisches Performance-Logging ohne Parameter
    """
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        operation_name = func.__name__
        start_time = time.time()
        
        # GC-Safe Logger Discovery
        func_logger = None
        try:
            if (args and 
                hasattr(args[0], 'logger') and 
                args[0].logger is not None and
                hasattr(args[0].logger, 'info')):
                # Methode in Klasse mit self.logger
                func_logger = args[0].logger
            else:
                # Fallback auf default ComponentLogger
                func_logger = ComponentLogger("performance")
        except Exception:
            # GC-Safety: Erstelle immer einen Logger
            func_logger = ComponentLogger("performance")
        
        try:
            # Function execution
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Performance-Logging (GC-Safe)
            try:
                if func_logger and hasattr(func_logger, 'info'):
                    func_logger.info(
                        f"Performance: {operation_name} completed",
                        operation_name=operation_name,
                        duration_seconds=round(duration, 4),
                        success=True,
                    )
            except Exception:
                # GC-Safety: Ignore logging errors
                pass
            
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            
            # Error-Performance-Logging (GC-Safe)
            try:
                if func_logger and hasattr(func_logger, 'error'):
                    func_logger.error(
                        f"Performance: {operation_name} failed",
                        error=e,
                        operation_name=operation_name,
                        duration_seconds=round(duration, 4),
                        success=False,
                    )
            except Exception:
                # GC-Safety: Ignore logging errors
                pass
            
            # Re-raise original exception
            raise
    
    return wrapper  # type: ignore


def log_function_calls(func: F) -> F:
    """
    GC-Safe Decorator für function_calls (parameterlos)
    Für DEBUG-Level Function-Entry/Exit Logging
    """
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__name__
        
        # GC-Safe Logger Discovery
        func_logger = None
        try:
            if (args and 
                hasattr(args[0], 'logger') and 
                args[0].logger is not None and
                hasattr(args[0].logger, 'debug')):
                func_logger = args[0].logger
            else:
                func_logger = ComponentLogger("function_calls")
        except Exception:
            # GC-Safety: Erstelle immer einen Fallback-Logger
            try:
                func_logger = ComponentLogger("function_calls")
            except Exception:
                # Extreme GC-Safety: Wenn auch das fehlschlägt, ignore logging
                func_logger = None
        
        # Function-Entry DEBUG Logging (GC-Safe)
        try:
            if func_logger and hasattr(func_logger, 'debug'):
                func_logger.debug(
                    f"Function '{func_name}' called",
                    function_name=func_name,
                    args_count=len(args),
                )
        except Exception:
            # GC-Safety: Ignore logging errors
            pass
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Function-Success DEBUG Logging (GC-Safe)
            try:
                if func_logger and hasattr(func_logger, 'debug'):
                    func_logger.debug(
                        f"Function '{func_name}' completed successfully",
                        function_name=func_name,
                        duration_seconds=round(duration, 4),
                    )
            except Exception:
                # GC-Safety: Ignore logging errors
                pass
            
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            
            # Function-Error DEBUG Logging (GC-Safe)
            try:
                if func_logger and hasattr(func_logger, 'debug'):
                    func_logger.debug(
                        f"Function '{func_name}' failed",
                        function_name=func_name,
                        duration_seconds=round(duration, 4),
                        error_type=type(e).__name__,
                    )
            except Exception:
                # GC-Safety: Ignore logging errors
                pass
            
            raise
    
    return wrapper  # type: ignore


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@contextmanager
def log_feature_execution(
    component_logger: ComponentLogger,
    feature_name: str,
    **initial_context: Any
) -> Generator[FeatureLogger, None, None]:
    """Context Manager für Feature-Level Logging"""
    
    feature_logger = FeatureLogger(component_logger, feature_name)
    feature_logger.started(**initial_context)
    
    try:
        yield feature_logger
        feature_logger.completed()
    
    except Exception as e:
        feature_logger.failed(e)
        raise


# =============================================================================
# LOGGER FACTORY
# =============================================================================

def get_logger(component_name: str) -> ComponentLogger:
    """Factory function für Component-Logger"""
    return ComponentLogger(component_name)


def get_feature_logger(component_logger: ComponentLogger, feature_name: str) -> FeatureLogger:
    """Factory function für Feature-Logger"""
    return FeatureLogger(component_logger, feature_name)


def get_processing_logger(component_logger: ComponentLogger, process_id: str) -> ProcessingLogger:
    """Factory function für Processing-Logger"""
    return ProcessingLogger(component_logger, process_id)


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

def get_development_config() -> LoggingConfig:
    """Development logging configuration"""
    return LoggingConfig(
        log_level=LogLevel.DEBUG,
        enable_console=True,
        enable_file=True,
        enable_structured=False,
        component_name="youtube_analyzer_dev",
    )


def get_production_config() -> LoggingConfig:
    """Production logging configuration"""
    return LoggingConfig(
        log_level=LogLevel.INFO,
        enable_console=False,
        enable_file=True,
        enable_structured=True,
        component_name="youtube_analyzer",
    )


def get_testing_config() -> LoggingConfig:
    """Testing logging configuration"""
    return LoggingConfig(
        log_level=LogLevel.WARNING,
        enable_console=True,
        enable_file=False,
        enable_structured=False,
        component_name="youtube_analyzer_test",
    )
