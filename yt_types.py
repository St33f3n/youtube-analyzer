"""
Type System für YouTube Analyzer
Vollständige Type-Definitionen für alle Module
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import TypeAlias
from typing import TypeVar
from typing import Union

import numpy as np
from pydantic import BaseModel
from pydantic import Field

# =============================================================================
# CORE RESULT TYPES
# =============================================================================

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Success result container"""
    value: T


@dataclass(frozen=True)
class Err(Generic[E]):
    """Error result container"""
    error: E


Result: TypeAlias = Union[Ok[T], Err[E]]


# =============================================================================
# CONFIGURATION TYPES
# =============================================================================

class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ProcessingStage(str, Enum):
    """Processing pipeline stages"""
    INFO = "info"
    AUDIO = "audio"
    TRANSCRIPTION = "transcription"
    ANALYSIS = "analysis"
    STORAGE = "storage"
    COMPLETED = "completed"


class AnalysisDecision(str, Enum):
    """Analysis decision outcomes"""
    APPROVE = "approve"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"


# =============================================================================
# ERROR TYPES
# =============================================================================

class YouTubeAnalyzerError(Exception):
    """Base exception for YouTube Analyzer"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ConfigurationError(YouTubeAnalyzerError):
    """Configuration related errors"""
    pass


class ValidationError(YouTubeAnalyzerError):
    """Data validation errors"""
    pass


class ProcessingError(YouTubeAnalyzerError):
    """Data processing errors"""
    pass


class DownloadError(YouTubeAnalyzerError):
    """Download related errors"""
    pass


class TranscriptionError(YouTubeAnalyzerError):
    """Audio transcription errors"""
    pass


class AnalysisError(YouTubeAnalyzerError):
    """AI analysis errors"""
    pass


class StorageError(YouTubeAnalyzerError):
    """Storage operation errors"""
    pass


class ServiceUnavailableError(YouTubeAnalyzerError):
    """External service unavailable errors"""
    pass


# =============================================================================
# DATA MODELS
# =============================================================================

class VideoMetadata(BaseModel):
    """YouTube video metadata"""
    id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    uploader: str = Field(..., description="Channel name")
    duration: int = Field(..., description="Duration in seconds")
    view_count: int = Field(default=0, description="View count")
    upload_date: str = Field(..., description="Upload date (YYYYMMDD)")
    description: str = Field(default="", description="Video description")
    webpage_url: str = Field(..., description="Full YouTube URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    format_count: int = Field(default=0, description="Available formats")
    
    class Config:
        frozen = True


class AudioMetadata(BaseModel):
    """Audio file metadata"""
    file_path: Path = Field(..., description="Audio file path")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Audio format (wav, mp3, etc.)")
    duration: float = Field(..., description="Duration in seconds")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    channels: int = Field(..., description="Number of audio channels")
    
    class Config:
        frozen = True


class TranscriptionResult(BaseModel):
    """Whisper transcription result"""
    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Detected language")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_name: str = Field(..., description="Whisper model used")
    device: str = Field(..., description="Processing device (cpu/cuda)")
    
    class Config:
        frozen = True


class AnalysisScore(BaseModel):
    """Individual analysis rule score"""
    rule_name: str = Field(..., description="Name of the analysis rule")
    score: float = Field(..., ge=0.0, le=1.0, description="Score value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level")
    reason: str = Field(..., description="Explanation for the score")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    categories: List[str] = Field(default_factory=list, description="Content categories")
    
    class Config:
        frozen = True


class AnalysisResult(BaseModel):
    """Complete analysis result"""
    video_id: str = Field(..., description="YouTube video ID")
    rule_scores: List[AnalysisScore] = Field(..., description="Individual rule scores")
    final_score: float = Field(..., ge=0.0, le=1.0, description="Weighted final score")
    decision: AnalysisDecision = Field(..., description="Final decision")
    processing_time: float = Field(..., description="Total processing time")
    created_at: str = Field(..., description="Analysis timestamp")
    
    class Config:
        frozen = True


class ProcessingStatus(BaseModel):
    """Processing pipeline status"""
    video_id: str = Field(..., description="YouTube video ID")
    current_stage: ProcessingStage = Field(..., description="Current stage")
    progress_percent: int = Field(..., ge=0, le=100, description="Progress percentage")
    status_message: str = Field(..., description="Current status message")
    started_at: str = Field(..., description="Processing start time")
    errors: List[str] = Field(default_factory=list, description="Accumulated errors")
    
    class Config:
        frozen = True


# =============================================================================
# SERVICE INTERFACES (PROTOCOLS)
# =============================================================================

class DownloadService(Protocol):
    """Interface for download services"""
    
    def get_video_info(self, url: str) -> Result[VideoMetadata, DownloadError]:
        """Get video metadata without downloading"""
        ...
    
    def download_audio(self, url: str) -> Result[AudioMetadata, DownloadError]:
        """Download audio to temporary file"""
        ...
    
    def download_video(self, url: str) -> Result[Path, DownloadError]:
        """Download video to temporary file"""
        ...
    
    def cleanup(self) -> None:
        """Clean up temporary files"""
        ...


class TranscriptionService(Protocol):
    """Interface for transcription services"""
    
    def transcribe_audio(self, audio_path: Path) -> Result[TranscriptionResult, TranscriptionError]:
        """Transcribe audio file to text"""
        ...
    
    def is_ready(self) -> bool:
        """Check if service is ready for transcription"""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        ...


class AnalysisService(Protocol):
    """Interface for analysis services"""
    
    def analyze_transcript(self, transcript: str) -> Result[AnalysisResult, AnalysisError]:
        """Analyze transcript using all rules"""
        ...
    
    def analyze_with_rule(self, transcript: str, rule_name: str) -> Result[AnalysisScore, AnalysisError]:
        """Analyze transcript with specific rule"""
        ...
    
    def is_ready(self) -> bool:
        """Check if service is ready for analysis"""
        ...


class StorageService(Protocol):
    """Interface for storage services"""
    
    def store_video(self, video_path: Path, metadata: VideoMetadata) -> Result[str, StorageError]:
        """Store video file and return storage ID"""
        ...
    
    def store_analysis(self, analysis: AnalysisResult) -> Result[str, StorageError]:
        """Store analysis results and return storage ID"""
        ...
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage service information"""
        ...


class ConfigurationService(Protocol):
    """Interface for configuration services"""
    
    def get_analysis_rules(self) -> Dict[str, Any]:
        """Get analysis rules configuration"""
        ...
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration"""
        ...
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get specific service configuration"""
        ...


# =============================================================================
# CALLBACK TYPES
# =============================================================================

ProgressCallback: TypeAlias = Callable[[int, str], None]
ErrorCallback: TypeAlias = Callable[[str, Dict[str, Any]], None]
StatusCallback: TypeAlias = Callable[[ProcessingStatus], None]
CompletionCallback: TypeAlias = Callable[[AnalysisResult], None]


# =============================================================================
# UTILITY TYPES
# =============================================================================

class GPUInfo(BaseModel):
    """GPU information"""
    device: str = Field(..., description="Device name (cpu/cuda/rocm)")
    name: str = Field(..., description="GPU name")
    memory_total: int = Field(default=0, description="Total memory in MB")
    memory_free: int = Field(default=0, description="Free memory in MB")
    available: bool = Field(..., description="GPU available")
    
    class Config:
        frozen = True


class ServiceStatus(BaseModel):
    """Service status information"""
    service_name: str = Field(..., description="Service name")
    status: Literal["ready", "loading", "error", "unavailable"] = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    
    class Config:
        frozen = True


class LogEntry(BaseModel):
    """Structured log entry"""
    timestamp: str = Field(..., description="Log timestamp")
    level: LogLevel = Field(..., description="Log level")
    component: str = Field(..., description="Component name")
    message: str = Field(..., description="Log message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    class Config:
        frozen = True


# =============================================================================
# NUMPY TYPES (for audio processing)
# =============================================================================

AudioArray: TypeAlias = np.ndarray[Any, np.dtype[np.float32]]
AudioBuffer: TypeAlias = bytes


# =============================================================================
# TYPE GUARDS
# =============================================================================

def is_ok(result: Result[T, E]) -> bool:
    """Type guard for Ok result"""
    return isinstance(result, Ok)


def is_err(result: Result[T, E]) -> bool:
    """Type guard for Err result"""
    return isinstance(result, Err)


def unwrap_ok(result: Result[T, E]) -> T:
    """Unwrap Ok result, raise if Err"""
    if isinstance(result, Ok):
        return result.value
    raise ValueError(f"Expected Ok, got Err: {result.error}")


def unwrap_err(result: Result[T, E]) -> E:
    """Unwrap Err result, raise if Ok"""
    if isinstance(result, Err):
        return result.error
    raise ValueError(f"Expected Err, got Ok: {result.value}")


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_youtube_url(url: str) -> Result[str, ValidationError]:
    """Validate YouTube URL format"""
    if not url or not isinstance(url, str):
        return Err(ValidationError("URL must be a non-empty string"))
    
    valid_patterns = [
        "youtube.com/watch",
        "youtu.be/",
        "youtube.com/shorts/",
        "youtube.com/embed/",
    ]
    
    if not any(pattern in url for pattern in valid_patterns):
        return Err(ValidationError(f"Invalid YouTube URL format: {url}"))
    
    return Ok(url)


def validate_score(score: float) -> Result[float, ValidationError]:
    """Validate score is between 0.0 and 1.0"""
    if not isinstance(score, (int, float)):
        return Err(ValidationError("Score must be a number"))
    
    if not 0.0 <= score <= 1.0:
        return Err(ValidationError(f"Score must be between 0.0 and 1.0, got {score}"))
    
    return Ok(float(score))


def validate_path(path: Path) -> Result[Path, ValidationError]:
    """Validate path exists and is accessible"""
    if not isinstance(path, Path):
        return Err(ValidationError("Path must be a pathlib.Path object"))
    
    if not path.exists():
        return Err(ValidationError(f"Path does not exist: {path}"))
    
    if not path.is_file():
        return Err(ValidationError(f"Path is not a file: {path}"))
    
    return Ok(path)
