"""
core_types.py – Universelles Kern-Typensystem für Python-Projekte

Enthält:
  - Funktionale Fehlerbehandlung via `Result`, `Ok` und `Err`
  - Erweiterte funktionale Operationen (map, and_then, unwrap_or)
  - Typ-Guards für sicheren Umgang mit `Result`
  - Strukturierte Fehlertypen mit Context
  - Allgemeine Validierungsfunktionen
  - Logging-Integration vorbereitet

Beispiel:

    from core_types import Ok, Err, is_ok, unwrap_ok, validate_score

    result = validate_score(0.85)
    if is_ok(result):
        score = unwrap_ok(result)
    else:
        error = result.error

    # Funktionale Operationen
    result.map(lambda x: x * 100).and_then(validate_percentage)
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeAlias, TypeVar, Union

# =============================================================================
# GENERIC RESULT TYPES
# =============================================================================

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")

@dataclass(frozen=True)
class Ok(Generic[T]):
    """Erfolgsergebnis mit Wert"""
    value: T
    
    def map(self, func: Callable[[T], U]) -> Result[U, E]:
        """Transformiert den Wert bei Erfolg"""
        try:
            return Ok(func(self.value))
        except Exception as e:
            # Bei map-Fehler: Exception in CoreError umwandeln
            return Err(CoreError(f"Map operation failed: {e}"))
    
    def and_then(self, func: Callable[[T], Result[U, F]]) -> Result[U, Union[E, F]]:
        """Kettet Result-produzierende Operationen"""
        return func(self.value)
    
    def unwrap_or(self, default: U) -> T:
        """Gibt Wert zurück oder Default bei Fehler"""
        return self.value

@dataclass(frozen=True)
class Err(Generic[E]):
    """Fehlerergebnis mit Fehlerobjekt"""
    error: E
    
    def map(self, func: Callable[[Any], U]) -> Result[U, E]:
        """Map bei Fehler: Fehler durchreichen"""
        return self
    
    def and_then(self, func: Callable[[Any], Result[U, F]]) -> Result[U, Union[E, F]]:
        """And_then bei Fehler: Fehler durchreichen"""
        return self
    
    def unwrap_or(self, default: U) -> U:
        """Gibt Default zurück bei Fehler"""
        return default

Result: TypeAlias = Union[Ok[T], Err[E]]

# =============================================================================
# ENHANCED ERROR TYPES
# =============================================================================

@dataclass
class ErrorContext:
    """Strukturierter Fehlerkontext für bessere Debugging-Info"""
    operation: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    suggestions: list[str] = field(default_factory=list)
    
    @classmethod
    def create(cls, operation: str, **kwargs) -> ErrorContext:
        """Factory für ErrorContext mit automatischem Stacktrace"""
        return cls(
            operation=operation,
            input_data=kwargs.get('input_data', {}),
            system_state=kwargs.get('system_state', {}),
            stack_trace=traceback.format_exc() if kwargs.get('include_trace', False) else None,
            suggestions=kwargs.get('suggestions', [])
        )

class CoreError(Exception):
    """Basisfehler für Kernbibliothek mit strukturiertem Context"""
    
    def __init__(
        self, 
        message: str, 
        context: Optional[ErrorContext] = None,
        **legacy_context: Any
    ) -> None:
        super().__init__(message)
        self.message = message
        # Backward compatibility: legacy dict context
        if legacy_context and context is None:
            self.context = ErrorContext.create("legacy", input_data=legacy_context)
        else:
            self.context = context or ErrorContext.create("unknown")
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert Fehler zu Dictionary für Logging"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'operation': self.context.operation,
            'input_data': self.context.input_data,
            'system_state': self.context.system_state,
            'suggestions': self.context.suggestions
        }

class ValidationError(CoreError):
    """Fehler bei Validierung von Eingabedaten"""
    pass

class ProcessingError(CoreError):
    """Fehler bei Datenverarbeitung"""
    pass

class FileSystemError(CoreError):
    """Fehler bei Dateisystem-Operationen"""
    pass

# =============================================================================
# TYPE GUARDS UND UNWRAP HELPERS
# =============================================================================

def is_ok(result: Result[T, E]) -> bool:
    """Gibt True zurück, wenn `result` ein Ok ist"""
    return isinstance(result, Ok)

def is_err(result: Result[T, E]) -> bool:
    """Gibt True zurück, wenn `result` ein Err ist"""
    return isinstance(result, Err)

def unwrap_ok(result: Result[T, E]) -> T:
    """Entpackt den Wert aus Ok oder wirft ValueError, wenn Err"""
    if isinstance(result, Ok):
        return result.value
    raise ValueError(f"Expected Ok, got Err: {result.error}")

def unwrap_err(result: Result[T, E]) -> E:
    """Entpackt den Fehler aus Err oder wirft ValueError, wenn Ok"""
    if isinstance(result, Err):
        return result.error
    raise ValueError(f"Expected Err, got Ok: {result.value}")

def unwrap_or(result: Result[T, E], default: U) -> Union[T, U]:
    """Entpackt Wert oder gibt Default zurück"""
    return result.unwrap_or(default)

# =============================================================================
# RESULT HELPER FUNCTIONS
# =============================================================================

def combine_results(*results: Result[Any, E]) -> Result[tuple[Any, ...], E]:
    """Kombiniert mehrere Results - Fehler bei erstem Err"""
    values = []
    for result in results:
        if is_err(result):
            return result
        values.append(unwrap_ok(result))
    return Ok(tuple(values))

def collect_results(results: list[Result[T, E]]) -> Result[list[T], list[E]]:
    """Sammelt alle Erfolge und Fehler getrennt"""
    successes = []
    errors = []
    
    for result in results:
        if is_ok(result):
            successes.append(unwrap_ok(result))
        else:
            errors.append(unwrap_err(result))
    
    if errors:
        return Err(errors)
    return Ok(successes)

# =============================================================================
# ENHANCED VALIDATORS
# =============================================================================

def validate_score(score: float) -> Result[float, ValidationError]:
    """
    Validiert, dass `score` eine Zahl zwischen 0.0 und 1.0 ist.

    Returns:
        Ok(float): Validierte Bewertung
        Err(ValidationError): Bei ungültigem Wert
    """
    context = ErrorContext.create(
        "validate_score",
        input_data={'score': score, 'type': type(score).__name__}
    )
    
    if not isinstance(score, (int, float)):
        context.suggestions = ["Provide numeric value (int or float)"]
        return Err(ValidationError("Score must be a number", context))
    
    if not 0.0 <= score <= 1.0:
        context.suggestions = ["Provide value between 0.0 and 1.0", "Check if percentage needs conversion"]
        return Err(ValidationError(f"Score must be between 0.0 and 1.0, got {score}", context))
    
    return Ok(float(score))

def validate_path(path: Path) -> Result[Path, FileSystemError]:
    """
    Validiert, dass `path` existiert und eine Datei ist.

    Returns:
        Ok(Path): Validierter Pfad
        Err(FileSystemError): Bei ungültigem Pfad
    """
    context = ErrorContext.create(
        "validate_path",
        input_data={'path': str(path), 'type': type(path).__name__}
    )
    
    if not isinstance(path, Path):
        context.suggestions = ["Convert string to Path: Path(your_string)"]
        return Err(FileSystemError("Path must be a pathlib.Path object", context))
    
    if not path.exists():
        context.suggestions = [
            "Check if file path is correct",
            "Verify file permissions",
            "Create file if it should exist"
        ]
        return Err(FileSystemError(f"Path does not exist: {path}", context))
    
    if not path.is_file():
        context.suggestions = [
            "Use path.is_dir() if directory expected",
            "Check if path points to special file type"
        ]
        return Err(FileSystemError(f"Path is not a file: {path}", context))
    
    return Ok(path)

def validate_range(value: float, min_val: float, max_val: float) -> Result[float, ValidationError]:
    """Validiert, dass Wert in Bereich liegt"""
    context = ErrorContext.create(
        "validate_range",
        input_data={'value': value, 'min': min_val, 'max': max_val}
    )
    
    if not isinstance(value, (int, float)):
        context.suggestions = ["Provide numeric value"]
        return Err(ValidationError("Value must be numeric", context))
    
    if not min_val <= value <= max_val:
        context.suggestions = [f"Provide value between {min_val} and {max_val}"]
        return Err(ValidationError(f"Value {value} not in range [{min_val}, {max_val}]", context))
    
    return Ok(float(value))

# =============================================================================
# OPTIONAL: BEISPIEL UND TESTS
# =============================================================================

if __name__ == "__main__":
    # Erweiterte Selbsttests
    print("=== Basic Validation ===")
    r1 = validate_score(0.7)
    print("validate_score(0.7):", r1)
    
    r2 = validate_score(1.2)
    print("validate_score(1.2):", r2)
    if is_err(r2):
        print("Error context:", unwrap_err(r2).to_dict())
    
    print("\n=== Functional Operations ===")
    r3 = Ok(0.5).map(lambda x: x * 100)
    print("Ok(0.5).map(x * 100):", r3)
    
    r4 = validate_score(0.8).and_then(lambda x: validate_range(x * 100, 0, 100))
    print("Chained validation:", r4)
    
    print("\n=== Result Combination ===")
    results = [validate_score(0.1), validate_score(0.5), validate_score(0.9)]
    combined = collect_results(results)
    print("Collect results:", combined)
