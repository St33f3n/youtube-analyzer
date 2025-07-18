# 📚 Grundlagen-Bibliothek - Vollständige Verwendungsanleitung

## 🎯 Übersicht

Die Grundlagen-Bibliothek besteht aus zwei Modulen:
- **`core_types.py`** - Funktionale Fehlerbehandlung mit Result-Types
- **`logging_plus.py`** - Multi-Level-Logging mit Result-Integration

## 🚀 Setup und Import

### Grundsetup
```python
# Basis-Imports
from core_types import Ok, Err, Result, is_ok, is_err, unwrap_ok, unwrap_err
from logging_plus import setup_logging, get_logger, log_feature, log_function

# Setup Logging am Anfang deiner Anwendung
setup_logging(
    name="my_app",
    level="INFO",           # oder "DEBUG" für Entwicklung
    log_file="app.log",     # Optional: File-Logging
    json_file="app.json"    # Optional: Structured Logging
)
```

### Dependencies
```bash
pip install loguru psutil
```

---

## 📦 core_types.py - Result-Types System

### Basic Usage

#### ✅ Erfolgreiche Results
```python
from core_types import Ok, Err, validate_score

# Erstelle Success-Result
success = Ok("Daten erfolgreich verarbeitet")
score_result = Ok(0.85)

# Checke ob Success
if is_ok(success):
    value = unwrap_ok(success)
    print(f"Erfolg: {value}")
```

#### ❌ Error-Results
```python
from core_types import ValidationError, ErrorContext

# Erstelle Error mit Context
context = ErrorContext.create(
    "user_validation",
    input_data={"email": "invalid-email"},
    suggestions=["Use valid email format", "Check @ symbol"]
)
error_result = Err(ValidationError("Invalid email format", context))

# Checke Errors
if is_err(error_result):
    error = unwrap_err(error_result)
    print(f"Fehler: {error.message}")
    print(f"Suggestions: {error.context.suggestions}")
```

### Validierungsfunktionen

#### Eingebaute Validators
```python
from core_types import validate_score, validate_path, validate_range

# Score-Validierung (0.0 - 1.0)
score_result = validate_score(0.75)
if is_ok(score_result):
    score = unwrap_ok(score_result)

# Pfad-Validierung
from pathlib import Path
path_result = validate_path(Path("data.csv"))

# Range-Validierung
range_result = validate_range(85.0, 0.0, 100.0)
```

#### Custom Validators schreiben
```python
from core_types import Result, Ok, Err, ValidationError, ErrorContext

def validate_email(email: str) -> Result[str, ValidationError]:
    """Custom Email-Validator"""
    context = ErrorContext.create(
        "email_validation",
        input_data={"email": email}
    )
    
    if not isinstance(email, str):
        context.suggestions = ["Provide string value"]
        return Err(ValidationError("Email must be string", context))
    
    if "@" not in email:
        context.suggestions = ["Add @ symbol", "Use format: user@domain.com"]
        return Err(ValidationError("Email must contain @", context))
    
    return Ok(email.lower())

# Verwendung
email_result = validate_email("user@example.com")
```

### Funktionale Operationen

#### Map - Transformiere Success-Values
```python
# Transformiere Wert bei Erfolg
result = Ok(0.5).map(lambda x: x * 100)  # Ok(50.0)

# Bei Error: Error bleibt erhalten
error_result = Err("Fehler").map(lambda x: x * 100)  # Err("Fehler")

# Praktisches Beispiel
score_percent = validate_score(0.85).map(lambda x: f"{x * 100}%")
if is_ok(score_percent):
    print(f"Score: {unwrap_ok(score_percent)}")  # "Score: 85.0%"
```

#### And_then - Kette Result-produzierende Funktionen
```python
def convert_to_grade(score: float) -> Result[str, ValidationError]:
    if score >= 0.9:
        return Ok("A")
    elif score >= 0.8:
        return Ok("B") 
    elif score >= 0.7:
        return Ok("C")
    else:
        return Ok("F")

# Kette Validierung + Konvertierung
final_result = validate_score(0.85).and_then(convert_to_grade)
# Ok("B")

# Bei Fehler in der Kette
invalid_result = validate_score(1.5).and_then(convert_to_grade)  
# Err(ValidationError("Score must be between 0.0 and 1.0, got 1.5"))
```

#### Unwrap_or - Sichere Default-Werte
```python
# Bei Success: Gibt Wert zurück
success_value = Ok("success").unwrap_or("default")  # "success"

# Bei Error: Gibt Default zurück  
error_value = Err("error").unwrap_or("default")  # "default"

# Praktisch für Konfiguration
config_timeout = validate_range(settings.timeout, 1, 300).unwrap_or(30)
```

### Result-Kombinatoren

#### Combine_results - Alle müssen erfolgreich sein
```python
from core_types import combine_results

# Alle Results erfolgreich
results = combine_results(
    validate_score(0.8),
    validate_range(25, 18, 65),
    validate_email("user@test.com")
)
if is_ok(results):
    score, age, email = unwrap_ok(results)
    print(f"Alle Validierungen erfolgreich: {score}, {age}, {email}")

# Ein Fehler = Gesamtergebnis Fehler
results_with_error = combine_results(
    validate_score(0.8),
    validate_score(1.5),  # Fehler!
    validate_score(0.9)
)
# Err(ValidationError(...)) - Erster Fehler wird zurückgegeben
```

#### Collect_results - Sammle Erfolge und Fehler getrennt
```python
from core_types import collect_results

scores = [0.8, 1.5, 0.3, 2.0, 0.9]  # Mischung valide/invalide
validation_results = [validate_score(s) for s in scores]

collected = collect_results(validation_results)
if is_ok(collected):
    valid_scores = unwrap_ok(collected)
    print(f"Alle Scores valide: {valid_scores}")
else:
    errors = unwrap_err(collected)
    print(f"Fehler gefunden: {len(errors)} Validierungsfehler")
```

---

## 📊 logging_plus.py - Multi-Level-Logging

### Setup und Konfiguration

#### Basic Setup
```python
from logging_plus import setup_logging, get_logger

# Entwicklung: Console + Debug
setup_logging(
    name="my_app",
    level="DEBUG",
    enable_console=True,
    log_file="debug.log"
)

# Produktion: File + JSON
setup_logging(
    name="my_app", 
    level="INFO",
    enable_console=False,
    log_file="app.log",
    json_file="structured.json"
)
```

#### Logger für Module
```python
# In jedem Modul
logger = get_logger(__name__)

# Oder spezifisch
data_logger = get_logger("data_processing")
api_logger = get_logger("api_client")
```

### Function-Level-Logging (DEBUG)

#### Decorator für automatisches Function-Logging
```python
from logging_plus import log_function

@log_function(log_args=True, log_performance=True)
def process_user_data(user_id: int, options: dict = None) -> str:
    """Verarbeitet User-Daten mit automatischem Logging"""
    time.sleep(0.1)  # Simulate work
    return f"processed_user_{user_id}"

# Automatisches Logging:
# DEBUG: → process_user_data() called (args_count=1, kwargs_keys=['options'])
# DEBUG: ✓ process_user_data() completed (duration_ms=101.2, performance_category='normal')

result = process_user_data(123, options={"format": "json"})
```

#### Manuelle Function-Level-Logs
```python
logger = get_logger("my_module")

def complex_calculation(data):
    logger.debug("Starting complex calculation", extra={
        'function': 'complex_calculation',
        'data_size': len(data),
        'stage': 'entry'
    })
    
    # ... processing ...
    
    logger.debug("Intermediate result calculated", extra={
        'stage': 'intermediate',
        'intermediate_value': intermediate
    })
    
    # ... more processing ...
    
    return result
```

### Feature-Level-Logging (INFO)

#### Context-Manager für Features
```python
from logging_plus import log_feature

# Basic Feature-Logging
with log_feature("user_registration") as feature:
    user_data = validate_user_input(input_data)
    user = create_user(user_data)
    send_welcome_email(user)
    feature.add_metric("user_id", user.id)

# INFO: ▶ Feature 'user_registration' started
# INFO: ✓ Feature 'user_registration' completed (duration_ms=234.5, metrics={'user_id': 123})
```

#### Erweiterte Feature-Logging
```python
with log_feature("data_processing", expected_duration=2.0) as feature:
    feature.checkpoint("validation_start")
    
    # Validierung
    validation_results = validate_batch(data)
    feature.add_metric("validation_errors", len(validation_results.errors))
    
    if validation_results.errors:
        feature.add_warning("Some validation errors found")
    
    feature.checkpoint("processing_start")
    
    # Verarbeitung
    processed_data = process_valid_data(validation_results.valid_data)
    feature.add_metric("items_processed", len(processed_data))
    
    feature.checkpoint("saving_start")
    
    # Speichern
    save_results(processed_data)
    feature.set_progress(len(processed_data), len(data))
    
    feature.checkpoint("completion")

# Automatische Performance-Bewertung basierend auf expected_duration
```

### Result-Integration

#### Automatisches Result-Logging
```python
from logging_plus import log_result, log_and_unwrap

# Log Result automatisch
user_result = validate_user_data(input_data)
logged_result = log_result(user_result, "user_validation")

# Bei Ok: INFO-Log mit Success
# Bei Err: ERROR-Log mit Error-Context

# Entpacke sicher mit Logging + Default
user = log_and_unwrap(user_result, "user_validation", default=None)
if user is None:
    print("User validation failed - check logs")
```

#### Result-Function-Decorator
```python
from logging_plus import log_result_function

@log_result_function("email_validation")
def validate_user_email(email: str) -> Result[str, ValidationError]:
    return validate_email(email)

# Automatisches Logging für jede Funktion die Results zurückgibt
email_result = validate_user_email("test@example.com")
# INFO: ✓ email_validation succeeded (bei Ok)
# ERROR: ✗ email_validation failed (bei Err mit Error-Context)
```

### Error-Level-Logging (ERROR)

#### Critical Error-Logging
```python
from logging_plus import log_critical_error

try:
    risky_operation()
except Exception as e:
    log_critical_error(
        e,
        "database_connection", 
        context={"host": "db.example.com", "timeout": 30},
        recovery_suggestions=[
            "Check database connectivity",
            "Verify credentials", 
            "Try backup database"
        ]
    )
    # Umfangreiches ERROR-Log mit System-State, Stack-Trace, Recovery-Hints
```

#### Automatischer Error-Context
```python
# Bei CoreError: Automatische Context-Extraktion
context = ErrorContext.create(
    "file_processing",
    input_data={"file": "data.csv", "size": "10MB"},
    suggestions=["Check file permissions", "Verify file format"]
)
error = ValidationError("File format invalid", context)

# log_critical_error extrahiert automatisch alle Context-Informationen
log_critical_error(error, "file_import")
```

---

## 🔄 Praktische Workflow-Patterns

### Pattern 1: Data Processing Pipeline
```python
from core_types import *
from logging_plus import *

@log_result_function("data_pipeline")
def process_data_pipeline(input_file: str) -> Result[dict, CoreError]:
    with log_feature("data_processing_pipeline") as feature:
        feature.checkpoint("file_validation")
        
        # 1. Validiere Input
        path_result = validate_path(Path(input_file))
        if is_err(path_result):
            return path_result
        
        feature.checkpoint("data_loading")
        
        # 2. Lade Daten (mit automatischer Result-Logging)
        data_result = load_csv_data(unwrap_ok(path_result))
        if is_err(data_result):
            return data_result
        
        feature.checkpoint("data_transformation")
        
        # 3. Transformiere mit funktionalen Operationen
        processed_result = data_result.and_then(validate_data_format) \
                                    .and_then(clean_data) \
                                    .and_then(enrich_data)
        
        if is_ok(processed_result):
            feature.add_metric("rows_processed", len(unwrap_ok(processed_result)))
            feature.checkpoint("completion")
        
        return processed_result

# Usage
result = process_data_pipeline("data/input.csv")
final_data = log_and_unwrap(result, "pipeline_execution", default={})
```

### Pattern 2: API Request mit Retry-Logic
```python
@log_function(log_performance=True)
def fetch_user_data(user_id: int) -> Result[dict, CoreError]:
    with log_feature(f"fetch_user_{user_id}") as feature:
        max_retries = 3
        
        for attempt in range(max_retries):
            feature.checkpoint(f"attempt_{attempt + 1}")
            
            try:
                response = api_client.get(f"/users/{user_id}")
                if response.status_code == 200:
                    feature.add_metric("attempts_needed", attempt + 1)
                    return Ok(response.json())
                else:
                    feature.add_warning(f"HTTP {response.status_code} on attempt {attempt + 1}")
                    
            except Exception as e:
                if attempt == max_retries - 1:  # Letzter Versuch
                    context = ErrorContext.create(
                        "api_request",
                        input_data={"user_id": user_id, "attempts": max_retries},
                        suggestions=["Check API connectivity", "Verify user_id exists"]
                    )
                    return Err(CoreError(f"API request failed after {max_retries} attempts: {e}", context))
                
                feature.add_warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return Err(CoreError("Unexpected: Should not reach here"))

# Usage mit automatischem Logging
user_result = fetch_user_data(123)
user = log_and_unwrap(user_result, "user_fetch", default=None)
```

### Pattern 3: Batch-Verarbeitung mit Fehlersammlung
```python
def process_user_batch(user_ids: list[int]) -> Result[dict, list[CoreError]]:
    with log_feature("batch_user_processing", expected_duration=len(user_ids) * 0.1) as feature:
        results = []
        errors = []
        
        for i, user_id in enumerate(user_ids):
            feature.checkpoint(f"user_{user_id}")
            
            user_result = fetch_user_data(user_id)
            if is_ok(user_result):
                results.append(unwrap_ok(user_result))
            else:
                errors.append(unwrap_err(user_result))
            
            feature.set_progress(i + 1, len(user_ids))
        
        feature.add_metric("successful_users", len(results))
        feature.add_metric("failed_users", len(errors))
        feature.add_metric("success_rate", len(results) / len(user_ids) * 100)
        
        if errors:
            return Err(errors)
        
        return Ok({
            "users": results,
            "total_processed": len(results)
        })

# Usage
batch_result = process_user_batch([1, 2, 3, 4, 5])
if is_ok(batch_result):
    data = unwrap_ok(batch_result)
    print(f"Processed {data['total_processed']} users successfully")
else:
    errors = unwrap_err(batch_result)
    print(f"Batch failed with {len(errors)} errors - check logs for details")
```

---

## 🎯 Best Practices

### Do's ✅

1. **Immer Type-Hints verwenden**
   ```python
   def process_data(input: str) -> Result[ProcessedData, ValidationError]:
   ```

2. **Result-Chains für komplexe Validierung**
   ```python
   result = validate_input(data).and_then(process).and_then(save)
   ```

3. **Feature-Logging für Business-Operationen**
   ```python
   with log_feature("user_registration"):
       # Business logic here
   ```

4. **Strukturierte Error-Contexts**
   ```python
   context = ErrorContext.create("operation", input_data={}, suggestions=[])
   ```

5. **unwrap_or für graceful degradation**
   ```python
   timeout = validate_timeout(config.timeout).unwrap_or(30)
   ```

### Don'ts ❌

1. **Niemals unwrap_ok ohne Check**
   ```python
   # ❌ FALSCH
   value = unwrap_ok(result)  # Kann crashen!
   
   # ✅ RICHTIG
   if is_ok(result):
       value = unwrap_ok(result)
   # oder
   value = result.unwrap_or(default_value)
   ```

2. **Exceptions in Result-Funktionen nicht fangen**
   ```python
   # ❌ FALSCH
   def validate_data(data):
       validate_format(data)  # Kann Exception werfen
       return Ok(data)
   
   # ✅ RICHTIG  
   def validate_data(data):
       try:
           validate_format(data)
           return Ok(data)
       except ValidationException as e:
           return Err(ValidationError(str(e)))
   ```

3. **Logging-Setup vergessen**
   ```python
   # ❌ FALSCH: Logging vor setup_logging verwenden
   
   # ✅ RICHTIG: Erst setup, dann logging
   setup_logging("app", "INFO")
   logger = get_logger("module")
   ```

---

## 🔧 Troubleshooting

### Häufige Probleme

**Problem: "Expected Ok, got Err"**
```python
# Ursache: unwrap_ok auf Err-Result
error_result = validate_score(1.5)  # Err
value = unwrap_ok(error_result)  # ❌ Crash!

# Lösung: Immer checken oder unwrap_or verwenden
if is_ok(error_result):
    value = unwrap_ok(error_result)
else:
    print("Validation failed")

# Oder:
value = error_result.unwrap_or(default_value)
```

**Problem: Import-Fehler**
```python
# ModuleNotFoundError: No module named 'loguru'
pip install loguru psutil

# ImportError: cannot import name 'setup_logging'
# → Prüfe ob logging_plus.py im Python-Path ist
```

**Problem: Logging funktioniert nicht**
```python
# Keine Logs sichtbar
setup_logging("app", "INFO")  # Erst setup...
logger = get_logger("module")  # ...dann logger

# Log-Level zu hoch
setup_logging("app", "DEBUG")  # Für alle Logs
```

**Problem: Type-Checker-Fehler**
```python
# mypy: Incompatible return value type
def my_function() -> Result[str, ValidationError]:
    return "string"  # ❌ Falsch
    
def my_function() -> Result[str, ValidationError]:
    return Ok("string")  # ✅ Richtig
```

---

## 📖 Weiterführende Beispiele

Schaue in `test_demo.py` für umfassende Beispiele aller Features oder erstelle eigene Test-Szenarien basierend auf deinen spezifischen Anwendungsfällen.

Die Bibliothek ist darauf ausgelegt, schrittweise eingeführt zu werden - beginne mit einfachen Result-Types und erweitere dann um Logging-Features nach Bedarf.