#!/usr/bin/env python3
"""
test_demo.py - Comprehensive Test für core_types + logging_plus

Testet alle Features der Grundlagen-Bibliothek:
- Result-Types und funktionale Operationen
- Multi-Level-Logging 
- Error-Handling mit Context
- Performance-Tracking
- System-Integration
"""

import time
from pathlib import Path

# Unsere Bibliotheken
from core_types import (
    Ok, Err, Result, is_ok, is_err, unwrap_ok,
    validate_score, validate_path, validate_range,
    unwrap_err, combine_results, collect_results,
    CoreError, ValidationError, ErrorContext
)

from logging_plus import (
    setup_logging, get_logger, log_feature, log_function,
    log_result, log_and_unwrap, log_critical_error,
    log_result_function
)

# =============================================================================
# 1. BASIC RESULT-TYPES TESTS
# =============================================================================

def test_basic_result_types():
    """Test: Grundlegende Result-Types Funktionalität"""
    print("\n🔍 === BASIC RESULT-TYPES TESTS ===")
    
    # Success Case
    score_result = validate_score(0.85)
    print(f"validate_score(0.85): {score_result}")
    
    if is_ok(score_result):
        print(f"✓ Score value: {unwrap_ok(score_result)}")
    
    # Error Case
    invalid_score = validate_score(1.5)
    print(f"validate_score(1.5): {invalid_score}")
    
    if is_err(invalid_score):
        error = unwrap_err(invalid_score)  # Korrekt: unwrap_err für Errors
        print(f"✗ Error: {error}")
        print(f"  Context: {error.to_dict()}")

def test_functional_operations():
    """Test: Funktionale Result-Operationen"""
    print("\n⚙️ === FUNCTIONAL OPERATIONS TESTS ===")
    
    # Map Operation
    result = Ok(0.5).map(lambda x: x * 100)
    print(f"Ok(0.5).map(x * 100): {result}")
    
    # Chaining mit and_then
    chained = validate_score(0.8).and_then(
        lambda x: validate_range(x * 100, 0, 100)
    )
    print(f"Chained validation: {chained}")
    
    # unwrap_or mit Default
    error_result = Err(ValidationError("Test error"))
    safe_value = error_result.unwrap_or("default_value")
    print(f"Error with default: {safe_value}")
    
    # Kombiniere mehrere Results
    results = [validate_score(0.1), validate_score(0.5), validate_score(0.9)]
    combined = collect_results(results)
    print(f"Collected results: {combined}")

# =============================================================================
# 2. LOGGING INTEGRATION TESTS
# =============================================================================

@log_function(log_args=True, log_performance=True)
def example_processing_function(data: str, multiplier: int = 2) -> str:
    """Beispiel-Funktion für Function-Level-Logging"""
    time.sleep(0.05)  # Simulate work
    return f"processed_{data}_x{multiplier}"

@log_result_function("score_validation")
def validate_user_score(score: float) -> Result[float, ValidationError]:
    """Beispiel für Result-Function-Decorator"""
    return validate_score(score)

def test_multi_level_logging():
    """Test: Multi-Level-Logging System"""
    print("\n📊 === MULTI-LEVEL LOGGING TESTS ===")
    
    # Feature-Level mit verschiedenen Scenarios
    with log_feature("user_data_processing", expected_duration=0.2) as feature:
        feature.checkpoint("validation_start")
        
        # Function-Level (DEBUG)
        processed = example_processing_function("user_data", multiplier=3)
        feature.add_metric("items_processed", 1)
        
        feature.checkpoint("score_validation")
        
        # Result-Function (automatisches Result-Logging)
        score_result = validate_user_score(0.75)
        feature.add_metric("validation_success", is_ok(score_result))
        
        if is_err(score_result):
            feature.add_warning("Score validation failed")
        
        feature.checkpoint("completion")
        feature.set_progress(1, 1)

def test_error_scenarios():
    """Test: Error-Handling und Critical-Error-Logging"""
    print("\n🚨 === ERROR SCENARIOS TESTS ===")
    
    # 1. Validation Error mit Context
    invalid_result = validate_score(-0.5)
    logged_result = log_result(invalid_result, "negative_score_test")
    
    # 2. File System Error
    fake_path = Path("/non/existent/file.txt")
    path_result = validate_path(fake_path)
    log_result(path_result, "file_access_test")
    
    # 3. Critical Error mit Custom Context
    try:
        raise ValueError("Simulated critical error")
    except Exception as e:
        log_critical_error(
            e, 
            "simulation_test",
            context={"test_phase": "error_simulation", "expected": True},
            recovery_suggestions=["This is a test - no action needed"]
        )

def test_performance_scenarios():
    """Test: Performance-Tracking"""
    print("\n⚡ === PERFORMANCE SCENARIOS TESTS ===")
    
    @log_function(log_performance=True)
    def fast_operation():
        time.sleep(0.001)  # Very fast
        return "fast_result"
    
    @log_function(log_performance=True)
    def slow_operation():
        time.sleep(0.5)  # Slow operation
        return "slow_result"
    
    # Feature mit Performance-Erwartung
    with log_feature("performance_test", expected_duration=0.1) as feature:
        feature.checkpoint("fast_op_start")
        fast_result = fast_operation()
        feature.add_metric("fast_op_result", fast_result)
        
        feature.checkpoint("slow_op_start")
        slow_result = slow_operation()
        feature.add_metric("slow_op_result", slow_result)
        
        feature.set_progress(2, 2)

# =============================================================================
# 3. INTEGRATION SCENARIOS
# =============================================================================

def test_real_world_scenario():
    """Test: Realistische Integration aller Features"""
    print("\n🌍 === REAL-WORLD SCENARIO TEST ===")
    
    @log_result_function("file_processing")
    def process_config_file(file_path: str) -> Result[dict, CoreError]:
        """Simuliert echte File-Processing mit Result-Types"""
        path = Path(file_path)
        
        # Validation Chain
        path_result = validate_path(path)
        if is_err(path_result):
            return path_result
        
        # Simulate file processing
        try:
            # In echt: JSON/YAML parsing etc.
            config = {
                "app_name": "test_app",
                "version": "1.0.0",
                "score_threshold": 0.8
            }
            return Ok(config)
        except Exception as e:
            context = ErrorContext.create(
                "config_parsing",
                input_data={"file_path": file_path},
                suggestions=["Check file format", "Validate JSON/YAML syntax"]
            )
            return Err(CoreError(f"Config parsing failed: {e}", context))
    
    with log_feature("application_startup", expected_duration=1.0) as feature:
        feature.checkpoint("config_loading")
        
        # Test mit existierender Datei (diese Datei selbst)
        config_result = process_config_file(__file__)
        
        if is_ok(config_result):
            config = unwrap_ok(config_result)
            feature.add_metric("config_loaded", True)
            feature.add_metric("config_keys", len(config))
            
            feature.checkpoint("score_validation")
            
            # Validate score from config
            score_result = validate_user_score(0.85)  # Simuliert Wert aus Config
            
            if is_ok(score_result):
                feature.add_metric("score_valid", True)
                feature.checkpoint("startup_complete")
            else:
                feature.add_warning("Invalid score in configuration")
                feature.add_metric("score_valid", False)
        else:
            feature.add_warning("Configuration loading failed")
            feature.add_metric("config_loaded", False)
        
        feature.set_progress(1, 1)

# =============================================================================
# 4. MAIN TEST RUNNER
# =============================================================================

def main():
    """Führt alle Tests aus"""
    print("🚀 STARTING COMPREHENSIVE LIBRARY TEST")
    print("=" * 50)
    
    # Setup Logging für Tests
    setup_logging(
        name="test_demo",
        level="DEBUG",
        log_file="test_demo.log",
        json_file="test_demo.json",
        enable_console=True,
        enable_structured=True
    )
    
    logger = get_logger("test_runner")
    logger.info("Test suite started", extra={"test_suite": "comprehensive_demo"})
    
    try:
        # Alle Test-Funktionen ausführen
        test_basic_result_types()
        test_functional_operations() 
        test_multi_level_logging()
        test_error_scenarios()
        test_performance_scenarios()
        test_real_world_scenario()
        
        logger.info("✅ All tests completed successfully")
        print("\n✅ ALL TESTS COMPLETED!")
        print("\n📁 Check log files:")
        print("  - test_demo.log (human-readable)")
        print("  - test_demo.json (structured)")
        
    except Exception as e:
        log_critical_error(e, "test_execution", log=logger)
        print(f"\n❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    main()
