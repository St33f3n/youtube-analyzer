"""
Trilium Note Uploader
Upload von LLM-processed Transkripten zu Trilium mit ETAPI
Config-Dict Support mit resolved Secrets (keine Backward Compatibility)
"""

from __future__ import annotations
import markdown
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import threading

# Trilium-py für ETAPI-Client
try:
    from trilium_py.client import ETAPI
    TRILIUM_PY_AVAILABLE = True
except ImportError:
    TRILIUM_PY_AVAILABLE = False
    ETAPI = None

# Import our core libraries
from core_types import Result, Ok, Err, CoreError, ErrorContext, unwrap_err, unwrap_ok
from yt_analyzer_core import TranskriptObject
from logging_plus import get_logger, log_function

# =============================================================================
# TRILIUM UPLOAD ENGINE - Config-Dict Support
# =============================================================================

class TriliumUploader:
    """ETAPI-basierter Trilium-Uploader mit resolved Secrets"""
    
    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self.logger = get_logger("TriliumUploader")
        
        # Validate trilium-py availability
        if not TRILIUM_PY_AVAILABLE:
            self.logger.error("trilium-py not available - install with: pip install trilium-py")
            return
        
        # Upload statistics
        self.upload_stats = {
            'total_uploads': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'total_notes_created': 0,
            'total_attributes_created': 0,
            'total_upload_time_seconds': 0.0,
            'skipped_existing': 0
        }
        
        # Trilium Client (initialized on first use)
        self.trilium_client: Optional[ETAPI] = None
        self.client_lock = threading.Lock()
    
    def _init_trilium_client(self) -> Result[ETAPI, CoreError]:
        """Initialisiert Trilium ETAPI-Client mit resolved Credentials"""
        if not TRILIUM_PY_AVAILABLE:
            context = ErrorContext.create(
                "init_trilium_client",
                suggestions=["Install trilium-py: pip install trilium-py"]
            )
            return Err(CoreError("trilium-py not available", context))
        
        with self.client_lock:
            if self.trilium_client:
                return Ok(self.trilium_client)
            
            try:
                # Get config values
                trilium_config = self.config_dict.get('trilium_upload', {})
                base_url = trilium_config.get('base_url')
                timeout = trilium_config.get('timeout', 60)
                
                # Get resolved token
                resolved_secrets = self.config_dict.get('resolved_secrets', {})
                token = resolved_secrets.get('trilium_token')
                
                if not base_url:
                    context = ErrorContext.create(
                        "init_trilium_client",
                        input_data={'config': trilium_config},
                        suggestions=["Set trilium_upload.base_url in config.yaml"]
                    )
                    return Err(CoreError("Trilium base_url not configured", context))
                
                if not token:
                    context = ErrorContext.create(
                        "init_trilium_client",
                        suggestions=[
                            "Ensure trilium token is available in resolved_secrets",
                            "Check KeyPass/secret resolution in pipeline manager"
                        ]
                    )
                    return Err(CoreError("Trilium token not resolved", context))
                
                # Initialize ETAPI client
                self.trilium_client = ETAPI(base_url, token)
                
                # Test connection
                app_info = self.trilium_client.app_info()
                if not app_info:
                    return Err(CoreError("Failed to connect to Trilium server"))
                
                self.logger.info(
                    f"Trilium client initialized successfully",
                    extra={
                        'base_url': base_url,
                        'trilium_version': app_info.get('appVersion', 'unknown'),
                        'timeout': timeout
                    }
                )
                
                return Ok(self.trilium_client)
                
            except Exception as e:
                context = ErrorContext.create(
                    "init_trilium_client",
                    input_data={
                        'base_url': base_url,
                        'has_token': bool(token),
                        'error': str(e)
                    },
                    suggestions=[
                        "Check Trilium server is running",
                        "Verify base_url is correct",
                        "Validate trilium token permissions"
                    ]
                )
                return Err(CoreError(f"Failed to initialize Trilium client: {e}", context))
    
    @log_function("trilium_upload")
    def upload_transcript(self, transcript_obj: TranskriptObject) -> Result[TranskriptObject, CoreError]:
        """
        Uploads LLM-processed transcript to Trilium as structured note
        
        Args:
            transcript_obj: TranskriptObject with bearbeiteter_transkript
            
        Returns:
            Ok(TranskriptObject): Updated object with trilium_note_id and trilium_link
            Err: Upload error
        """
        start_time = time.time()
        self.upload_stats['total_uploads'] += 1
        
        self.logger.info(
            f"Starting Trilium upload: {transcript_obj.titel}",
            extra={
                'video_title': transcript_obj.titel,
                'has_processed_transcript': bool(transcript_obj.bearbeiteter_transkript),
                'llm_model': transcript_obj.model
            }
        )
        
        # Validate prerequisites
        if not transcript_obj.bearbeiteter_transkript:
            error = CoreError("Cannot upload to Trilium: No processed transcript available")
            self.upload_stats['failed_uploads'] += 1
            return Err(error)
        
        try:
            # Initialize client
            client_result = self._init_trilium_client()
            if isinstance(client_result, Err):
                self.upload_stats['failed_uploads'] += 1
                return client_result
            
            client = unwrap_ok(client_result)
            
            # Get parent note configuration
            trilium_config = self.config_dict.get('trilium_upload', {})
            parent_note_id = trilium_config.get('parent_note_id', 'root')
            
            # Format note content (clean, no metadata)
            content = self._format_note_content(transcript_obj)
            
            # Create note
            note_result = client.create_note(
                parentNoteId=parent_note_id,
                title=transcript_obj.titel,  # Clean title only
                content=content,
                type="text"
            )
            print(note_result)      
            if not note_result or not note_result.get('note'):
                error = CoreError("Failed to create Trilium note: Invalid response")
                self.upload_stats['failed_uploads'] += 1
                return Err(error)
            
            note_id = note_result['note']['noteId']

            
            self.upload_stats['total_notes_created'] += 1
            
            self.logger.debug(
                f"Trilium note created successfully",
                extra={
                    'note_id': note_id,
                    'title': transcript_obj.titel,
                    'parent_note_id': parent_note_id
                }
            )
            
            # Create metadata attributes
            attributes_result = self._create_note_attributes(client, note_id, transcript_obj)
            if isinstance(attributes_result, Err):
                # Note created but attributes failed - warn but continue
                self.logger.warning(
                    f"Note created but attributes failed: {unwrap_err(attributes_result).message}",
                    extra={'note_id': note_id}
                )
            
            # Create tags
            tags_result = self._create_note_tags(client, note_id, trilium_config)
            if isinstance(tags_result, Err):
                # Tags failed - warn but continue
                self.logger.warning(
                    f"Note created but tags failed: {unwrap_err(tags_result).message}",
                    extra={'note_id': note_id}
                )
            
            # Update TranskriptObject with Trilium info
            base_url = trilium_config.get('base_url')
            transcript_obj.trilium_note_id = note_id  # For database storage
            transcript_obj.trilium_link = f"{base_url}/#note/{note_id}"
            transcript_obj.update_stage("trilium_upload_completed")
            
            # Update statistics
            upload_time = time.time() - start_time
            self.upload_stats['successful_uploads'] += 1
            self.upload_stats['total_upload_time_seconds'] += upload_time
            
            self.logger.info(
                f"Trilium upload completed successfully",
                extra={
                    'video_title': transcript_obj.titel,
                    'note_id': note_id,
                    'trilium_link': transcript_obj.trilium_link,
                    'upload_time_seconds': round(upload_time, 2),
                    'content_length': len(content)
                }
            )
            
            return Ok(transcript_obj)
            
        except Exception as e:
            upload_time = time.time() - start_time
            self.upload_stats['failed_uploads'] += 1
            
            context = ErrorContext.create(
                "upload_transcript",
                input_data={
                    'title': transcript_obj.titel,
                    'has_content': bool(transcript_obj.bearbeiteter_transkript),
                    'upload_time': upload_time
                },
                suggestions=[
                    "Check Trilium server connectivity",
                    "Verify token permissions",
                    "Check parent note exists"
                ]
            )
            
            error = CoreError(f"Trilium upload failed: {e}", context)
            transcript_obj.update_stage("trilium_upload_failed")
            
            self.logger.error(
                f"Trilium upload failed",
                extra={
                    'video_title': transcript_obj.titel,
                    'error': str(e),
                    'upload_time_seconds': round(upload_time, 2)
                }
            )
            
            return Err(error)

    def _format_note_content(self, transcript_obj: TranskriptObject) -> str:
        """
        Formats clean note content as HTML
    
        Args:
            transcript_obj: TranskriptObject with content
        
        Returns:
            Formatted note content as HTML
        """
        content_parts = []
    
        # LLM-processed content (primary)
        if transcript_obj.bearbeiteter_transkript:
            content_parts.append("## LLM-Processed Content")
            content_parts.append(transcript_obj.bearbeiteter_transkript.strip())
            content_parts.append("")  # Empty line
    
        # Original transcript (reference)
        if transcript_obj.transkript:
            content_parts.append("## Original Transcript")
            content_parts.append(transcript_obj.transkript.strip())
    
        # Convert markdown to HTML
        markdown_content = "\n".join(content_parts)
        html_content = markdown.markdown(markdown_content)
    
        return html_content
        
    @log_function("create_note_attributes")
    def _create_note_attributes(self, client: ETAPI, note_id: str, transcript_obj: TranskriptObject) -> Result[None, CoreError]:
        """
        Creates metadata attributes for the note
        
        Args:
            client: Trilium ETAPI client
            note_id: Target note ID
            transcript_obj: Source object with metadata
            
        Returns:
            Ok(None): All attributes created successfully
            Err: Attribute creation error
        """
        try:
            # Metadata mapping for individual labels
            metadata_mapping = {
                'sourceType': 'youtube',
                'llmModel': transcript_obj.model or 'unknown',
                'tokens': str(transcript_obj.tokens or 0),
                'cost': str(transcript_obj.cost or 0.0),
                'processingTime': str(transcript_obj.processing_time or 0.0),
                'originalUrl': transcript_obj.original_url or '',
                'kanal': transcript_obj.kanal or '',
                'länge': str(transcript_obj.länge or ''),
                'uploadDate': transcript_obj.upload_date.isoformat() if transcript_obj.upload_date else '',
                'processingDate': transcript_obj.date_created.isoformat() if transcript_obj.date_created else ''
            }
            
            # Create individual metadata labels
            attributes_created = 0
            for name, value in metadata_mapping.items():
                if value:  # Only create non-empty attributes
                    try:
                        result = client.create_attribute(
                            noteId=note_id,
                            type='label',
                            name=name,
                            value=str(value),
                            isInheritable=False
                        )
                        if result:
                            attributes_created += 1
                            self.logger.debug(f"Created attribute: {name}={value}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create attribute {name}: {e}")
            
            self.upload_stats['total_attributes_created'] += attributes_created
            
            self.logger.debug(
                f"Created {attributes_created} metadata attributes",
                extra={
                    'note_id': note_id,
                    'attributes_created': attributes_created,
                    'total_possible': len(metadata_mapping)
                }
            )
            
            return Ok(None)
            
        except Exception as e:
            context = ErrorContext.create(
                "create_note_attributes",
                input_data={'note_id': note_id, 'metadata_count': len(metadata_mapping)},
                suggestions=["Check note exists", "Verify attribute creation permissions"]
            )
            return Err(CoreError(f"Failed to create note attributes: {e}", context))
    
    @log_function("create_note_tags")
    def _create_note_tags(self, client: ETAPI, note_id: str, trilium_config: dict) -> Result[None, CoreError]:
        """
        Creates tags for the note using multi-text tags label
        
        Args:
            client: Trilium ETAPI client
            note_id: Target note ID
            trilium_config: Trilium configuration with auto_tags
            
        Returns:
            Ok(None): All tags created successfully
            Err: Tag creation error
        """
        try:
            auto_tags = trilium_config.get('auto_tags', [])
            if not auto_tags:
                self.logger.debug("No auto_tags configured, skipping tag creation")
                return Ok(None)
            
            # Create tags as individual labels with name 'tags'
            tags_created = 0
            for tag in auto_tags:
                try:
                    result = client.create_attribute(
                        noteId=note_id,
                        type='label',
                        name='tags',  # Multi-text label name
                        value=tag,
                        isInheritable=False
                    )
                    if result:
                        tags_created += 1
                        self.logger.debug(f"Created tag: {tag}")
                except Exception as e:
                    self.logger.warning(f"Failed to create tag {tag}: {e}")
            
            self.upload_stats['total_attributes_created'] += tags_created
            
            self.logger.debug(
                f"Created {tags_created} tags",
                extra={
                    'note_id': note_id,
                    'tags_created': tags_created,
                    'total_tags': len(auto_tags)
                }
            )
            
            return Ok(None)
            
        except Exception as e:
            context = ErrorContext.create(
                "create_note_tags",
                input_data={'note_id': note_id, 'auto_tags': auto_tags},
                suggestions=["Check note exists", "Verify tag creation permissions"]
            )
            return Err(CoreError(f"Failed to create note tags: {e}", context))
    
    def get_upload_statistics(self) -> Dict[str, Any]:
        """Gibt Upload-Statistiken zurück"""
        return {
            **self.upload_stats,
            'success_rate': (
                self.upload_stats['successful_uploads'] / self.upload_stats['total_uploads'] * 100
                if self.upload_stats['total_uploads'] > 0 else 0
            ),
            'average_upload_time_seconds': (
                self.upload_stats['total_upload_time_seconds'] / self.upload_stats['successful_uploads']
                if self.upload_stats['successful_uploads'] > 0 else 0
            ),
            'attributes_per_note': (
                self.upload_stats['total_attributes_created'] / self.upload_stats['total_notes_created']
                if self.upload_stats['total_notes_created'] > 0 else 0
            )
        }

# =============================================================================
# INTEGRATION FUNCTION - Config-Dict Support
# =============================================================================

def upload_to_trilium_dict(transcript_obj: TranskriptObject, config_dict: dict) -> Result[TranskriptObject, CoreError]:
    """
    Integration-Funktion für TranskriptObject-Trilium-Upload mit Config-Dict
    
    Args:
        transcript_obj: TranskriptObject mit bearbeiteter_transkript
        config_dict: Config-Dictionary mit resolved_secrets
        
    Returns:
        Ok(TranskriptObject): TranskriptObject mit trilium_note_id und trilium_link
        Err: Upload-Fehler
    """
    uploader = TriliumUploader(config_dict)
    return uploader.upload_transcript(transcript_obj)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from logging_plus import setup_logging
    from yt_analyzer_config import SecureConfigManager
    from datetime import datetime, time as dt_time
    
    # Setup
    setup_logging("trilium_uploader_test", "DEBUG")
    
    # Test configuration mit Secret-Resolution
    config_manager = SecureConfigManager()
    config_result = config_manager.load_config()
    print(config_result)
    if isinstance(config_result, Ok):
        config = unwrap_ok(config_result)
        
        # Simulate Pipeline Manager Secret Resolution
        config_dict = config.dict()
        
        # Resolve secrets
        trilium_token_result = config_manager.get_trilium_token()
        
        config_dict['resolved_secrets'] = {
            'trilium_token': unwrap_ok(trilium_token_result) if isinstance(trilium_token_result, Ok) else None
        }
        
        # Create test TranskriptObject
        test_obj = TranskriptObject(
            titel="Test Trilium Upload",
            kanal="Test Channel",
            länge=dt_time(0, 10, 30),
            upload_date=datetime.now(),
            original_url="https://www.youtube.com/watch?v=test123",
            transkript="This is the original transcript before processing."
        )
        
        # Mock required fields
        test_obj.bearbeiteter_transkript = "This is a test LLM-processed transcript with important insights."
        test_obj.model = "claude-sonnet-4-20250514"
        test_obj.tokens = 150
        test_obj.cost = 0.0023
        test_obj.processing_time = 5.2
        test_obj.update_stage("llm_processing_completed")
        
        # Test Trilium upload mit Config-Dict
        result = upload_to_trilium_dict(test_obj, config_dict)
        
        if isinstance(result, Ok):
            uploaded_obj = unwrap_ok(result)
            print(f"✅ Trilium upload successful:")
            print(f"  Note ID: {uploaded_obj.trilium_note_id}")
            print(f"  Link: {uploaded_obj.trilium_link}")
            print(f"  Title: {uploaded_obj.titel}")
        else:
            error = unwrap_err(result)
            print(f"❌ Trilium upload failed: {error.message}")
    else:
        print("❌ Config loading failed")
