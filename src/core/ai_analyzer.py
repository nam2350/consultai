"""
Unified AI Analyzer - Dual-Tier Architecture Orchestrator

Orchestrates AI analysis using realtime/batch tiers.
- Realtime: consultation support (fast summary ~3 seconds)
- Batch: full analysis (summary + category + title)
"""
import time
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Batch tier components - shared summarizer instance for memory optimization
from .models.qwen3_4b.summarizer import Qwen2507Summarizer
from .models.qwen3_4b.classifier import CategoryClassifier
from .models.qwen3_4b.title_generator import TitleGenerator

# Realtime tier components
from .models.qwen3_1_7b.summarizer import Qwen3Summarizer


from .quality_validator import quality_validator
from .config import get_application_settings
from .model_registry import (
    REALTIME_TIER,
    BATCH_TIER,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_BATCH_MODEL,
    normalize_tier,
    resolve_realtime_model,
    resolve_batch_model,
    get_model_display_name,
    check_model_files,
    MODEL_KEY_ALIASES,
    REALTIME_MODELS,
)
import hashlib

logger = logging.getLogger(__name__)
settings = get_application_settings()

class AIAnalyzer:
    """Unified AI Analyzer Orchestrator (Dual-Tier Architecture)"""

    def __init__(self, model_path: str):
        """
        Args:
            model_path: AI model base path (batch tier)
        """
        self.model_path = model_path

        # Batch tier components (lazy loading)
        self.batch_summarizer = None
        self.batch_category_classifier = None
        self.batch_title_generator = None
        self.current_batch_model = None

        # Realtime tier components (lazy loading)
        self.realtime_qwen3 = None  # Qwen3-1.7B

        self.batch_model_loaded = False
        self.realtime_models_loaded = {}  # realtime model loading status

        # Real-time processing cache (future optimization)
        self.cache_enabled = False

        self.legacy_input_counts = {
            "model_tier": 0,
            "realtime_model": 0,
            "batch_model": 0,
            "model_key_alias": 0,
        }
    
    def _load_batch_model(self, model_name: str) -> bool:
        """Load batch model (lazy loading)"""
        try:
            # If the same model is already loaded, skip
            if self.current_batch_model == model_name and self.batch_model_loaded:
                logger.info(f"[AIAnalyzer] batch model {model_name} already loaded")
                return True

            logger.info(f"[AIAnalyzer] start loading batch model: {model_name}")

            # Clean up existing model
            if self.batch_summarizer:
                del self.batch_summarizer
                del self.batch_category_classifier
                del self.batch_title_generator
                self.batch_summarizer = None
                self.batch_category_classifier = None
                self.batch_title_generator = None

            # Pre-flight model path validation
            model_path = Path(self.model_path)
            valid, missing = check_model_files(model_path)
            if not valid:
                logger.error(
                    "[AIAnalyzer] batch model files missing: %s (path: %s)",
                    ", ".join(missing),
                    model_path,
                )
                return False

            # Load model
            if model_name == "qwen3_4b":
                from .models.qwen3_4b.summarizer import Qwen2507Summarizer
                from .models.qwen3_4b.classifier import CategoryClassifier
                from .models.qwen3_4b.title_generator import TitleGenerator

                self.batch_summarizer = Qwen2507Summarizer(self.model_path)
                self.batch_category_classifier = CategoryClassifier(shared_summarizer=self.batch_summarizer)
                self.batch_title_generator = TitleGenerator(shared_summarizer=self.batch_summarizer)
            else:
                logger.error(f"[AIAnalyzer] unsupported batch model: {model_name}")
                return False
            # Initialize model
            success = self.batch_summarizer.load_model()
            if success:
                self.current_batch_model = model_name
                self.batch_model_loaded = True
                logger.info(f"[AIAnalyzer] finished loading batch model: {model_name}")
            else:
                logger.error(f"[AIAnalyzer] failed to initialize batch model: {model_name}")
                self.batch_model_loaded = False

            return success

        except Exception as e:
            logger.error(f"[AIAnalyzer] load failure ({model_name}): {e}")
            self.batch_model_loaded = False
            return False

    def initialize(self) -> bool:
        """Initialize AI analyzer - Load default batch model (qwen3_4b)"""
        try:
            logger.info("[AIAnalyzer] initialization sequence started")
            logger.info(f"[AIAnalyzer] current working directory: {os.getcwd()}")

            # Load default batch model (qwen3_4b)
            success = self._load_batch_model(DEFAULT_BATCH_MODEL)
            logger.info(f"[AIAnalyzer] default batch load result: {success}")

            if success:
                logger.info("[AIAnalyzer] shared components initialized")
                logger.info("[AIAnalyzer] summary/classification/title available in single pass")
                return True
            else:
                logger.error("[AIAnalyzer] model load failed")
                return False
                
        except (OSError, IOError) as e:
            logger.error(f"[AIAnalyzer] model file access error: {e}")
            return False
        except (RuntimeError, ImportError) as e:
            logger.error(f"[AIAnalyzer] model initialization error: {e}")
            return False
        except Exception as e:
            logger.error(f"[AIAnalyzer] unexpected initialization failure: {e}")
            return False

    def _load_realtime_model(self, model_name: str) -> bool:
        """Load realtime model (lazy loading)"""
        try:
            if model_name in self.realtime_models_loaded and self.realtime_models_loaded[model_name]:
                return True

            logger.info(f"[AIAnalyzer] start loading realtime model: {model_name}")

            if model_name == "qwen3":
                if self.realtime_qwen3 is None:
                    model_path = settings.REALTIME_MODEL_PATH_QWEN3
                    valid, missing = check_model_files(Path(model_path))
                    if not valid:
                        logger.error(
                            "[AIAnalyzer] realtime model files missing: %s (path: %s)",
                            ", ".join(missing),
                            model_path,
                        )
                        return False
                    self.realtime_qwen3 = Qwen3Summarizer(model_path)
                success = self.realtime_qwen3.load_model()
                self.realtime_models_loaded["qwen3"] = success
            else:
                logger.error(f"[AIAnalyzer] unsupported realtime model: {model_name}")
                return False

            logger.info(f"[AIAnalyzer] finished loading realtime model: {model_name} - {success}")
            return success

        except Exception as e:
            logger.error(f"[AIAnalyzer] realtime load failure ({model_name}): {e}")
            return False
    
    def analyze(self, conversation_text: str, analysis_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simple analyze method for compatibility with ConsultationService"""
        try:
            logger.info(f"[AIAnalyzer] analyze method called with options: {analysis_options}")
            # Default options for basic analysis
            default_options = {
                "model_tier": BATCH_TIER,
                "batch_model": DEFAULT_BATCH_MODEL,
                "include_summary": True,
                "include_category_recommendation": True,
                "include_title_generation": True,
            }

            # Merge with provided options
            options = default_options.copy()
            if analysis_options:
                options.update(analysis_options)

            # Call the full analysis method
            result = self.analyze_consultation(
                consultation_content=conversation_text,
                stt_conversation=conversation_text,
                options=options
            )

            # Return success format expected by ConsultationService
            model_tier = normalize_tier(options.get("model_tier", BATCH_TIER))
            realtime_model = options.get("realtime_model")
            batch_model = options.get("batch_model")
            model_name = get_model_display_name(
                model_tier,
                realtime_model if model_tier == REALTIME_TIER else batch_model,
            )

            return {
                "success": True,
                "summary": result.get("summary", ""),
                "recommended_categories": result.get("recommended_categories", []),
                "generated_titles": result.get("generated_titles", []),
                "processing_time": result.get("processing_time", 0),
                "model_name": model_name,
                "model_tier": model_tier
            }

        except Exception as e:
            logger.error(f"[AIAnalyzer] analyze method error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _warn_legacy_options(self, options: Dict[str, Any]) -> None:
        raw_tier = str(options.get("model_tier", "")).strip().lower()
        if raw_tier and raw_tier not in {REALTIME_TIER, BATCH_TIER}:
            self.legacy_input_counts["model_tier"] += 1
            logger.warning(
                "[AIAnalyzer] Unsupported model_tier value received; use 'realtime' or 'batch'."
            )
        for key_name in ("realtime_model", "batch_model"):
            raw_value = options.get(key_name)
            if not raw_value:
                continue
            normalized_value = MODEL_KEY_ALIASES.get(str(raw_value).strip().lower())
            if normalized_value and normalized_value != str(raw_value).strip().lower():
                self.legacy_input_counts["model_key_alias"] += 1
                logger.warning(
                    "[AIAnalyzer] Legacy model key '%s' received; use '%s'.",
                    raw_value,
                    normalized_value,
                )

    def analyze_consultation(
        self, 
        consultation_content: str,
        stt_conversation: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dual-tier AI analysis processing"""

        start_time = time.time()
        results = {}

        try:
            logger.info(f"[AIAnalyzer] analyze_consultation method called with options: {options}")
            self._warn_legacy_options(options)
            # Determine model tier
            model_tier = normalize_tier(options.get("model_tier", BATCH_TIER))
            realtime_model = options.get("realtime_model")
            batch_model = options.get("batch_model")
            realtime_model = resolve_realtime_model(realtime_model)
            batch_model = resolve_batch_model(batch_model)

            logger.info(
                "[AIAnalyzer] DEBUG: model_tier=%s, realtime_model=%s, batch_model=%s",
                model_tier,
                realtime_model,
                batch_model,
            )

            if model_tier == REALTIME_TIER:
                # Realtime mode: Fast summary only
                logger.info(f"[AIAnalyzer] realtime mode started - model: {realtime_model}")

                # Load realtime model
                if not self._load_realtime_model(realtime_model):
                    raise RuntimeError(f"Realtime model loading failed: {realtime_model}")

                # Generate realtime summary
                if options.get("include_summary", True):
                    summary_result = self._generate_realtime_summary(stt_conversation, realtime_model)
                    results.update(summary_result)

                # Category/title generation disabled for realtime (cache optimization)
                logger.info("[AIAnalyzer] realtime mode summary finished")

            else:
                # Batch mode: Full features processing
                logger.info(f"[AIAnalyzer] batch mode started - {batch_model}")

                # Load selected batch model
                if not self._load_batch_model(batch_model):
                    raise RuntimeError(f"Batch model loading failed: {batch_model}")

                # 1. Generate summary
                if options.get("include_summary", True):
                    logger.info(f"[AIAnalyzer] generating summary with {batch_model}...")
                    summary_result = self._generate_batch_summary(stt_conversation)
                    results.update(summary_result)

                # 2. Category recommendation
                if options.get("include_category_recommendation", True):
                    logger.info("[AIAnalyzer] generating category recommendations...")
                    categories = self.batch_category_classifier.classify(consultation_content)
                    results["recommended_categories"] = categories

                # 3. Title generation
                if options.get("include_title_generation", True):
                    logger.info("[AIAnalyzer] generating titles...")
                    titles = self.batch_title_generator.generate(consultation_content)
                    results["generated_titles"] = titles

            # Record processing time
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            results["model_tier"] = model_tier

            if model_tier == REALTIME_TIER:
                results["realtime_model"] = realtime_model
            else:
                results["batch_model"] = batch_model

            # Memory cleanup for next processing
            self.reset_for_next_analysis()

            logger.info(f"[AIAnalyzer] completed ({model_tier}) - {processing_time:.2f}s")
            return results

        except Exception as e:
            logger.error(f"[AIAnalyzer] processing failure: {e}")
            # Memory cleanup on failure
            try:
                self.reset_for_next_analysis()
            except:
                pass
            raise
    
    def _generate_batch_summary(self, conversation_text: str) -> Dict[str, Any]:
        """Generate batch summary (using selected model)"""
        try:
            # Generate summary with selected batch model
            summary_result = self.batch_summarizer.summarize_consultation(conversation_text)
            
            if not summary_result.get('success', False):
                raise RuntimeError(f"Summary generation failed: {summary_result.get('error', 'Unknown error')}")
            
            summary = summary_result['summary']
            
            # Apply existing quality validation system
            quality_validation = quality_validator.validate_summary(summary, conversation_text)
            
            result = {
                "summary": summary,
                "quality_score": quality_validation.get('quality_score', 0.0),
                "quality_warnings": quality_validation.get('warnings', []),
                "ai_processing_time": summary_result.get('processing_time', 0.0)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"[AIAnalyzer] summary validation error: {e}")
            raise

    def _generate_realtime_summary(self, conversation_text: str, model_name: str) -> Dict[str, Any]:
        """Generate realtime fast summary"""
        try:
            if model_name == "qwen3":
                if not self.realtime_qwen3:
                    raise RuntimeError("Qwen3-1.7B realtime model is not loaded")
                result = self.realtime_qwen3.summarize_consultation(conversation_text)

            else:
                raise ValueError(f"Unsupported realtime model: {model_name}")

            # Format result
            if result.get("success", False):
                summary = result.get("summary", "")

                # Realtime 전용 품질 검증 (완화된 기준)
                quality_validation = quality_validator.validate_summary_realtime(summary, conversation_text)

                return {
                    "summary": summary,
                    "processing_time": result.get("processing_time", 0),
                    "model_name": result.get("model_name", model_name),
                    "quality_score": quality_validation.get('quality_score', 0.0),
                    "quality_warnings": quality_validation.get('warnings', []),
                    "is_acceptable": quality_validation.get('is_acceptable', False),
                    "realtime_tier": True
                }
            else:
                logger.warning(f"[AIAnalyzer] realtime summary failure - {model_name}: {result.get('error', 'Unknown error')}")
                return {
                    "summary": "",
                    "error": result.get("error", "Realtime summary generation failed"),
                    "realtime_tier": True
                }

        except Exception as e:
            logger.error(f"[AIAnalyzer] realtime summary exception ({model_name}): {e}")
            return {
                "summary": "",
                "error": str(e),
                "realtime_tier": True
            }

    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        batch_model_key = self.current_batch_model or DEFAULT_BATCH_MODEL
        batch_display = get_model_display_name(BATCH_TIER, batch_model_key)
        realtime_loaded = [key for key, loaded in self.realtime_models_loaded.items() if loaded]
        realtime_display = [
            get_model_display_name(REALTIME_TIER, key) for key in realtime_loaded
        ]
        return {
            "model_loaded": self.batch_model_loaded,
            "model_name": batch_display,
            "available_features": [
                "summary_generation",
                "category_recommendation", 
                "title_generation"
            ],
            "batch": {
                "loaded": self.batch_model_loaded,
                "model_key": batch_model_key,
                "model_display": batch_display,
                "model_path": self.model_path,
            },
            "realtime": {
                "loaded_models": realtime_loaded,
                "loaded_displays": realtime_display,
                "available_models": list(REALTIME_MODELS.keys()),
                "model_path_qwen3": settings.REALTIME_MODEL_PATH_QWEN3,
            },
            "legacy_input_counts": dict(self.legacy_input_counts),
        }
    
    def cleanup(self):
        """Cleanup all component resources"""
        try:
            logger.info("[AIAnalyzer] memory cleanup started")
            
            # Component cleanup
            if self.batch_summarizer:
                self.batch_summarizer.cleanup()
            if self.realtime_qwen3 and hasattr(self.realtime_qwen3, "cleanup"):
                self.realtime_qwen3.cleanup()
            if self.batch_category_classifier and hasattr(self.batch_category_classifier, "cleanup"):
                self.batch_category_classifier.cleanup()
            if self.batch_title_generator and hasattr(self.batch_title_generator, "cleanup"):
                self.batch_title_generator.cleanup()
            
            # Reset status
            self.batch_model_loaded = False
            self.current_batch_model = None
            self.batch_summarizer = None
            self.batch_category_classifier = None
            self.batch_title_generator = None
            
            logger.info("[AIAnalyzer] memory cleanup finished")
            
        except Exception as e:
            logger.error(f"[AIAnalyzer] memory cleanup error: {e}")
    
    def reset_for_next_analysis(self):
        """Memory cleanup for next processing (optimized periodic)"""
        try:
            # Perform cleanup every 5 times (optimized frequency)
            if not hasattr(self, '_cleanup_counter'):
                self._cleanup_counter = 0
            
            self._cleanup_counter += 1
            
            if self._cleanup_counter % 5 == 0:  # Cleanup every 5 times
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # logger.debug(f"[AIAnalyzer] periodic cleanup ({self._cleanup_counter})")  # disabled in production
            
        except Exception as e:
            logger.warning(f"[AIAnalyzer] cleanup warning: {e}")

