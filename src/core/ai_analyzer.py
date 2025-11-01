"""
Unified AI Analyzer - Dual-Tier Architecture Orchestrator

Orchestrates AI analysis using SLM/LLM dual-tier architecture.
- SLM: Real-time consultation support (fast summary ~3 seconds)
- LLM: Batch analysis (full features, ~20 seconds)
"""
import time
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# LLM tier components - Shared summarizer instance for memory optimization
from .models.qwen3_4b.summarizer import Qwen2507Summarizer
from .models.qwen3_4b.classifier import CategoryClassifier
from .models.qwen3_4b.title_generator import TitleGenerator

# SLM tier components
from .models.qwen3_1_7b.summarizer import Qwen3Summarizer
from .models.midm_mini.summarizer import MidmSummarizer

from .quality_validator import quality_validator
import hashlib

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """Unified AI Analyzer Orchestrator (Dual-Tier Architecture)"""

    def __init__(self, model_path: str):
        """
        Args:
            model_path: AI model base path (LLM tier)
        """
        self.model_path = model_path

        # LLM tier components (lazy loading)
        self.llm_summarizer = None
        self.category_classifier = None
        self.title_generator = None
        self.current_llm_model = None

        # SLM tier components (lazy loading)
        self.slm_qwen3 = None  # Qwen3-1.7B
        self.slm_midm = None   # Midm-2.0-Mini

        self.model_loaded = False
        self.slm_models_loaded = {}  # SLM model loading status

        # Real-time processing cache (future optimization)
        self.cache_enabled = False
    
    def _load_llm_model(self, model_name: str) -> bool:
        """Load LLM model (lazy loading)"""
        try:
            # If the same model is already loaded, skip
            if self.current_llm_model == model_name and self.model_loaded:
                logger.info(f"[AIAnalyzer] LLM model {model_name} already loaded")
                return True

            logger.info(f"[AIAnalyzer] start loading LLM model: {model_name}")

            # Clean up existing model
            if self.llm_summarizer:
                del self.llm_summarizer
                del self.category_classifier
                del self.title_generator
                self.llm_summarizer = None
                self.category_classifier = None
                self.title_generator = None

            # Load model
            if model_name == "qwen3_4b":
                from .models.qwen3_4b.summarizer import Qwen2507Summarizer
                from .models.qwen3_4b.classifier import CategoryClassifier
                from .models.qwen3_4b.title_generator import TitleGenerator

                self.llm_summarizer = Qwen2507Summarizer(self.model_path)
                self.category_classifier = CategoryClassifier(shared_summarizer=self.llm_summarizer)
                self.title_generator = TitleGenerator(shared_summarizer=self.llm_summarizer)


            elif model_name == "midm_base":
                from .models.midm_base.summarizer import MidmBaseSummarizer
                from .models.midm_base.classifier import CategoryClassifier
                from .models.midm_base.title_generator import TitleGenerator

                self.llm_summarizer = MidmBaseSummarizer(self.model_path)
                self.category_classifier = CategoryClassifier(shared_summarizer=self.llm_summarizer)
                self.title_generator = TitleGenerator(shared_summarizer=self.llm_summarizer)

            elif model_name == "ax_light":
                from .models.ax_light.summarizer import AXLightSummarizer
                from .models.ax_light.classifier import CategoryClassifier
                from .models.ax_light.title_generator import TitleGenerator

                self.llm_summarizer = AXLightSummarizer(self.model_path)
                self.category_classifier = CategoryClassifier(shared_summarizer=self.llm_summarizer)
                self.title_generator = TitleGenerator(shared_summarizer=self.llm_summarizer)

            else:
                logger.error(f"[AIAnalyzer] unsupported LLM model: {model_name}")
                return False
            # Initialize model
            success = self.llm_summarizer.load_model()
            if success:
                self.current_llm_model = model_name
                self.model_loaded = True
                logger.info(f"[AIAnalyzer] finished loading LLM model: {model_name}")
            else:
                logger.error(f"[AIAnalyzer] failed to initialize LLM model: {model_name}")
                self.model_loaded = False

            return success

        except Exception as e:
            logger.error(f"[AIAnalyzer] load failure ({model_name}): {e}")
            self.model_loaded = False
            return False

    def initialize(self) -> bool:
        """Initialize AI analyzer - Load default LLM model (qwen3_4b)"""
        try:
            logger.info("[AIAnalyzer] initialization sequence started")
            logger.info(f"[AIAnalyzer] current working directory: {os.getcwd()}")

            # Load default LLM model (qwen3_4b)
            success = self._load_llm_model("qwen3_4b")
            logger.info(f"[AIAnalyzer] default LLM load result: {success}")

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

    def _load_slm_model(self, model_name: str) -> bool:
        """Load SLM model (lazy loading)"""
        try:
            if model_name in self.slm_models_loaded and self.slm_models_loaded[model_name]:
                return True

            logger.info(f"[AIAnalyzer] start loading SLM model: {model_name}")

            if model_name == "qwen3":
                if self.slm_qwen3 is None:
                    model_path = r"models\Qwen3-1.7B"
                    self.slm_qwen3 = Qwen3Summarizer(model_path)
                success = self.slm_qwen3.load_model()
                self.slm_models_loaded["qwen3"] = success

            elif model_name == "midm":
                if self.slm_midm is None:
                    model_path = r"models\Midm-2.0-Mini"
                    self.slm_midm = MidmSummarizer(model_path)
                success = self.slm_midm.load_model()
                self.slm_models_loaded["midm"] = success

            else:
                logger.error(f"[AIAnalyzer] unsupported SLM model: {model_name}")
                return False

            logger.info(f"[AIAnalyzer] finished loading SLM model: {model_name} - {success}")
            return success

        except Exception as e:
            logger.error(f"[AIAnalyzer] SLM load failure ({model_name}): {e}")
            return False
    
    def analyze(self, conversation_text: str, analysis_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simple analyze method for compatibility with ConsultationService"""
        try:
            logger.info(f"[AIAnalyzer] analyze method called with options: {analysis_options}")
            # Default options for basic analysis
            default_options = {
                "model_tier": "llm",
                "llm_model": "qwen3_4b",
                "include_summary": True,
                "include_category_recommendation": True,
                "include_title_generation": True
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
            model_tier = options.get("model_tier", "llm")
            if model_tier == "slm":
                model_name = f"{result.get('slm_model', 'qwen3')}_slm"
            else:
                model_name = result.get('llm_model', 'qwen3_4b')

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
            # Determine model tier
            model_tier = options.get("model_tier", "llm")
            slm_model = options.get("slm_model", "qwen3")
            llm_model = options.get("llm_model", "qwen3_4b")

            logger.info(f"[AIAnalyzer] DEBUG: model_tier={model_tier}, slm_model={slm_model}, llm_model={llm_model}")

            if model_tier == "slm":
                # SLM mode: Fast summary only
                logger.info(f"[AIAnalyzer] SLM mode started - model: {slm_model}")

                # Load SLM model
                if slm_model == "both":
                    # Load both models
                    if not self._load_slm_model("qwen3") or not self._load_slm_model("midm"):
                        raise RuntimeError("SLM dual model loading failed")
                else:
                    if not self._load_slm_model(slm_model):
                        raise RuntimeError(f"SLM model loading failed: {slm_model}")

                # Generate SLM summary
                if options.get("include_summary", True):
                    if slm_model == "both":
                        # Generate with both models
                        qwen3_result = self._generate_slm_summary(stt_conversation, "qwen3")
                        midm_result = self._generate_slm_summary(stt_conversation, "midm")

                        # Combine dual model results
                        results.update({
                            "summary": f"[Qwen3-1.7B] {qwen3_result.get('summary', '')}\n[Midm-2.0-Mini] {midm_result.get('summary', '')}",
                            "qwen3_result": qwen3_result,
                            "midm_result": midm_result,
                            "dual_model": True,
                            "processing_time": qwen3_result.get("processing_time", 0) + midm_result.get("processing_time", 0)
                        })
                    else:
                        summary_result = self._generate_slm_summary(stt_conversation, slm_model)
                        results.update(summary_result)

                # Category/title generation disabled for SLM (cache optimization)
                logger.info("[AIAnalyzer] SLM mode summary finished")

            else:
                # LLM mode: Full features processing
                logger.info(f"[AIAnalyzer] LLM mode started - {llm_model}")

                # Load selected LLM model
                if not self._load_llm_model(llm_model):
                    raise RuntimeError(f"LLM model loading failed: {llm_model}")

                # 1. Generate summary
                if options.get("include_summary", True):
                    logger.info(f"[AIAnalyzer] generating summary with {llm_model}...")
                    summary_result = self._generate_summary(stt_conversation)
                    results.update(summary_result)

                # 2. Category recommendation
                if options.get("include_category_recommendation", True):
                    logger.info("[AIAnalyzer] generating category recommendations...")
                    categories = self.category_classifier.classify(consultation_content)
                    results["recommended_categories"] = categories

                # 3. Title generation
                if options.get("include_title_generation", True):
                    logger.info("[AIAnalyzer] generating titles...")
                    titles = self.title_generator.generate(consultation_content)
                    results["generated_titles"] = titles

            # Record processing time
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            results["model_tier"] = model_tier

            if model_tier == "slm":
                results["slm_model"] = slm_model
            else:
                results["llm_model"] = llm_model

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
    
    def _generate_summary(self, conversation_text: str) -> Dict[str, Any]:
        """Generate LLM summary (using selected model)"""
        try:
            # Generate summary with selected LLM model
            summary_result = self.llm_summarizer.summarize_consultation(conversation_text)
            
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

    def _generate_slm_summary(self, conversation_text: str, model_name: str) -> Dict[str, Any]:
        """Generate SLM fast summary"""
        try:
            if model_name == "qwen3":
                if not self.slm_qwen3:
                    raise RuntimeError("Qwen3-1.7B SLM model is not loaded")
                result = self.slm_qwen3.summarize_consultation(conversation_text)
            elif model_name == "midm":
                if not self.slm_midm:
                    raise RuntimeError("Midm-2.0-Mini SLM model is not loaded")
                result = self.slm_midm.summarize_consultation(conversation_text)
            else:
                raise ValueError(f"Unsupported SLM model: {model_name}")

            # Format result
            if result.get("success", False):
                summary = result.get("summary", "")

                # SLM 전용 품질 검증 (완화된 기준)
                quality_validation = quality_validator.validate_summary_slm(summary, conversation_text)

                return {
                    "summary": summary,
                    "processing_time": result.get("processing_time", 0),
                    "model_name": result.get("model_name", model_name),
                    "quality_score": quality_validation.get('quality_score', 0.0),
                    "quality_warnings": quality_validation.get('warnings', []),
                    "is_acceptable": quality_validation.get('is_acceptable', False),
                    "slm_tier": True
                }
            else:
                logger.warning(f"[AIAnalyzer] SLM summary failure - {model_name}: {result.get('error', 'Unknown error')}")
                return {
                    "summary": "",
                    "error": result.get("error", "SLM summary generation failed"),
                    "slm_tier": True
                }

        except Exception as e:
            logger.error(f"[AIAnalyzer] SLM summary exception ({model_name}): {e}")
            return {
                "summary": "",
                "error": str(e),
                "slm_tier": True
            }

    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            "model_loaded": self.model_loaded,
            "model_name": "Qwen3-4B-Instruct-2507",
            "available_features": [
                "summary_generation",
                "category_recommendation", 
                "title_generation"
            ]
        }
    
    def cleanup(self):
        """Cleanup all component resources"""
        try:
            logger.info("[AIAnalyzer] memory cleanup started")
            
            # Component cleanup
            if hasattr(self, 'qwen_summarizer') and self.qwen_summarizer:
                self.qwen_summarizer.cleanup()
                
            if hasattr(self, 'category_classifier') and self.category_classifier:
                self.category_classifier.cleanup()
                
            if hasattr(self, 'title_generator') and self.title_generator:
                self.title_generator.cleanup()
            
            # Reset status
            self.model_loaded = False
            
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

