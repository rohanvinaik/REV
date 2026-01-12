#!/usr/bin/env python3
"""
Strategic Testing Orchestrator

This module orchestrates multi-stage model testing based on fingerprint
identification, adapting strategies based on detected architecture.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging

from .model_library import ModelFingerprintLibrary, ModelIdentificationResult, BaseModelFingerprint
from ..hdc.unified_fingerprint import UnifiedFingerprintGenerator, UnifiedFingerprint
from ..analysis.unified_model_analysis import UnifiedModelAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class TestingStage:
    """Represents a stage in the multi-stage testing process"""
    stage_name: str
    stage_type: str  # "identification", "light_pass", "targeted", "comprehensive"
    duration_estimate: float  # Estimated time in seconds
    required_resources: Dict[str, Any]
    cassettes: List[str]
    focus_layers: List[int]
    configuration: Dict[str, Any]


@dataclass
class OrchestrationPlan:
    """Complete testing plan for a model"""
    model_path: str
    identified_architecture: Optional[str]
    confidence: float
    stages: List[TestingStage]
    total_estimated_time: float
    optimization_settings: Dict[str, Any]
    reasoning: List[str]


class StrategicTestingOrchestrator:
    """
    Orchestrates intelligent, multi-stage model testing based on
    fingerprint identification and adaptive strategies.
    """
    
    def __init__(self,
                 library_path: str = "./fingerprint_library",
                 enable_caching: bool = True):
        """
        Initialize the orchestrator.
        
        Args:
            library_path: Path to fingerprint library
            enable_caching: Whether to cache identification results
        """
        self.library = ModelFingerprintLibrary(library_path)
        self.fingerprint_generator = None  # Will be initialized as needed
        self.analyzer = UnifiedModelAnalyzer()
        self.enable_caching = enable_caching
        
        # Cache for identification results
        self.identification_cache = {}
        
        # Stage configurations
        self.stage_configs = self._initialize_stage_configs()
    
    def _initialize_stage_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize configuration templates for different testing stages."""
        return {
            "quick_identification": {
                "max_layers": 10,
                "challenges": 3,
                "cassettes": ["syntactic", "semantic"],
                "time_limit": 300,  # 5 minutes
                "purpose": "Fast architecture identification"
            },
            "light_pass": {
                "max_layers": 20,
                "challenges": 5,
                "cassettes": ["syntactic", "semantic", "transform"],
                "time_limit": 1800,  # 30 minutes
                "purpose": "Basic model characterization"
            },
            "targeted_analysis": {
                "focus_on_vulnerabilities": True,
                "challenges": 10,
                "time_limit": 7200,  # 2 hours
                "purpose": "Focused testing on known weaknesses"
            },
            "comprehensive_analysis": {
                "all_layers": True,
                "all_cassettes": True,
                "challenges": 20,
                "time_limit": 36000,  # 10 hours
                "purpose": "Complete model profiling"
            },
            "novel_exploration": {
                "discovery_mode": True,
                "adaptive_sampling": True,
                "challenges": 15,
                "time_limit": 14400,  # 4 hours
                "purpose": "Explore unknown architecture"
            }
        }
    
    def create_orchestration_plan(self,
                                 model_path: str,
                                 claimed_family: Optional[str] = None,
                                 force_comprehensive: bool = False,
                                 time_budget: Optional[float] = None) -> OrchestrationPlan:
        """
        Create a multi-stage testing plan for a model.
        
        Args:
            model_path: Path to the model to test
            claimed_family: Optional claimed architecture family
            force_comprehensive: Force comprehensive analysis regardless of identification
            time_budget: Optional time budget in seconds
            
        Returns:
            Complete orchestration plan with stages and strategies
        """
        logger.info(f"Creating orchestration plan for {model_path}")
        
        stages = []
        reasoning = []
        total_time = 0
        
        # Stage 1: Quick Identification
        if not force_comprehensive:
            identification_stage = self._create_identification_stage(
                model_path, claimed_family
            )
            stages.append(identification_stage)
            total_time += identification_stage.duration_estimate
            reasoning.append("Stage 1: Quick identification to determine architecture")
        
        # Use dual library system for identification
        from .dual_library_system import identify_and_strategize
        identification, strategy = identify_and_strategize(model_path)
        
        # Log identification result
        logger.info(f"Model identification: {identification.identified_family} "
                   f"(confidence: {identification.confidence:.1%}, method: {identification.method})")
        
        if identification.method == "unknown":
            # Unknown model - need diagnostic first
            reasoning.append("Unknown model - running diagnostic fingerprinting first")
            
            diagnostic_stage = TestingStage(
                stage_name="Diagnostic Fingerprinting",
                stage_type="diagnostic",
                duration_estimate=600,  # 10 minutes
                required_resources={"memory_gb": 4, "compute": "low"},
                cassettes=["syntactic"],
                focus_layers=strategy.get("sample_layers", list(range(0, 100, 10))),
                configuration={"diagnostic": True, "quick_scan": True},
                metrics_to_collect=["layer_norms", "attention_patterns"],
                validation_criteria={"min_challenges": strategy.get("challenges", 5)}
            )
            stages.append(diagnostic_stage)
            total_time += diagnostic_stage.duration_estimate
            
            # After diagnostic, run standard analysis
            analysis_stage = TestingStage(
                stage_name="Standard Behavioral Analysis",
                stage_type="standard",
                duration_estimate=1200,  # 20 minutes
                required_resources={"memory_gb": 8, "compute": "medium"},
                cassettes=["syntactic", "semantic", "arithmetic"],
                focus_layers=list(range(0, 100, 5)),
                configuration={"standard": True},
                metrics_to_collect=["divergence", "attention", "mlp"],
                validation_criteria={"min_challenges": 10}
            )
            stages.append(analysis_stage)
            total_time += analysis_stage.duration_estimate
            
        elif identification.method == "name_match":
            # Known family - use targeted testing
            reasoning.append(f"Identified as {identification.identified_family} family - using targeted testing")
            reasoning.append(f"Reference model: {identification.reference_model}")
            
            targeted_stage = TestingStage(
                stage_name=f"Targeted {identification.identified_family.upper()} Testing",
                stage_type="targeted",
                duration_estimate=900,  # 15 minutes
                required_resources={"memory_gb": 6, "compute": "medium"},
                cassettes=strategy.get("cassettes", ["syntactic", "semantic"]),
                focus_layers=strategy.get("focus_layers", []),
                configuration={
                    "targeted": True,
                    "family": identification.identified_family,
                    "reference": identification.reference_model
                },
                metrics_to_collect=["divergence", "family_specific"],
                validation_criteria={"min_challenges": strategy.get("challenges", 10)}
            )
            stages.append(targeted_stage)
            total_time += targeted_stage.duration_estimate
        
        # Additional exploratory stage if confidence is low
        if identification.confidence < 0.5:
            # Novel architecture - exploratory approach
            reasoning.append("Novel architecture detected - using exploratory approach")
            
            exploration_stage = self._create_exploration_stage(model_path)
            stages.append(exploration_stage)
            total_time += exploration_stage.duration_estimate
            
            # Stage 3: Build base fingerprint
            fingerprint_stage = self._create_fingerprint_building_stage(model_path)
            stages.append(fingerprint_stage)
            total_time += fingerprint_stage.duration_estimate
            
        elif identification.confidence >= 0.85:
            # High confidence match - targeted testing
            reasoning.append(f"High confidence match with {identification.identified_family}")
            
            strategy = self.library.get_testing_strategy(identification)
            
            targeted_stage = self._create_targeted_stage(
                model_path, strategy, identification
            )
            stages.append(targeted_stage)
            total_time += targeted_stage.duration_estimate
            
            # Optional adversarial testing for known architectures
            if strategy.get("adversarial_config"):
                adversarial_stage = self._create_adversarial_stage(
                    model_path, strategy["adversarial_config"]
                )
                stages.append(adversarial_stage)
                total_time += adversarial_stage.duration_estimate
                reasoning.append("Added adversarial testing for known vulnerabilities")
                
        else:
            # Medium confidence - adaptive approach
            reasoning.append(f"Medium confidence match - using adaptive approach")
            
            # Light pass to gather more information
            light_stage = self._create_light_pass_stage(model_path)
            stages.append(light_stage)
            total_time += light_stage.duration_estimate
            
            # Adaptive testing based on light pass results
            adaptive_stage = self._create_adaptive_stage(model_path, identification)
            stages.append(adaptive_stage)
            total_time += adaptive_stage.duration_estimate
        
        # Stage 3: Comparison with claimed architecture (if provided)
        if claimed_family and claimed_family != identification.identified_family:
            reasoning.append(f"Verifying claim of {claimed_family} architecture")
            
            verification_stage = self._create_verification_stage(
                model_path, claimed_family, identification.identified_family
            )
            stages.append(verification_stage)
            total_time += verification_stage.duration_estimate
        
        # Apply time budget constraints if specified
        if time_budget and total_time > time_budget:
            stages = self._optimize_for_time_budget(stages, time_budget)
            reasoning.append(f"Optimized plan to fit {time_budget/3600:.1f} hour budget")
            total_time = sum(s.duration_estimate for s in stages)
        
        # Create optimization settings based on identification
        optimization = self._create_optimization_settings(identification)
        
        return OrchestrationPlan(
            model_path=model_path,
            identified_architecture=identification.identified_family,
            confidence=identification.confidence,
            stages=stages,
            total_estimated_time=total_time,
            optimization_settings=optimization,
            reasoning=reasoning
        )
    
    def _create_identification_stage(self, 
                                    model_path: str,
                                    claimed_family: Optional[str]) -> TestingStage:
        """Create quick identification stage."""
        config = self.stage_configs["quick_identification"]
        
        return TestingStage(
            stage_name="Architecture Identification",
            stage_type="identification",
            duration_estimate=config["time_limit"],
            required_resources={"memory_gb": 8, "compute": "low"},
            cassettes=config["cassettes"],
            focus_layers=list(range(0, config["max_layers"], 2)),
            configuration={
                "challenges": config["challenges"],
                "quick_mode": True,
                "claimed_family": claimed_family,
                "purpose": config["purpose"]
            }
        )
    
    def _create_exploration_stage(self, model_path: str) -> TestingStage:
        """Create exploration stage for novel architectures."""
        config = self.stage_configs["novel_exploration"]
        
        return TestingStage(
            stage_name="Novel Architecture Exploration",
            stage_type="exploration",
            duration_estimate=config["time_limit"],
            required_resources={"memory_gb": 16, "compute": "medium"},
            cassettes=["syntactic", "semantic", "recursive", "transform", "meta"],
            focus_layers=list(range(0, 100, 5)),  # Sample every 5th layer
            configuration={
                "discovery_mode": True,
                "adaptive_sampling": True,
                "challenges": config["challenges"],
                "purpose": config["purpose"]
            }
        )
    
    def _create_fingerprint_building_stage(self, model_path: str) -> TestingStage:
        """Create stage for building new base fingerprint."""
        return TestingStage(
            stage_name="Base Fingerprint Construction",
            stage_type="fingerprint_building",
            duration_estimate=3600,  # 1 hour
            required_resources={"memory_gb": 12, "compute": "medium"},
            cassettes=["syntactic", "semantic", "transform"],
            focus_layers=list(range(0, 100, 10)),
            configuration={
                "create_base": True,
                "comprehensive_profiling": True,
                "save_to_library": True
            }
        )
    
    def _create_targeted_stage(self,
                              model_path: str,
                              strategy: Dict[str, Any],
                              identification: ModelIdentificationResult) -> TestingStage:
        """Create targeted testing stage based on known architecture."""
        config = self.stage_configs["targeted_analysis"]
        
        return TestingStage(
            stage_name=f"Targeted {identification.identified_family} Analysis",
            stage_type="targeted",
            duration_estimate=config["time_limit"],
            required_resources={"memory_gb": 16, "compute": "high"},
            cassettes=strategy["cassettes"],
            focus_layers=strategy["focus_layers"],
            configuration={
                "strategy": strategy,
                "known_vulnerabilities": strategy.get("focus_layers", []),
                "optimization": strategy.get("optimization", {}),
                "challenges": config["challenges"],
                "purpose": f"Target known {identification.identified_family} characteristics"
            }
        )
    
    def _create_adversarial_stage(self,
                                 model_path: str,
                                 adversarial_config: Dict[str, Any]) -> TestingStage:
        """Create adversarial testing stage."""
        return TestingStage(
            stage_name="Adversarial Security Testing",
            stage_type="adversarial",
            duration_estimate=7200,  # 2 hours
            required_resources={"memory_gb": 16, "compute": "high"},
            cassettes=["adversarial", "extraction", "inversion"],
            focus_layers=adversarial_config.get("target_layers", []),
            configuration={
                "attack_types": adversarial_config.get("attack_types", []),
                "sensitivity": adversarial_config.get("sensitivity", "medium"),
                "security_focus": True
            }
        )
    
    def _create_light_pass_stage(self, model_path: str) -> TestingStage:
        """Create light pass stage for initial characterization."""
        config = self.stage_configs["light_pass"]
        
        return TestingStage(
            stage_name="Light Characterization Pass",
            stage_type="light_pass",
            duration_estimate=config["time_limit"],
            required_resources={"memory_gb": 8, "compute": "low"},
            cassettes=config["cassettes"],
            focus_layers=list(range(0, config["max_layers"], 4)),
            configuration={
                "challenges": config["challenges"],
                "quick_profiling": True,
                "purpose": config["purpose"]
            }
        )
    
    def _create_adaptive_stage(self,
                              model_path: str,
                              identification: ModelIdentificationResult) -> TestingStage:
        """Create adaptive testing stage."""
        return TestingStage(
            stage_name="Adaptive Analysis",
            stage_type="adaptive",
            duration_estimate=10800,  # 3 hours
            required_resources={"memory_gb": 12, "compute": "medium"},
            cassettes=["semantic", "recursive", "transform", "theory_of_mind"],
            focus_layers=list(range(0, 80, 8)),
            configuration={
                "adaptive_mode": True,
                "confidence_threshold": identification.confidence,
                "expand_on_anomalies": True
            }
        )
    
    def _create_verification_stage(self,
                                  model_path: str,
                                  claimed: str,
                                  detected: str) -> TestingStage:
        """Create verification stage for architecture claims."""
        return TestingStage(
            stage_name=f"Verify {claimed} vs {detected}",
            stage_type="verification",
            duration_estimate=3600,  # 1 hour
            required_resources={"memory_gb": 8, "compute": "medium"},
            cassettes=["comparative", "differential"],
            focus_layers=[0, 10, 20, 40, 60],
            configuration={
                "compare_architectures": True,
                "claimed": claimed,
                "detected": detected,
                "differential_analysis": True
            }
        )
    
    def _optimize_for_time_budget(self,
                                 stages: List[TestingStage],
                                 budget: float) -> List[TestingStage]:
        """Optimize stages to fit within time budget."""
        # Sort by priority (identification > targeted > comprehensive)
        priority_order = ["identification", "targeted", "light_pass", 
                         "adaptive", "verification", "exploration", 
                         "adversarial", "fingerprint_building"]
        
        stages_sorted = sorted(stages, 
                             key=lambda s: priority_order.index(s.stage_type) 
                             if s.stage_type in priority_order else 999)
        
        optimized = []
        remaining_budget = budget
        
        for stage in stages_sorted:
            if stage.duration_estimate <= remaining_budget:
                optimized.append(stage)
                remaining_budget -= stage.duration_estimate
            elif remaining_budget > 600:  # At least 10 minutes
                # Create reduced version of stage
                reduced = TestingStage(
                    stage_name=f"{stage.stage_name} (Reduced)",
                    stage_type=stage.stage_type,
                    duration_estimate=remaining_budget * 0.8,
                    required_resources=stage.required_resources,
                    cassettes=stage.cassettes[:2],  # Reduce cassettes
                    focus_layers=stage.focus_layers[::2],  # Sample half
                    configuration={**stage.configuration, "reduced": True}
                )
                optimized.append(reduced)
                break
        
        return optimized
    
    def _create_optimization_settings(self,
                                     identification: ModelIdentificationResult) -> Dict[str, Any]:
        """Create optimization settings based on identification."""
        settings = {
            "parallel_execution": True,
            "cache_intermediates": True,
            "skip_redundant": True
        }
        
        if identification.confidence > 0.9:
            # High confidence - aggressive optimization
            settings.update({
                "skip_stable_layers": True,
                "focus_on_boundaries": True,
                "use_cached_baseline": True,
                "parallel_workers": 8
            })
        elif identification.confidence > 0.7:
            # Medium confidence - balanced optimization
            settings.update({
                "adaptive_sampling": True,
                "parallel_workers": 4
            })
        else:
            # Low confidence - conservative approach
            settings.update({
                "comprehensive_sampling": True,
                "parallel_workers": 2,
                "verify_all": True
            })
        
        return settings
    
    def _create_mock_fingerprint(self, model_path: str) -> UnifiedFingerprint:
        """Create mock fingerprint for testing (would be real in production)."""
        import numpy as np
        
        return UnifiedFingerprint(
            unified_hypervector=np.random.randn(10000),
            prompt_hypervector=np.random.randn(10000),
            pathway_hypervector=np.random.randn(10000),
            response_hypervector=np.random.randn(10000),
            model_id=Path(model_path).stem,
            prompt_text="Mock prompt for orchestration",
            response_text="Mock response for orchestration",
            layer_count=32,  # Default for testing
            layers_sampled=list(range(0, 32, 4)),  # Sample every 4th layer
            fingerprint_quality=0.9,
            divergence_stats={'mean': 0.5, 'std': 0.1, 'max': 0.7, 'min': 0.3},
            binding_strength=0.85
        )
    
    def execute_plan(self,
                    plan: OrchestrationPlan,
                    pipeline_interface: Any,
                    progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Execute an orchestration plan.
        
        Args:
            plan: Orchestration plan to execute
            pipeline_interface: Interface to REV pipeline
            progress_callback: Optional callback for progress updates
            
        Returns:
            Execution results with all stage outputs
        """
        logger.info(f"Executing orchestration plan with {len(plan.stages)} stages")
        
        results = {
            "model_path": plan.model_path,
            "identified_architecture": plan.identified_architecture,
            "confidence": plan.confidence,
            "stages": {},
            "overall_status": "in_progress",
            "start_time": datetime.now()
        }
        
        for i, stage in enumerate(plan.stages):
            stage_start = time.time()
            
            logger.info(f"Executing stage {i+1}/{len(plan.stages)}: {stage.stage_name}")
            
            if progress_callback:
                progress_callback(i / len(plan.stages), stage.stage_name)
            
            try:
                # Execute stage based on type
                if stage.stage_type == "identification":
                    stage_result = self._execute_identification(
                        plan.model_path, stage, pipeline_interface
                    )
                elif stage.stage_type == "targeted":
                    stage_result = self._execute_targeted(
                        plan.model_path, stage, pipeline_interface
                    )
                elif stage.stage_type == "exploration":
                    stage_result = self._execute_exploration(
                        plan.model_path, stage, pipeline_interface
                    )
                else:
                    stage_result = self._execute_generic(
                        plan.model_path, stage, pipeline_interface
                    )
                
                stage_result["duration"] = time.time() - stage_start
                results["stages"][stage.stage_name] = stage_result
                
            except Exception as e:
                logger.error(f"Stage {stage.stage_name} failed: {e}")
                results["stages"][stage.stage_name] = {
                    "status": "failed",
                    "error": str(e),
                    "duration": time.time() - stage_start
                }
        
        results["end_time"] = datetime.now()
        results["total_duration"] = (results["end_time"] - results["start_time"]).total_seconds()
        results["overall_status"] = "completed"
        
        # Generate final report
        results["summary"] = self._generate_execution_summary(results, plan)
        
        return results
    
    def _execute_identification(self, 
                               model_path: str,
                               stage: TestingStage,
                               pipeline: Any) -> Dict[str, Any]:
        """Execute identification stage."""
        # Simplified implementation - would call actual pipeline
        return {
            "status": "completed",
            "identified_family": "llama",
            "confidence": 0.92,
            "layers_analyzed": stage.focus_layers,
            "cassettes_used": stage.cassettes
        }
    
    def _execute_targeted(self,
                         model_path: str,
                         stage: TestingStage,
                         pipeline: Any) -> Dict[str, Any]:
        """Execute targeted testing stage."""
        return {
            "status": "completed",
            "vulnerabilities_tested": stage.configuration.get("known_vulnerabilities", []),
            "anomalies_found": [],
            "optimization_applied": stage.configuration.get("optimization", {})
        }
    
    def _execute_exploration(self,
                            model_path: str,
                            stage: TestingStage,
                            pipeline: Any) -> Dict[str, Any]:
        """Execute exploration stage."""
        return {
            "status": "completed",
            "novel_patterns": [],
            "architecture_features": {},
            "recommended_family": "unknown"
        }
    
    def _execute_generic(self,
                        model_path: str,
                        stage: TestingStage,
                        pipeline: Any) -> Dict[str, Any]:
        """Execute generic stage."""
        return {
            "status": "completed",
            "stage_type": stage.stage_type,
            "configuration": stage.configuration
        }
    
    def _generate_execution_summary(self,
                                   results: Dict[str, Any],
                                   plan: OrchestrationPlan) -> Dict[str, Any]:
        """Generate summary of execution results."""
        successful_stages = sum(1 for s in results["stages"].values() 
                              if s.get("status") == "completed")
        
        return {
            "total_stages": len(plan.stages),
            "successful_stages": successful_stages,
            "total_duration": results["total_duration"],
            "identified_architecture": plan.identified_architecture,
            "confidence": plan.confidence,
            "optimization_achieved": results["total_duration"] <= plan.total_estimated_time
        }