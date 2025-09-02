#!/usr/bin/env python3
"""
Cassette Executor - Integrates advanced probe cassettes with REV pipeline.
Runs as Phase 2 after baseline topology discovery.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass, asdict

from .advanced_probe_cassettes import (
    CassetteLibrary, ProbeType, Probe, ProbeCassette
)

logger = logging.getLogger(__name__)

@dataclass 
class CassetteExecutionConfig:
    """Configuration for cassette-based probe execution."""
    topology_file: str
    output_dir: str = "./cassette_results"
    max_probes_per_layer: int = 10
    probe_timeout: float = 120.0  # seconds
    parallel_execution: bool = False
    probe_types: List[ProbeType] = None  # None means all types
    complexity_range: tuple = (1, 10)  # Min and max complexity
    
    def __post_init__(self):
        if self.probe_types is None:
            self.probe_types = list(ProbeType)

class CassetteExecutor:
    """Executes probe cassettes based on discovered topology."""
    
    def __init__(self, config: CassetteExecutionConfig):
        self.config = config
        self.library = CassetteLibrary()
        self.topology = self._load_topology()
        self.results = {}
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def _load_topology(self) -> Dict[str, Any]:
        """Load topology from phase 1 analysis."""
        with open(self.config.topology_file, 'r') as f:
            return json.load(f)
    
    def generate_execution_plan(self) -> Dict[int, List[Probe]]:
        """
        Generate execution plan based on topology and configuration.
        
        Returns:
            Mapping of layer index to probes to execute
        """
        logger.info("Generating cassette execution plan from topology")
        
        # Use library's intelligent scheduling
        base_schedule = self.library.generate_probe_schedule(self.topology)
        
        # Filter by configuration
        filtered_schedule = {}
        for layer, probes in base_schedule.items():
            filtered_probes = []
            
            for probe in probes:
                # Check probe type filter
                if probe.type not in self.config.probe_types:
                    continue
                    
                # Check complexity range
                if not (self.config.complexity_range[0] <= 
                       probe.complexity <= 
                       self.config.complexity_range[1]):
                    continue
                    
                filtered_probes.append(probe)
            
            # Limit probes per layer
            if filtered_probes:
                filtered_schedule[layer] = filtered_probes[:self.config.max_probes_per_layer]
        
        logger.info(f"Execution plan covers {len(filtered_schedule)} layers "
                   f"with {sum(len(p) for p in filtered_schedule.values())} total probes")
        
        return filtered_schedule
    
    def execute_probe(self, probe: Probe, layer: int, model_interface) -> Dict[str, Any]:
        """
        Execute a single probe at a specific layer.
        
        Args:
            probe: The probe to execute
            layer: Layer index for execution
            model_interface: Interface to the model being tested
            
        Returns:
            Execution results including divergence score and timing
        """
        start_time = time.time()
        
        try:
            # Execute probe through model interface
            # This would integrate with true_segment_execution.py
            result = model_interface.execute_at_layer(
                prompt=probe.text,
                layer=layer,
                timeout=self.config.probe_timeout
            )
            
            execution_time = time.time() - start_time
            
            return {
                "probe_id": probe.id,
                "layer": layer,
                "divergence": result.get("divergence", 0.0),
                "execution_time": execution_time,
                "success": True,
                "probe_type": probe.type.value,
                "complexity": probe.complexity,
                "tags": probe.tags
            }
            
        except Exception as e:
            logger.error(f"Failed to execute probe {probe.id} at layer {layer}: {e}")
            return {
                "probe_id": probe.id,
                "layer": layer,
                "divergence": 1.0,  # Max divergence for failure
                "execution_time": time.time() - start_time,
                "success": False,
                "error": str(e),
                "probe_type": probe.type.value,
                "complexity": probe.complexity
            }
    
    def execute_layer_probes(self, layer: int, probes: List[Probe], 
                            model_interface) -> List[Dict[str, Any]]:
        """Execute all probes for a specific layer."""
        logger.info(f"Executing {len(probes)} probes at layer {layer}")
        
        results = []
        for i, probe in enumerate(probes):
            logger.debug(f"  Probe {i+1}/{len(probes)}: {probe.type.value} "
                        f"(complexity {probe.complexity})")
            
            result = self.execute_probe(probe, layer, model_interface)
            results.append(result)
            
            # Log progress
            if result["success"]:
                logger.info(f"    ✓ Divergence: {result['divergence']:.4f} "
                          f"in {result['execution_time']:.2f}s")
            else:
                logger.warning(f"    ✗ Failed: {result.get('error', 'Unknown error')}")
        
        return results
    
    def execute_plan(self, execution_plan: Dict[int, List[Probe]], 
                     model_interface) -> Dict[str, Any]:
        """
        Execute the complete probe plan.
        
        Args:
            execution_plan: Mapping of layer to probes
            model_interface: Interface to the model being tested
            
        Returns:
            Complete execution results
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: CASSETTE-BASED ADVANCED PROBING")
        logger.info("=" * 60)
        
        all_results = {
            "metadata": {
                "topology_file": self.config.topology_file,
                "start_time": time.time(),
                "config": asdict(self.config)
            },
            "layer_results": {},
            "summary": {}
        }
        
        # Execute by layer
        total_layers = len(execution_plan)
        for idx, (layer, probes) in enumerate(sorted(execution_plan.items())):
            logger.info(f"\nLayer {layer} ({idx+1}/{total_layers})")
            logger.info("-" * 40)
            
            layer_results = self.execute_layer_probes(layer, probes, model_interface)
            all_results["layer_results"][layer] = layer_results
            
            # Calculate layer statistics
            successful_results = [r for r in layer_results if r["success"]]
            if successful_results:
                avg_divergence = sum(r["divergence"] for r in successful_results) / len(successful_results)
                logger.info(f"Layer {layer} summary: "
                          f"{len(successful_results)}/{len(layer_results)} successful, "
                          f"avg divergence: {avg_divergence:.4f}")
        
        # Generate summary analysis
        all_results["summary"] = self._generate_summary(all_results["layer_results"])
        all_results["metadata"]["end_time"] = time.time()
        all_results["metadata"]["total_duration"] = (
            all_results["metadata"]["end_time"] - all_results["metadata"]["start_time"]
        )
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _generate_summary(self, layer_results: Dict[int, List[Dict]]) -> Dict[str, Any]:
        """Generate summary analysis of execution results."""
        summary = {
            "total_probes": 0,
            "successful_probes": 0,
            "by_probe_type": {},
            "by_complexity": {},
            "layer_specializations": {},
            "anomalies": []
        }
        
        # Flatten results for analysis
        flat_results = {}
        for layer, results in layer_results.items():
            flat_results[layer] = {r["probe_id"]: r["divergence"] 
                                  for r in results if r["success"]}
            
            # Count statistics
            for result in results:
                summary["total_probes"] += 1
                if result["success"]:
                    summary["successful_probes"] += 1
                    
                    # By probe type
                    probe_type = result["probe_type"]
                    if probe_type not in summary["by_probe_type"]:
                        summary["by_probe_type"][probe_type] = {
                            "count": 0,
                            "avg_divergence": 0,
                            "layers": []
                        }
                    summary["by_probe_type"][probe_type]["count"] += 1
                    summary["by_probe_type"][probe_type]["layers"].append(layer)
                    
                    # By complexity
                    complexity = result["complexity"]
                    if complexity not in summary["by_complexity"]:
                        summary["by_complexity"][complexity] = {
                            "count": 0,
                            "avg_divergence": 0
                        }
                    summary["by_complexity"][complexity]["count"] += 1
        
        # Calculate averages
        for probe_type, stats in summary["by_probe_type"].items():
            type_divergences = []
            for layer, results in layer_results.items():
                for r in results:
                    if r["success"] and r["probe_type"] == probe_type:
                        type_divergences.append(r["divergence"])
            if type_divergences:
                stats["avg_divergence"] = sum(type_divergences) / len(type_divergences)
        
        for complexity, stats in summary["by_complexity"].items():
            comp_divergences = []
            for layer, results in layer_results.items():
                for r in results:
                    if r["success"] and r["complexity"] == complexity:
                        comp_divergences.append(r["divergence"])
            if comp_divergences:
                stats["avg_divergence"] = sum(comp_divergences) / len(comp_divergences)
        
        # Use library's analysis
        execution_plan = self.generate_execution_plan()
        detailed_analysis = self.library.analyze_results(flat_results, execution_plan)
        summary.update(detailed_analysis)
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save execution results to file."""
        output_file = Path(self.config.output_dir) / "cassette_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_file}")
        
        # Also save summary separately
        summary_file = Path(self.config.output_dir) / "cassette_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results["summary"], f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable report from results."""
        report = []
        report.append("=" * 70)
        report.append("CASSETTE EXECUTION REPORT")
        report.append("=" * 70)
        report.append("")
        
        summary = results["summary"]
        
        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 40)
        report.append(f"Total probes executed: {summary['total_probes']}")
        report.append(f"Successful probes: {summary['successful_probes']} "
                     f"({summary['successful_probes']/summary['total_probes']*100:.1f}%)")
        report.append(f"Execution time: {results['metadata']['total_duration']:.2f} seconds")
        report.append("")
        
        # By probe type
        report.append("PERFORMANCE BY PROBE TYPE")
        report.append("-" * 40)
        for probe_type, stats in sorted(summary["by_probe_type"].items()):
            report.append(f"{probe_type:20s}: {stats['count']:3d} probes, "
                         f"avg divergence: {stats['avg_divergence']:.4f}")
        report.append("")
        
        # By complexity
        report.append("PERFORMANCE BY COMPLEXITY")
        report.append("-" * 40)
        for complexity in sorted(summary["by_complexity"].keys()):
            stats = summary["by_complexity"][complexity]
            report.append(f"Complexity {complexity:2d}: {stats['count']:3d} probes, "
                         f"avg divergence: {stats['avg_divergence']:.4f}")
        report.append("")
        
        # Layer specializations
        if "layer_specializations" in summary:
            report.append("LAYER SPECIALIZATIONS")
            report.append("-" * 40)
            for layer, spec in sorted(summary["layer_specializations"].items()):
                report.append(f"Layer {layer:3d}: Best at {spec['best_capability']:15s}, "
                             f"Worst at {spec['worst_capability']:15s} "
                             f"(spread: {spec['score_spread']:.3f})")
        report.append("")
        
        # Anomalies
        if summary.get("anomalies"):
            report.append("ANOMALIES DETECTED")
            report.append("-" * 40)
            for anomaly in summary["anomalies"][:10]:  # Show first 10
                report.append(f"Layer {anomaly['layer']:3d}: Probe {anomaly['probe_id']} "
                             f"expected {anomaly['expected']:.3f}, "
                             f"got {anomaly['actual']:.3f} "
                             f"(Δ={anomaly['deviation']:+.3f})")
        
        return "\n".join(report)


# Integration with main pipeline
def run_cassette_phase(topology_file: str, model_path: str, 
                       output_dir: str = "./cassette_results",
                       probe_types: List[str] = None) -> Dict[str, Any]:
    """
    Run Phase 2 cassette-based analysis.
    
    Args:
        topology_file: Path to topology JSON from Phase 1
        model_path: Path to model for testing
        output_dir: Directory for results
        probe_types: List of probe types to include (None = all)
        
    Returns:
        Execution results
    """
    # Configure execution
    config = CassetteExecutionConfig(
        topology_file=topology_file,
        output_dir=output_dir,
        probe_types=[ProbeType(pt) for pt in probe_types] if probe_types else None
    )
    
    # Initialize executor
    executor = CassetteExecutor(config)
    
    # Generate execution plan
    execution_plan = executor.generate_execution_plan()
    
    # Create model interface (would integrate with existing REV code)
    from ..models.true_segment_execution import REVTrueExecution
    model_interface = REVTrueExecution(
        model_path=model_path,
        device="auto",
        memory_limit=4096
    )
    
    # Execute plan
    results = executor.execute_plan(execution_plan, model_interface)
    
    # Generate and print report
    report = executor.generate_report(results)
    print(report)
    
    # Save report
    report_file = Path(output_dir) / "cassette_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Phase 2 Cassette Analysis")
    parser.add_argument("topology", help="Path to topology JSON from Phase 1")
    parser.add_argument("model", help="Path to model to test")
    parser.add_argument("--output", default="./cassette_results",
                       help="Output directory for results")
    parser.add_argument("--types", nargs="+",
                       help="Probe types to include (default: all)")
    
    args = parser.parse_args()
    
    results = run_cassette_phase(
        topology_file=args.topology,
        model_path=args.model,
        output_dir=args.output,
        probe_types=args.types
    )