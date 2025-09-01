#!/usr/bin/env python3
"""
Demonstration of REV behavioral probing diagnostics.
Shows how the enhanced logging makes it immediately obvious when 
the system is using behavioral probing vs falling back to hardcoded sites.
"""

import logging
import time
from datetime import datetime
from src.diagnostics.probe_monitor import get_probe_monitor, ProbeExecutionRecord
from src.models.true_segment_execution import SegmentExecutionConfig

def demonstrate_diagnostics():
    """Demonstrate the comprehensive behavioral probing diagnostics."""
    
    print("🔬" * 20)
    print("🔬 REV BEHAVIORAL PROBING DIAGNOSTICS DEMO")
    print("🔬" * 20)
    
    # Configure detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('behavioral_diagnostics_demo.log')
        ]
    )
    
    # Get probe monitor
    monitor = get_probe_monitor()
    
    print("\n📊 1. PROBE MONITOR INITIALIZATION")
    print("-" * 40)
    print(f"✅ Probe monitor initialized")
    print(f"✅ Log file: {monitor.log_file}")
    print(f"✅ HTML reports enabled: {monitor.enable_html_reports}")
    
    # Simulate probe executions
    print("\n🧪 2. SIMULATING PROBE EXECUTIONS")
    print("-" * 40)
    
    # Simulate successful probe executions
    success_probes = [
        ("Calculate 2+2*3", 0.45),
        ("Explain quantum mechanics", 0.67),
        ("Translate hello to French", 0.32),
        ("Solve logic puzzle", 0.78),
        ("Generate creative story", 0.56)
    ]
    
    for probe_text, divergence in success_probes:
        record = ProbeExecutionRecord(
            timestamp=datetime.now().isoformat(),
            probe_text=probe_text,
            layer_idx=25,
            execution_time_ms=150 + (divergence * 200),  # Realistic timing
            device="cpu",
            dtype="float32",
            success=True,
            divergence_score=divergence,
            tensor_shapes={"hidden_states": "(1, 8, 4096)", "attention": "(1, 32, 8, 8)"},
            memory_usage_mb=2.5 + (divergence * 3.0)
        )
        monitor.log_probe_execution(record)
        time.sleep(0.1)  # Brief pause for realistic timing
    
    # Simulate failed probe executions
    failed_probes = [
        ("Complex mathematical proof", "CUDA out of memory"),
        ("Multi-step reasoning chain", "Tensor shape mismatch: expected [1, 8, 4096], got [1, 10, 4096]"),
    ]
    
    for probe_text, error in failed_probes:
        record = ProbeExecutionRecord(
            timestamp=datetime.now().isoformat(),
            probe_text=probe_text,
            layer_idx=35,
            execution_time_ms=85.0,
            device="cuda",
            dtype="float16",
            success=False,
            error_message=error,
            memory_usage_mb=1.2
        )
        monitor.log_probe_execution(record)
        time.sleep(0.1)
    
    # Simulate fallback scenarios
    print("\n⚠️  3. SIMULATING FALLBACK SCENARIOS")
    print("-" * 40)
    
    fallback_scenarios = [
        ("Insufficient behavioral profiles: 2 < 3", 12, 3, ["math probe", "reasoning probe"]),
        ("Discovery pipeline exception: CUDA device assertion failed", 28, 5, ["complex probe 1", "complex probe 2", "complex probe 3"])
    ]
    
    for reason, layer_idx, probe_count, failed_probes in fallback_scenarios:
        monitor.log_fallback(reason, layer_idx, probe_count, failed_probes)
        time.sleep(0.1)
    
    print("\n📈 4. GENERATING DIAGNOSTIC REPORT")
    print("-" * 40)
    
    # Generate comprehensive report
    report = monitor.generate_report()
    
    print(f"✅ Diagnostic Report Generated:")
    print(f"   Total Executions: {report['summary']['total_executions']}")
    print(f"   Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"   Fallback Count: {report['summary']['fallback_count']}")
    print(f"   🎯 Using Behavioral Probing: {'✅ YES' if report['summary']['using_behavioral_probing'] else '❌ NO (HARDCODED FALLBACK)'}")
    
    if report['summary']['fallback_count'] > 0:
        print(f"\n⚠️  FALLBACK DETECTED - System is NOT using behavioral probing!")
        print(f"   Fallback events: {report['summary']['fallback_count']}")
        print(f"   Recommendations: {len(report['recommendations'])}")
        for rec in report['recommendations'][:3]:
            print(f"   • {rec}")
    
    # Show performance metrics
    print(f"\n⚡ Performance Metrics:")
    print(f"   Average Execution Time: {report['performance']['average_execution_time_ms']:.1f}ms")
    print(f"   Median Execution Time: {report['performance']['median_execution_time_ms']:.1f}ms")
    
    # Show divergence analysis
    print(f"\n🧬 Behavioral Analysis:")
    print(f"   Divergence Samples: {report['divergence_analysis']['total_scores']}")
    print(f"   Mean Divergence: {report['divergence_analysis']['mean_divergence']:.3f}")
    print(f"   Divergence Range: {report['divergence_analysis']['min_divergence']:.3f} - {report['divergence_analysis']['max_divergence']:.3f}")
    
    # Save detailed report
    report_path = monitor.save_report()
    
    print(f"\n💾 5. DETAILED REPORTS SAVED")
    print("-" * 40)
    print(f"✅ JSON Report: {report_path}")
    if monitor.enable_html_reports:
        html_path = report_path.replace('.json', '.html')
        print(f"✅ HTML Report: {html_path}")
    
    print(f"\n📋 6. KEY DIAGNOSTIC INDICATORS")
    print("-" * 40)
    print(f"🔍 How to tell if behavioral probing is working:")
    print(f"   1. ✅ Success rate > 70%")
    print(f"   2. ✅ Fallback count = 0") 
    print(f"   3. ✅ Divergence variance > 0.01")
    print(f"   4. ✅ Multiple probe types executed")
    print(f"   5. ✅ No 'hardcoded_fallback' site types")
    
    current_status = "✅ WORKING" if report['summary']['using_behavioral_probing'] else "❌ NOT WORKING"
    print(f"\n🎯 CURRENT STATUS: {current_status}")
    
    if not report['summary']['using_behavioral_probing']:
        print(f"\n🚨 SYSTEM IS USING HARDCODED SITES!")
        print(f"   Check the detailed reports for specific failure reasons.")
        print(f"   Look for device/memory errors, tensor shape mismatches, or discovery failures.")
    
    print(f"\n{'🔬' * 20}")
    print(f"🔬 DIAGNOSTICS DEMO COMPLETE")
    print(f"{'🔬' * 20}")


if __name__ == "__main__":
    demonstrate_diagnostics()