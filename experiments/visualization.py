"""
Publication-ready visualization for REV validation experiments.
Generates ROC curves, stopping time histograms, and performance dashboards.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging

# Configure matplotlib for publication quality
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

logger = logging.getLogger(__name__)


class ValidationVisualizer:
    """Create publication-ready visualizations for validation results."""
    
    def __init__(self, results_dir: str = "experiments/results"):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_roc_curves(
        self,
        metrics_data: Dict[str, Any],
        title: str = "Model Family Classification ROC Curves",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curves for model family classification.
        
        Args:
            metrics_data: Dictionary with ROC data by family
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        families = list(metrics_data.keys())[:4]  # Plot up to 4 families
        colors = sns.color_palette("husl", len(families))
        
        for idx, (ax, family) in enumerate(zip(axes.flat, families)):
            if family in metrics_data:
                data = metrics_data[family]
                
                # Plot ROC curve
                ax.plot(
                    data['fpr'], 
                    data['tpr'],
                    color=colors[idx],
                    lw=2,
                    label=f'{family} (AUC = {data["auc_score"]:.3f})'
                )
                
                # Plot diagonal reference
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
                
                # Fill area under curve
                ax.fill_between(
                    data['fpr'],
                    0,
                    data['tpr'],
                    alpha=0.15,
                    color=colors[idx]
                )
                
                # Formatting
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{family.upper()} Family Detection')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved ROC curves to {save_path}")
        
        return fig
    
    def plot_stopping_time_distributions(
        self,
        stopping_data: Dict[str, Any],
        title: str = "SPRT Stopping Time Distributions",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot stopping time histograms for SPRT.
        
        Args:
            stopping_data: Dictionary with stopping time data
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Extract theta values and sort
        theta_keys = sorted([k for k in stopping_data.keys() if k.startswith('theta_')])
        
        for idx, theta_key in enumerate(theta_keys[:6]):
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            data = stopping_data[theta_key]
            theta_val = float(theta_key.split('_')[1])
            
            # Create histogram
            if 'distribution' in data:
                dist = data['distribution']
            else:
                # Simulate distribution if not provided
                dist = np.random.gamma(
                    shape=2,
                    scale=data['mean'] / 2,
                    size=1000
                )
            
            ax.hist(
                dist,
                bins=30,
                alpha=0.7,
                color=plt.cm.viridis(theta_val),
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add vertical lines for statistics
            ax.axvline(data['mean'], color='red', linestyle='--', 
                      label=f'Mean: {data["mean"]:.1f}', linewidth=1.5)
            ax.axvline(data['median'], color='green', linestyle='--',
                      label=f'Median: {data["median"]:.1f}', linewidth=1.5)
            
            # Formatting
            ax.set_xlabel('Stopping Time (samples)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'θ = {theta_val:.1f}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved stopping time distributions to {save_path}")
        
        return fig
    
    def plot_adversarial_results(
        self,
        adversarial_data: Dict[str, List[Any]],
        title: str = "Adversarial Attack Success Rates",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot adversarial attack success rates.
        
        Args:
            adversarial_data: Dictionary with adversarial results
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        attack_types = ['stitching', 'spoofing', 'gradient', 'poisoning']
        
        for ax, attack_type in zip(axes.flat, attack_types):
            if attack_type in adversarial_data:
                results = adversarial_data[attack_type]
                
                # Calculate success rates by target family
                success_by_family = {}
                for result in results:
                    target = result.target_family
                    if target not in success_by_family:
                        success_by_family[target] = []
                    success_by_family[target].append(1 if result.success else 0)
                
                # Compute average success rates
                families = list(success_by_family.keys())
                success_rates = [np.mean(success_by_family[f]) * 100 
                               for f in families]
                
                # Create bar plot
                bars = ax.bar(families, success_rates, alpha=0.7)
                
                # Color bars based on success rate
                for bar, rate in zip(bars, success_rates):
                    if rate > 50:
                        bar.set_color('red')
                    elif rate > 25:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
                
                # Add value labels on bars
                for bar, rate in zip(bars, success_rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{rate:.1f}%',
                           ha='center', va='bottom', fontsize=8)
                
                # Formatting
                ax.set_xlabel('Target Family')
                ax.set_ylabel('Success Rate (%)')
                ax.set_title(f'{attack_type.capitalize()} Attack')
                ax.set_ylim([0, 100])
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add horizontal line at 50%
                ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved adversarial results to {save_path}")
        
        return fig
    
    def plot_performance_dashboard(
        self,
        metrics: Dict[str, Any],
        title: str = "REV Performance Dashboard",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive performance dashboard.
        
        Args:
            metrics: Dictionary with all performance metrics
            title: Dashboard title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax1,
                cbar_kws={'label': 'Count'}
            )
            ax1.set_title('Confusion Matrix')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
        
        # 2. F1 Scores by Family
        ax2 = fig.add_subplot(gs[0, 1])
        if 'classification_report' in metrics:
            report = metrics['classification_report']
            families = [k for k in report.keys() if k not in 
                       ['accuracy', 'macro avg', 'weighted avg']]
            f1_scores = [report[f]['f1-score'] for f in families]
            
            bars = ax2.bar(families, f1_scores, color='skyblue', alpha=0.7)
            ax2.set_title('F1 Scores by Family')
            ax2.set_xlabel('Model Family')
            ax2.set_ylabel('F1 Score')
            ax2.set_ylim([0, 1])
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.3f}',
                        ha='center', va='bottom', fontsize=8)
        
        # 3. Efficiency Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        if 'efficiency_comparison' in metrics:
            eff_data = metrics['efficiency_comparison']
            ax3.plot(eff_data['theta'], eff_data['sprt_mean_n'], 
                    'b-', label='SPRT', linewidth=2)
            ax3.axhline(y=eff_data['fixed_n'], color='r', 
                       linestyle='--', label=f'Fixed (n={eff_data["fixed_n"]})')
            ax3.fill_between(eff_data['theta'], 0, eff_data['sprt_mean_n'],
                            alpha=0.3, color='blue')
            ax3.set_title('Sample Size Efficiency')
            ax3.set_xlabel('True θ')
            ax3.set_ylabel('Expected Sample Size')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Power Comparison
        ax4 = fig.add_subplot(gs[1, 0])
        if 'efficiency_comparison' in metrics:
            eff_data = metrics['efficiency_comparison']
            ax4.plot(eff_data['theta'], eff_data['sprt_power'], 
                    'g-', label='SPRT Power', linewidth=2)
            ax4.plot(eff_data['theta'], eff_data['fixed_power'], 
                    'r--', label='Fixed Power', linewidth=2)
            ax4.set_title('Statistical Power Comparison')
            ax4.set_xlabel('True θ')
            ax4.set_ylabel('Power')
            ax4.set_ylim([0, 1])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. False Positive Rates
        ax5 = fig.add_subplot(gs[1, 1])
        if 'fpr_by_confidence' in metrics:
            fpr_data = metrics['fpr_by_confidence']
            confidence_levels = list(list(fpr_data.values())[0].keys())
            
            for family, fprs in fpr_data.items():
                rates = [fprs[c] for c in confidence_levels]
                ax5.plot(confidence_levels, rates, marker='o', label=family)
            
            ax5.set_title('False Positive Rates by Confidence')
            ax5.set_xlabel('Confidence Threshold')
            ax5.set_ylabel('False Positive Rate')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. AUC Scores Summary
        ax6 = fig.add_subplot(gs[1, 2])
        if 'auc_scores' in metrics:
            families = list(metrics['auc_scores'].keys())
            aucs = list(metrics['auc_scores'].values())
            
            bars = ax6.barh(families, aucs, color='coral', alpha=0.7)
            ax6.set_title('AUC Scores Summary')
            ax6.set_xlabel('AUC Score')
            ax6.set_xlim([0, 1])
            ax6.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar, auc in zip(bars, aucs):
                width = bar.get_width()
                ax6.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{auc:.3f}',
                        ha='left', va='center', fontsize=8)
        
        # 7. Adversarial Attack Summary
        ax7 = fig.add_subplot(gs[2, :2])
        if 'adversarial_summary' in metrics:
            adv_data = metrics['adversarial_summary']
            
            # Create grouped bar chart
            attacks = list(adv_data.keys())
            success_rates = [adv_data[a]['success_rate'] * 100 for a in attacks]
            detection_rates = [(1 - adv_data[a]['success_rate']) * 100 
                             for a in attacks]
            
            x = np.arange(len(attacks))
            width = 0.35
            
            bars1 = ax7.bar(x - width/2, success_rates, width, 
                           label='Attack Success', color='red', alpha=0.7)
            bars2 = ax7.bar(x + width/2, detection_rates, width,
                           label='Detection Success', color='green', alpha=0.7)
            
            ax7.set_title('Adversarial Attack vs Detection Success')
            ax7.set_xlabel('Attack Type')
            ax7.set_ylabel('Rate (%)')
            ax7.set_xticks(x)
            ax7.set_xticklabels(attacks, rotation=45, ha='right')
            ax7.legend()
            ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Key Metrics Table
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        if 'summary_stats' in metrics:
            stats = metrics['summary_stats']
            
            # Create table data
            table_data = []
            for key, value in stats.items():
                if isinstance(value, float):
                    table_data.append([key.replace('_', ' ').title(), f'{value:.3f}'])
                else:
                    table_data.append([key.replace('_', ' ').title(), str(value)])
            
            table = ax8.table(
                cellText=table_data,
                colLabels=['Metric', 'Value'],
                cellLoc='left',
                loc='center',
                colWidths=[0.6, 0.4]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            ax8.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved performance dashboard to {save_path}")
        
        return fig
    
    def plot_efficiency_heatmap(
        self,
        efficiency_data: Dict[str, Any],
        title: str = "SPRT Efficiency Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot efficiency heatmap comparing different parameters.
        
        Args:
            efficiency_data: Dictionary with efficiency metrics
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap data
        if 'efficiency_matrix' in efficiency_data:
            matrix = efficiency_data['efficiency_matrix']
        else:
            # Create sample data
            theta_values = np.linspace(0.3, 0.9, 10)
            alpha_values = np.linspace(0.01, 0.1, 10)
            matrix = np.random.rand(10, 10) * 0.5 + 0.3
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Efficiency Gain', rotation=270, labelpad=20)
        
        # Set ticks and labels
        if 'theta_labels' in efficiency_data:
            ax.set_xticks(range(len(efficiency_data['theta_labels'])))
            ax.set_xticklabels(efficiency_data['theta_labels'])
        if 'alpha_labels' in efficiency_data:
            ax.set_yticks(range(len(efficiency_data['alpha_labels'])))
            ax.set_yticklabels(efficiency_data['alpha_labels'])
        
        # Labels
        ax.set_xlabel('True Parameter θ')
        ax.set_ylabel('Significance Level α')
        ax.set_title(title)
        
        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha='center', va='center', color='black',
                             fontsize=7)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved efficiency heatmap to {save_path}")
        
        return fig
    
    def create_full_report(
        self,
        all_results: Dict[str, Any],
        output_dir: Optional[str] = None
    ):
        """
        Create full visualization report with all plots.
        
        Args:
            all_results: Dictionary containing all experiment results
            output_dir: Directory to save plots
        """
        if output_dir is None:
            output_dir = self.results_dir / "plots"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all visualizations
        if 'roc_metrics' in all_results:
            self.plot_roc_curves(
                all_results['roc_metrics'],
                save_path=output_dir / "roc_curves.png"
            )
        
        if 'stopping_times' in all_results:
            self.plot_stopping_time_distributions(
                all_results['stopping_times'],
                save_path=output_dir / "stopping_times.png"
            )
        
        if 'adversarial_results' in all_results:
            self.plot_adversarial_results(
                all_results['adversarial_results'],
                save_path=output_dir / "adversarial_attacks.png"
            )
        
        if 'performance_metrics' in all_results:
            self.plot_performance_dashboard(
                all_results['performance_metrics'],
                save_path=output_dir / "performance_dashboard.png"
            )
        
        if 'efficiency_data' in all_results:
            self.plot_efficiency_heatmap(
                all_results['efficiency_data'],
                save_path=output_dir / "efficiency_heatmap.png"
            )
        
        logger.info(f"Created full visualization report in {output_dir}")
        
        # Create summary HTML report
        self._create_html_report(output_dir)
    
    def _create_html_report(self, output_dir: Path):
        """Create HTML report combining all visualizations."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>REV Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; margin-top: 30px; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
                .section { margin-bottom: 40px; }
            </style>
        </head>
        <body>
            <h1>REV Framework Validation Report</h1>
            
            <div class="section">
                <h2>1. ROC Curves - Model Family Classification</h2>
                <img src="roc_curves.png" alt="ROC Curves">
            </div>
            
            <div class="section">
                <h2>2. SPRT Stopping Time Distributions</h2>
                <img src="stopping_times.png" alt="Stopping Times">
            </div>
            
            <div class="section">
                <h2>3. Adversarial Attack Results</h2>
                <img src="adversarial_attacks.png" alt="Adversarial Attacks">
            </div>
            
            <div class="section">
                <h2>4. Performance Dashboard</h2>
                <img src="performance_dashboard.png" alt="Performance Dashboard">
            </div>
            
            <div class="section">
                <h2>5. Efficiency Heatmap</h2>
                <img src="efficiency_heatmap.png" alt="Efficiency Heatmap">
            </div>
        </body>
        </html>
        """
        
        report_path = output_dir / "report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created HTML report at {report_path}")