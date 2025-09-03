"""
Stopping time analysis for Sequential Probability Ratio Test (SPRT).
Analyzes decision efficiency and stopping time distributions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
from scipy import stats
from scipy.special import gammaln
import logging

logger = logging.getLogger(__name__)


@dataclass
class StoppingTimeResult:
    """Container for stopping time analysis results."""
    
    mean_stopping_time: float
    median_stopping_time: float
    std_stopping_time: float
    min_stopping_time: int
    max_stopping_time: int
    quantiles: Dict[float, float]
    distribution: np.ndarray
    decision: str  # 'accept', 'reject', or 'inconclusive'
    confidence: float
    power: float
    type1_error: float
    type2_error: float


class SPRTAnalyzer:
    """Analyze SPRT stopping times and decision efficiency."""
    
    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.10,
        theta0: float = 0.5,
        theta1: float = 0.7
    ):
        """
        Initialize SPRT analyzer.
        
        Args:
            alpha: Type I error probability
            beta: Type II error probability
            theta0: Null hypothesis parameter
            theta1: Alternative hypothesis parameter
        """
        self.alpha = alpha
        self.beta = beta
        self.theta0 = theta0
        self.theta1 = theta1
        
        # Compute SPRT boundaries
        self.upper_bound = np.log((1 - beta) / alpha)
        self.lower_bound = np.log(beta / (1 - alpha))
        
        # Log likelihood ratio parameters
        self.log_lr_num = np.log(theta1 / (1 - theta1))
        self.log_lr_den = np.log(theta0 / (1 - theta0))
    
    def simulate_stopping_times(
        self,
        true_theta: float,
        num_simulations: int = 10000,
        max_samples: int = 1000
    ) -> StoppingTimeResult:
        """
        Simulate SPRT stopping times under different conditions.
        
        Args:
            true_theta: True parameter value
            num_simulations: Number of simulations to run
            max_samples: Maximum samples before stopping
            
        Returns:
            StoppingTimeResult with analysis
        """
        stopping_times = []
        decisions = []
        log_likelihood_paths = []
        
        for _ in range(num_simulations):
            # Run single SPRT
            samples, decision, log_lr_path = self._run_single_sprt(
                true_theta, max_samples
            )
            
            stopping_times.append(samples)
            decisions.append(decision)
            log_likelihood_paths.append(log_lr_path)
        
        stopping_times = np.array(stopping_times)
        
        # Compute statistics
        accept_rate = sum(d == 'accept' for d in decisions) / num_simulations
        reject_rate = sum(d == 'reject' for d in decisions) / num_simulations
        inconclusive_rate = sum(d == 'inconclusive' for d in decisions) / num_simulations
        
        # Compute error rates
        if true_theta <= self.theta0:
            type1_error = reject_rate  # Rejecting H0 when true
            type2_error = 0  # Can't make type II error when H0 is true
            power = 0
        else:
            type1_error = 0  # Can't make type I error when H1 is true
            type2_error = accept_rate  # Accepting H0 when false
            power = reject_rate
        
        # Compute quantiles
        quantiles = {
            0.1: np.percentile(stopping_times, 10),
            0.25: np.percentile(stopping_times, 25),
            0.5: np.percentile(stopping_times, 50),
            0.75: np.percentile(stopping_times, 75),
            0.9: np.percentile(stopping_times, 90),
            0.95: np.percentile(stopping_times, 95),
            0.99: np.percentile(stopping_times, 99)
        }
        
        return StoppingTimeResult(
            mean_stopping_time=np.mean(stopping_times),
            median_stopping_time=np.median(stopping_times),
            std_stopping_time=np.std(stopping_times),
            min_stopping_time=int(np.min(stopping_times)),
            max_stopping_time=int(np.max(stopping_times)),
            quantiles=quantiles,
            distribution=stopping_times,
            decision='reject' if reject_rate > accept_rate else 'accept',
            confidence=max(accept_rate, reject_rate),
            power=power,
            type1_error=type1_error,
            type2_error=type2_error
        )
    
    def _run_single_sprt(
        self,
        true_theta: float,
        max_samples: int
    ) -> Tuple[int, str, List[float]]:
        """
        Run a single SPRT simulation.
        
        Args:
            true_theta: True parameter value
            max_samples: Maximum samples
            
        Returns:
            Tuple of (stopping_time, decision, log_likelihood_path)
        """
        log_lr_sum = 0
        log_lr_path = []
        
        for n in range(1, max_samples + 1):
            # Generate sample
            x = np.random.binomial(1, true_theta)
            
            # Update log likelihood ratio
            if x == 1:
                log_lr_sum += self.log_lr_num - self.log_lr_den
            else:
                log_lr_sum += np.log((1 - self.theta1) / (1 - self.theta0))
            
            log_lr_path.append(log_lr_sum)
            
            # Check stopping conditions
            if log_lr_sum >= self.upper_bound:
                return n, 'reject', log_lr_path
            elif log_lr_sum <= self.lower_bound:
                return n, 'accept', log_lr_path
        
        return max_samples, 'inconclusive', log_lr_path
    
    def analyze_efficiency(
        self,
        theta_values: List[float] = None,
        num_simulations: int = 1000
    ) -> Dict[float, StoppingTimeResult]:
        """
        Analyze SPRT efficiency across different parameter values.
        
        Args:
            theta_values: Parameter values to test
            num_simulations: Simulations per parameter
            
        Returns:
            Dictionary of results by parameter value
        """
        if theta_values is None:
            theta_values = np.linspace(0.3, 0.9, 13)
        
        results = {}
        
        for theta in theta_values:
            logger.info(f"Analyzing efficiency for theta={theta:.2f}")
            results[theta] = self.simulate_stopping_times(
                theta, 
                num_simulations
            )
        
        return results
    
    def compute_expected_stopping_time(
        self,
        theta: float
    ) -> float:
        """
        Compute theoretical expected stopping time using Wald's approximation.
        
        Args:
            theta: True parameter value
            
        Returns:
            Expected stopping time
        """
        # Kullback-Leibler divergence
        if theta == self.theta0:
            kl_divergence = 0
        else:
            kl_divergence = theta * np.log(theta / self.theta0) + \
                          (1 - theta) * np.log((1 - theta) / (1 - self.theta0))
        
        if kl_divergence == 0:
            # At the boundary, use different approximation
            return 100  # Placeholder value
        
        # Wald's approximation
        if theta <= self.theta0:
            # Under H0
            prob_reject = self.alpha
            expected_log_lr = -kl_divergence
        else:
            # Under H1
            prob_reject = 1 - self.beta
            expected_log_lr = theta * np.log(self.theta1 / self.theta0) + \
                            (1 - theta) * np.log((1 - self.theta1) / (1 - self.theta0))
        
        if expected_log_lr == 0:
            return float('inf')
        
        expected_stopping_time = (
            prob_reject * self.upper_bound + 
            (1 - prob_reject) * self.lower_bound
        ) / expected_log_lr
        
        return abs(expected_stopping_time)
    
    def compare_with_fixed_sample(
        self,
        fixed_n: int = 100,
        theta_values: List[float] = None,
        num_simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Compare SPRT with fixed sample size test.
        
        Args:
            fixed_n: Fixed sample size
            theta_values: Parameter values to test
            num_simulations: Number of simulations
            
        Returns:
            Comparison results
        """
        if theta_values is None:
            theta_values = np.linspace(0.3, 0.9, 13)
        
        comparison = {
            'theta': theta_values,
            'sprt_mean_n': [],
            'fixed_n': fixed_n,
            'efficiency_gain': [],
            'sprt_power': [],
            'fixed_power': []
        }
        
        for theta in theta_values:
            # SPRT analysis
            sprt_result = self.simulate_stopping_times(theta, num_simulations)
            comparison['sprt_mean_n'].append(sprt_result.mean_stopping_time)
            comparison['sprt_power'].append(sprt_result.power)
            
            # Fixed sample test power
            fixed_power = self._compute_fixed_sample_power(theta, fixed_n)
            comparison['fixed_power'].append(fixed_power)
            
            # Efficiency gain
            efficiency = (fixed_n - sprt_result.mean_stopping_time) / fixed_n
            comparison['efficiency_gain'].append(efficiency)
        
        return comparison
    
    def _compute_fixed_sample_power(
        self,
        theta: float,
        n: int
    ) -> float:
        """
        Compute power of fixed sample size test.
        
        Args:
            theta: True parameter
            n: Sample size
            
        Returns:
            Statistical power
        """
        # Critical value for fixed sample test
        critical_value = stats.binom.ppf(1 - self.alpha, n, self.theta0)
        
        # Power under alternative
        power = 1 - stats.binom.cdf(critical_value, n, theta)
        
        return power
    
    def analyze_sequential_boundaries(
        self,
        max_samples: int = 500
    ) -> Dict[str, np.ndarray]:
        """
        Analyze how SPRT boundaries evolve with sample size.
        
        Args:
            max_samples: Maximum number of samples
            
        Returns:
            Dictionary with boundary evolution
        """
        n_values = np.arange(1, max_samples + 1)
        
        # Compute boundaries normalized by sample size
        upper_boundaries = self.upper_bound * np.ones(max_samples)
        lower_boundaries = self.lower_bound * np.ones(max_samples)
        
        # Compute continuation region
        continuation_width = upper_boundaries - lower_boundaries
        
        # Compute approximate boundaries for different confidence levels
        boundaries = {
            'n': n_values,
            'upper': upper_boundaries,
            'lower': lower_boundaries,
            'continuation_width': continuation_width,
            'normalized_upper': upper_boundaries / np.sqrt(n_values),
            'normalized_lower': lower_boundaries / np.sqrt(n_values)
        }
        
        return boundaries
    
    def compute_operating_characteristics(
        self,
        theta_range: Tuple[float, float] = (0.0, 1.0),
        num_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute operating characteristic (OC) curve.
        
        Args:
            theta_range: Range of parameter values
            num_points: Number of points to evaluate
            
        Returns:
            OC curve data
        """
        theta_values = np.linspace(theta_range[0], theta_range[1], num_points)
        prob_accept_h0 = []
        expected_n = []
        
        for theta in theta_values:
            # Probability of accepting H0
            if theta <= self.theta0:
                p_accept = 1 - self.alpha
            elif theta >= self.theta1:
                p_accept = self.beta
            else:
                # Interpolate
                p_accept = self.beta + (1 - self.alpha - self.beta) * \
                          (self.theta1 - theta) / (self.theta1 - self.theta0)
            
            prob_accept_h0.append(p_accept)
            
            # Expected sample size
            expected_n.append(self.compute_expected_stopping_time(theta))
        
        return {
            'theta': theta_values,
            'prob_accept_h0': np.array(prob_accept_h0),
            'expected_n': np.array(expected_n)
        }
    
    def analyze_empirical_bernstein_bounds(
        self,
        samples: np.ndarray,
        delta: float = 0.05
    ) -> Dict[str, Any]:
        """
        Analyze empirical Bernstein bounds for stopping times.
        
        Args:
            samples: Sample data
            delta: Confidence level
            
        Returns:
            Bernstein bound analysis
        """
        n = len(samples)
        sample_mean = np.mean(samples)
        sample_var = np.var(samples, ddof=1)
        
        # Empirical Bernstein bound
        b = 1  # Assume bounded in [0, 1]
        t = np.sqrt(2 * sample_var * np.log(2 / delta) / n) + \
            7 * b * np.log(2 / delta) / (3 * (n - 1))
        
        # Confidence interval
        lower_bound = sample_mean - t
        upper_bound = sample_mean + t
        
        # Hoeffding bound for comparison
        hoeffding_t = np.sqrt(np.log(2 / delta) / (2 * n))
        hoeffding_lower = sample_mean - hoeffding_t
        hoeffding_upper = sample_mean + hoeffding_t
        
        return {
            'sample_mean': sample_mean,
            'sample_variance': sample_var,
            'bernstein_lower': lower_bound,
            'bernstein_upper': upper_bound,
            'bernstein_width': upper_bound - lower_bound,
            'hoeffding_lower': hoeffding_lower,
            'hoeffding_upper': hoeffding_upper,
            'hoeffding_width': hoeffding_upper - hoeffding_lower,
            'improvement_ratio': (hoeffding_upper - hoeffding_lower) / \
                               (upper_bound - lower_bound) if upper_bound > lower_bound else 1
        }
    
    def generate_stopping_time_report(
        self,
        output_path: str = "experiments/results/stopping_time_analysis.json"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive stopping time analysis report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Complete analysis report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'configuration': {
                'alpha': self.alpha,
                'beta': self.beta,
                'theta0': self.theta0,
                'theta1': self.theta1,
                'upper_bound': self.upper_bound,
                'lower_bound': self.lower_bound
            },
            'stopping_times': {},
            'efficiency_comparison': {},
            'operating_characteristics': {},
            'boundaries': {}
        }
        
        # Analyze stopping times for different theta values
        theta_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for theta in theta_values:
            result = self.simulate_stopping_times(theta, num_simulations=5000)
            report['stopping_times'][f'theta_{theta:.1f}'] = {
                'mean': result.mean_stopping_time,
                'median': result.median_stopping_time,
                'std': result.std_stopping_time,
                'min': result.min_stopping_time,
                'max': result.max_stopping_time,
                'quantiles': result.quantiles,
                'power': result.power,
                'type1_error': result.type1_error,
                'type2_error': result.type2_error
            }
        
        # Efficiency comparison
        report['efficiency_comparison'] = self.compare_with_fixed_sample(
            fixed_n=100,
            theta_values=theta_values
        )
        
        # Operating characteristics
        report['operating_characteristics'] = self.compute_operating_characteristics()
        
        # Sequential boundaries
        report['boundaries'] = self.analyze_sequential_boundaries()
        
        # Save report
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            else:
                return obj
        
        report_serializable = convert_arrays(report)
        
        with open(output_path, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        logger.info(f"Saved stopping time analysis to {output_path}")
        
        return report