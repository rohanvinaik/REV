#!/usr/bin/env python3
"""
Optimization Integration for Response Prediction System

This module integrates response prediction with the REV pipeline to optimize
prompt selection, execution scheduling, and resource allocation.
"""

import os
import time
import heapq
import json
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from .response_predictor import ResponsePredictor, ResponsePrediction, HistoricalResponse, PromptFeatures
from .pattern_recognition import TemplateResponseMapper, PromptClusterAnalyzer, AnomalyDetector


@dataclass
class ExecutionBudget:
    """Resource budget constraints for prompt execution"""
    max_total_cost: float = 100.0
    max_execution_time: float = 3600.0  # seconds
    max_memory_usage: float = 8192.0  # MB
    max_api_calls: int = 1000
    max_prompts: int = 100
    priority_threshold: float = 0.5
    
    # Resource unit costs
    cost_per_second: float = 0.1
    cost_per_mb: float = 0.01
    cost_per_api_call: float = 0.001


@dataclass
class ExecutionPlan:
    """Optimized execution plan for prompt set"""
    selected_prompts: List[str]
    execution_order: List[int]  # Indices into selected_prompts
    batch_groups: List[List[int]]  # Batching for parallel execution
    estimated_total_cost: float
    estimated_total_time: float
    estimated_memory_peak: float
    expected_information_gain: float
    confidence_score: float
    optimization_method: str
    
    # Resource allocation
    resource_allocation: Dict[str, float]
    parallel_groups: List[List[str]]
    
    # Metadata
    plan_timestamp: float = field(default_factory=time.time)
    plan_id: str = field(default_factory=lambda: f"plan_{int(time.time())}")


@dataclass
class PromptExecutionItem:
    """Item in execution queue with predictions and priorities"""
    prompt: str
    template_id: Optional[str]
    prediction: ResponsePrediction
    priority_score: float
    execution_cost: float
    information_value: float
    dependencies: List[str] = field(default_factory=list)
    
    # Execution metadata
    cluster_id: Optional[int] = None
    anomaly_score: float = 0.0
    estimated_duration: float = 0.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering (higher priority first)"""
        return self.priority_score > other.priority_score


class InformationGainEstimator:
    """Estimates information gain from executing prompts"""
    
    def __init__(self):
        self.coverage_weights = {
            'domain_diversity': 0.25,
            'difficulty_spread': 0.20,
            'template_coverage': 0.20,
            'novelty_score': 0.15,
            'uncertainty_reduction': 0.10,
            'complexity_balance': 0.10
        }
        
        # Historical information gain tracking
        self.domain_coverage: Dict[str, float] = defaultdict(float)
        self.difficulty_coverage: Dict[int, float] = defaultdict(float)
        self.template_coverage: Dict[str, float] = defaultdict(float)
        
        # Diminishing returns tracking
        self.domain_saturation: Dict[str, float] = defaultdict(float)
        self.template_saturation: Dict[str, float] = defaultdict(float)
    
    def estimate_information_gain(self, prompt: str, prediction: ResponsePrediction,
                                 features: PromptFeatures, context: Dict[str, Any]) -> float:
        """Estimate information gain from executing this prompt"""
        gain_components = {}
        
        # Domain diversity gain
        domain_indicators = features.domain_indicators
        domain_gain = 0.0
        for domain in domain_indicators:
            current_coverage = self.domain_coverage[domain]
            # Diminishing returns: gain decreases with coverage
            domain_gain += (1.0 - current_coverage) * (1.0 - self.domain_saturation[domain])
        domain_gain = domain_gain / max(len(domain_indicators), 1)
        gain_components['domain_diversity'] = domain_gain
        
        # Difficulty spread gain
        difficulty = features.difficulty_level
        current_difficulty_coverage = self.difficulty_coverage[difficulty]
        difficulty_gain = 1.0 - current_difficulty_coverage
        gain_components['difficulty_spread'] = difficulty_gain
        
        # Template coverage gain
        template_id = features.template_id or "unknown"
        current_template_coverage = self.template_coverage[template_id]
        template_gain = (1.0 - current_template_coverage) * (1.0 - self.template_saturation[template_id])
        gain_components['template_coverage'] = template_gain
        
        # Novelty score (based on prediction uncertainty)
        novelty_gain = prediction.uncertainty_score
        gain_components['novelty_score'] = novelty_gain
        
        # Uncertainty reduction potential
        uncertainty_gain = prediction.uncertainty_score * prediction.predicted_informativeness
        gain_components['uncertainty_reduction'] = uncertainty_gain
        
        # Complexity balance gain
        current_complexity_balance = context.get('complexity_balance', 0.5)
        target_complexity = features.flesch_kincaid_grade / 15.0  # Normalize
        complexity_gap = abs(target_complexity - current_complexity_balance)
        complexity_gain = min(complexity_gap, 0.5)  # Cap at 0.5
        gain_components['complexity_balance'] = complexity_gain
        
        # Calculate weighted information gain
        total_gain = sum(
            gain_components[component] * self.coverage_weights[component]
            for component in gain_components
        )
        
        return min(total_gain, 1.0)  # Cap at 1.0
    
    def update_coverage(self, prompt: str, features: PromptFeatures, 
                       actual_response: str) -> None:
        """Update coverage tracking after prompt execution"""
        # Update domain coverage
        for domain in features.domain_indicators:
            self.domain_coverage[domain] = min(1.0, self.domain_coverage[domain] + 0.1)
            
            # Update saturation (diminishing returns)
            self.domain_saturation[domain] = min(0.9, self.domain_saturation[domain] + 0.05)
        
        # Update difficulty coverage
        difficulty = features.difficulty_level
        self.difficulty_coverage[difficulty] = min(1.0, self.difficulty_coverage[difficulty] + 0.15)
        
        # Update template coverage
        template_id = features.template_id or "unknown"
        self.template_coverage[template_id] = min(1.0, self.template_coverage[template_id] + 0.2)
        self.template_saturation[template_id] = min(0.85, self.template_saturation[template_id] + 0.1)
    
    def get_coverage_summary(self) -> Dict[str, Any]:
        """Get current coverage summary"""
        return {
            'domain_coverage': dict(self.domain_coverage),
            'difficulty_coverage': dict(self.difficulty_coverage),
            'template_coverage': dict(self.template_coverage),
            'total_domains_covered': len(self.domain_coverage),
            'avg_domain_coverage': np.mean(list(self.domain_coverage.values())) if self.domain_coverage else 0.0,
            'coverage_balance': self._calculate_coverage_balance()
        }
    
    def _calculate_coverage_balance(self) -> float:
        """Calculate how balanced the coverage is across different dimensions"""
        # Calculate coefficient of variation for different coverage dimensions
        domain_values = list(self.domain_coverage.values())
        difficulty_values = list(self.difficulty_coverage.values())
        
        balance_scores = []
        
        if len(domain_values) > 1:
            domain_cv = np.std(domain_values) / max(np.mean(domain_values), 1e-6)
            balance_scores.append(1.0 - min(domain_cv, 1.0))
        
        if len(difficulty_values) > 1:
            difficulty_cv = np.std(difficulty_values) / max(np.mean(difficulty_values), 1e-6)
            balance_scores.append(1.0 - min(difficulty_cv, 1.0))
        
        return np.mean(balance_scores) if balance_scores else 1.0


class PromptSelectionOptimizer:
    """Optimizes prompt selection using various optimization algorithms"""
    
    def __init__(self, predictor: ResponsePredictor):
        self.predictor = predictor
        self.information_gain_estimator = InformationGainEstimator()
        
        # Optimization methods
        self.optimization_methods = {
            'greedy': self._greedy_selection,
            'genetic_algorithm': self._genetic_algorithm_selection,
            'simulated_annealing': self._simulated_annealing_selection,
            'multi_objective': self._multi_objective_optimization,
            'dynamic_programming': self._dynamic_programming_selection
        }
        
        # Caching for optimization speedup
        self.prediction_cache: Dict[str, ResponsePrediction] = {}
        self.information_gain_cache: Dict[str, float] = {}
        
    def optimize_prompt_selection(self, candidate_prompts: List[str],
                                 budget: ExecutionBudget,
                                 method: str = 'multi_objective',
                                 context: Dict[str, Any] = None) -> ExecutionPlan:
        """Optimize prompt selection given budget constraints"""
        context = context or {}
        
        if method not in self.optimization_methods:
            method = 'greedy'  # Fallback
        
        # Generate predictions for all candidates
        prompt_items = []
        for prompt in candidate_prompts:
            prediction = self._get_prediction(prompt)
            features = self.predictor.feature_extractor.extract_features(prompt)
            
            # Calculate costs and benefits
            execution_cost = self._calculate_execution_cost(prediction, budget)
            information_gain = self.information_gain_estimator.estimate_information_gain(
                prompt, prediction, features, context
            )
            
            priority_score = information_gain / max(execution_cost, 0.01)
            
            item = PromptExecutionItem(
                prompt=prompt,
                template_id=features.template_id,
                prediction=prediction,
                priority_score=priority_score,
                execution_cost=execution_cost,
                information_value=information_gain,
                estimated_duration=prediction.estimated_computation_time,
                resource_requirements={
                    'memory': prediction.estimated_memory_usage,
                    'computation': prediction.estimated_computation_time
                }
            )
            prompt_items.append(item)
        
        # Apply optimization method
        optimization_func = self.optimization_methods[method]
        selected_items = optimization_func(prompt_items, budget, context)
        
        # Create execution plan
        execution_plan = self._create_execution_plan(selected_items, budget, method, context)
        
        return execution_plan
    
    def _greedy_selection(self, prompt_items: List[PromptExecutionItem],
                         budget: ExecutionBudget, context: Dict[str, Any]) -> List[PromptExecutionItem]:
        """Greedy selection based on priority score"""
        # Sort by priority score
        sorted_items = sorted(prompt_items, key=lambda x: x.priority_score, reverse=True)
        
        selected = []
        total_cost = 0.0
        total_time = 0.0
        total_memory = 0.0
        
        for item in sorted_items:
            # Check budget constraints
            new_cost = total_cost + item.execution_cost
            new_time = total_time + item.estimated_duration
            new_memory = max(total_memory, item.resource_requirements.get('memory', 0))
            
            if (new_cost <= budget.max_total_cost and 
                new_time <= budget.max_execution_time and
                new_memory <= budget.max_memory_usage and
                len(selected) < budget.max_prompts):
                
                selected.append(item)
                total_cost = new_cost
                total_time = new_time
                total_memory = new_memory
            
            if len(selected) >= budget.max_prompts:
                break
        
        return selected
    
    def _genetic_algorithm_selection(self, prompt_items: List[PromptExecutionItem],
                                    budget: ExecutionBudget, context: Dict[str, Any]) -> List[PromptExecutionItem]:
        """Genetic algorithm optimization for prompt selection"""
        population_size = min(50, len(prompt_items))
        generations = 20
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        # Initialize population (binary vectors indicating selection)
        population = []
        for _ in range(population_size):
            individual = np.random.random(len(prompt_items)) < 0.3  # 30% selection probability
            population.append(individual)
        
        def fitness_function(individual):
            selected_items = [prompt_items[i] for i, selected in enumerate(individual) if selected]
            
            if not selected_items:
                return 0.0
            
            total_cost = sum(item.execution_cost for item in selected_items)
            total_time = sum(item.estimated_duration for item in selected_items)
            total_memory = max((item.resource_requirements.get('memory', 0) for item in selected_items), default=0)
            total_information = sum(item.information_value for item in selected_items)
            
            # Penalty for violating constraints
            penalty = 0.0
            if total_cost > budget.max_total_cost:
                penalty += (total_cost - budget.max_total_cost) / budget.max_total_cost
            if total_time > budget.max_execution_time:
                penalty += (total_time - budget.max_execution_time) / budget.max_execution_time
            if total_memory > budget.max_memory_usage:
                penalty += (total_memory - budget.max_memory_usage) / budget.max_memory_usage
            if len(selected_items) > budget.max_prompts:
                penalty += (len(selected_items) - budget.max_prompts) / budget.max_prompts
            
            return total_information / (1 + penalty)
        
        # Evolution
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [fitness_function(individual) for individual in population]
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                tournament_size = 3
                tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                new_population.append(population[winner_idx].copy())
            
            population = new_population
            
            # Crossover
            for i in range(0, population_size - 1, 2):
                if np.random.random() < crossover_rate:
                    crossover_point = np.random.randint(1, len(prompt_items))
                    child1 = np.concatenate([population[i][:crossover_point], population[i+1][crossover_point:]])
                    child2 = np.concatenate([population[i+1][:crossover_point], population[i][crossover_point:]])
                    population[i] = child1
                    population[i+1] = child2
            
            # Mutation
            for individual in population:
                for j in range(len(individual)):
                    if np.random.random() < mutation_rate:
                        individual[j] = not individual[j]
        
        # Return best individual
        final_fitness = [fitness_function(individual) for individual in population]
        best_individual = population[np.argmax(final_fitness)]
        
        return [prompt_items[i] for i, selected in enumerate(best_individual) if selected]
    
    def _simulated_annealing_selection(self, prompt_items: List[PromptExecutionItem],
                                     budget: ExecutionBudget, context: Dict[str, Any]) -> List[PromptExecutionItem]:
        """Simulated annealing optimization"""
        def objective_function(selection_vector):
            selected_items = [prompt_items[i] for i, selected in enumerate(selection_vector) if selected]
            
            if not selected_items:
                return -1000.0  # Heavy penalty for empty selection
            
            total_cost = sum(item.execution_cost for item in selected_items)
            total_time = sum(item.estimated_duration for item in selected_items)
            total_memory = max((item.resource_requirements.get('memory', 0) for item in selected_items), default=0)
            total_information = sum(item.information_value for item in selected_items)
            
            # Hard constraints
            if (total_cost > budget.max_total_cost or 
                total_time > budget.max_execution_time or
                total_memory > budget.max_memory_usage or
                len(selected_items) > budget.max_prompts):
                return -1000.0
            
            return total_information
        
        # Initialize with greedy solution
        current_solution = np.zeros(len(prompt_items), dtype=bool)
        greedy_items = self._greedy_selection(prompt_items, budget, context)
        for item in greedy_items:
            idx = prompt_items.index(item)
            current_solution[idx] = True
        
        current_score = objective_function(current_solution)
        best_solution = current_solution.copy()
        best_score = current_score
        
        # Simulated annealing parameters
        initial_temp = 10.0
        cooling_rate = 0.95
        min_temp = 0.01
        
        temperature = initial_temp
        while temperature > min_temp:
            # Generate neighbor solution
            neighbor_solution = current_solution.copy()
            
            # Random flip
            flip_idx = np.random.randint(len(prompt_items))
            neighbor_solution[flip_idx] = not neighbor_solution[flip_idx]
            
            neighbor_score = objective_function(neighbor_solution)
            
            # Acceptance probability
            if neighbor_score > current_score:
                accept = True
            else:
                delta = neighbor_score - current_score
                accept_prob = np.exp(delta / temperature)
                accept = np.random.random() < accept_prob
            
            if accept:
                current_solution = neighbor_solution
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_solution = current_solution.copy()
                    best_score = current_score
            
            temperature *= cooling_rate
        
        return [prompt_items[i] for i, selected in enumerate(best_solution) if selected]
    
    def _multi_objective_optimization(self, prompt_items: List[PromptExecutionItem],
                                    budget: ExecutionBudget, context: Dict[str, Any]) -> List[PromptExecutionItem]:
        """Multi-objective optimization balancing cost and information gain"""
        # Use NSGA-II style approach simplified for binary selection
        
        def evaluate_objectives(selection_vector):
            selected_items = [prompt_items[i] for i, selected in enumerate(selection_vector) if selected]
            
            if not selected_items:
                return [0.0, 1000.0]  # No information, high cost penalty
            
            total_information = sum(item.information_value for item in selected_items)
            total_cost = sum(item.execution_cost for item in selected_items)
            
            # Check constraints
            total_time = sum(item.estimated_duration for item in selected_items)
            total_memory = max((item.resource_requirements.get('memory', 0) for item in selected_items), default=0)
            
            # Penalty for constraint violation
            constraint_penalty = 0.0
            if total_cost > budget.max_total_cost:
                constraint_penalty += total_cost - budget.max_total_cost
            if total_time > budget.max_execution_time:
                constraint_penalty += (total_time - budget.max_execution_time) * 0.1
            if total_memory > budget.max_memory_usage:
                constraint_penalty += (total_memory - budget.max_memory_usage) * 0.01
            if len(selected_items) > budget.max_prompts:
                constraint_penalty += (len(selected_items) - budget.max_prompts) * 10
            
            # Objectives: maximize information, minimize cost
            return [total_information, total_cost + constraint_penalty]
        
        # Generate multiple solutions with different trade-offs
        solutions = []
        
        # Greedy solution (high information)
        greedy_items = self._greedy_selection(prompt_items, budget, context)
        greedy_vector = np.zeros(len(prompt_items), dtype=bool)
        for item in greedy_items:
            idx = prompt_items.index(item)
            greedy_vector[idx] = True
        solutions.append(greedy_vector)
        
        # Cost-optimized solution (low cost)
        cost_sorted = sorted(prompt_items, key=lambda x: x.execution_cost)
        cost_vector = np.zeros(len(prompt_items), dtype=bool)
        total_cost = 0
        for item in cost_sorted:
            idx = prompt_items.index(item)
            if total_cost + item.execution_cost <= budget.max_total_cost:
                cost_vector[idx] = True
                total_cost += item.execution_cost
        solutions.append(cost_vector)
        
        # Random solutions for diversity
        for _ in range(10):
            random_vector = np.random.random(len(prompt_items)) < 0.4
            solutions.append(random_vector)
        
        # Evaluate all solutions
        objectives = [evaluate_objectives(sol) for sol in solutions]
        
        # Find Pareto optimal solutions
        pareto_solutions = []
        for i, (sol, obj) in enumerate(zip(solutions, objectives)):
            is_dominated = False
            for j, (_, other_obj) in enumerate(zip(solutions, objectives)):
                if i != j:
                    # Check if solution i is dominated by solution j
                    if (other_obj[0] >= obj[0] and other_obj[1] <= obj[1] and
                        (other_obj[0] > obj[0] or other_obj[1] < obj[1])):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_solutions.append((sol, obj))
        
        # Select solution with best trade-off (e.g., highest information/cost ratio)
        if pareto_solutions:
            best_ratio = -float('inf')
            best_solution = None
            
            for sol, obj in pareto_solutions:
                if obj[1] > 0:  # Avoid division by zero
                    ratio = obj[0] / obj[1]
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_solution = sol
            
            if best_solution is not None:
                return [prompt_items[i] for i, selected in enumerate(best_solution) if selected]
        
        # Fallback to greedy
        return self._greedy_selection(prompt_items, budget, context)
    
    def _dynamic_programming_selection(self, prompt_items: List[PromptExecutionItem],
                                     budget: ExecutionBudget, context: Dict[str, Any]) -> List[PromptExecutionItem]:
        """Dynamic programming optimization (knapsack-style)"""
        # Simplify to 0-1 knapsack with cost constraint
        n_items = len(prompt_items)
        max_cost = int(budget.max_total_cost * 100)  # Scale for integer DP
        
        # DP table: dp[i][c] = maximum information value using first i items with cost <= c
        dp = [[0.0 for _ in range(max_cost + 1)] for _ in range(n_items + 1)]
        
        # Fill DP table
        for i in range(1, n_items + 1):
            item = prompt_items[i - 1]
            item_cost = int(item.execution_cost * 100)
            item_value = item.information_value
            
            for c in range(max_cost + 1):
                # Don't take item i
                dp[i][c] = dp[i-1][c]
                
                # Take item i if possible
                if c >= item_cost:
                    dp[i][c] = max(dp[i][c], dp[i-1][c - item_cost] + item_value)
        
        # Backtrack to find selected items
        selected_indices = []
        i, c = n_items, max_cost
        
        while i > 0 and c > 0:
            if dp[i][c] != dp[i-1][c]:
                # Item i-1 was selected
                selected_indices.append(i - 1)
                item_cost = int(prompt_items[i-1].execution_cost * 100)
                c -= item_cost
            i -= 1
        
        selected_items = [prompt_items[idx] for idx in selected_indices]
        
        # Additional constraint checking
        total_time = sum(item.estimated_duration for item in selected_items)
        total_memory = max((item.resource_requirements.get('memory', 0) for item in selected_items), default=0)
        
        # Remove items if other constraints are violated
        if (total_time > budget.max_execution_time or 
            total_memory > budget.max_memory_usage or
            len(selected_items) > budget.max_prompts):
            # Fallback to greedy with constraints
            return self._greedy_selection(prompt_items, budget, context)
        
        return selected_items
    
    def _get_prediction(self, prompt: str) -> ResponsePrediction:
        """Get prediction with caching"""
        if prompt not in self.prediction_cache:
            self.prediction_cache[prompt] = self.predictor.predict_response(prompt)
        return self.prediction_cache[prompt]
    
    def _calculate_execution_cost(self, prediction: ResponsePrediction, 
                                 budget: ExecutionBudget) -> float:
        """Calculate total execution cost for a prediction"""
        return (
            prediction.estimated_computation_time * budget.cost_per_second +
            prediction.estimated_memory_usage * budget.cost_per_mb +
            budget.cost_per_api_call
        )
    
    def _create_execution_plan(self, selected_items: List[PromptExecutionItem],
                             budget: ExecutionBudget, method: str,
                             context: Dict[str, Any]) -> ExecutionPlan:
        """Create detailed execution plan from selected items"""
        if not selected_items:
            return ExecutionPlan(
                selected_prompts=[],
                execution_order=[],
                batch_groups=[],
                estimated_total_cost=0.0,
                estimated_total_time=0.0,
                estimated_memory_peak=0.0,
                expected_information_gain=0.0,
                confidence_score=0.0,
                optimization_method=method,
                resource_allocation={},
                parallel_groups=[]
            )
        
        # Calculate totals
        total_cost = sum(item.execution_cost for item in selected_items)
        total_time = sum(item.estimated_duration for item in selected_items)
        peak_memory = max(item.resource_requirements.get('memory', 0) for item in selected_items)
        total_information = sum(item.information_value for item in selected_items)
        avg_confidence = np.mean([item.prediction.prediction_confidence for item in selected_items])
        
        # Create execution order (by priority, then by dependencies)
        execution_order = self._optimize_execution_order(selected_items)
        
        # Create batch groups for parallel execution
        batch_groups = self._create_batch_groups(selected_items, budget)
        
        # Resource allocation
        resource_allocation = {
            'cpu_allocation': total_time,
            'memory_allocation': peak_memory,
            'cost_allocation': total_cost,
            'parallel_workers': min(len(batch_groups), 4)
        }
        
        # Create parallel groups
        parallel_groups = [[item.prompt for item in group] for group in batch_groups]
        
        return ExecutionPlan(
            selected_prompts=[item.prompt for item in selected_items],
            execution_order=execution_order,
            batch_groups=[[selected_items.index(item) for item in group] for group in batch_groups],
            estimated_total_cost=total_cost,
            estimated_total_time=total_time,
            estimated_memory_peak=peak_memory,
            expected_information_gain=total_information,
            confidence_score=avg_confidence,
            optimization_method=method,
            resource_allocation=resource_allocation,
            parallel_groups=parallel_groups
        )
    
    def _optimize_execution_order(self, items: List[PromptExecutionItem]) -> List[int]:
        """Optimize execution order considering dependencies and priorities"""
        # Simple topological sort with priority weighting
        ordered_indices = []
        remaining_items = list(enumerate(items))
        
        while remaining_items:
            # Find items with no unmet dependencies
            available_items = []
            for i, (original_idx, item) in enumerate(remaining_items):
                if not item.dependencies:  # No dependencies
                    available_items.append((i, original_idx, item))
                # In practice, would check if dependencies are satisfied
            
            if not available_items:
                # Break circular dependencies by selecting highest priority
                i, original_idx, item = max(remaining_items, key=lambda x: x[1].priority_score)
                available_items = [(remaining_items.index((original_idx, item)), original_idx, item)]
            
            # Select highest priority available item
            selected_i, selected_original_idx, selected_item = max(available_items, 
                                                                  key=lambda x: x[2].priority_score)
            
            ordered_indices.append(selected_original_idx)
            remaining_items.pop(selected_i)
        
        return ordered_indices
    
    def _create_batch_groups(self, items: List[PromptExecutionItem],
                           budget: ExecutionBudget) -> List[List[PromptExecutionItem]]:
        """Create batches for parallel execution"""
        # Group items by similar resource requirements
        batch_groups = []
        remaining_items = items.copy()
        
        while remaining_items:
            current_batch = []
            batch_memory = 0.0
            batch_time = 0.0
            
            # Greedy batching by resource compatibility
            items_to_remove = []
            for item in remaining_items:
                item_memory = item.resource_requirements.get('memory', 0)
                item_time = item.estimated_duration
                
                # Check if item fits in current batch
                if (batch_memory + item_memory <= budget.max_memory_usage and
                    len(current_batch) < 10):  # Max 10 items per batch
                    current_batch.append(item)
                    batch_memory += item_memory
                    batch_time = max(batch_time, item_time)  # Parallel execution
                    items_to_remove.append(item)
            
            # Remove batched items
            for item in items_to_remove:
                remaining_items.remove(item)
            
            if current_batch:
                batch_groups.append(current_batch)
            else:
                # Force add one item to avoid infinite loop
                if remaining_items:
                    batch_groups.append([remaining_items.pop(0)])
        
        return batch_groups


class REVPipelineIntegration:
    """Integration with REV pipeline for optimized prompt execution"""
    
    def __init__(self, predictor: ResponsePredictor, optimizer: PromptSelectionOptimizer):
        self.predictor = predictor
        self.optimizer = optimizer
        self.execution_monitor = ExecutionMonitor()
        
        # Pipeline state
        self.current_plan: Optional[ExecutionPlan] = None
        self.execution_results: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        
    def optimize_and_execute(self, candidate_prompts: List[str],
                           budget: ExecutionBudget,
                           rev_pipeline_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize prompt selection and execute through REV pipeline"""
        rev_pipeline_params = rev_pipeline_params or {}
        
        # Step 1: Optimize prompt selection
        optimization_context = {
            'coverage_balance': 0.5,
            'diversity_requirement': 0.7,
            'quality_threshold': 0.6
        }
        
        execution_plan = self.optimizer.optimize_prompt_selection(
            candidate_prompts, budget, method='multi_objective', context=optimization_context
        )
        
        self.current_plan = execution_plan
        
        # Step 2: Execute optimized plan
        execution_results = self._execute_plan(execution_plan, rev_pipeline_params)
        
        # Step 3: Update predictor with results
        self._update_predictor_with_results(execution_results)
        
        # Step 4: Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(execution_plan, execution_results)
        
        return {
            'execution_plan': execution_plan,
            'execution_results': execution_results,
            'performance_metrics': performance_metrics,
            'optimization_summary': self._create_optimization_summary(execution_plan, execution_results)
        }
    
    def _execute_plan(self, plan: ExecutionPlan, rev_pipeline_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the optimized plan through REV pipeline"""
        results = {
            'successful_executions': [],
            'failed_executions': [],
            'execution_times': [],
            'actual_costs': [],
            'response_qualities': []
        }
        
        # Monitor execution
        start_time = time.time()
        
        try:
            # Execute in parallel batches
            for batch_idx, batch_prompts in enumerate(plan.parallel_groups):
                batch_results = self._execute_batch(batch_prompts, rev_pipeline_params)
                
                for prompt, result in batch_results.items():
                    if result['success']:
                        results['successful_executions'].append({
                            'prompt': prompt,
                            'response': result['response'],
                            'execution_time': result['execution_time'],
                            'memory_usage': result['memory_usage']
                        })
                    else:
                        results['failed_executions'].append({
                            'prompt': prompt,
                            'error': result['error']
                        })
                    
                    results['execution_times'].append(result['execution_time'])
                    results['actual_costs'].append(result['cost'])
                    
                    if 'quality_score' in result:
                        results['response_qualities'].append(result['quality_score'])
        
        except Exception as e:
            results['execution_error'] = str(e)
        
        results['total_execution_time'] = time.time() - start_time
        
        return results
    
    def _execute_batch(self, prompts: List[str], rev_pipeline_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a batch of prompts"""
        batch_results = {}
        
        # Simulate REV pipeline execution
        # In practice, this would call the actual REV pipeline
        for prompt in prompts:
            start_time = time.time()
            
            try:
                # Simulate execution
                time.sleep(0.1)  # Simulate processing time
                
                # Mock response
                response = f"Response to: {prompt[:50]}..."
                execution_time = time.time() - start_time
                
                batch_results[prompt] = {
                    'success': True,
                    'response': response,
                    'execution_time': execution_time,
                    'memory_usage': 50.0,  # Mock memory usage
                    'cost': execution_time * 0.1,  # Mock cost
                    'quality_score': 0.8  # Mock quality
                }
                
            except Exception as e:
                batch_results[prompt] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': time.time() - start_time,
                    'cost': 0.0
                }
        
        return batch_results
    
    def _update_predictor_with_results(self, execution_results: Dict[str, Any]) -> None:
        """Update predictor with actual execution results"""
        for execution in execution_results['successful_executions']:
            prompt = execution['prompt']
            response = execution['response']
            actual_time = execution['execution_time']
            actual_memory = execution['memory_usage']
            
            # Extract features
            features = self.predictor.feature_extractor.extract_features(prompt)
            
            # Create historical response
            historical_response = HistoricalResponse(
                prompt=prompt,
                response=response,
                features=features,
                actual_length=len(response),
                actual_word_count=len(response.split()),
                actual_tokens=int(len(response.split()) * 1.3),
                execution_time=actual_time,
                memory_usage=actual_memory,
                coherence_score=0.8,  # Would calculate from response
                informativeness_score=0.7,  # Would calculate from response
                diversity_score=0.6,  # Would calculate from response
                model_id="test_model",
                timestamp=time.time(),
                template_id=features.template_id
            )
            
            # Add to predictor
            self.predictor.add_historical_response(historical_response)
            
            # Update information gain estimator
            self.optimizer.information_gain_estimator.update_coverage(prompt, features, response)
    
    def _calculate_performance_metrics(self, plan: ExecutionPlan, 
                                     results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics for optimization evaluation"""
        metrics = {}
        
        # Success rate
        total_executions = len(results['successful_executions']) + len(results['failed_executions'])
        if total_executions > 0:
            metrics['success_rate'] = len(results['successful_executions']) / total_executions
        else:
            metrics['success_rate'] = 0.0
        
        # Cost efficiency
        actual_total_cost = sum(results['actual_costs'])
        if plan.estimated_total_cost > 0:
            metrics['cost_accuracy'] = 1.0 - abs(actual_total_cost - plan.estimated_total_cost) / plan.estimated_total_cost
        else:
            metrics['cost_accuracy'] = 0.0
        
        # Time efficiency
        actual_total_time = results['total_execution_time']
        if plan.estimated_total_time > 0:
            metrics['time_accuracy'] = 1.0 - abs(actual_total_time - plan.estimated_total_time) / plan.estimated_total_time
        else:
            metrics['time_accuracy'] = 0.0
        
        # Information gain
        if results['response_qualities']:
            metrics['avg_response_quality'] = np.mean(results['response_qualities'])
        else:
            metrics['avg_response_quality'] = 0.0
        
        # Resource utilization
        if results['execution_times']:
            metrics['avg_execution_time'] = np.mean(results['execution_times'])
            metrics['execution_time_std'] = np.std(results['execution_times'])
        
        return metrics
    
    def _create_optimization_summary(self, plan: ExecutionPlan, 
                                   results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of optimization results"""
        return {
            'optimization_method': plan.optimization_method,
            'selected_prompts_count': len(plan.selected_prompts),
            'planned_cost': plan.estimated_total_cost,
            'actual_cost': sum(results['actual_costs']),
            'planned_time': plan.estimated_total_time,
            'actual_time': results['total_execution_time'],
            'expected_information_gain': plan.expected_information_gain,
            'success_rate': len(results['successful_executions']) / max(len(plan.selected_prompts), 1),
            'resource_efficiency': self._calculate_resource_efficiency(plan, results),
            'recommendation': self._generate_optimization_recommendation(plan, results)
        }
    
    def _calculate_resource_efficiency(self, plan: ExecutionPlan, 
                                     results: Dict[str, Any]) -> float:
        """Calculate overall resource efficiency"""
        cost_efficiency = 1.0 - abs(sum(results['actual_costs']) - plan.estimated_total_cost) / max(plan.estimated_total_cost, 1.0)
        time_efficiency = 1.0 - abs(results['total_execution_time'] - plan.estimated_total_time) / max(plan.estimated_total_time, 1.0)
        success_rate = len(results['successful_executions']) / max(len(plan.selected_prompts), 1)
        
        return (cost_efficiency + time_efficiency + success_rate) / 3.0
    
    def _generate_optimization_recommendation(self, plan: ExecutionPlan, 
                                            results: Dict[str, Any]) -> str:
        """Generate recommendation for future optimizations"""
        success_rate = len(results['successful_executions']) / max(len(plan.selected_prompts), 1)
        resource_efficiency = self._calculate_resource_efficiency(plan, results)
        
        if success_rate > 0.9 and resource_efficiency > 0.8:
            return "Optimization performed well. Consider using similar parameters for future runs."
        elif success_rate < 0.7:
            return "Low success rate. Consider more conservative prompt selection or better error handling."
        elif resource_efficiency < 0.6:
            return "Poor resource efficiency. Review cost estimation and consider different optimization method."
        else:
            return "Mixed results. Consider tuning optimization parameters based on specific performance issues."


class ExecutionMonitor:
    """Monitors execution performance and provides real-time feedback"""
    
    def __init__(self):
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
    def start_execution(self, execution_id: str, plan: ExecutionPlan) -> None:
        """Start monitoring an execution"""
        self.active_executions[execution_id] = {
            'plan': plan,
            'start_time': time.time(),
            'completed_prompts': 0,
            'failed_prompts': 0,
            'current_cost': 0.0
        }
    
    def update_execution(self, execution_id: str, prompt_result: Dict[str, Any]) -> None:
        """Update execution progress"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            
            if prompt_result['success']:
                execution['completed_prompts'] += 1
            else:
                execution['failed_prompts'] += 1
            
            execution['current_cost'] += prompt_result.get('cost', 0.0)
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get current execution status"""
        if execution_id not in self.active_executions:
            return {'error': 'Execution not found'}
        
        execution = self.active_executions[execution_id]
        plan = execution['plan']
        
        total_prompts = len(plan.selected_prompts)
        completed = execution['completed_prompts']
        failed = execution['failed_prompts']
        progress = (completed + failed) / max(total_prompts, 1)
        
        elapsed_time = time.time() - execution['start_time']
        estimated_remaining_time = (elapsed_time / max(progress, 0.01)) - elapsed_time
        
        return {
            'execution_id': execution_id,
            'progress': progress,
            'completed_prompts': completed,
            'failed_prompts': failed,
            'total_prompts': total_prompts,
            'elapsed_time': elapsed_time,
            'estimated_remaining_time': estimated_remaining_time,
            'current_cost': execution['current_cost'],
            'estimated_total_cost': plan.estimated_total_cost,
            'success_rate': completed / max(completed + failed, 1)
        }


if __name__ == "__main__":
    # Example usage
    from .response_predictor import ResponsePredictor
    
    # Initialize components
    predictor = ResponsePredictor()
    optimizer = PromptSelectionOptimizer(predictor)
    integration = REVPipelineIntegration(predictor, optimizer)
    
    # Example prompts
    candidate_prompts = [
        "What is machine learning?",
        "Explain neural networks in detail.",
        "How does blockchain technology work?",
        "Compare supervised and unsupervised learning.",
        "Describe the principles of deep learning."
    ]
    
    # Create budget
    budget = ExecutionBudget(
        max_total_cost=50.0,
        max_execution_time=300.0,
        max_prompts=3
    )
    
    # Optimize and execute
    results = integration.optimize_and_execute(candidate_prompts, budget)
    
    print(f"Selected {len(results['execution_plan'].selected_prompts)} prompts")
    print(f"Success rate: {results['performance_metrics']['success_rate']:.2f}")
    print(f"Cost efficiency: {results['performance_metrics']['cost_accuracy']:.2f}")