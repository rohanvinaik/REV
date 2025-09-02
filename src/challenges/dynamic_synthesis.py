#!/usr/bin/env python3
"""
Dynamic Prompt Synthesis System for REV Framework

Real-time generation of novel prompts through template combination,
context-aware adaptation, and domain-specific synthesis.
"""

import random
import re
import itertools
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model for NLP analysis
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


class TemplateType(Enum):
    """Types of prompt templates."""
    QUESTION = "question"
    INSTRUCTION = "instruction"
    COMPLETION = "completion"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    SCENARIO = "scenario"
    COMPARATIVE = "comparative"
    HYPOTHETICAL = "hypothetical"


class DomainType(Enum):
    """Domain specializations."""
    SCIENTIFIC = "scientific"
    MATHEMATICAL = "mathematical"
    PHILOSOPHICAL = "philosophical"
    TECHNICAL = "technical"
    CREATIVE_WRITING = "creative_writing"
    BUSINESS = "business"
    MEDICAL = "medical"
    LEGAL = "legal"
    EDUCATIONAL = "educational"
    SOCIAL = "social"


@dataclass
class PromptTemplate:
    """Individual prompt template."""
    id: str
    type: TemplateType
    pattern: str
    variables: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    complexity: float = 1.0
    domains: List[DomainType] = field(default_factory=list)
    
    def fill(self, values: Dict[str, str]) -> str:
        """Fill template with values."""
        result = self.pattern
        for var in self.variables:
            if var in values:
                result = result.replace(f"{{{var}}}", values[var])
        return result


@dataclass
class GenerationContext:
    """Context for prompt generation."""
    previous_prompts: List[str] = field(default_factory=list)
    model_responses: List[str] = field(default_factory=list)
    performance_scores: List[float] = field(default_factory=list)
    current_difficulty: float = 1.0
    domain_focus: Optional[DomainType] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    def update(self, prompt: str, response: str = "", score: float = 0.5):
        """Update context with new interaction."""
        self.previous_prompts.append(prompt)
        if response:
            self.model_responses.append(response)
        if score is not None:
            self.performance_scores.append(score)
            # Adaptive difficulty adjustment
            if len(self.performance_scores) >= 3:
                recent_avg = np.mean(self.performance_scores[-3:])
                if recent_avg > 0.8:
                    self.current_difficulty = min(5.0, self.current_difficulty * 1.2)
                elif recent_avg < 0.3:
                    self.current_difficulty = max(0.5, self.current_difficulty * 0.8)


class TemplateMixer:
    """Combines multiple templates to create novel prompts."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.combination_rules = self._define_combination_rules()
        self.semantic_bridges = self._create_semantic_bridges()
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize base templates."""
        templates = {}
        
        # Question templates
        templates["q_basic"] = PromptTemplate(
            id="q_basic",
            type=TemplateType.QUESTION,
            pattern="What is {concept}?",
            variables=["concept"],
            complexity=1.0
        )
        
        templates["q_comparative"] = PromptTemplate(
            id="q_comparative",
            type=TemplateType.COMPARATIVE,
            pattern="How does {concept1} differ from {concept2} in terms of {aspect}?",
            variables=["concept1", "concept2", "aspect"],
            complexity=2.5
        )
        
        templates["q_hypothetical"] = PromptTemplate(
            id="q_hypothetical",
            type=TemplateType.HYPOTHETICAL,
            pattern="If {condition}, what would be the implications for {domain}?",
            variables=["condition", "domain"],
            complexity=3.0
        )
        
        # Instruction templates
        templates["i_explain"] = PromptTemplate(
            id="i_explain",
            type=TemplateType.INSTRUCTION,
            pattern="Explain {topic} as if you were {perspective}.",
            variables=["topic", "perspective"],
            complexity=2.0
        )
        
        templates["i_analyze"] = PromptTemplate(
            id="i_analyze",
            type=TemplateType.ANALYTICAL,
            pattern="Analyze the {aspect} of {subject} considering {factors}.",
            variables=["aspect", "subject", "factors"],
            complexity=3.5
        )
        
        # Reasoning templates
        templates["r_chain"] = PromptTemplate(
            id="r_chain",
            type=TemplateType.REASONING,
            pattern="Given that {premise1} and {premise2}, what can we conclude about {conclusion_topic}?",
            variables=["premise1", "premise2", "conclusion_topic"],
            complexity=3.0
        )
        
        templates["r_counterfactual"] = PromptTemplate(
            id="r_counterfactual",
            type=TemplateType.REASONING,
            pattern="If {counterfactual} had been true, how would {outcome} have changed?",
            variables=["counterfactual", "outcome"],
            complexity=4.0
        )
        
        # Creative templates
        templates["c_narrative"] = PromptTemplate(
            id="c_narrative",
            type=TemplateType.CREATIVE,
            pattern="Write a {genre} story about {protagonist} who {challenge}.",
            variables=["genre", "protagonist", "challenge"],
            complexity=2.5
        )
        
        templates["c_metaphor"] = PromptTemplate(
            id="c_metaphor",
            type=TemplateType.CREATIVE,
            pattern="Create a metaphor that explains {concept} using {familiar_domain}.",
            variables=["concept", "familiar_domain"],
            complexity=3.0
        )
        
        # Scenario templates
        templates["s_problem"] = PromptTemplate(
            id="s_problem",
            type=TemplateType.SCENARIO,
            pattern="You are {role}. How would you solve {problem} given {constraints}?",
            variables=["role", "problem", "constraints"],
            complexity=3.5
        )
        
        return templates
    
    def _define_combination_rules(self) -> Dict[Tuple[TemplateType, TemplateType], float]:
        """Define compatibility scores for template combinations."""
        rules = {}
        
        # Compatible combinations
        rules[(TemplateType.QUESTION, TemplateType.REASONING)] = 0.9
        rules[(TemplateType.INSTRUCTION, TemplateType.ANALYTICAL)] = 0.85
        rules[(TemplateType.HYPOTHETICAL, TemplateType.SCENARIO)] = 0.8
        rules[(TemplateType.COMPARATIVE, TemplateType.ANALYTICAL)] = 0.85
        rules[(TemplateType.CREATIVE, TemplateType.SCENARIO)] = 0.75
        
        # Less compatible combinations
        rules[(TemplateType.CREATIVE, TemplateType.ANALYTICAL)] = 0.4
        rules[(TemplateType.QUESTION, TemplateType.INSTRUCTION)] = 0.3
        
        # Make rules symmetric
        for (t1, t2), score in list(rules.items()):
            rules[(t2, t1)] = score
        
        return rules
    
    def _create_semantic_bridges(self) -> Dict[str, List[str]]:
        """Create semantic bridges for smooth transitions."""
        bridges = {
            "causal": ["Therefore", "As a result", "Consequently", "This leads to"],
            "additive": ["Furthermore", "Additionally", "Moreover", "Also consider"],
            "contrastive": ["However", "On the other hand", "In contrast", "Alternatively"],
            "temporal": ["Subsequently", "Following this", "After which", "Then"],
            "exemplification": ["For example", "Such as", "To illustrate", "Consider"],
            "elaboration": ["Specifically", "In particular", "To elaborate", "More precisely"]
        }
        return bridges
    
    def combine_templates(self, 
                          template_ids: List[str],
                          variables: Dict[str, str],
                          use_bridges: bool = True) -> str:
        """Combine multiple templates into a coherent prompt."""
        if not template_ids:
            return ""
        
        templates = [self.templates[tid] for tid in template_ids if tid in self.templates]
        if not templates:
            return ""
        
        # Fill each template
        filled = [t.fill(variables) for t in templates]
        
        if len(filled) == 1:
            return filled[0]
        
        # Combine with semantic bridges
        result = filled[0]
        for i in range(1, len(filled)):
            if use_bridges:
                # Select appropriate bridge based on template types
                bridge_type = self._select_bridge_type(templates[i-1].type, templates[i].type)
                bridge = random.choice(self.semantic_bridges[bridge_type])
                result += f" {bridge}, {filled[i].lower()}"
            else:
                result += f" {filled[i]}"
        
        return result
    
    def _select_bridge_type(self, type1: TemplateType, type2: TemplateType) -> str:
        """Select appropriate bridge type based on template types."""
        if type1 == TemplateType.REASONING and type2 == TemplateType.REASONING:
            return "causal"
        elif type1 == TemplateType.QUESTION and type2 == TemplateType.QUESTION:
            return "additive"
        elif type1 == TemplateType.HYPOTHETICAL:
            return "temporal"
        elif type2 == TemplateType.COMPARATIVE:
            return "contrastive"
        else:
            return random.choice(["additive", "elaboration"])
    
    def blend_templates(self, template1_id: str, template2_id: str, blend_ratio: float = 0.5) -> str:
        """Semantically blend two templates."""
        t1 = self.templates.get(template1_id)
        t2 = self.templates.get(template2_id)
        
        if not t1 or not t2:
            return ""
        
        # Extract key components from each template
        pattern1_parts = t1.pattern.split()
        pattern2_parts = t2.pattern.split()
        
        # Blend based on ratio
        blend_point1 = int(len(pattern1_parts) * blend_ratio)
        blend_point2 = int(len(pattern2_parts) * (1 - blend_ratio))
        
        blended = pattern1_parts[:blend_point1] + ["..."] + pattern2_parts[blend_point2:]
        
        return " ".join(blended)
    
    def satisfy_constraints(self, prompt: str, constraints: Dict[str, Any]) -> bool:
        """Check if prompt satisfies given constraints."""
        # Length constraint
        if "min_length" in constraints:
            if len(prompt.split()) < constraints["min_length"]:
                return False
        
        if "max_length" in constraints:
            if len(prompt.split()) > constraints["max_length"]:
                return False
        
        # Complexity constraint
        if "min_complexity" in constraints:
            if flesch_kincaid_grade(prompt) < constraints["min_complexity"]:
                return False
        
        # Required keywords
        if "required_keywords" in constraints:
            for keyword in constraints["required_keywords"]:
                if keyword.lower() not in prompt.lower():
                    return False
        
        # Forbidden words
        if "forbidden_words" in constraints:
            for word in constraints["forbidden_words"]:
                if word.lower() in prompt.lower():
                    return False
        
        return True


class ContextAwareGenerator:
    """Generates prompts based on context and model feedback."""
    
    def __init__(self):
        self.context_patterns = self._initialize_context_patterns()
        self.difficulty_modifiers = self._create_difficulty_modifiers()
        self.conversation_templates = self._create_conversation_templates()
    
    def _initialize_context_patterns(self) -> Dict[str, List[str]]:
        """Initialize context-based patterns."""
        patterns = {
            "follow_up": [
                "Building on the previous point about {topic}, {question}",
                "Given your response about {topic}, {elaboration}",
                "To clarify your answer on {topic}, {clarification}"
            ],
            "contradiction": [
                "You mentioned {claim}, but what about {counter_example}?",
                "How do you reconcile {statement1} with {statement2}?",
                "Isn't there a contradiction between {point1} and {point2}?"
            ],
            "deepening": [
                "Can you elaborate on the {aspect} you mentioned?",
                "What are the deeper implications of {concept}?",
                "How does {detail} relate to the broader context of {topic}?"
            ],
            "application": [
                "How would {concept} apply to {scenario}?",
                "Can you provide a practical example of {theory}?",
                "What would happen if we applied {principle} to {situation}?"
            ]
        }
        return patterns
    
    def _create_difficulty_modifiers(self) -> Dict[float, Dict[str, Any]]:
        """Create difficulty-based modifications."""
        modifiers = {
            0.5: {  # Easy
                "vocab_level": "simple",
                "sentence_complexity": "simple",
                "concepts": ["basic", "familiar"],
                "reasoning_steps": 1
            },
            1.0: {  # Normal
                "vocab_level": "standard",
                "sentence_complexity": "moderate",
                "concepts": ["intermediate"],
                "reasoning_steps": 2
            },
            2.0: {  # Challenging
                "vocab_level": "advanced",
                "sentence_complexity": "complex",
                "concepts": ["abstract", "technical"],
                "reasoning_steps": 3
            },
            3.0: {  # Expert
                "vocab_level": "specialized",
                "sentence_complexity": "sophisticated",
                "concepts": ["cutting-edge", "interdisciplinary"],
                "reasoning_steps": 4
            },
            4.0: {  # Extreme
                "vocab_level": "esoteric",
                "sentence_complexity": "highly_complex",
                "concepts": ["paradoxical", "meta-cognitive"],
                "reasoning_steps": 5
            }
        }
        return modifiers
    
    def _create_conversation_templates(self) -> List[Dict[str, str]]:
        """Create multi-turn conversation templates."""
        templates = [
            {
                "opening": "Let's explore {topic} through a series of questions.",
                "follow_ups": [
                    "First, what is your understanding of {aspect1}?",
                    "How does that relate to {aspect2}?",
                    "What would be the implications for {application}?"
                ],
                "closing": "Based on our discussion, summarize the key insights about {topic}."
            },
            {
                "opening": "I'd like to challenge your thinking on {topic}.",
                "follow_ups": [
                    "What assumptions underlie your position on {aspect}?",
                    "How would you respond to the criticism that {objection}?",
                    "Can you provide evidence for {claim}?"
                ],
                "closing": "How has this discussion changed your perspective on {topic}?"
            },
            {
                "opening": "Let's conduct a thought experiment about {scenario}.",
                "follow_ups": [
                    "What would be the immediate consequences?",
                    "How would different stakeholders react?",
                    "What long-term changes might occur?"
                ],
                "closing": "What lessons can we draw from this hypothetical scenario?"
            }
        ]
        return templates
    
    def inject_context(self, base_prompt: str, context: GenerationContext) -> str:
        """Inject context into prompt based on history."""
        if not context.previous_prompts:
            return base_prompt
        
        # Extract key topics from previous interactions
        last_prompt = context.previous_prompts[-1]
        last_response = context.model_responses[-1] if context.model_responses else ""
        
        # Extract key entities/topics using spaCy
        doc = nlp(last_response if last_response else last_prompt)
        entities = [ent.text for ent in doc.ents]
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Select context pattern based on performance
        if context.performance_scores and context.performance_scores[-1] < 0.5:
            pattern_type = "clarification"
        elif len(context.previous_prompts) > 3:
            pattern_type = "deepening"
        else:
            pattern_type = "follow_up"
        
        # Build contextual prompt
        if pattern_type in self.context_patterns:
            pattern = random.choice(self.context_patterns[pattern_type])
            
            # Fill in context-specific information
            topic = random.choice(noun_phrases) if noun_phrases else "this concept"
            
            contextual_addition = pattern.format(
                topic=topic,
                question=base_prompt,
                elaboration=f"how does this connect to {base_prompt}",
                clarification=f"specifically regarding {base_prompt}"
            )
            
            return contextual_addition
        
        return base_prompt
    
    def scale_difficulty(self, prompt: str, target_difficulty: float) -> str:
        """Adjust prompt difficulty based on target level."""
        # Find closest difficulty level
        levels = sorted(self.difficulty_modifiers.keys())
        closest_level = min(levels, key=lambda x: abs(x - target_difficulty))
        modifiers = self.difficulty_modifiers[closest_level]
        
        # Apply vocabulary modifications
        if modifiers["vocab_level"] == "simple":
            prompt = self._simplify_vocabulary(prompt)
        elif modifiers["vocab_level"] in ["advanced", "specialized", "esoteric"]:
            prompt = self._complexify_vocabulary(prompt)
        
        # Add reasoning steps if needed
        if modifiers["reasoning_steps"] > 2:
            prompt = self._add_reasoning_steps(prompt, modifiers["reasoning_steps"])
        
        return prompt
    
    def _simplify_vocabulary(self, text: str) -> str:
        """Simplify vocabulary in text."""
        doc = nlp(text)
        simplified = []
        
        for token in doc:
            # Replace complex words with simpler synonyms
            if len(token.text) > 8 and token.pos_ in ["NOUN", "VERB", "ADJ"]:
                synsets = wordnet.synsets(token.text)
                if synsets:
                    # Get simpler synonyms
                    simpler = [lemma.name() for synset in synsets 
                              for lemma in synset.lemmas() 
                              if len(lemma.name()) < len(token.text)]
                    if simpler:
                        simplified.append(simpler[0].replace("_", " "))
                    else:
                        simplified.append(token.text)
                else:
                    simplified.append(token.text)
            else:
                simplified.append(token.text)
        
        return " ".join(simplified)
    
    def _complexify_vocabulary(self, text: str) -> str:
        """Add complexity to vocabulary."""
        doc = nlp(text)
        complexified = []
        
        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ"] and random.random() < 0.3:
                synsets = wordnet.synsets(token.text)
                if synsets:
                    # Get more complex synonyms
                    complex_synonyms = [lemma.name() for synset in synsets 
                                       for lemma in synset.lemmas() 
                                       if len(lemma.name()) > len(token.text)]
                    if complex_synonyms:
                        complexified.append(complex_synonyms[0].replace("_", " "))
                    else:
                        complexified.append(token.text)
                else:
                    complexified.append(token.text)
            else:
                complexified.append(token.text)
        
        return " ".join(complexified)
    
    def _add_reasoning_steps(self, prompt: str, num_steps: int) -> str:
        """Add reasoning steps to prompt."""
        steps = []
        for i in range(num_steps):
            step_phrases = [
                f"Step {i+1}: Consider the implications of this.",
                f"Next, analyze the underlying assumptions.",
                f"Then, evaluate potential counterarguments.",
                f"Subsequently, synthesize the different perspectives.",
                f"Finally, draw a reasoned conclusion."
            ]
            if i < len(step_phrases):
                steps.append(step_phrases[i])
        
        return f"{prompt} {' '.join(steps)}"
    
    def generate_conversation(self, topic: str, context: GenerationContext, num_turns: int = 3) -> List[str]:
        """Generate multi-turn conversation."""
        template = random.choice(self.conversation_templates)
        conversation = []
        
        # Opening
        opening = template["opening"].format(topic=topic)
        conversation.append(opening)
        
        # Follow-ups based on context
        for i, follow_up_template in enumerate(template["follow_ups"][:num_turns]):
            # Adapt based on context
            if context.model_responses and i < len(context.model_responses):
                # Extract information from previous response
                prev_response = context.model_responses[-1]
                doc = nlp(prev_response)
                entities = [ent.text for ent in doc.ents]
                
                # Customize follow-up
                if entities:
                    follow_up = follow_up_template.format(
                        aspect=entities[0] if entities else topic,
                        aspect1=entities[0] if entities else f"the basics of {topic}",
                        aspect2=entities[1] if len(entities) > 1 else f"advanced {topic}",
                        application=f"real-world {topic}",
                        objection=f"{topic} might not work in all cases",
                        claim=f"your assertion about {topic}"
                    )
                else:
                    follow_up = follow_up_template.format(
                        aspect=topic,
                        aspect1=f"the fundamentals of {topic}",
                        aspect2=f"the complexities of {topic}",
                        application=f"practical {topic}",
                        objection=f"{topic} has limitations",
                        claim=f"the importance of {topic}"
                    )
            else:
                follow_up = follow_up_template.format(
                    aspect=topic,
                    aspect1=f"basic {topic}",
                    aspect2=f"advanced {topic}",
                    application=f"{topic} applications",
                    objection=f"{topic} criticism",
                    claim=f"{topic} benefits"
                )
            
            conversation.append(follow_up)
        
        # Closing
        closing = template["closing"].format(topic=topic)
        conversation.append(closing)
        
        return conversation


class DomainSynthesizer:
    """Generates domain-specific prompts."""
    
    def __init__(self):
        self.domain_vocabularies = self._load_domain_vocabularies()
        self.domain_patterns = self._create_domain_patterns()
        self.cross_domain_bridges = self._create_cross_domain_bridges()
        self.edge_case_generators = self._initialize_edge_case_generators()
    
    def _load_domain_vocabularies(self) -> Dict[DomainType, Dict[str, List[str]]]:
        """Load domain-specific vocabularies."""
        vocabs = {}
        
        vocabs[DomainType.SCIENTIFIC] = {
            "concepts": ["hypothesis", "experiment", "variable", "control", "data", "analysis", "theory", "observation"],
            "verbs": ["hypothesize", "test", "measure", "analyze", "conclude", "replicate", "validate", "falsify"],
            "adjectives": ["empirical", "quantitative", "reproducible", "significant", "controlled", "systematic"],
            "jargon": ["p-value", "confidence interval", "standard deviation", "correlation", "causation", "peer review"]
        }
        
        vocabs[DomainType.MATHEMATICAL] = {
            "concepts": ["theorem", "proof", "equation", "function", "variable", "integral", "derivative", "matrix"],
            "verbs": ["prove", "derive", "calculate", "solve", "integrate", "differentiate", "factorize", "optimize"],
            "adjectives": ["continuous", "discrete", "linear", "non-linear", "convergent", "divergent", "orthogonal"],
            "jargon": ["bijection", "homeomorphism", "eigenvalue", "Lagrangian", "Hamiltonian", "tensor", "manifold"]
        }
        
        vocabs[DomainType.PHILOSOPHICAL] = {
            "concepts": ["epistemology", "ontology", "ethics", "metaphysics", "logic", "consciousness", "free will"],
            "verbs": ["argue", "deduce", "infer", "contemplate", "critique", "synthesize", "deconstruct"],
            "adjectives": ["existential", "phenomenological", "dialectical", "teleological", "deontological"],
            "jargon": ["qualia", "noumenon", "categorical imperative", "dialectic", "solipsism", "nihilism"]
        }
        
        vocabs[DomainType.TECHNICAL] = {
            "concepts": ["algorithm", "architecture", "protocol", "interface", "implementation", "optimization", "scalability"],
            "verbs": ["implement", "debug", "optimize", "refactor", "deploy", "integrate", "authenticate", "encrypt"],
            "adjectives": ["asynchronous", "distributed", "concurrent", "stateless", "idempotent", "polymorphic"],
            "jargon": ["API", "REST", "GraphQL", "microservices", "containerization", "CI/CD", "load balancing"]
        }
        
        vocabs[DomainType.MEDICAL] = {
            "concepts": ["diagnosis", "treatment", "symptom", "pathology", "prognosis", "etiology", "epidemiology"],
            "verbs": ["diagnose", "treat", "prescribe", "examine", "monitor", "intervene", "prevent", "rehabilitate"],
            "adjectives": ["chronic", "acute", "benign", "malignant", "systemic", "localized", "idiopathic"],
            "jargon": ["comorbidity", "contraindication", "prophylaxis", "metastasis", "anamnesis", "differential diagnosis"]
        }
        
        vocabs[DomainType.LEGAL] = {
            "concepts": ["precedent", "jurisdiction", "liability", "statute", "tort", "contract", "evidence", "testimony"],
            "verbs": ["prosecute", "defend", "appeal", "arbitrate", "litigate", "adjudicate", "legislate", "enforce"],
            "adjectives": ["statutory", "constitutional", "precedential", "punitive", "compensatory", "procedural"],
            "jargon": ["mens rea", "actus reus", "habeas corpus", "voir dire", "pro bono", "amicus curiae", "stare decisis"]
        }
        
        # Add default vocab for other domains
        for domain in DomainType:
            if domain not in vocabs:
                vocabs[domain] = {
                    "concepts": ["concept", "idea", "principle", "method", "approach", "framework"],
                    "verbs": ["analyze", "evaluate", "consider", "develop", "apply", "understand"],
                    "adjectives": ["important", "significant", "relevant", "comprehensive", "detailed"],
                    "jargon": []
                }
        
        return vocabs
    
    def _create_domain_patterns(self) -> Dict[DomainType, List[str]]:
        """Create domain-specific prompt patterns."""
        patterns = {}
        
        patterns[DomainType.SCIENTIFIC] = [
            "Design an experiment to test {hypothesis} while controlling for {variables}.",
            "What would be the implications if {finding} were replicated across {contexts}?",
            "Analyze the statistical significance of {result} given {sample_size} and {confidence_level}.",
            "How would you falsify the theory that {claim}?"
        ]
        
        patterns[DomainType.MATHEMATICAL] = [
            "Prove that {statement} for all {domain}.",
            "Find the {operation} of {expression} with respect to {variable}.",
            "Demonstrate the relationship between {concept1} and {concept2} using {method}.",
            "Solve the {type} equation: {equation}"
        ]
        
        patterns[DomainType.PHILOSOPHICAL] = [
            "What are the {ethical_framework} implications of {scenario}?",
            "How does {philosopher}'s concept of {idea} relate to {contemporary_issue}?",
            "Critically examine the assumption that {belief}.",
            "Construct an argument for/against {position} using {logical_framework}."
        ]
        
        patterns[DomainType.TECHNICAL] = [
            "Design a {architecture_type} architecture for {system} that handles {requirements}.",
            "Optimize the {algorithm} for {metric} given {constraints}.",
            "Implement {feature} ensuring {non_functional_requirements}.",
            "Debug the {issue} in {component} considering {dependencies}."
        ]
        
        patterns[DomainType.MEDICAL] = [
            "Given symptoms {symptoms}, what differential diagnoses would you consider?",
            "Explain the pathophysiology of {condition} and its {complications}.",
            "Design a treatment plan for {patient_profile} with {condition}.",
            "Evaluate the efficacy of {intervention} for {indication}."
        ]
        
        patterns[DomainType.LEGAL] = [
            "Analyze the {case} under {jurisdiction}'s {area_of_law}.",
            "What precedents support/challenge {legal_position}?",
            "Draft {document_type} considering {legal_requirements}.",
            "Evaluate the liability of {party} in {scenario}."
        ]
        
        # Add generic patterns for other domains
        for domain in DomainType:
            if domain not in patterns:
                patterns[domain] = [
                    "Analyze {topic} from a {domain} perspective.",
                    "What are the {domain} implications of {scenario}?",
                    "Apply {domain} principles to solve {problem}.",
                    "Evaluate {subject} using {domain} methodology."
                ]
        
        return patterns
    
    def _create_cross_domain_bridges(self) -> Dict[Tuple[DomainType, DomainType], List[str]]:
        """Create bridges between different domains."""
        bridges = {}
        
        bridges[(DomainType.SCIENTIFIC, DomainType.PHILOSOPHICAL)] = [
            "What are the epistemological implications of {scientific_finding}?",
            "How does {scientific_method} relate to {philosophical_concept}?"
        ]
        
        bridges[(DomainType.TECHNICAL, DomainType.BUSINESS)] = [
            "What is the ROI of implementing {technical_solution}?",
            "How does {technology} impact {business_process}?"
        ]
        
        bridges[(DomainType.MEDICAL, DomainType.LEGAL)] = [
            "What are the legal implications of {medical_decision}?",
            "How does {medical_condition} affect {legal_capacity}?"
        ]
        
        bridges[(DomainType.MATHEMATICAL, DomainType.TECHNICAL)] = [
            "Apply {mathematical_concept} to optimize {algorithm}.",
            "What is the computational complexity of {mathematical_operation}?"
        ]
        
        return bridges
    
    def _initialize_edge_case_generators(self) -> Dict[DomainType, callable]:
        """Initialize edge case generators for each domain."""
        generators = {}
        
        def scientific_edge_case():
            scenarios = [
                "What if the control group shows stronger effects than the treatment group?",
                "How would you handle p-hacking accusations in your research?",
                "What if your results contradict established theory?",
                "How do you address non-reproducible results?"
            ]
            return random.choice(scenarios)
        
        def mathematical_edge_case():
            scenarios = [
                "What happens when the function is undefined at critical points?",
                "How do you handle division by zero in this context?",
                "What if the series doesn't converge?",
                "How do you prove this for infinite sets?"
            ]
            return random.choice(scenarios)
        
        def technical_edge_case():
            scenarios = [
                "How does the system handle race conditions?",
                "What happens during network partition?",
                "How do you ensure data consistency in distributed failure?",
                "What's the behavior under memory pressure?"
            ]
            return random.choice(scenarios)
        
        generators[DomainType.SCIENTIFIC] = scientific_edge_case
        generators[DomainType.MATHEMATICAL] = mathematical_edge_case
        generators[DomainType.TECHNICAL] = technical_edge_case
        
        # Default edge case generator
        def default_edge_case():
            return "What happens in the worst-case scenario?"
        
        for domain in DomainType:
            if domain not in generators:
                generators[domain] = default_edge_case
        
        return generators
    
    def generate_domain_prompt(self, domain: DomainType, complexity: float = 1.0) -> str:
        """Generate a domain-specific prompt."""
        vocab = self.domain_vocabularies[domain]
        patterns = self.domain_patterns[domain]
        
        # Select pattern based on complexity
        pattern_idx = min(int(complexity * len(patterns) / 5), len(patterns) - 1)
        pattern = patterns[pattern_idx]
        
        # Fill pattern with domain-specific vocabulary
        filled = pattern
        for placeholder in re.findall(r'\{(\w+)\}', pattern):
            if placeholder in ["hypothesis", "claim", "finding", "theory", "concept", "idea"]:
                replacement = random.choice(vocab["concepts"])
            elif placeholder in ["variables", "constraints", "requirements", "dependencies"]:
                replacement = ", ".join(random.sample(vocab["concepts"], min(2, len(vocab["concepts"]))))
            elif placeholder in ["operation", "method", "algorithm", "framework"]:
                replacement = random.choice(vocab["verbs"])
            else:
                replacement = placeholder.replace("_", " ")
            
            filled = filled.replace(f"{{{placeholder}}}", replacement)
        
        return filled
    
    def inject_jargon(self, prompt: str, domain: DomainType, density: float = 0.1) -> str:
        """Inject domain-specific jargon into prompt."""
        vocab = self.domain_vocabularies[domain]
        if not vocab["jargon"]:
            return prompt
        
        words = prompt.split()
        num_injections = int(len(words) * density)
        
        for _ in range(num_injections):
            jargon = random.choice(vocab["jargon"])
            position = random.randint(0, len(words))
            words.insert(position, f"({jargon})")
        
        return " ".join(words)
    
    def create_cross_domain_prompt(self, domain1: DomainType, domain2: DomainType) -> str:
        """Create a prompt bridging two domains."""
        key = (domain1, domain2)
        reverse_key = (domain2, domain1)
        
        if key in self.cross_domain_bridges:
            patterns = self.cross_domain_bridges[key]
        elif reverse_key in self.cross_domain_bridges:
            patterns = self.cross_domain_bridges[reverse_key]
        else:
            # Create generic cross-domain prompt
            return f"How do principles from {domain1.value} apply to problems in {domain2.value}?"
        
        pattern = random.choice(patterns)
        
        # Fill with vocabulary from both domains
        vocab1 = self.domain_vocabularies[domain1]
        vocab2 = self.domain_vocabularies[domain2]
        
        filled = pattern
        for placeholder in re.findall(r'\{(\w+)\}', pattern):
            if domain1.value in placeholder.lower():
                replacement = random.choice(vocab1["concepts"])
            elif domain2.value in placeholder.lower():
                replacement = random.choice(vocab2["concepts"])
            else:
                replacement = placeholder.replace("_", " ")
            
            filled = filled.replace(f"{{{placeholder}}}", replacement)
        
        return filled
    
    def generate_edge_case(self, domain: DomainType) -> str:
        """Generate edge case prompt for domain."""
        if domain in self.edge_case_generators:
            return self.edge_case_generators[domain]()
        return "What happens in an edge case scenario?"


class QualityController:
    """Controls quality of generated prompts."""
    
    def __init__(self):
        self.grammar_checker = self._initialize_grammar_checker()
        self.coherence_scorer = self._initialize_coherence_scorer()
        self.complexity_estimator = self._initialize_complexity_estimator()
        self.redundancy_filter = RedundancyFilter()
    
    def _initialize_grammar_checker(self):
        """Initialize grammar checking."""
        return nlp  # Using spaCy for grammar checking
    
    def _initialize_coherence_scorer(self):
        """Initialize coherence scoring."""
        return CoherenceScorer()
    
    def _initialize_complexity_estimator(self):
        """Initialize complexity estimation."""
        return ComplexityEstimator()
    
    def validate_grammar(self, prompt: str) -> Tuple[bool, List[str]]:
        """Validate grammatical correctness."""
        doc = self.grammar_checker(prompt)
        errors = []
        
        # Check for basic grammar issues
        for token in doc:
            # Check for repeated words
            if token.i > 0 and token.text == doc[token.i - 1].text:
                errors.append(f"Repeated word: {token.text}")
            
            # Check for missing determiners
            if token.pos_ == "NOUN" and token.dep_ == "nsubj":
                if token.i == 0 or doc[token.i - 1].pos_ not in ["DET", "PRON"]:
                    if token.text[0].islower():
                        errors.append(f"Missing determiner before: {token.text}")
        
        # Check sentence structure
        sentences = list(doc.sents)
        for sent in sentences:
            # Check for fragments
            has_subject = any(token.dep_ == "nsubj" for token in sent)
            has_verb = any(token.pos_ == "VERB" for token in sent)
            
            if not (has_subject and has_verb):
                if len(sent) > 3:  # Ignore very short sentences
                    errors.append(f"Sentence fragment: {sent.text[:30]}...")
        
        return len(errors) == 0, errors
    
    def score_coherence(self, prompt: str) -> float:
        """Score semantic coherence of prompt."""
        return self.coherence_scorer.score(prompt)
    
    def estimate_complexity(self, prompt: str) -> Dict[str, float]:
        """Estimate various complexity metrics."""
        return self.complexity_estimator.estimate(prompt)
    
    def filter_redundant(self, prompts: List[str]) -> List[str]:
        """Filter redundant or too similar prompts."""
        return self.redundancy_filter.filter(prompts)


class CoherenceScorer:
    """Scores semantic coherence of text."""
    
    def score(self, text: str) -> float:
        """Score coherence from 0 to 1."""
        doc = nlp(text)
        
        # Check entity consistency
        entities = [ent.text for ent in doc.ents]
        entity_score = 1.0
        if entities:
            # Check if entities are referenced consistently
            entity_mentions = defaultdict(int)
            for ent in entities:
                entity_mentions[ent.lower()] += 1
            
            # Penalize single mentions of important entities
            single_mentions = sum(1 for count in entity_mentions.values() if count == 1)
            entity_score = 1.0 - (single_mentions / len(entity_mentions)) * 0.3
        
        # Check topical coherence
        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
        topic_score = 1.0
        if len(noun_chunks) > 3:
            # Check for topic drift
            first_half = set(noun_chunks[:len(noun_chunks)//2])
            second_half = set(noun_chunks[len(noun_chunks)//2:])
            overlap = len(first_half & second_half)
            topic_score = overlap / min(len(first_half), len(second_half)) if min(len(first_half), len(second_half)) > 0 else 0.5
        
        # Check sentence connectivity
        sentences = list(doc.sents)
        connectivity_score = 1.0
        if len(sentences) > 1:
            # Check for transition words or repeated concepts
            transitions = ["however", "therefore", "moreover", "furthermore", "additionally", "consequently"]
            has_transitions = sum(1 for sent in sentences[1:] 
                                 if any(trans in sent.text.lower() for trans in transitions))
            connectivity_score = 0.7 + (has_transitions / (len(sentences) - 1)) * 0.3
        
        # Combine scores
        final_score = (entity_score * 0.3 + topic_score * 0.4 + connectivity_score * 0.3)
        return final_score


class ComplexityEstimator:
    """Estimates complexity of prompts."""
    
    def estimate(self, text: str) -> Dict[str, float]:
        """Estimate various complexity metrics."""
        metrics = {}
        
        # Readability scores
        metrics["flesch_reading_ease"] = flesch_reading_ease(text)
        metrics["flesch_kincaid_grade"] = flesch_kincaid_grade(text)
        
        # Structural complexity
        doc = nlp(text)
        sentences = list(doc.sents)
        
        if sentences:
            # Average sentence length
            metrics["avg_sentence_length"] = np.mean([len(sent.text.split()) for sent in sentences])
            
            # Maximum dependency tree depth
            max_depth = 0
            for sent in sentences:
                for token in sent:
                    depth = 0
                    current = token
                    while current.head != current:
                        depth += 1
                        current = current.head
                        if depth > 20:  # Prevent infinite loops
                            break
                    max_depth = max(max_depth, depth)
            metrics["max_dependency_depth"] = max_depth
        
        # Vocabulary complexity
        words = [token.text.lower() for token in doc if token.is_alpha]
        if words:
            metrics["vocabulary_size"] = len(set(words))
            metrics["avg_word_length"] = np.mean([len(word) for word in words])
            
            # Type-token ratio
            metrics["type_token_ratio"] = len(set(words)) / len(words)
        
        # Conceptual complexity (based on named entities and noun phrases)
        metrics["entity_density"] = len(doc.ents) / len(doc) if len(doc) > 0 else 0
        metrics["noun_phrase_density"] = len(list(doc.noun_chunks)) / len(sentences) if sentences else 0
        
        return metrics


class RedundancyFilter:
    """Filters redundant prompts."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_prompts = deque(maxlen=1000)
    
    def filter(self, prompts: List[str]) -> List[str]:
        """Filter redundant prompts."""
        filtered = []
        
        for prompt in prompts:
            if not self._is_redundant(prompt):
                filtered.append(prompt)
                self.seen_prompts.append(prompt.lower())
        
        return filtered
    
    def _is_redundant(self, prompt: str) -> bool:
        """Check if prompt is redundant."""
        prompt_lower = prompt.lower()
        
        # Check exact duplicates
        if prompt_lower in self.seen_prompts:
            return True
        
        # Check similarity with recent prompts
        prompt_tokens = set(word_tokenize(prompt_lower))
        
        for seen in self.seen_prompts:
            seen_tokens = set(word_tokenize(seen))
            
            # Jaccard similarity
            intersection = len(prompt_tokens & seen_tokens)
            union = len(prompt_tokens | seen_tokens)
            
            if union > 0:
                similarity = intersection / union
                if similarity > self.similarity_threshold:
                    return True
        
        return False


class DynamicSynthesisSystem:
    """Main system for dynamic prompt synthesis."""
    
    def __init__(self):
        self.template_mixer = TemplateMixer()
        self.context_generator = ContextAwareGenerator()
        self.domain_synthesizer = DomainSynthesizer()
        self.quality_controller = QualityController()
        self.generation_cache = {}
    
    def generate_prompt(self,
                        template_types: Optional[List[TemplateType]] = None,
                        domain: Optional[DomainType] = None,
                        context: Optional[GenerationContext] = None,
                        complexity: float = 1.0,
                        ensure_quality: bool = True) -> str:
        """Generate a single dynamic prompt."""
        
        # Select template types if not specified
        if not template_types:
            template_types = [random.choice(list(TemplateType))]
        
        # Generate base prompt from templates
        template_ids = []
        for ttype in template_types:
            matching = [tid for tid, t in self.template_mixer.templates.items() 
                       if t.type == ttype]
            if matching:
                template_ids.append(random.choice(matching))
        
        # Generate variables for templates
        variables = self._generate_template_variables()
        
        # Combine templates
        base_prompt = self.template_mixer.combine_templates(template_ids, variables)
        
        # Apply domain-specific modifications
        if domain:
            base_prompt = self.domain_synthesizer.inject_jargon(base_prompt, domain)
        
        # Apply context if available
        if context:
            base_prompt = self.context_generator.inject_context(base_prompt, context)
            base_prompt = self.context_generator.scale_difficulty(base_prompt, context.current_difficulty)
        else:
            base_prompt = self.context_generator.scale_difficulty(base_prompt, complexity)
        
        # Quality control
        if ensure_quality:
            is_valid, errors = self.quality_controller.validate_grammar(base_prompt)
            if not is_valid:
                # Try to fix common issues
                base_prompt = self._fix_grammar_issues(base_prompt, errors)
            
            # Check coherence
            coherence = self.quality_controller.score_coherence(base_prompt)
            if coherence < 0.5:
                # Regenerate with different parameters
                return self.generate_prompt(template_types, domain, context, complexity, ensure_quality)
        
        return base_prompt
    
    def _generate_template_variables(self) -> Dict[str, str]:
        """Generate variables for template filling."""
        concepts = ["artificial intelligence", "quantum computing", "climate change", 
                   "democracy", "consciousness", "evolution", "blockchain", "genetics"]
        
        aspects = ["implications", "challenges", "benefits", "limitations", 
                  "applications", "theoretical foundations", "future developments"]
        
        roles = ["expert", "beginner", "critic", "advocate", "researcher", 
                "policy maker", "educator", "practitioner"]
        
        concept1 = random.choice(concepts)
        concept2 = random.choice([c for c in concepts if c != concept1])
        
        variables = {
            "concept": random.choice(concepts),
            "concept1": concept1,
            "concept2": concept2,
            "aspect": random.choice(aspects),
            "topic": random.choice(concepts),
            "perspective": random.choice(roles),
            "subject": random.choice(concepts),
            "factors": ", ".join(random.sample(aspects, 2)),
            "premise1": f"{random.choice(concepts)} is important",
            "premise2": f"{random.choice(aspects)} must be considered",
            "conclusion_topic": random.choice(concepts),
            "condition": f"{random.choice(concepts)} becomes widespread",
            "domain": random.choice(["society", "technology", "science", "ethics"]),
            "role": random.choice(roles),
            "problem": f"implementing {random.choice(concepts)}",
            "constraints": "limited resources and time",
            "genre": random.choice(["science fiction", "mystery", "adventure"]),
            "protagonist": random.choice(["a scientist", "an AI", "a detective"]),
            "challenge": f"must solve the mystery of {random.choice(concepts)}",
            "counterfactual": f"{random.choice(concepts)} had never been discovered",
            "outcome": "technological progress",
            "familiar_domain": random.choice(["cooking", "sports", "music", "nature"])
        }
        
        return variables
    
    def _fix_grammar_issues(self, prompt: str, errors: List[str]) -> str:
        """Attempt to fix common grammar issues."""
        fixed = prompt
        
        for error in errors:
            if "Repeated word" in error:
                # Remove repeated words
                words = fixed.split()
                fixed_words = [words[0]]
                for word in words[1:]:
                    if word != fixed_words[-1]:
                        fixed_words.append(word)
                fixed = " ".join(fixed_words)
            
            elif "Missing determiner" in error:
                # Add 'the' before nouns that need it
                fixed = re.sub(r'\b([A-Z][a-z]+)\b', r'the \1', fixed, count=1)
        
        return fixed
    
    def generate_batch(self,
                       num_prompts: int,
                       domains: Optional[List[DomainType]] = None,
                       context: Optional[GenerationContext] = None,
                       complexity_range: Tuple[float, float] = (0.5, 3.0),
                       ensure_diversity: bool = True) -> List[str]:
        """Generate a batch of diverse prompts."""
        prompts = []
        
        for i in range(num_prompts):
            # Vary parameters for diversity
            if domains:
                domain = random.choice(domains)
            else:
                domain = random.choice(list(DomainType)) if random.random() < 0.5 else None
            
            complexity = random.uniform(*complexity_range)
            
            # Select varied template types
            num_templates = random.randint(1, 2)
            template_types = random.sample(list(TemplateType), num_templates)
            
            prompt = self.generate_prompt(
                template_types=template_types,
                domain=domain,
                context=context,
                complexity=complexity
            )
            
            prompts.append(prompt)
        
        # Filter redundant if ensuring diversity
        if ensure_diversity:
            prompts = self.quality_controller.filter_redundant(prompts)
            
            # Generate more if needed
            while len(prompts) < num_prompts:
                additional = self.generate_batch(
                    num_prompts - len(prompts),
                    domains,
                    context,
                    complexity_range,
                    ensure_diversity=False
                )
                prompts.extend(additional)
                prompts = self.quality_controller.filter_redundant(prompts)
        
        return prompts[:num_prompts]
    
    def generate_conversation_sequence(self,
                                      topic: str,
                                      num_turns: int = 5,
                                      domain: Optional[DomainType] = None) -> List[str]:
        """Generate a coherent conversation sequence."""
        context = GenerationContext()
        
        if domain:
            context.domain_focus = domain
        
        # Generate conversation
        conversation = self.context_generator.generate_conversation(
            topic, context, num_turns
        )
        
        # Add domain-specific elements if specified
        if domain:
            conversation = [
                self.domain_synthesizer.inject_jargon(turn, domain, density=0.05)
                for turn in conversation
            ]
        
        return conversation
    
    def generate_edge_cases(self,
                           domain: DomainType,
                           num_cases: int = 5) -> List[str]:
        """Generate edge case prompts for a domain."""
        edge_cases = []
        
        for _ in range(num_cases):
            edge_case = self.domain_synthesizer.generate_edge_case(domain)
            
            # Add complexity
            edge_case = self.context_generator.scale_difficulty(edge_case, 3.5)
            
            edge_cases.append(edge_case)
        
        return edge_cases