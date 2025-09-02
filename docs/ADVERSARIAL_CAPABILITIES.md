# Advanced Adversarial Prompt Generation Capabilities

## ⚠️ IMPORTANT DISCLAIMER

**These capabilities are for authorized security research and model robustness testing ONLY.**

The adversarial generation methods implemented in `src/challenges/kdf_prompts.py` are designed to help researchers:
- Test model safety boundaries
- Identify potential vulnerabilities
- Improve model robustness
- Develop better safety mechanisms

**DO NOT use these techniques for:**
- Unauthorized data extraction
- Bypassing safety measures in production systems
- Extracting personally identifiable information
- Any malicious or harmful purposes

## 🛡️ Safety Controls

All high-risk adversarial prompts include:
1. **Safety wrappers** - Dangerous prompts are wrapped with clear research context
2. **Warning labels** - Each prompt includes safety notes
3. **Research-only flags** - Metadata marking prompts for research use
4. **Opt-in dangerous generation** - High-risk prompts require explicit `include_dangerous=True`

## 📊 Implemented Attack Strategies

### 1. Advanced Jailbreak Techniques

#### Divergence Attack (150x faster extraction)
- **Success Rate**: 150x faster than baseline methods
- **Attack Vector**: Memorization extraction through prefix divergence
- **Method**: `generate_divergence_attack()`
- **Research Basis**: Exploits model tendency to complete memorized sequences

#### Multi-Round Conversational Jailbreaking (MRCJ)
- **Success Rate**: >90% on tested models
- **Attack Vector**: Trust building over multiple conversation turns
- **Method**: `generate_multi_round_conversational_jailbreak()`
- **Strategy**: 
  - Round 1: Establish benign research context
  - Round 2: Gradual boundary testing
  - Round 3: Actual jailbreak attempt

#### Special Character Triggers
- **Attack Vector**: Unicode exploitation and control characters
- **Method**: `generate_special_character_triggers()`
- **Techniques**:
  - Null bytes and zero-width spaces
  - RTL overrides and ANSI escape codes
  - Invalid UTF-8 sequences
  - Format string injections

#### Temperature Exploitation
- **Attack Vector**: Sampling parameter manipulation
- **Method**: `generate_temperature_exploitation()`
- **Exploits**: Temperature, top_p, top_k parameter overrides

### 2. Model Inversion Attacks

#### Two-Stage Inversion
- **Success Rate**: 38-75% on personalized LLMs
- **Attack Vector**: Representation projection + text recovery
- **Method**: `generate_two_stage_inversion_attack()`
- **Targets**: PII extraction from fine-tuned models

#### Cross-Lingual Inversion
- **Attack Vector**: Language switching to bypass safety
- **Method**: `generate_cross_lingual_inversion()`
- **Languages**: English, Spanish, Chinese, Arabic, Russian, German

### 3. Membership Inference Probes

#### SPV-MIA (Self-calibrated Probabilistic Variation)
- **Attack Vector**: Training data membership detection
- **Method**: `generate_spv_mia_probe()`
- **Technique**: Confidence calibration for membership inference

#### Dataset Extraction
- **Success Rate**: >50% dataset recovery potential
- **Attack Vector**: Targeted queries for training data
- **Method**: `generate_dataset_extraction_probe()`
- **Targets**: Common dataset markers and formats

### 4. Safety Mechanism Analysis

#### Alignment Faking Detection
- **Purpose**: Detect when models pretend to be more aligned
- **Method**: `generate_alignment_faking_detector()`
- **Tests**: Preference conflicts and authenticity

#### PAIR Algorithm Jailbreaks
- **Attack Vector**: Automatic iterative refinement
- **Method**: `generate_pair_algorithm_jailbreak()`
- **Strategy**: Progressive obfuscation and refinement

#### Deception Pattern Detection
- **Purpose**: Reveal inconsistencies in model responses
- **Method**: `generate_deception_pattern_detector()`
- **Tests**: Honesty and response authenticity

## 🔧 Usage Examples

### Basic Usage (Safe Prompts Only)

```python
from src.challenges.kdf_prompts import KDFPromptGenerator
import os

generator = KDFPromptGenerator(prf_key=os.urandom(32))

# Generate safe adversarial suite
safe_suite = generator.generate_comprehensive_adversarial_suite(
    base_index=0,
    include_dangerous=False  # Only safe prompts
)

# Individual safe tests
alignment_test = generator.generate_alignment_faking_detector(index=1)
deception_test = generator.generate_deception_pattern_detector(index=2)
```

### Advanced Usage (Research Context)

```python
# For authorized security research only
# Requires explicit opt-in for dangerous prompts

# Generate full suite with safety wrappers
full_suite = generator.generate_comprehensive_adversarial_suite(
    base_index=100,
    include_dangerous=True  # Includes high-risk prompts with safety wrappers
)

# Each dangerous prompt includes:
# - Safety notice wrapper
# - Research-only flag
# - Attack vector metadata
# - Expected behavior documentation
```

### Integration with REV Pipeline

```python
# Generate adversarial challenges for REV verification
challenges = generator.generate_challenge_set(
    n_challenges=100,
    adversarial_ratio=0.2,  # 20% adversarial prompts
    behavioral_probe_ratio=0.15
)

# Export for REV integration
rev_challenges = generator.export_for_integration(challenges)
```

## 📈 Performance Metrics

| Attack Type | Success Rate | Research Value | Risk Level |
|-------------|--------------|----------------|------------|
| Divergence Attack | 150x faster | High | High |
| MRCJ | >90% | High | High |
| Two-Stage Inversion | 38-75% | High | High |
| SPV-MIA | Variable | Medium | Medium |
| Dataset Extraction | >50% | High | High |
| Alignment Faking | N/A | High | Low |
| Temperature Exploit | Variable | Medium | Medium |
| Cross-Lingual | Variable | Medium | Medium |

## 🔒 Ethical Guidelines

1. **Authorization Required**: Only use on systems you own or have explicit permission to test
2. **Research Context**: Always operate within research ethics guidelines
3. **Documentation**: Keep detailed logs of all testing activities
4. **Responsible Disclosure**: Report vulnerabilities through appropriate channels
5. **No Malicious Use**: Never use these techniques to harm or exploit
6. **Safety First**: Always use safety wrappers when testing dangerous prompts
7. **User Protection**: Never target real user data or PII

## 🔗 References

- Divergence Attacks: Based on research showing prefix-based memorization extraction
- MRCJ: Multi-round conversational jailbreaking research
- SPV-MIA: Self-calibrated probabilistic variation for membership inference
- PAIR Algorithm: Prompt Automatic Iterative Refinement techniques
- Two-Stage Inversion: Representation projection and text recovery methods

## 🚨 Security Notice

This module implements advanced adversarial techniques that could potentially:
- Extract training data from models
- Bypass safety mechanisms
- Reveal model internals
- Identify fine-tuning datasets

**Use responsibly and ethically for improving AI safety only.**

---

*Last Updated: September 2025*
*Version: 1.0*
*Module: src/challenges/kdf_prompts.py*