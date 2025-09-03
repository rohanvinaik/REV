# Contributing to REV

Thank you for your interest in contributing to the REV (Restriction Enzyme Verification) System! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our Code of Conduct:

- **Be respectful**: Treat everyone with respect. No harassment, discrimination, or inappropriate behavior.
- **Be collaborative**: Work together to solve problems and improve the project.
- **Be constructive**: Provide helpful feedback and accept criticism gracefully.
- **Be responsible**: Take ownership of your contributions and their impact.

## Getting Started

### Prerequisites

- Python 3.9+ (3.10+ recommended)
- Git and Git LFS
- 16GB+ RAM (64GB+ for large model testing)
- CUDA 12.1+ (optional, for GPU acceleration)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/REV.git
cd REV
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/rohanvinaik/REV.git
```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Development Dependencies

```bash
# Install all dependencies including dev tools
pip install -r requirements.txt -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Installation

```bash
# Run tests
pytest tests/

# Check code style
black --check src/
flake8 src/
mypy src/
```

## How to Contribute

### Types of Contributions

#### 🐛 Bug Reports
- Use GitHub Issues to report bugs
- Include detailed reproduction steps
- Provide system information (OS, Python version, GPU)
- Attach relevant logs and error messages

#### ✨ Feature Requests
- Open a GitHub Issue with the "enhancement" label
- Describe the use case and benefits
- Discuss implementation approach if possible

#### 📝 Documentation
- Fix typos, improve clarity, add examples
- Update docstrings and API documentation
- Create tutorials and guides

#### 🔧 Code Contributions
- Fix bugs from the issue tracker
- Implement approved feature requests
- Optimize performance
- Add tests

### Finding Issues to Work On

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - Community help needed
- `bug` - Bug fixes
- `enhancement` - New features
- `documentation` - Documentation improvements

## Pull Request Process

### 1. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, readable code
- Follow the coding standards
- Add/update tests as needed
- Update documentation

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add unified fingerprint comparison

- Implement Hamming distance optimization
- Add Jaccard similarity metric
- Update tests and documentation"
```

#### Commit Message Format

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_integration.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### 5. Update Documentation

- Update CLAUDE.md if changing user-facing features
- Update API_REFERENCE.md for API changes
- Add docstrings for new functions/classes

### 6. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Link to related issues
- Description of what changed and why
- Screenshots/examples if applicable
- Test results

## Coding Standards

### Python Style Guide

We follow PEP 8 with these specifications:

```python
# Imports - grouped and sorted
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.core.sequential import SequentialTest
from src.hdc.encoder import HDCEncoder


# Class definitions
class ExampleClass:
    """
    Brief description of the class.
    
    Longer description explaining purpose and usage.
    
    Attributes:
        attribute1: Description of attribute1
        attribute2: Description of attribute2
    """
    
    def __init__(self, param1: str, param2: int = 10):
        """
        Initialize ExampleClass.
        
        Args:
            param1: Description of param1
            param2: Description of param2 (default: 10)
        """
        self.param1 = param1
        self.param2 = param2
    
    def method_example(self, arg: np.ndarray) -> Dict[str, float]:
        """
        Brief method description.
        
        Args:
            arg: Input array description
            
        Returns:
            Dictionary containing:
                - 'key1': Description of value1
                - 'key2': Description of value2
                
        Raises:
            ValueError: If arg is empty
            
        Example:
            >>> obj = ExampleClass("test")
            >>> result = obj.method_example(np.array([1, 2, 3]))
            >>> print(result['key1'])
        """
        if len(arg) == 0:
            raise ValueError("Input array cannot be empty")
        
        # Implementation
        return {"key1": float(np.mean(arg)), "key2": float(np.std(arg))}
```

### Code Quality Tools

Configure your editor to use:

```bash
# Formatting
black src/ --line-length 100

# Linting
flake8 src/ --max-line-length 100 --ignore E203,W503

# Type checking
mypy src/ --ignore-missing-imports

# Import sorting
isort src/ --profile black

# Security scanning
bandit -r src/
```

### Performance Guidelines

- Use NumPy operations instead of loops when possible
- Profile code for bottlenecks: `python -m cProfile`
- Minimize memory allocations in hot paths
- Use generators for large data streams
- Cache expensive computations

## Testing Guidelines

### Test Structure

```python
# tests/test_example.py
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.module.example import ExampleClass


class TestExampleClass:
    """Test suite for ExampleClass."""
    
    @pytest.fixture
    def example_instance(self):
        """Create example instance for testing."""
        return ExampleClass("test", param2=20)
    
    def test_initialization(self, example_instance):
        """Test proper initialization."""
        assert example_instance.param1 == "test"
        assert example_instance.param2 == 20
    
    def test_method_with_valid_input(self, example_instance):
        """Test method with valid input."""
        result = example_instance.method_example(np.array([1, 2, 3]))
        assert "key1" in result
        assert isinstance(result["key1"], float)
        assert result["key1"] == pytest.approx(2.0)
    
    def test_method_with_invalid_input(self, example_instance):
        """Test method raises error for invalid input."""
        with pytest.raises(ValueError, match="empty"):
            example_instance.method_example(np.array([]))
    
    @patch('src.module.example.external_function')
    def test_with_mock(self, mock_func, example_instance):
        """Test with mocked external dependency."""
        mock_func.return_value = {"mocked": True}
        # Test implementation
```

### Test Categories

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical paths
- **Regression Tests**: Prevent bug reoccurrence

### Running Tests

```bash
# All tests
make test

# Specific category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v

# With markers
pytest -m "not slow"  # Skip slow tests
pytest -m "gpu"        # Only GPU tests
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: Optional[int] = None) -> Dict[str, Any]:
    """
    Brief description of function.
    
    Longer description if needed, explaining the purpose,
    algorithm, or important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to None.
        
    Returns:
        Dictionary containing:
            - 'result': The computation result
            - 'metadata': Additional information
            
    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is not an integer
        
    Example:
        >>> result = function_name("test", param2=42)
        >>> print(result['result'])
        
    Note:
        This function is memory-intensive for large inputs.
        
    See Also:
        related_function: For alternative approach
    """
    pass
```

### Documentation Updates

When adding new features:

1. Update relevant `.md` files in `docs/`
2. Add/update docstrings
3. Include usage examples
4. Update CLAUDE.md if user-facing
5. Add to API_REFERENCE.md if public API

## Community

### Getting Help

- 📧 Email: support@rev-system.ai
- 💬 Discussions: GitHub Discussions
- 🐛 Issues: GitHub Issues
- 📖 Docs: https://rev-system.readthedocs.io

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, community chat
- **Pull Requests**: Code contributions, reviews

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Documentation credits

## Review Process

### PR Review Checklist

Reviewers will check:

- [ ] Code follows style guidelines
- [ ] Tests pass and coverage maintained
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Breaking changes noted
- [ ] Commit messages clear

### Review Timeline

- Small PRs: 1-2 days
- Medium PRs: 3-5 days
- Large PRs: 1 week+

## Advanced Contributing

### Adding New Model Support

1. Create adapter in `src/models/adapters/`
2. Register in model factory
3. Add tests
4. Update compatibility matrix

### Adding New Prompt System

1. Implement in `src/orchestration/`
2. Register with orchestrator
3. Define weight and strategy
4. Add tests and documentation

### Performance Optimization

1. Profile with `cProfile` or `line_profiler`
2. Identify bottlenecks
3. Optimize with NumPy/PyTorch operations
4. Benchmark improvements
5. Document changes

## Release Process

### Version Numbering

We use Semantic Versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

1. Update version in `__version__.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Build documentation
5. Create git tag
6. Push to PyPI

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing to REV! Your efforts help make model verification accessible and reliable for everyone. 🚀