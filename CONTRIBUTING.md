# Contributing to Yakub Promotion Analysis

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- A clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

For feature requests, please:

- Describe the enhancement clearly
- Explain why it would be useful
- Provide examples if possible

### Pull Requests

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Test your changes** (`pytest tests/`)
5. **Commit** with a clear message (`git commit -m 'Add amazing feature'`)
6. **Push** to your fork (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/yakub-promotion-analysis.git
cd yakub-promotion-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small

### Formatting

```bash
# Format code with black
black src/ tests/

# Check style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

## Commit Messages

Use conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Formatting
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance

**Example:**
```
feat(models): Add cross-validation support

Implement k-fold cross-validation for model evaluation.
This provides more robust performance estimates.

Closes #123
```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_processing.py
```

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions
- Update data dictionary if adding new fields

## Questions?

Feel free to open an issue for any questions!

## Code of Conduct

Be respectful and constructive in all interactions.
