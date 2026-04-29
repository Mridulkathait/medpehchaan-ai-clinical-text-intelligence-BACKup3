# Contributing to MedPehchaan AI+

Thank you for your interest in contributing to MedPehchaan AI+! This document provides guidelines and information for contributors.

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Use the [GitHub Issues](https://github.com/Mridulkathait/medpehchaan-ai-clinical-text-intelligence-BACKup3/issues) page
- Provide detailed steps to reproduce the issue
- Include error messages, screenshots, and system information
- Check if the issue already exists before creating a new one

### ✨ Feature Requests
- Open a [GitHub Issue](https://github.com/Mridulkathait/medpehchaan-ai-clinical-text-intelligence-BACKup3/issues) with the "enhancement" label
- Describe the proposed feature and its benefits
- Consider how it fits with the project's goals

### 🔧 Code Contributions
- Fork the repository
- Create a feature branch from `main`
- Make your changes
- Ensure tests pass
- Submit a pull request

## 🚀 Development Setup

### Prerequisites
- Python 3.8+
- Git
- pip

### Local Development

1. **Clone and setup:**
   ```bash
   git clone https://github.com/Mridulkathait/medpehchaan-ai-clinical-text-intelligence-BACKup3.git
   cd medpehchaan-ai-clinical-text-intelligence-BACKup3
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Run tests:**
   ```bash
   python demo.py  # Basic functionality test
   ```

## 📝 Coding Standards

### Python Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused on single responsibilities

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Reference issue numbers when applicable
- Example: `feat: Add support for PDF file processing (#123)`

### Documentation
- Update README.md for significant changes
- Add docstrings to new functions
- Update comments for complex logic

## 🧪 Testing

### Manual Testing
- Test with various input formats (text, CSV, PDF)
- Verify output accuracy
- Check error handling
- Test edge cases

### Automated Testing
- Run the demo script to verify basic functionality
- Check that all imports work correctly
- Verify that the Streamlit app starts without errors

## 🔒 Security Considerations

### Medical Data Handling
- Never commit real patient data
- Use anonymized sample data only
- Be aware of HIPAA and privacy regulations
- This is educational software - not for clinical use

### API Keys and Secrets
- Never commit API keys or credentials
- Use environment variables for sensitive data
- Document required environment variables

## 📋 Pull Request Process

1. **Create a Fork**: Fork the repository to your GitHub account
2. **Create a Branch**: `git checkout -b feature/your-feature-name`
3. **Make Changes**: Implement your feature or fix
4. **Test Thoroughly**: Ensure everything works as expected
5. **Update Documentation**: Update README and docstrings as needed
6. **Commit Changes**: `git commit -m "feat: Add your feature description"`
7. **Push to Branch**: `git push origin feature/your-feature-name`
8. **Create PR**: Open a pull request with a clear description

### PR Requirements
- Clear description of changes
- Reference any related issues
- Include screenshots for UI changes
- Ensure CI/CD checks pass
- Request review from maintainers

## 🎓 Educational Context

This is a **class project** for learning purposes. Contributions should:
- Enhance educational value
- Maintain code quality standards
- Follow best practices
- Include proper documentation

## 📞 Getting Help

- **Issues**: [GitHub Issues](https://github.com/Mridulkathait/medpehchaan-ai-clinical-text-intelligence-BACKup3/issues)
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check README.md and inline code comments

## 🙏 Recognition

Contributors will be acknowledged in the project documentation. Thank you for helping improve MedPehchaan AI+!

---

**Remember**: This project is for educational purposes. Always consult healthcare professionals for medical decisions.