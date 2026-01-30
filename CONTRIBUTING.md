# Contributing to Multimodal Vision-Language Reasoning System

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to the Multimodal Vision-Language Reasoning System. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## How Can I Contribute?

### Reporting Bugs
*   **Check the existing issues** to see if the bug has already been reported.
*   **Provide a clear, descriptive title** for the issue.
*   **Describe the exact steps to reproduce the bug.**
*   **Include screenshots or videos** if applicable.
*   **Provide information about your environment** (OS, GPU, CUDA version, Python version).

### Suggesting Enhancements
*   **Check the existing issues** to see if the enhancement has already been suggested.
*   **Explain why this enhancement would be useful** to most users.

### Pull Requests
1.  **Fork the repository** and create your branch from `main`.
2.  **Ensure your code follows the existing style.**
3.  **Update documentation** if you are adding new features or changing behavior.
4.  **Run existing tests** to ensure no regressions.
5.  **Submit a pull request** with a comprehensive description of your changes.

## Directory Structure Overview
-   `src/core/`: Main entry point, configuration, and shared state.
-   `src/vision/`: Computer vision pipelines (YOLO, CLIP, FAISS).
-   `src/reasoning/`: High-level AI reasoning (LLaVA).
-   `src/audio/`: Voice interface and transcription.
-   `src/utils/`: Maintenance and setup utilities.

## Setup for Development
1.  Clone your fork.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Install development dependencies (if any).
4.  Follow [`docs/INSTALL_REASONING.md`](./docs/INSTALL_REASONING.md) to set up model weights.

## Code Style
*   Use PEP 8 for Python code.
*   Use descriptive variable and function names.
*   Include docstrings for public classes and functions.

## Questions?
Feel free to open an issue for any questions or reach out to [Shubham Kambli](mailto:shubhamkambli1112@gmail.com).

Happy coding! 🚀
