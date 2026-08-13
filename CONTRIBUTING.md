# Contributing to VitalSense (Your Personal Health Insights Agent) ❖

Thank you for considering contributing to VitalSense! This document provides guidelines and instructions to help you get started.

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior by opening an issue on the repository.

## Getting Started

### Prerequisites

*   Python 3.10+
*   Git
*   Supabase Account (for database/auth)
*   Google Gemini API Key (or other supported LLM)
*   Streamlit Knowledge

### Local Development Setup

1.  **Fork the Repository:** Click the "Fork" button at the top right of this page to create your own copy.
2.  **Clone your Fork:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/VitalSense.git
    cd VitalSense
    ```
3.  **Add Upstream Remote:**
    ```bash
    git remote add upstream https://github.com/GitSamarth/VitalSense.git
    ```
4.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Set up Environment Variables:**
    *   Create a `.streamlit/secrets.toml` file based on the provided template in the README.
    *   Fill in your Supabase and API credentials.
7.  **Run the Application:**
    ```bash
    streamlit run src/main.py
    ```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue using the bug report template (if available) or providing the following information:

*   **Description:** A clear and concise description of what the bug is.
*   **Steps to Reproduce:** Exactly how to trigger the bug.
*   **Expected Behavior:** What you expected to happen.
*   **Actual Behavior:** What actually happened.
*   **Environment:** OS, Python version, Streamlit version, Browser.
*   **Screenshots:** If applicable.

### Suggesting Enhancements

We welcome ideas for new features or improvements! Create an issue and provide:

*   **Description:** A clear description of the proposed enhancement.
*   **Motivation:** Why this would be useful.
*   **Possible Implementation (Optional):** How you think it could be built.

### Submitting Pull Requests

1.  **Ensure there is an issue:** Before starting work, make sure there is an open issue discussing the change. This prevents wasted effort.
2.  **Create a Branch:**
    ```bash
    git checkout -b feature/your-feature-name
    # OR
    git checkout -b fix/your-bug-fix-name
    ```
3.  **Make your Changes:** Follow the coding standards outlined below.
4.  **Test your Changes:** Ensure the application runs locally and your changes don't break existing functionality. (Add unit tests if applicable).
5.  **Commit your Changes:** Write clear, concise commit messages.
    ```bash
    git commit -m "feat: add user profile page" 
    ```
    *(Consider using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/))*
6.  **Push to your Fork:**
    ```bash
    git push origin your-branch-name
    ```
7.  **Open a Pull Request:** Go to the original repository and click "Compare & pull request". Provide a detailed description of your changes and link the relevant issue.

## Coding Standards

*   **PEP 8:** Follow PEP 8 guidelines for Python code.
*   **Docstrings:** Include docstrings for all functions, classes, and modules describing their purpose, arguments, and return values.
*   **Type Hinting:** Use type hints where possible to improve code readability and maintainability.
*   **Modularity:** Keep functions and components small and focused on a single task.
*   **UI Consistency:** Ensure new UI components match the existing styling and layout of the application.
*   **Error Handling:** Implement robust error handling, especially for API calls and database operations.

## Pull Request Review Process

*   Maintainers will review your PR and may request changes.
*   Please be responsive to feedback.
*   Once approved, a maintainer will merge your PR.

## Community

*   Be respectful and considerate of others.
*   Help answer questions from other contributors.

We appreciate your contributions to making VitalSense better! If you have questions, feel free to open an issue or contact the maintainers.