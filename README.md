<div align="center">
  <h1>❖ VitalSense</h1>
  <p>Your Personal AI Health Assistant</p>

  <a href="https://github.com/GitSamarth/VitalSense/issues"><img src="https://img.shields.io/github/issues/GitSamarth/VitalSense"></a> 
  <a href="https://github.com/GitSamarth/VitalSense/stargazers"><img src="https://img.shields.io/github/stars/GitSamarth/VitalSense"></a>
  <a href="https://github.com/GitSamarth/VitalSense/network/members"><img src="https://img.shields.io/github/forks/GitSamarth/VitalSense"></a>
  <a href="https://github.com/GitSamarth/VitalSense/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
</div>

## 🌟 What is VitalSense?

**VitalSense** is a modern, AI-powered health assistant application designed to securely manage, analyze, and gain insights from your medical reports. It leverages advanced Retrieval-Augmented Generation (RAG) capabilities to allow users to intuitively chat with and extract insights from their health documents.

## 🚀 Current Features

*   **Secure Authentication:** User login and signup powered by Supabase.
*   **Medical Report Analysis:** Upload your medical reports (PDFs) and extract structured insights instantly.
*   **Interactive RAG Chat:** Chat directly with your reports to ask follow-up questions, understand complex medical jargon, and track health trends.
*   **Modern UI/UX:** A clean, visually consistent interface with a dark Navy/Teal aesthetic.
*   **Database Management:** Structured storage for user sessions, chats, and uploaded reports.

## 🛠️ Technologies Used

*   **Frontend & Application Logic:** [Streamlit](https://streamlit.io/)
*   **Authentication & Database:** [Supabase](https://supabase.com/)
*   **LLM & RAG Engine:** Gemini / Supported AI APIs
*   **Styling:** Custom Streamlit CSS & Theming

## 🏗️ Project Structure

```text
VitalSense/
├── .streamlit/             # Streamlit configuration and themes
├── public/                 # Static assets
├── src/                    # Source code
│   ├── auth/               # Authentication logic and session management
│   ├── components/         # Reusable UI components (Sidebar, Footer, Analysis Forms)
│   ├── config/             # App configurations and constants
│   ├── services/           # External API integrations (AI, RAG logic)
│   ├── utils/              # Helper functions and validators
│   └── main.py             # Application entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## ⚙️ Local Development Setup

To run VitalSense locally, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/GitSamarth/VitalSense.git
cd VitalSense
```

### 2. Install dependencies
It is highly recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Configure Secrets
Create a `.streamlit/secrets.toml` file with your credentials:
```toml
SUPABASE_URL = "your-supabase-url"
SUPABASE_KEY = "your-supabase-key"
# Add your specific LLM API key here
```

### 4. Run the Application
```bash
streamlit run src/main.py
```

## 📈 Project Evolution

### Current Version (VitalSense)
- Rebranded to **VitalSense** from its predecessor.
- Retained and optimized existing RAG functionality.
- Updated UI/UX with a new dark-mode visual theme (Navy & Teal).
- Removed unused assets and streamlined the repository.
- Improved code consistency and updated documentation.

### Planned Development
I will continue extending VitalSense with additional AI-engineering capabilities in future development stages. **Note: These advanced capabilities are currently in planning and are not yet part of the codebase.** Future updates will include advanced agentic health analysis, enhanced data visualization, and improved longitudinal health tracking.

---

## 🤝 Contributing
Contributions are always welcome! Please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
