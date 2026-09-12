# 🥗 NutriGuide AI

NutriGuide AI is an AI-powered nutrition guidance web application built with **Python, Streamlit, and Groq AI**. It uses a multi-stage AI workflow to provide general, educational nutrition guidance through a simple and user-friendly interface.

## ✨ Features

* 🤖 AI-powered nutrition assessment
* 🛡️ AI safety review
* 🥗 General nutrition guidance
* 🍽️ Meal ideas based on user preferences
* 📊 Results dashboard
* 📄 Downloadable PDF report
* 🔄 Multi-stage AI workflow
* ⚠️ Error handling for API issues
* 🎨 Interactive Streamlit interface
* 💫 Lottie animation

## 🔄 AI Workflow

NutriGuide AI uses three stages:

1. **Assessment Agent**
   Reviews the information provided by the user.

2. **Safety Agent**
   Reviews the assessment and checks the planned guidance for safety.

3. **Nutrition Guidance Agent**
   Uses the previous stages as context to generate general nutrition guidance and meal ideas.

This workflow allows information to pass from one stage to the next.

## 🛠️ Technologies

* **Python**
* **Streamlit**
* **Groq AI**
* **ReportLab**
* **Requests**
* **Streamlit-Lottie**
* **python-dotenv**

## 📁 Project Structure

```text
NutriGuide-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

> `.env` should not be uploaded to GitHub because it contains your API key.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/habibaakbar/NutriGuide-AI.git
```

Open the project folder:

```bash
cd NutriGuide-AI
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 🔑 API Key Setup

Create a `.env` file in the project folder:

```env
GROQ_API_KEY=your_groq_api_key
```

Never share or commit your API key.

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🚀 Deployment

The application can be deployed using **Streamlit Community Cloud**.

Add the following secret in the Streamlit app settings:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

## 📊 Application Flow

```text
User
  ↓
Nutrition Assessment
  ↓
Assessment Agent
  ↓
Safety Agent
  ↓
Nutrition Guidance Agent
  ↓
Results Dashboard
  ↓
Download PDF Report
```

## 🔐 Safety & Disclaimer

NutriGuide AI provides **general educational nutrition information**. It does not diagnose medical conditions, prescribe treatment, or replace advice from a qualified healthcare or nutrition professional.

Users with specific health concerns, allergies, or other individual needs should seek appropriate professional guidance.

## 🎯 Project Goal

The goal of NutriGuide AI is to demonstrate how an AI-powered application can use a **multi-stage workflow, context passing, safety checking, and automated report generation** to provide useful general nutrition education.

## 👩‍💻 Author

**Habiba Akbar**

GitHub:
https://github.com/habibaakbar

---

⭐ If you find this project useful, consider giving it a star on GitHub.