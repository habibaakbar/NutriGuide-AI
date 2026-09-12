import os
import json
from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

import streamlit as st
import requests
from streamlit_lottie import st_lottie
from dotenv import load_dotenv
from groq import Groq

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NutriGuide AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

try:
    API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    API_KEY = None

if not API_KEY:
    API_KEY = os.getenv("GROQ_API_KEY")

if API_KEY:
    API_KEY = API_KEY.strip()


# ============================================================
# API CONFIGURATION
# ============================================================

if not API_KEY:
    st.error("🔑 Groq API key is missing.")

    st.info(
        "For Streamlit Cloud, add GROQ_API_KEY under "
        "Settings → Secrets. For local development, "
        "add it to your .env file."
    )

    st.stop()


client = Groq(api_key=API_KEY)

MODEL_NAME = "llama3-8b-8192"


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "user_data" not in st.session_state:
    st.session_state.user_data = {}

if "assessment" not in st.session_state:
    st.session_state.assessment = {}

if "safety_result" not in st.session_state:
    st.session_state.safety_result = {}

if "guidance" not in st.session_state:
    st.session_state.guidance = {}

if "workflow_context" not in st.session_state:
    st.session_state.workflow_context = {}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 1rem;
    }

    .hero-text {
        font-size: 1.15rem;
        line-height: 1.7;
        color: #475569;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 2rem;
        font-weight: 750;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        height: 100%;
    }

    .card h3 {
        margin-top: 0;
    }

    .workflow-card {
        background: white;
        padding: 1.3rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        min-height: 180px;
    }

    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .status-safe {
        padding: 1rem;
        border-radius: 12px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
    }

    .status-warning {
        padding: 1rem;
        border-radius: 12px;
        background: #fffbeb;
        border: 1px solid #fde68a;
    }

    .small-text {
        color: #64748b;
        font-size: 0.9rem;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOTTIE LOADER
# ============================================================

@st.cache_data
def load_lottieurl(url: str):
    """
    Safely load a Lottie animation from a URL.
    """

    if not url:
        return None

    if not url.startswith(("http://", "https://")):
        return None

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        return response.json()

    except (requests.RequestException, ValueError):
        return None


# ============================================================
# JSON CLEANER
# ============================================================

def clean_json_response(text):
    """
    Clean AI response and convert it into JSON.
    """

    if not text:
        return {}

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return {}

    return {}


# ============================================================
# AI RESPONSE GENERATOR
# ============================================================

def generate_ai_response(prompt, temperature=0.3):
    """
    Send a prompt to Groq and return the AI response.
    """

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are NutriGuide AI, a safe educational "
                        "nutrition guidance assistant. "
                        "Provide general educational information only. "
                        "Do not diagnose medical conditions. "
                        "Do not prescribe treatment. "
                        "Do not provide restrictive eating plans, "
                        "calorie targets, or weight-loss instructions. "
                        "Be especially careful with young users."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
            max_tokens=1800,
        )

        return response.choices[0].message.content

    except Exception as e:

        error_message = str(e).lower()

        if "rate" in error_message or "quota" in error_message:
            st.error(
                "⚠️ The AI service has reached its usage limit. "
                "Please try again later."
            )

        elif "api" in error_message or "authentication" in error_message:
            st.error(
                "⚠️ There is a problem with the AI API configuration."
            )

        else:
            st.error(
                "⚠️ Something went wrong while generating the AI response."
            )

        return ""


# ============================================================
# DISPLAY LIST ITEMS
# ============================================================

def display_list_items(items):

    if not items:
        st.write("No information available.")

        return

    if isinstance(items, str):

        st.write(items)

        return

    for item in items:

        if isinstance(item, dict):

            title = item.get("title", "")
            description = item.get("description", "")

            if title:
                st.markdown(f"**{title}**")

            if description:
                st.write(description)

        else:

            st.markdown(f"• {item}")


# ============================================================
# BOOLEAN HELPER
# ============================================================

def get_bool(value):

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        return value.lower() in [
            "yes",
            "true",
            "1",
            "y",
        ]

    return bool(value)


# ============================================================
# PDF TEXT HELPER
# ============================================================

def pdf_text(value):

    if value is None:
        return ""

    if isinstance(value, list):

        value = "\n".join(str(x) for x in value)

    elif isinstance(value, dict):

        value = json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )

    value = str(value)

    return escape(value).replace("\n", "<br/>")


# ============================================================
# CREATE PDF REPORT
# ============================================================

def create_pdf_report(
    user_data,
    assessment,
    safety_result,
    guidance,
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=15,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=6,
    )

    story = []

    story.append(
        Paragraph(
            "NutriGuide AI",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Nutrition Guidance Report",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            body_style,
        )
    )

    story.append(Spacer(1, 10))

    # --------------------------------------------------------
    # USER INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. User Information",
            heading_style,
        )
    )

    user_rows = [
        [
            Paragraph("<b>Field</b>", body_style),
            Paragraph("<b>Information</b>", body_style),
        ],
        [
            "Age",
            pdf_text(user_data.get("age", "Not provided")),
        ],
        [
            "Activity Level",
            pdf_text(user_data.get("activity_level", "Not provided")),
        ],
        [
            "Diet Preference",
            pdf_text(user_data.get("diet_preference", "Not provided")),
        ],
        [
            "Allergies",
            pdf_text(user_data.get("allergies", "None reported")),
        ],
    ]

    user_table = Table(
        user_rows,
        colWidths=[150, 330],
    )

    user_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(user_table)

    # --------------------------------------------------------
    # ASSESSMENT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. AI Assessment",
            heading_style,
        )
    )

    assessment_summary = assessment.get(
        "summary",
        assessment.get(
            "assessment",
            "No assessment summary available.",
        ),
    )

    story.append(
        Paragraph(
            pdf_text(assessment_summary),
            body_style,
        )
    )

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Safety Review",
            heading_style,
        )
    )

    safety_status = safety_result.get(
        "status",
        "Reviewed",
    )

    safety_message = safety_result.get(
        "message",
        safety_result.get(
            "reason",
            "General educational guidance is provided.",
        ),
    )

    story.append(
        Paragraph(
            f"<b>Status:</b> {pdf_text(safety_status)}",
            body_style,
        )
    )

    story.append(
        Paragraph(
            pdf_text(safety_message),
            body_style,
        )
    )

    # --------------------------------------------------------
    # GUIDANCE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. General Nutrition Guidance",
            heading_style,
        )
    )

    tips = guidance.get("tips", [])

    if tips:

        for tip in tips:

            story.append(
                Paragraph(
                    f"• {pdf_text(tip)}",
                    body_style,
                )
            )

    else:

        general_guidance = guidance.get(
            "guidance",
            guidance.get(
                "summary",
                "No additional guidance available.",
            ),
        )

        story.append(
            Paragraph(
                pdf_text(general_guidance),
                body_style,
            )
        )

    # --------------------------------------------------------
    # MEAL IDEAS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. Meal Ideas",
            heading_style,
        )
    )

    meal_ideas = guidance.get(
        "meal_ideas",
        guidance.get(
            "meals",
            [],
        ),
    )

    if meal_ideas:

        for meal in meal_ideas:

            if isinstance(meal, dict):

                meal_name = meal.get(
                    "name",
                    "Meal idea",
                )

                meal_description = meal.get(
                    "description",
                    "",
                )

                story.append(
                    Paragraph(
                        f"<b>{pdf_text(meal_name)}</b>: "
                        f"{pdf_text(meal_description)}",
                        body_style,
                    )
                )

            else:

                story.append(
                    Paragraph(
                        f"• {pdf_text(meal)}",
                        body_style,
                    )
                )

    else:

        story.append(
            Paragraph(
                "No meal ideas available.",
                body_style,
            )
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> NutriGuide AI provides general "
            "educational nutrition information. It is not a "
            "replacement for advice from a qualified healthcare "
            "or nutrition professional.",
            body_style,
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# RESET APP
# ============================================================

def reset_app():

    st.session_state.page = "home"

    st.session_state.user_data = {}

    st.session_state.assessment = {}

    st.session_state.safety_result = {}

    st.session_state.guidance = {}

    st.session_state.workflow_context = {}


# ============================================================
# HOME PAGE
# ============================================================

def show_home():

    st.markdown(
        '<div class="hero-title">🥗 NutriGuide AI</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(
        [1.4, 1],
        gap="large",
    )

    with col1:

        st.markdown(
            """
            <div class="hero-text">
            NutriGuide AI is an AI-powered educational nutrition
            assistant that provides general nutrition guidance
            through a simple multi-stage AI workflow.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "🚀 Start Nutrition Assessment",
            type="primary",
            use_container_width=True,
        ):

            st.session_state.page = "assessment"

            st.rerun()

    with col2:

        # IMPORTANT:
        # This must be a plain URL, NOT a Markdown link.

        lottie_url = (
            "https://lottie.host/"
            "020bd926-2a7f-4bba-9577-fb1777265a7f/"
            "p1yWpY1j7c.json"
        )

        lottie_anim = load_lottieurl(lottie_url)

        if lottie_anim:

            st_lottie(
                lottie_anim,
                height=250,
                key="food_animation",
            )

        else:

            st.markdown(
                """
                <div style="
                    text-align:center;
                    font-size:100px;
                    padding:40px;
                ">
                🥗
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # FEATURES
    # ========================================================

    st.markdown(
        '<div class="section-title">✨ Features</div>',
        unsafe_allow_html=True,
    )

    feature_cols = st.columns(4)

    features = [
        (
            "🤖",
            "AI Assessment",
            "Analyse your general nutrition information "
            "using an AI workflow.",
        ),
        (
            "🛡️",
            "Safety Review",
            "A separate safety stage checks the generated "
            "guidance before it is shown.",
        ),
        (
            "🍎",
            "Meal Ideas",
            "Receive simple, general meal ideas based on "
            "your preferences.",
        ),
        (
            "📄",
            "PDF Report",
            "Download your results as a professional PDF report.",
        ),
    ]

    for column, feature in zip(
        feature_cols,
        features,
    ):

        icon, title, description = feature

        with column:

            st.markdown(
                f"""
                <div class="card">
                    <div style="font-size:2rem;">{icon}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # WORKFLOW
    # ========================================================

    st.markdown(
        '<div class="section-title">🔄 AI Workflow</div>',
        unsafe_allow_html=True,
    )

    workflow_cols = st.columns(3)

    workflow = [
        (
            "1️⃣",
            "Assessment Agent",
            "Reviews the information entered by the user "
            "and creates an assessment summary.",
        ),
        (
            "2️⃣",
            "Safety Agent",
            "Reviews the assessment and checks whether the "
            "planned guidance is appropriate.",
        ),
        (
            "3️⃣",
            "Nutrition Guidance Agent",
            "Uses the previous stages as context and creates "
            "general educational nutrition guidance.",
        ),
    ]

    for column, item in zip(
        workflow_cols,
        workflow,
    ):

        number, title, description = item

        with column:

            st.markdown(
                f"""
                <div class="workflow-card">
                    <div style="font-size:1.8rem;">{number}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
        """
        <div class="footer">
            <b>NutriGuide AI</b><br>
            General educational nutrition guidance only.
            This application does not provide medical diagnosis
            or professional treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ASSESSMENT PAGE
# ============================================================

def show_assessment():

    st.title("📝 Nutrition Assessment")

    st.write(
        "Enter the information below to generate your "
        "personalised general nutrition guidance."
    )

    st.info(
        "This application provides general educational information "
        "and does not replace advice from a qualified professional."
    )

    # ========================================================
    # FORM
    # ========================================================

    with st.form("nutrition_assessment_form"):

        st.subheader("👤 Basic Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=100,
                value=18,
                step=1,
            )

        with col2:

            activity_level = st.selectbox(
                "Activity Level",
                [
                    "Low",
                    "Moderate",
                    "Active",
                    "Very Active",
                ],
            )

        st.subheader("🍽️ Food Preferences")

        diet_preference = st.selectbox(
            "Diet Preference",
            [
                "No specific preference",
                "Vegetarian",
                "Vegan",
                "Halal",
                "Other",
            ],
        )

        allergies = st.text_area(
            "Food Allergies or Intolerances",
            placeholder=(
                "Example: peanuts, milk, eggs, "
                "or write 'None'"
            ),
        )

        food_preferences = st.text_area(
            "Foods You Like",
            placeholder=(
                "Example: rice, vegetables, chicken, "
                "fruit, yoghurt..."
            ),
        )

        food_dislikes = st.text_area(
            "Foods You Dislike",
            placeholder=(
                "Example: spicy foods, certain vegetables..."
            ),
        )

        health_notes = st.text_area(
            "Additional Information",
            placeholder=(
                "Optional. Add general information that "
                "may help provide safer educational guidance."
            ),
        )

        submitted = st.form_submit_button(
            "🤖 Generate Nutrition Guidance",
            type="primary",
            use_container_width=True,
        )

    # ========================================================
    # FORM SUBMITTED
    # ========================================================

    if submitted:

        user_data = {
            "age": age,
            "activity_level": activity_level,
            "diet_preference": diet_preference,
            "allergies": allergies or "None",
            "food_preferences": food_preferences or "Not provided",
            "food_dislikes": food_dislikes or "None",
            "health_notes": health_notes or "None",
        }

        st.session_state.user_data = user_data

        # ====================================================
        # STAGE 1 - ASSESSMENT AGENT
        # ====================================================

        with st.status(
            "🔄 Running AI workflow...",
            expanded=True,
        ) as status:

            st.write("1️⃣ Running Assessment Agent...")

            assessment_prompt = f"""
You are the Assessment Agent in a nutrition education application.

Review the following user information:

{json.dumps(user_data, indent=2)}

Create a concise educational assessment.

Important rules:
- Do not diagnose diseases.
- Do not provide medical treatment.
- Do not recommend restrictive diets.
- Do not give calorie targets.
- Do not provide weight-loss or weight-gain targets.
- Focus on general nutrition habits.
- Respect allergies and food preferences.
- Be age-appropriate.
- If the user appears to be young, use extra caution.

Return ONLY valid JSON.

Use this structure:

{{
    "summary": "short educational assessment",
    "positive_habits": [
        "habit 1",
        "habit 2"
    ],
    "areas_to_focus": [
        "area 1",
        "area 2"
    ]
}}
"""

            assessment_response = generate_ai_response(
                assessment_prompt,
                temperature=0.2,
            )

            assessment = clean_json_response(
                assessment_response
            )

            if not assessment:

                assessment = {
                    "summary": (
                        "The assessment could not be generated. "
                        "Please try again."
                    ),
                    "positive_habits": [],
                    "areas_to_focus": [],
                }

            st.session_state.assessment = assessment

            st.write("✅ Assessment Agent completed.")

            # =================================================
            # STAGE 2 - SAFETY AGENT
            # =================================================

            st.write("2️⃣ Running Safety Agent...")

            safety_prompt = f"""
You are the Safety Agent for NutriGuide AI.

Review the user's information and the assessment.

USER INFORMATION:
{json.dumps(user_data, indent=2)}

ASSESSMENT:
{json.dumps(assessment, indent=2)}

Check the planned educational guidance for safety.

Rules:
- Do not diagnose.
- Do not prescribe medication.
- Do not create restrictive diets.
- Do not recommend calorie restriction.
- Do not provide weight-loss targets.
- Do not provide weight-gain targets.
- Do not encourage skipping meals.
- Do not promote unhealthy eating behaviour.
- Respect allergies.
- Respect the user's age.
- General balanced nutrition education is acceptable.

Return ONLY valid JSON.

Structure:

{{
    "status": "Safe",
    "message": "short safety explanation",
    "important_notes": [
        "note 1",
        "note 2"
    ]
}}
"""

            safety_response = generate_ai_response(
                safety_prompt,
                temperature=0.1,
            )

            safety_result = clean_json_response(
                safety_response
            )

            if not safety_result:

                safety_result = {
                    "status": "Reviewed",
                    "message": (
                        "The guidance should remain general "
                        "and educational."
                    ),
                    "important_notes": [],
                }

            st.session_state.safety_result = safety_result

            st.write("✅ Safety Agent completed.")

            # =================================================
            # STAGE 3 - GUIDANCE AGENT
            # =================================================

            st.write(
                "3️⃣ Running Nutrition Guidance Agent..."
            )

            guidance_prompt = f"""
You are the Nutrition Guidance Agent.

Generate general educational nutrition guidance using
the context produced by the previous AI stages.

USER INFORMATION:
{json.dumps(user_data, indent=2)}

ASSESSMENT:
{json.dumps(assessment, indent=2)}

SAFETY REVIEW:
{json.dumps(safety_result, indent=2)}

Your response must:
- Be educational and practical.
- Encourage balanced meals and variety.
- Respect allergies and dietary preferences.
- Avoid medical diagnosis.
- Avoid medical treatment.
- Avoid restrictive diets.
- Avoid calorie counting.
- Avoid weight-loss targets.
- Avoid weight-gain targets.
- Avoid body-size comparisons.
- Avoid encouraging meal skipping.
- Avoid unhealthy eating behaviour.
- Be suitable for the user's age.
- If there is a potentially serious health concern,
  recommend discussing it with a parent/guardian or
  qualified healthcare professional rather than diagnosing it.

Return ONLY valid JSON.

Structure:

{{
    "summary": "short general nutrition guidance summary",
    "tips": [
        "practical tip 1",
        "practical tip 2",
        "practical tip 3",
        "practical tip 4"
    ],
    "meal_ideas": [
        {{
            "name": "Breakfast",
            "description": "balanced general meal idea"
        }},
        {{
            "name": "Lunch",
            "description": "balanced general meal idea"
        }},
        {{
            "name": "Dinner",
            "description": "balanced general meal idea"
        }},
        {{
            "name": "Snack",
            "description": "simple snack idea"
        }}
    ],
    "hydration_tip": "general hydration advice",
    "professional_note": "when professional advice may be useful"
}}
"""

            guidance_response = generate_ai_response(
                guidance_prompt,
                temperature=0.3,
            )

            guidance = clean_json_response(
                guidance_response
            )

            if not guidance:

                guidance = {
                    "summary": (
                        "General balanced nutrition habits "
                        "can support everyday wellbeing."
                    ),
                    "tips": [
                        "Include a variety of foods.",
                        "Try to include fruits and vegetables.",
                        "Choose regular balanced meals.",
                        "Drink water regularly.",
                    ],
                    "meal_ideas": [],
                    "hydration_tip": (
                        "Drink water regularly throughout the day."
                    ),
                    "professional_note": (
                        "Speak with a qualified professional "
                        "for individual health concerns."
                    ),
                }

            st.session_state.guidance = guidance

            # =================================================
            # PASS CONTEXT BETWEEN AGENTS
            # =================================================

            st.session_state.workflow_context = {
                "user": user_data,
                "assessment": assessment,
                "safety": safety_result,
                "guidance": guidance,
            }

            st.write(
                "✅ Nutrition Guidance Agent completed."
            )

            status.update(
                label="✅ AI workflow completed!",
                state="complete",
                expanded=False,
            )

        st.session_state.page = "results"

        st.rerun()

    # ========================================================
    # BACK BUTTON
    # ========================================================

    st.divider()

    if st.button(
        "← Back to Home",
        use_container_width=True,
    ):

        st.session_state.page = "home"

        st.rerun()


# ============================================================
# RESULTS PAGE
# ============================================================

def show_results():

    st.title("📊 Nutrition Results Dashboard")

    user_data = st.session_state.user_data

    assessment = st.session_state.assessment

    safety_result = st.session_state.safety_result

    guidance = st.session_state.guidance

    # ========================================================
    # TOP SUMMARY
    # ========================================================

    st.success(
        "Your AI nutrition workflow has been completed successfully."
    )

    st.markdown(
        '<div class="section-title">👤 Assessment Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Age",
            user_data.get(
                "age",
                "N/A",
            ),
        )

    with col2:

        st.metric(
            "Activity",
            user_data.get(
                "activity_level",
                "N/A",
            ),
        )

    with col3:

        st.metric(
            "Diet",
            user_data.get(
                "diet_preference",
                "N/A",
            ),
        )

    with col4:

        st.metric(
            "Safety",
            safety_result.get(
                "status",
                "Reviewed",
            ),
        )

    # ========================================================
    # SAFETY
    # ========================================================

    st.markdown(
        '<div class="section-title">🛡️ Safety Review</div>',
        unsafe_allow_html=True,
    )

    safety_status = safety_result.get(
        "status",
        "Reviewed",
    )

    safety_message = safety_result.get(
        "message",
        "General educational guidance is provided.",
    )

    if safety_status.lower() == "safe":

        st.markdown(
            f"""
            <div class="status-safe">
                <h3>✅ {safety_status}</h3>
                <p>{safety_message}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="status-warning">
                <h3>⚠️ {safety_status}</h3>
                <p>{safety_message}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # ASSESSMENT SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">🤖 AI Assessment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True,
    )

    summary = assessment.get(
        "summary",
        "No assessment summary available.",
    )

    st.write(summary)

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("✅ Positive Habits")

        display_list_items(
            assessment.get(
                "positive_habits",
                [],
            )
        )

    with col2:

        st.subheader("🎯 Areas to Focus On")

        display_list_items(
            assessment.get(
                "areas_to_focus",
                [],
            )
        )

    # ========================================================
    # GUIDANCE
    # ========================================================

    st.markdown(
        '<div class="section-title">🥗 General Nutrition Guidance</div>',
        unsafe_allow_html=True,
    )

    guidance_summary = guidance.get(
        "summary",
        "No guidance summary available.",
    )

    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True,
    )

    st.write(guidance_summary)

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    # ========================================================
    # TIPS
    # ========================================================

    st.subheader("💡 Practical Tips")

    tips = guidance.get(
        "tips",
        [],
    )

    display_list_items(tips)

    # ========================================================
    # MEAL IDEAS
    # ========================================================

    st.subheader("🍽️ Meal Ideas")

    meal_ideas = guidance.get(
        "meal_ideas",
        [],
    )

    if meal_ideas:

        for meal in meal_ideas:

            if isinstance(meal, dict):

                meal_name = meal.get(
                    "name",
                    "Meal",
                )

                meal_description = meal.get(
                    "description",
                    "",
                )

                st.markdown(
                    f"""
                    <div class="card" style="margin-bottom:1rem;">
                        <h4>{meal_name}</h4>
                        <p>{meal_description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.write(f"• {meal}")

    else:

        st.info(
            "No meal ideas were generated."
        )

    # ========================================================
    # HYDRATION
    # ========================================================

    st.subheader("💧 Hydration")

    st.info(
        guidance.get(
            "hydration_tip",
            "Drink water regularly throughout the day.",
        )
    )

    # ========================================================
    # PROFESSIONAL NOTE
    # ========================================================

    professional_note = guidance.get(
        "professional_note",
        "",
    )

    if professional_note:

        st.warning(
            professional_note
        )

    # ========================================================
    # PDF REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">📄 Download Report</div>',
        unsafe_allow_html=True,
    )

    try:

        pdf_data = create_pdf_report(
            user_data,
            assessment,
            safety_result,
            guidance,
        )

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_data,
            file_name="NutriGuide_AI_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    except Exception:

        st.error(
            "The PDF report could not be generated. "
            "Please try again."
        )

    # ========================================================
    # ACTION BUTTONS
    # ========================================================

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔄 New Assessment",
            type="primary",
            use_container_width=True,
        ):

            reset_app()

            st.session_state.page = "assessment"

            st.rerun()

    with col2:

        if st.button(
            "🏠 Back to Home",
            use_container_width=True,
        ):

            reset_app()

            st.session_state.page = "home"

            st.rerun()


# ============================================================
# PAGE ROUTING
# ============================================================

if st.session_state.page == "home":

    show_home()

elif st.session_state.page == "assessment":

    show_assessment()

elif st.session_state.page == "results":

    show_results()

else:

    reset_app()

    show_home()