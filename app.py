import os
import json
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai


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

API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv(
    "GEMINI_API_KEY"
)

if API_KEY:
    API_KEY = API_KEY.strip()

if not API_KEY:
    st.error("🔑 Gemini API key is missing.")

    st.info(
        "For Streamlit Cloud, add GEMINI_API_KEY under "
        "Settings → Secrets. For local development, add it "
        "to your .env file."
    )

    st.stop()

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "page": "home",
    "user_data": {},
    "assessment": {},
    "safety_result": {},
    "guidance": {},
    "workflow_context": {},
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

        /* Main page */
        .block-container {
            max-width: 1150px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Headings */
        h1 {
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        h2,
        h3 {
            font-weight: 650;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            padding: 18px;
            border-radius: 16px;
            background: rgba(128, 128, 128, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            font-weight: 600;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            min-height: 45px;
            font-weight: 600;
        }

        /* Form inputs */
        div[data-baseweb="select"] > div {
            border-radius: 10px;
        }

        textarea,
        input {
            border-radius: 10px !important;
        }

        /* Alerts */
        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        /* Feature cards */
        .feature-card {
            padding: 1.25rem;
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 16px;
            min-height: 180px;
            background: rgba(128, 128, 128, 0.035);
        }

        .feature-card h3 {
            margin-bottom: 0.6rem;
        }

        /* Step cards */
        .step-card {
            padding: 1.2rem;
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 14px;
            min-height: 165px;
            background: rgba(128, 128, 128, 0.025);
        }

        .step-card h3 {
            margin-bottom: 0.6rem;
        }

        /* Meal cards */
        .meal-card {
            padding: 1rem 1.1rem;
            margin-bottom: 0.7rem;
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            background: rgba(128, 128, 128, 0.035);
        }

        /* Dashboard summary */
        .summary-card {
            padding: 1.25rem;
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 16px;
            background: rgba(128, 128, 128, 0.035);
            margin-bottom: 1rem;
        }

        /* Small text */
        .small-note {
            font-size: 0.9rem;
            opacity: 0.78;
        }

        /* Footer */
        .app-footer {
            text-align: center;
            padding: 1.5rem 0 0.5rem 0;
            opacity: 0.7;
            font-size: 0.85rem;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_json_response(text):
    """Clean Gemini output and convert it into a Python dictionary."""

    if not text:
        raise RuntimeError("INVALID_GEMINI_JSON")

    text = text.strip()

    # Remove Markdown code fences if Gemini returns them.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Try direct JSON parsing first.
    try:
        result = json.loads(text)

        if not isinstance(result, dict):
            raise RuntimeError("INVALID_GEMINI_JSON")

        return result

    except json.JSONDecodeError:
        pass

    # If extra text surrounds the JSON, extract the outermost object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]

        try:
            result = json.loads(json_text)

            if not isinstance(result, dict):
                raise RuntimeError("INVALID_GEMINI_JSON")

            return result

        except json.JSONDecodeError as error:
            raise RuntimeError("INVALID_GEMINI_JSON") from error

    raise RuntimeError("INVALID_GEMINI_JSON")


def generate_ai_response(prompt, retries=3):
    """Generate Gemini response with safe error handling."""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text

        except Exception as error:
            error_message = str(error).lower()

            # ------------------------------------------------
            # INVALID / UNAUTHORISED API KEY
            # ------------------------------------------------

            if (
                "401" in error_message
                or "unauthorized" in error_message
                or (
                    "api key" in error_message
                    and "invalid" in error_message
                )
            ):
                raise RuntimeError(
                    "INVALID_API_KEY"
                ) from error

            # ------------------------------------------------
            # QUOTA / RATE LIMIT
            # ------------------------------------------------

            if (
                "429" in error_message
                or "resource_exhausted" in error_message
                or "quota" in error_message
                or "rate limit" in error_message
            ):
                raise RuntimeError(
                    "API_QUOTA_EXCEEDED"
                ) from error

            # ------------------------------------------------
            # TEMPORARY GEMINI SERVER ERROR
            # ------------------------------------------------

            if (
                "503" in error_message
                or "unavailable" in error_message
                or "service unavailable" in error_message
            ):
                if attempt < retries - 1:
                    time.sleep(3)
                    continue

                raise RuntimeError(
                    "GEMINI_TEMPORARILY_UNAVAILABLE"
                ) from error

            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            raise RuntimeError(
                "GEMINI_REQUEST_FAILED"
            ) from error

    raise RuntimeError("GEMINI_REQUEST_FAILED")


def display_list_items(
    items,
    empty_message="No information available.",
    card_style=False,
):
    """Display a list safely in the Streamlit UI."""

    if not items:
        st.write(empty_message)
        return

    for item in items:

        if card_style:
            st.markdown(
                f"""
                <div class="meal-card">
                    🍽️ {item}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.write(f"• {item}")


def reset_app():
    """Reset the application for a new assessment."""

    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value


def get_bool(value):
    """Safely convert common AI boolean values to bool."""

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "yes",
            "1",
        }

    return bool(value)


# ============================================================
# HOME PAGE
# ============================================================

def show_home():

    st.title("🥗 NutriGuide AI")

    st.subheader("Your AI-Powered Nutrition Assistant")

    st.write(
        "Get personalised general nutrition guidance based on "
        "your preferences, goals and dietary needs."
    )

    st.divider()

    st.info(
        "💡 This app provides general nutrition guidance. "
        "For medical or condition-specific dietary advice, "
        "please consult a qualified healthcare professional."
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader("✨ What you can get")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🧠 Personalised Guidance</h3>
                <p>
                    Guidance based on your food preferences,
                    activity level, goals and dietary needs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🍽️ Meal Ideas</h3>
                <p>
                    Practical breakfast, lunch, snack and dinner
                    ideas for everyday nutrition.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🛡️ Safety-Aware</h3>
                <p>
                    Allergies, dietary restrictions and health
                    concerns are considered before guidance is generated.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # --------------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------------

    st.subheader("🚀 How it works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown(
            """
            <div class="step-card">
                <h3>1️⃣ Tell us about yourself</h3>
                <p>
                    Share your basic dietary preferences,
                    activity level, goals and food information.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step2:
        st.markdown(
            """
            <div class="step-card">
                <h3>2️⃣ AI reviews your information</h3>
                <p>
                    NutriGuide AI analyses your information
                    and checks important dietary considerations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step3:
        st.markdown(
            """
            <div class="step-card">
                <h3>3️⃣ Get your guidance</h3>
                <p>
                    Receive personalised meal ideas and
                    practical general nutrition guidance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button(
        "🥗 Start Nutrition Assessment",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.page = "assessment"
        st.rerun()

    st.markdown(
        """
        <div class="app-footer">
            NutriGuide AI is for general educational nutrition
            guidance and is not a substitute for professional medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# NUTRITION ASSESSMENT PAGE
# ============================================================

def show_assessment():

    st.title("📝 Nutrition Assessment")

    st.write(
        "Tell us a little about your nutrition preferences and "
        "goals. This information will be used to prepare general "
        "personalised guidance."
    )

    st.divider()

    with st.form("nutrition_assessment_form"):

        st.subheader("👤 Basic Information")

        col1, col2 = st.columns(2)

        with col1:
            age_group = st.selectbox(
                "Age Group",
                [
                    "Under 18",
                    "18–30",
                    "31–45",
                    "46–60",
                    "60+",
                ],
            )

        with col2:
            activity_level = st.selectbox(
                "Activity Level",
                [
                    "Low",
                    "Moderate",
                    "High",
                ],
            )

        st.subheader("🥗 Dietary Preferences")

        dietary_preference = st.selectbox(
            "Dietary Preference",
            [
                "No specific preference",
                "Vegetarian",
                "Vegan",
                "Other",
            ],
        )

        dietary_other = ""

        if dietary_preference == "Other":
            dietary_other = st.text_input(
                "Please describe your dietary preference"
            )

        food_allergies = st.text_area(
            "Food Allergies",
            placeholder="Example: peanuts, eggs, milk",
        )

        foods_to_avoid = st.text_area(
            "Foods You Avoid",
            placeholder=(
                "Example: very spicy food, certain vegetables"
            ),
        )

        favourite_foods = st.text_area(
            "Favourite / Available Foods",
            placeholder=(
                "Example: rice, roti, chicken, vegetables, "
                "fruit, yoghurt"
            ),
        )

        st.subheader("🎯 Your Goal")

        goal = st.selectbox(
            "What would you like help with?",
            [
                "Healthy eating",
                "Fitness / active lifestyle",
                "Better meal variety",
                "Healthy lifestyle",
                "General health concern",
                "Other",
            ],
        )

        other_goal = ""

        if goal == "Other":
            other_goal = st.text_input(
                "Please describe your goal"
            )

        health_information = st.text_area(
            "Health Information / Concerns",
            placeholder=(
                "Optional. Mention any health concern you want "
                "the AI to consider."
            ),
        )

        st.divider()

        submitted = st.form_submit_button(
            "🧠 Generate Nutrition Guidance",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        actual_preference = dietary_preference

        if dietary_preference == "Other" and dietary_other.strip():
            actual_preference = dietary_other.strip()

        actual_goal = goal

        if goal == "Other" and other_goal.strip():
            actual_goal = other_goal.strip()

        user_data = {
            "age_group": age_group,
            "activity_level": activity_level,
            "dietary_preference": actual_preference,
            "food_allergies": food_allergies.strip(),
            "foods_to_avoid": foods_to_avoid.strip(),
            "favourite_foods": favourite_foods.strip(),
            "goal": actual_goal,
            "health_information": health_information.strip(),
        }

        st.session_state.user_data = user_data

        # ----------------------------------------------------
        # AI WORKFLOW
        # ----------------------------------------------------

        try:

            # =================================================
            # STAGE 1 — ASSESSMENT
            # =================================================

            with st.spinner("🧠 Analysing your information..."):

                assessment_prompt = f"""
You are the Assessment Agent for NutriGuide AI.

Your job is to analyse the user's nutrition information and
prepare a structured profile for the next workflow stages.

USER INFORMATION:
{json.dumps(user_data, indent=2)}

IMPORTANT SAFETY RULES:
- This app provides general nutrition education.
- Do not diagnose medical conditions.
- Do not prescribe treatment diets.
- Do not create restrictive weight-loss plans.
- Do not provide calorie restriction targets.
- Do not set weight-loss or weight-gain targets.
- If the user mentions a medical condition or health concern,
  clearly recommend consultation with a qualified healthcare
  professional or registered dietitian for condition-specific
  advice.
- Respect allergies and foods the user avoids.
- The user's age group must be considered.
- If the user is under 18, provide only age-appropriate,
  general healthy-eating guidance and avoid dieting or
  weight-focused advice.

Return ONLY valid JSON.

Use exactly this structure:

{{
  "profile_summary": "Short general summary.",
  "dietary_preference": "Dietary preference",
  "allergies": [],
  "foods_to_avoid": [],
  "goal": "Main goal",
  "plan_type": "long_term_guidance",
  "plan_type_reason": "Short explanation.",
  "planning_considerations": [],
  "safety_notes": [],
  "professional_advice_recommended": false
}}

PLAN TYPE RULES:
- Use "long_term_guidance" for healthy eating, healthy
  lifestyle, fitness/active lifestyle and meal variety.
- Use "short_term_general_guidance" when the user mentions a
  health or medical concern where condition-specific advice
  would require professional guidance.
- Do NOT create a fixed 7-day or 30-day plan.
"""

                assessment_text = generate_ai_response(
                    assessment_prompt
                )

                assessment = clean_json_response(
                    assessment_text
                )

                st.session_state.assessment = assessment

            # =================================================
            # STAGE 2 — SAFETY CHECK
            # =================================================

            with st.spinner("🛡️ Checking your dietary needs..."):

                safety_prompt = f"""
You are the Safety Agent for NutriGuide AI.

Review the user's information and the Assessment Agent's
result before nutrition guidance is generated.

USER INFORMATION:
{json.dumps(user_data, indent=2)}

ASSESSMENT:
{json.dumps(assessment, indent=2)}

Check:
1. Food allergies.
2. Foods the user avoids.
3. Dietary preferences.
4. Health or medical concerns.
5. Age-related safety.
6. Whether professional advice should be recommended.

IMPORTANT:
- Never diagnose.
- Never prescribe a medical treatment diet.
- Never recommend dangerous or highly restrictive diets.
- Never provide weight-loss targets or calorie restriction.
- For users under 18, avoid weight-focused dieting advice.
- For health/medical concerns, recommend a qualified healthcare
  professional or registered dietitian for condition-specific
  advice.

Return ONLY valid JSON.

Use exactly this structure:

{{
  "status": "safe_to_continue",
  "allergy_restrictions": [],
  "dietary_restrictions": [],
  "foods_to_avoid": [],
  "safety_flags": [],
  "meal_planning_rules": [],
  "professional_advice_recommended": false
}}

STATUS OPTIONS:
- "safe_to_continue"
- "continue_with_caution"
- "professional_review_recommended"
"""

                safety_text = generate_ai_response(
                    safety_prompt
                )

                safety_result = clean_json_response(
                    safety_text
                )

                st.session_state.safety_result = safety_result

            # =================================================
            # STAGE 3 — NUTRITION GUIDANCE
            # =================================================

            with st.spinner("🍽️ Preparing your nutrition guidance..."):

                guidance_prompt = f"""
You are the Nutrition Guidance Agent for NutriGuide AI.

Create personalised GENERAL nutrition guidance using the
information passed from the previous workflow stages.

USER INFORMATION:
{json.dumps(user_data, indent=2)}

ASSESSMENT:
{json.dumps(assessment, indent=2)}

SAFETY CHECK:
{json.dumps(safety_result, indent=2)}

IMPORTANT RULES:
- This is general nutrition education, not medical advice.
- Do not diagnose diseases or health conditions.
- Do not prescribe treatment diets.
- Do not provide weight-loss or weight-gain targets.
- Do not provide calorie restriction targets.
- Do not encourage restrictive eating.
- Do not create a fixed 7-day plan.
- Do not create a fixed 30-day plan.
- Do not mention a specific duration such as "follow this for
  one week".
- Instead, provide flexible ongoing guidance and meal ideas.
- Respect all allergies and foods the user avoids.
- Respect dietary preferences.
- Use practical, familiar and culturally appropriate foods
  where possible.
- Keep meals balanced and varied.
- For users under 18, focus on healthy growth, regular meals,
  balanced food choices and overall wellbeing rather than
  weight change.
- If a medical/health concern is present, provide only general
  nutrition information and clearly recommend a qualified
  healthcare professional or registered dietitian for
  condition-specific advice.
- Do not claim that food can cure or treat a medical condition.

Return ONLY valid JSON.

Use exactly this structure:

{{
  "plan_title": "Personalised Nutrition Guidance",
  "guidance_type": "Long-term healthy lifestyle guidance",
  "duration": "Ongoing guidance — no fixed duration",
  "important_safety_note": "Short safety message.",
  "breakfast_ideas": [
    "Breakfast idea 1",
    "Breakfast idea 2",
    "Breakfast idea 3",
    "Breakfast idea 4"
  ],
  "lunch_ideas": [
    "Lunch idea 1",
    "Lunch idea 2",
    "Lunch idea 3",
    "Lunch idea 4"
  ],
  "snack_ideas": [
    "Snack idea 1",
    "Snack idea 2",
    "Snack idea 3",
    "Snack idea 4"
  ],
  "dinner_ideas": [
    "Dinner idea 1",
    "Dinner idea 2",
    "Dinner idea 3",
    "Dinner idea 4"
  ],
  "nutrition_tips": [
    "General nutrition tip 1",
    "General nutrition tip 2",
    "General nutrition tip 3",
    "General nutrition tip 4",
    "General nutrition tip 5"
  ]
}}
"""

                guidance_text = generate_ai_response(
                    guidance_prompt
                )

                guidance = clean_json_response(
                    guidance_text
                )

                st.session_state.guidance = guidance

            # ------------------------------------------------
            # SAVE WORKFLOW CONTEXT
            # ------------------------------------------------

            st.session_state.workflow_context = {
                "user_data": user_data,
                "assessment": assessment,
                "safety_result": safety_result,
                "guidance": guidance,
            }

            st.session_state.page = "results"

            st.rerun()

        except Exception as error:

            error_code = str(error)

            if error_code == "INVALID_API_KEY":

                st.error(
                    "🔑 Your Gemini API key is invalid or not authorised."
                )

                st.info(
                    "Please check GEMINI_API_KEY in your Streamlit "
                    "Secrets and try again."
                )

            elif error_code == "API_QUOTA_EXCEEDED":

                st.warning(
                    "⏳ Gemini API quota has been reached."
                )

                st.info(
                    "Please wait and try again later, or check your "
                    "Gemini API project quota."
                )

            elif error_code == "GEMINI_TEMPORARILY_UNAVAILABLE":

                st.warning(
                    "🔄 Gemini is temporarily unavailable."
                )

                st.info(
                    "Please wait a few moments and try again."
                )

            elif error_code == "INVALID_GEMINI_JSON":

                st.error(
                    "📄 The AI returned an unexpected response."
                )

                st.info(
                    "Please try generating the guidance again."
                )

            elif error_code == "GEMINI_REQUEST_FAILED":

                st.error(
                    "⚠️ We could not generate your nutrition guidance."
                )

                st.info(
                    "Please try again. If the problem continues, "
                    "check your Gemini API configuration."
                )

            else:

                st.error(
                    "⚠️ Something unexpected happened."
                )

                st.info(
                    "Please try again."
                )

            st.divider()

            if st.button("⬅️ Back to Home"):

                reset_app()

                st.session_state.page = "home"

                st.rerun()


# ============================================================
# RESULTS PAGE
# ============================================================

def show_results():

    assessment = st.session_state.get(
        "assessment",
        {},
    )

    safety_result = st.session_state.get(
        "safety_result",
        {},
    )

    guidance = st.session_state.get(
        "guidance",
        {},
    )

    user_data = st.session_state.get(
        "user_data",
        {},
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("🍽️ Your Nutrition Guidance")

    st.write(
        "Your personalised general nutrition guidance is ready."
    )

    # --------------------------------------------------------
    # TOP SUMMARY CARD
    # --------------------------------------------------------

    guidance_type = guidance.get(
        "guidance_type",
        "General nutrition guidance",
    )

    duration = guidance.get(
        "duration",
        "Ongoing guidance — no fixed duration",
    )

    plan_title = guidance.get(
        "plan_title",
        "Personalised Nutrition Guidance",
    )

    st.markdown(
        f"""
        <div class="summary-card">
            <h3>🌱 {plan_title}</h3>
            <p><strong>Guidance:</strong> {guidance_type}</p>
            <p><strong>Approach:</strong> {duration}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if assessment.get("plan_type") == "short_term_general_guidance":

        st.info(
            "ℹ️ A health-related concern was detected. The "
            "information below is general nutrition education only. "
            "For condition-specific advice, please consult a "
            "qualified healthcare professional or registered dietitian."
        )

    else:

        st.success(
            "🌱 Your guidance has been prepared using your "
            "dietary preferences and safety information."
        )

    # --------------------------------------------------------
    # QUICK PROFILE
    # --------------------------------------------------------

    st.subheader("👤 Your Nutrition Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Age Group",
            user_data.get(
                "age_group",
                "Not provided",
            ),
        )

    with col2:
        st.metric(
            "Activity",
            user_data.get(
                "activity_level",
                "Not provided",
            ),
        )

    with col3:
        st.metric(
            "Diet",
            user_data.get(
                "dietary_preference",
                "Not provided",
            ),
        )

    with col4:
        st.metric(
            "Goal",
            user_data.get(
                "goal",
                "Not provided",
            ),
        )

    # --------------------------------------------------------
    # AI ASSESSMENT
    # --------------------------------------------------------

    st.divider()

    with st.expander(
        "🧠 AI Assessment",
        expanded=True,
    ):

        profile_summary = assessment.get(
            "profile_summary",
            "No assessment summary available.",
        )

        st.write(profile_summary)

        planning_considerations = assessment.get(
            "planning_considerations",
            [],
        )

        if planning_considerations:

            st.markdown(
                "**Important considerations:**"
            )

            display_list_items(
                planning_considerations
            )

    # --------------------------------------------------------
    # SAFETY INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader("🛡️ Safety Information")

    safety_status = safety_result.get(
        "status",
        "safe_to_continue",
    )

    if safety_status == "professional_review_recommended":

        st.warning(
            "👩‍⚕️ Professional advice is recommended before "
            "making condition-specific dietary changes."
        )

    elif safety_status == "continue_with_caution":

        st.warning(
            "⚠️ Please use this guidance carefully and consider "
            "professional advice where appropriate."
        )

    else:

        st.success(
            "✅ The guidance respects the dietary information "
            "provided in your assessment."
        )

    safety_note = guidance.get(
        "important_safety_note",
        "",
    )

    if safety_note:

        st.info(
            safety_note
        )

    safety_flags = safety_result.get(
        "safety_flags",
        [],
    )

    allergy_restrictions = safety_result.get(
        "allergy_restrictions",
        [],
    )

    foods_to_avoid = safety_result.get(
        "foods_to_avoid",
        [],
    )

    if safety_flags:

        with st.expander(
            "⚠️ Safety Considerations",
            expanded=True,
        ):

            display_list_items(
                safety_flags
            )

    if allergy_restrictions:

        with st.expander(
            "🚫 Allergy Restrictions",
            expanded=True,
        ):

            display_list_items(
                allergy_restrictions
            )

    if foods_to_avoid:

        with st.expander(
            "🥗 Foods to Avoid",
            expanded=True,
        ):

            display_list_items(
                foods_to_avoid
            )

    professional_advice = (
        get_bool(
            assessment.get(
                "professional_advice_recommended",
                False,
            )
        )
        or
        get_bool(
            safety_result.get(
                "professional_advice_recommended",
                False,
            )
        )
    )

    if professional_advice:

        st.warning(
            "👩‍⚕️ For health or medical concerns, please consult "
            "a qualified healthcare professional or registered "
            "dietitian for personalised advice."
        )

    # --------------------------------------------------------
    # MEAL IDEAS
    # --------------------------------------------------------

    st.divider()

    st.subheader("🍽️ Meal Ideas")

    breakfast_ideas = guidance.get(
        "breakfast_ideas",
        [],
    )

    lunch_ideas = guidance.get(
        "lunch_ideas",
        [],
    )

    snack_ideas = guidance.get(
        "snack_ideas",
        [],
    )

    dinner_ideas = guidance.get(
        "dinner_ideas",
        [],
    )

    # Breakfast + Lunch
    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🍳 Breakfast")

        display_list_items(
            breakfast_ideas,
            card_style=True,
        )

    with col2:

        st.markdown("### 🥗 Lunch")

        display_list_items(
            lunch_ideas,
            card_style=True,
        )

    # Snack + Dinner
    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🍎 Snacks")

        display_list_items(
            snack_ideas,
            card_style=True,
        )

    with col2:

        st.markdown("### 🍽️ Dinner")

        display_list_items(
            dinner_ideas,
            card_style=True,
        )

    # --------------------------------------------------------
    # NUTRITION TIPS
    # --------------------------------------------------------

    st.divider()

    st.subheader("💡 General Nutrition Tips")

    nutrition_tips = guidance.get(
        "nutrition_tips",
        [],
    )

    display_list_items(
        nutrition_tips
    )

    # --------------------------------------------------------
    # FINAL SAFETY NOTE
    # --------------------------------------------------------

    st.divider()

    st.info(
        "🥗 These are general nutrition suggestions. Your food "
        "needs can change over time. For medical conditions or "
        "personalised dietary treatment, seek advice from a "
        "qualified healthcare professional."
    )

    # --------------------------------------------------------
    # FOOTER ACTIONS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔄 Create Another Nutrition Guidance",
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

    st.markdown(
        """
        <div class="app-footer">
            NutriGuide AI • General educational nutrition guidance
        </div>
        """,
        unsafe_allow_html=True,
    )


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