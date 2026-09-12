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

API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error(
        "Gemini API key is missing. Please configure "
        "GEMINI_API_KEY in Streamlit Secrets or your local .env file."
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
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            padding: 15px;
            border-radius: 12px;
        }

        .small-note {
            font-size: 0.9rem;
            opacity: 0.8;
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
        raise ValueError("Gemini returned an empty response.")

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
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # If extra text surrounds the JSON, extract the outermost object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Gemini returned invalid JSON: {error}"
            ) from error

    raise ValueError("No valid JSON object was found in Gemini's response.")


def generate_ai_response(prompt, retries=3):
    """Generate Gemini response with basic retry and quota handling."""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            return response.text

        except Exception as error:
            error_message = str(error)

            # Temporary Gemini server overload.
            if "503" in error_message or "UNAVAILABLE" in error_message:
                if attempt < retries - 1:
                    time.sleep(3)
                    continue

            # Free-tier/API quota issue.
            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            ):
                raise RuntimeError(
                    "Gemini API quota has been exceeded. "
                    "Please try again later or use a Gemini API "
                    "project with available quota."
                ) from error

            raise error


def display_list_items(items, empty_message="No information available."):
    """Display a list safely in the Streamlit UI."""

    if not items:
        st.write(empty_message)
        return

    for item in items:
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

    st.subheader("✨ What you can get")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🧠 Personalised Guidance")
        st.write(
            "Guidance based on your food preferences, activity "
            "level, goals and dietary needs."
        )

    with col2:
        st.markdown("### 🍽️ Meal Ideas")
        st.write(
            "Practical breakfast, lunch, snack and dinner ideas "
            "without a fixed plan duration."
        )

    with col3:
        st.markdown("### 🛡️ Safety-Aware")
        st.write(
            "The app checks allergies, foods to avoid and "
            "health-related concerns before generating guidance."
        )

    st.divider()

    st.subheader("🚀 How it works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("### 1️⃣ Tell us about yourself")
        st.write(
            "Share your basic dietary preferences, activity level, "
            "goals and food information."
        )

    with step2:
        st.markdown("### 2️⃣ AI reviews your information")
        st.write(
            "NutriGuide AI analyses your information and checks "
            "important dietary considerations."
        )

    with step3:
        st.markdown("### 3️⃣ Get your guidance")
        st.write(
            "Receive personalised meal ideas and practical general "
            "nutrition guidance."
        )

    st.divider()

    if st.button(
        "🥗 Start Nutrition Assessment",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.page = "assessment"
        st.rerun()

    st.caption(
        "NutriGuide AI is for general educational nutrition guidance "
        "and is not a substitute for professional medical advice."
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
            placeholder="Example: very spicy food, certain vegetables",
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
            st.error(
                "⚠️ Something went wrong while generating your "
                "nutrition guidance."
            )

            st.error(str(error))

            st.info(
                "Please check your Gemini API key/quota and try "
                "again."
            )

    st.divider()

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()


# ============================================================
# RESULTS PAGE
# ============================================================

def show_results():

    assessment = st.session_state.get("assessment", {})
    safety_result = st.session_state.get("safety_result", {})
    guidance = st.session_state.get("guidance", {})
    user_data = st.session_state.get("user_data", {})

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("🍽️ Your Nutrition Guidance")

    st.subheader("Personalised general nutrition guidance")

    st.write(
        "Your nutrition guidance has been prepared based on "
        "your preferences, goals and dietary needs."
    )

    st.divider()

    # --------------------------------------------------------
    # GUIDANCE TYPE
    # --------------------------------------------------------

    guidance_type = guidance.get(
        "guidance_type",
        "General nutrition guidance",
    )

    duration = guidance.get(
        "duration",
        "Ongoing guidance — no fixed duration",
    )

    if assessment.get("plan_type") == "short_term_general_guidance":
        st.info(
            "ℹ️ Health-related concern detected. The information "
            "below is general nutrition education only. For "
            "condition-specific advice, please consult a qualified "
            "healthcare professional or registered dietitian."
        )
    else:
        st.success(
            f"🌱 {guidance_type}"
        )

    st.caption(duration)

    # --------------------------------------------------------
    # USER PROFILE
    # --------------------------------------------------------

    st.subheader("👤 Your Nutrition Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Age Group",
            user_data.get("age_group", "Not provided"),
        )

    with col2:
        st.metric(
            "Activity",
            user_data.get("activity_level", "Not provided"),
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
            user_data.get("goal", "Not provided"),
        )

    # --------------------------------------------------------
    # AI ASSESSMENT
    # --------------------------------------------------------

    st.divider()

    st.subheader("🧠 AI Assessment")

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
        st.markdown("**Important considerations:**")
        display_list_items(planning_considerations)

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
            "Professional advice is recommended before making "
            "condition-specific dietary changes."
        )

    elif safety_status == "continue_with_caution":
        st.warning(
            "Please use this guidance carefully and consider "
            "professional advice where appropriate."
        )

    else:
        st.success(
            "The guidance respects the dietary information "
            "provided in your assessment."
        )

    safety_note = guidance.get(
        "important_safety_note",
        "",
    )

    if safety_note:
        st.write(safety_note)

    safety_flags = safety_result.get(
        "safety_flags",
        [],
    )

    if safety_flags:
        st.markdown("**Safety considerations:**")
        display_list_items(safety_flags)

    allergy_restrictions = safety_result.get(
        "allergy_restrictions",
        [],
    )

    if allergy_restrictions:
        st.markdown("**Allergy restrictions considered:**")
        display_list_items(allergy_restrictions)

    foods_to_avoid = safety_result.get(
        "foods_to_avoid",
        [],
    )

    if foods_to_avoid:
        st.markdown("**Foods to avoid:**")
        display_list_items(foods_to_avoid)

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

    st.subheader("🍳 Breakfast Ideas")

    breakfast_ideas = guidance.get(
        "breakfast_ideas",
        [],
    )

    display_list_items(breakfast_ideas)

    st.divider()

    st.subheader("🥗 Lunch Ideas")

    lunch_ideas = guidance.get(
        "lunch_ideas",
        [],
    )

    display_list_items(lunch_ideas)

    st.divider()

    st.subheader("🍎 Snack Ideas")

    snack_ideas = guidance.get(
        "snack_ideas",
        [],
    )

    display_list_items(snack_ideas)

    st.divider()

    st.subheader("🍽️ Dinner Ideas")

    dinner_ideas = guidance.get(
        "dinner_ideas",
        [],
    )

    display_list_items(dinner_ideas)

    # --------------------------------------------------------
    # NUTRITION TIPS
    # --------------------------------------------------------

    st.divider()

    st.subheader("💡 General Nutrition Tips")

    nutrition_tips = guidance.get(
        "nutrition_tips",
        [],
    )

    display_list_items(nutrition_tips)

    # --------------------------------------------------------
    # FOOTER ACTIONS
    # --------------------------------------------------------

    st.divider()

    st.info(
        "🥗 These are general nutrition suggestions. Your food "
        "needs can change over time. For medical conditions or "
        "personalised dietary treatment, seek advice from a "
        "qualified healthcare professional."
    )

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
