"""services/gemma_service.py

Service layer for analyzing Indian street food images using Google Gemma 4
and Gemini multimodal models via the official `google-genai` SDK.

Validates structured output through Pydantic and generates targeted,
culturally practical vendor inquiry questions.
"""

import json
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from PIL import Image
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Official model identifiers
DEFAULT_GEMMA_MODEL = os.getenv("DEFAULT_MODEL", "gemma-4-26b-a4b-it")
FALLBACK_GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================================
# Pydantic Output Schemas
# ============================================================================

class IngredientItem(BaseModel):
    """An identified or likely food ingredient."""
    name: str = Field(
        ...,
        description="Name of the ingredient (e.g., Potatoes / Aloo, Chickpeas / Chole, Gram Flour / Besan)."
    )
    is_allergen: bool = Field(
        ...,
        description="True if this ingredient directly matches user-specified allergens or common major allergens."
    )
    confidence: str = Field(
        ...,
        description="Visual detection confidence: 'High', 'Medium', or 'Low'."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Culinary context (e.g., 'May be seasoned with compounded hing containing wheat')."
    )


class AllergenAlert(BaseModel):
    """An allergen or dietary risk alert."""
    allergen: str = Field(
        ...,
        description="Allergen name (e.g., 'Peanuts / Groundnut', 'Dairy / Ghee / Paneer', 'Gluten / Wheat / Hing')."
    )
    risk_level: str = Field(
        ...,
        description="Risk severity: 'High', 'Moderate', or 'Potential Cross-Contamination'."
    )
    reason: str = Field(
        ...,
        description="Detailed explanation of why this risk is flagged based on visual cues or standard street recipe prep."
    )


class VendorQuestion(BaseModel):
    """Practical questions designed to be spoken to the food stall vendor."""
    question_english: str = Field(
        ...,
        description="Question phrased in clear English."
    )
    question_hindi: str = Field(
        ...,
        description="Question written in Hindi script and/or conversational Hinglish (e.g., 'Bhaiya, isme mungfali ka tel ya kaju paste use kiya hai kya?')."
    )
    reason: str = Field(
        ...,
        description="Why the customer should ask this question before consuming the food."
    )


class FoodAnalysisResult(BaseModel):
    """Structured response schema for food image analysis."""
    dish_name: str = Field(
        ...,
        description="Identified Indian street food or dish name (e.g., Samosa Chaat, Pav Bhaji, Pani Puri, Vada Pav)."
    )
    cuisine_region: str = Field(
        ...,
        description="Regional origin or style (e.g., Mumbai Street Food, North Indian Chaat, Delhi Street Food)."
    )
    confidence_summary: str = Field(
        ...,
        description="Summary of visual clarity and model confidence in identifying the dish."
    )
    identified_ingredients: List[IngredientItem] = Field(
        ...,
        description="List of ingredients visually spotted or customarily present in this dish."
    )
    uncertain_ingredients: List[str] = Field(
        ...,
        description="Ingredients that CANNOT be verified by photo alone (e.g., frying oil type, compounded hing/wheat carrier, dairy in gravies)."
    )
    allergen_warnings: List[AllergenAlert] = Field(
        ...,
        description="Alerts corresponding to user allergen/dietary filters and common street food pitfalls."
    )
    dietary_compliance_assessment: str = Field(
        ...,
        description="Evaluation against selected dietary preferences (e.g., Vegetarian, Jain, Vegan, Gluten-Free)."
    )
    vendor_questions: List[VendorQuestion] = Field(
        ...,
        description="2-4 respectful, direct questions to ask the vendor to verify safety."
    )
    safety_disclaimer: str = Field(
        ...,
        description="Mandatory disclaimer stating that photos cannot guarantee safety, oil purity, or lack of cross-contamination."
    )


# ============================================================================
# API Key and Client Helpers
# ============================================================================

def get_api_key(override_key: Optional[str] = None) -> Optional[str]:
    """Retrieve the API key from override parameter, Streamlit secrets, or environment variables."""
    if override_key and override_key.strip():
        return override_key.strip()

    # Check Streamlit secrets (for Streamlit Community Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                val = st.secrets["GEMINI_API_KEY"]
                if val:
                    return str(val).strip()
            if "GOOGLE_API_KEY" in st.secrets:
                val = st.secrets["GOOGLE_API_KEY"]
                if val:
                    return str(val).strip()
    except Exception:
        # st.secrets may raise FileNotFoundError or StreamlitSecretNotFoundError when run outside Streamlit or without secrets.toml
        pass

    # Fallback to standard environment variables (.env)
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def build_system_prompt(language: str = "English") -> str:
    """Build specialized system instructions for Indian street food analysis."""
    lang_instruction = (
        "Respond primarily in English with helpful conversational Hindi/Hinglish questions for the vendor."
        if language == "English"
        else "मुख्य विवरण और चेतावनी हिंदी में प्रदान करें, और विक्रेता से पूछने के लिए सरल हिंदी प्रश्न तैयार करें।"
    )

    return f"""You are Realbites, an expert culinary and food-safety assistant specializing in Indian street food.
Target language preference: {language}. {lang_instruction}

CRITICAL SAFETY DIRECTIVE:
1. NEVER claim or imply that an image proves food is safe or allergen-free.
2. In street food environments, ingredients are often hidden (e.g., peanut/groundnut oil, mustard oil, dalda, compounded asafoetida/hing containing wheat flour as an anti-caking agent, cashew paste in gravies, ghee brushed on bread).
3. Cross-contamination risk is HIGH on shared tawas, deep frying oil pots, and prep counters.
4. Always populate `uncertain_ingredients` with ingredients that cannot be verified by eyesight alone (especially cooking oils, secret spice blends, and binders).
5. Always generate respectful, actionable questions for the food stall vendor (using colloquial Indian street terms like 'Bhaiya, ...').
6. Provide a firm `safety_disclaimer` reinforcing that visual analysis has limitations and personal vendor inquiry is essential.
"""


def build_analysis_prompt(
    dietary_preferences: List[str],
    allergens: List[str],
    language: str = "English"
) -> str:
    """Construct the user prompt detailing customer dietary and allergen restrictions."""
    pref_str = ", ".join(dietary_preferences) if dietary_preferences else "Standard / No specific preference"
    allergen_str = ", ".join(allergens) if allergens else "None specified"

    return f"""Please inspect this Indian street food or menu image with rigorous attention to culinary composition and food allergens.

Customer Profile:
- Dietary Preferences: {pref_str}
- Critical Allergens to avoid: {allergen_str}
- Preferred Language: {language}

Instructions:
1. Identify the dish and regional style.
2. List visible and customary ingredients with your confidence level.
3. Explicitly check for user allergens:
   - If Peanuts/Nuts are selected: check for peanuts (singdana), peanut oil, cashew (kaju) paste, almond flakes.
   - If Gluten is selected: check for maida, atta, bread/pav/puri, and compounded hing (asafoetida mixed with wheat flour).
   - If Dairy is selected: check for ghee, butter, paneer, dahi (curd), malai, cheese, mawa.
   - If Jain is selected: flag any root vegetables (potatoes/aloo, onions/pyaz, garlic/lahsun, ginger, carrots) and non-veg items.
4. Identify uncertain elements that cannot be proven by sight (such as frying oil medium, shared tawa contact, compounded spices).
5. Generate 2 to 4 practical questions to ask the street food vendor (in both English and conversational Hindi/Hinglish).
6. Provide a thorough safety disclaimer reminding the diner that image analysis cannot detect microscopic traces or guarantee allergy safety.
"""


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_food_image(
    image: Image.Image,
    dietary_preferences: Optional[List[str]] = None,
    allergens: Optional[List[str]] = None,
    language: str = "English",
    model_id: str = DEFAULT_GEMMA_MODEL,
    api_key: Optional[str] = None,
    demo_mode: bool = False,
) -> FoodAnalysisResult:
    """Analyze a food image using Gemma 4 or Gemini multimodal models.

    Args:
        image: PIL Image instance of the food or menu.
        dietary_preferences: List of dietary restrictions (e.g., ['Vegetarian', 'Jain']).
        allergens: List of allergen flags (e.g., ['Peanuts', 'Dairy']).
        language: 'English' or 'Hindi'.
        model_id: The model identifier (e.g., 'gemma-4-26b-a4b-it', 'gemma-4-31b-it', 'gemini-2.5-flash').
        api_key: Optional override for Gemini API Key.
        demo_mode: If True, returns a mock analysis without making an external API call.

    Returns:
        A validated FoodAnalysisResult Pydantic instance.

    Raises:
        ValueError: If API key is missing or model returns invalid data.
        RuntimeError: If API call fails.
    """
    dietary_preferences = dietary_preferences or []
    allergens = allergens or []

    # If demo mode is active, return a realistic sample analysis
    if demo_mode:
        logger.info("Demo mode requested; generating synthetic sample analysis.")
        return get_sample_analysis(dietary_preferences, allergens, language)

    # Validate API key
    active_key = get_api_key(api_key)
    if not active_key:
        raise ValueError(
            "API key not found. Please provide a GEMINI_API_KEY in your .env file "
            "or enter it directly in the Streamlit sidebar."
        )

    # Initialize Google GenAI client
    client = genai.Client(api_key=active_key)

    # Ensure image is in RGB format for processing
    if image.mode != "RGB":
        image = image.convert("RGB")

    system_instruction = build_system_prompt(language)
    prompt = build_analysis_prompt(dietary_preferences, allergens, language)

    # Configure structured output using Pydantic schema
    config = GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=FoodAnalysisResult,
        temperature=0.2,
    )

    # Execute model call with fallback handling
    target_model = model_id
    try:
        logger.info("Invoking model: %s", target_model)
        response = client.models.generate_content(
            model=target_model,
            contents=[image, prompt],
            config=config,
        )
    except Exception as exc:
        err_str = str(exc)
        logger.warning("Primary model call '%s' failed: %s", target_model, err_str)

        # Check if the error is due to Gemma 4 endpoint availability
        if ("not found" in err_str.lower() or "404" in err_str or "unsupported" in err_str.lower()) and "gemma" in target_model.lower():
            logger.info("Attempting automatic fallback to verified Gemini model: %s", FALLBACK_GEMINI_MODEL)
            try:
                response = client.models.generate_content(
                    model=FALLBACK_GEMINI_MODEL,
                    contents=[image, prompt],
                    config=config,
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Model '{target_model}' was unavailable on your API key/endpoint, "
                    f"and the fallback model '{FALLBACK_GEMINI_MODEL}' also failed: {fallback_exc}. "
                    "Note: Gemma 4 hosted endpoints may require Google AI Studio beta/allowlist or Vertex AI Model Garden access."
                ) from fallback_exc
        else:
            raise RuntimeError(f"AI analysis failed ({target_model}): {err_str}") from exc

    # Parse and validate structured output
    raw_text = response.text or ""
    try:
        # Pydantic v2 validation from JSON string
        result = FoodAnalysisResult.model_validate_json(raw_text)
        return result
    except Exception as validation_error:
        logger.error("Failed to parse structured output: %s. Raw text: %s", validation_error, raw_text)
        # Attempt fallback JSON parsing if markdown fences exist
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            parsed = json.loads(cleaned.strip())
            return FoodAnalysisResult.model_validate(parsed)
        except Exception:
            raise ValueError(
                f"Model response could not be validated against the required safety schema: {validation_error}"
            ) from validation_error


# ============================================================================
# Demo / Sample Analysis Generator (For offline testing & immediate trial)
# ============================================================================

def get_sample_analysis(
    dietary_preferences: List[str],
    allergens: List[str],
    language: str = "English"
) -> FoodAnalysisResult:
    """Generate a realistic sample analysis for Indian street food (Samosa Chaat)."""
    is_hindi = language == "Hindi"

    # Base identified ingredients for Samosa Chaat
    ingredients = [
        IngredientItem(
            name="Potatoes / Aloo (Boiled & Spiced)" if not is_hindi else "उबले आलू और मसाले",
            is_allergen=False,
            confidence="High",
            notes="Core filling inside the crisp samosa pastry."
        ),
        IngredientItem(
            name="Refined Wheat Flour / Maida" if not is_hindi else "मैदा (गेहूं का आटा)",
            is_allergen=True,
            confidence="High",
            notes="Forms the outer pastry cone; contains gluten."
        ),
        IngredientItem(
            name="Chickpea Curry / Chole" if not is_hindi else "छोले / चना मसाला",
            is_allergen=False,
            confidence="High",
            notes="Poured over crushed samosa; seasoned with garam masala."
        ),
        IngredientItem(
            name="Sweet Tamarind Chutney (Saunth)" if not is_hindi else "मीठी इमली की चटनी",
            is_allergen=False,
            confidence="High",
            notes="Made with tamarind, jaggery, and ginger powder."
        ),
        IngredientItem(
            name="Spiced Mint-Coriander Chutney" if not is_hindi else "हरी पुदीना-धनिया चटनी",
            is_allergen=False,
            confidence="High",
            notes="May occasionally be thickened with roasted gram or peanut paste."
        ),
        IngredientItem(
            name="Crisp Chickpea Vermicelli / Sev" if not is_hindi else "बेसन की सेव",
            is_allergen=False,
            confidence="Medium",
            notes="Made from besan (gram flour), usually fried in commercial vegetable oil."
        ),
        IngredientItem(
            name="Yogurt / Dahi (Whisked & Sweetened)" if not is_hindi else "दही (Dahi)",
            is_allergen=True,
            confidence="High",
            notes="Key topping on Chaat; contains lactose and milk protein."
        ),
    ]

    warnings: List[AllergenAlert] = []

    # Check for Gluten
    if any("gluten" in a.lower() or "wheat" in a.lower() for a in allergens):
        warnings.append(
            AllergenAlert(
                allergen="Gluten / Wheat / Maida",
                risk_level="High",
                reason=(
                    "The samosa crust is prepared directly from refined wheat flour (maida). "
                    "Compounded asafoetida (hing) in the chole gravy also typically contains wheat starch."
                )
            )
        )

    # Check for Dairy
    if any("dairy" in a.lower() or "milk" in a.lower() for a in allergens):
        warnings.append(
            AllergenAlert(
                allergen="Dairy / Lactose",
                risk_level="High",
                reason="Dahi (whisked yogurt) is poured over the samosa chaat. Some stalls also fry samosas in vanaspati/ghee blend."
            )
        )

    # Check for Peanuts
    if any("peanut" in a.lower() or "singdana" in a.lower() for a in allergens):
        warnings.append(
            AllergenAlert(
                allergen="Peanuts / Singdana",
                risk_level="Moderate",
                reason=(
                    "While whole peanuts are not clearly visible on top, green chutneys at street stalls "
                    "frequently use ground peanuts for thickening, and the frying oil may be groundnut oil."
                )
            )
        )

    # Fallback general alert if none specific were flagged
    if not warnings:
        warnings.append(
            AllergenAlert(
                allergen="Hidden Dairy & Gluten",
                risk_level="Moderate",
                reason="Dish contains distinct wheat pastry (maida) and fresh curd (dahi)."
            )
        )

    # Questions to ask vendor
    vendor_questions = [
        VendorQuestion(
            question_english="What type of cooking oil is used for deep frying the samosas?",
            question_hindi="भैया, समोसे तलने के लिए कौन सा तेल इस्तेमाल किया है? (रिफाइंड या मूंगफली का तेल?)",
            reason="Street stalls may use groundnut/peanut oil, mustard oil, or generic vegetable blends."
        ),
        VendorQuestion(
            question_english="Does the green chutney or sweet chutney contain peanut paste, sesame, or cashew paste?",
            question_hindi="भैया, हरी या मीठी चटनी में मूंगफली (singdana), काजू या तिल मिलाया है क्या?",
            reason="Chutneys are frequently thickened with crushed peanuts or nut powder."
        ),
        VendorQuestion(
            question_english="Can you prepare a portion without dahi (curd) and without touching the dairy ladles?",
            question_hindi="भैया, बिना दही के अलग प्लेट में दे सकते हैं क्या?",
            reason="Necessary for dairy-free and vegan diners to prevent cross-contamination from the curd spoon."
        ),
    ]

    # Jain check
    if any("jain" in p.lower() for p in dietary_preferences):
        compliance = "NON-COMPLIANT: Contains potatoes (aloo), and chole gravy typically contains onions, garlic, and ginger."
    elif any("vegan" in p.lower() for p in dietary_preferences):
        compliance = "NON-COMPLIANT (As Shown): Contains dairy yogurt (dahi). Can potentially be made vegan if ordered without curd."
    elif any("gluten" in p.lower() for p in dietary_preferences):
        compliance = "NON-COMPLIANT: Contains wheat crust (maida) and potential compounded hing."
    else:
        compliance = "VEGETARIAN: Standard preparation is lacto-vegetarian, but contains dairy and gluten."

    return FoodAnalysisResult(
        dish_name="Dahi Samosa Chaat" if not is_hindi else "दही समोसा चाट",
        cuisine_region="North Indian / Delhi Street Food",
        confidence_summary="High visual confidence based on pastry shape, chickpea gravy, sweet chutney, and curd layer.",
        identified_ingredients=ingredients,
        uncertain_ingredients=[
            "Type of deep-frying oil (Peanut oil, Palm oil, or Soy blend)",
            "Presence of compounded Hing (Asafoetida) containing wheat starch in the chole gravy",
            "Chutney thickeners (roasted peanuts or chana dal)",
            "Cross-contamination from shared ladles and prep tawa"
        ],
        allergen_warnings=warnings,
        dietary_compliance_assessment=compliance,
        vendor_questions=vendor_questions,
        safety_disclaimer=(
            "⚠️ CRITICAL DISCLAIMER: AI image analysis cannot detect invisible allergens, frying oil composition, "
            "or kitchen cross-contamination. Never rely on an image alone to declare food safe. "
            "Always verbally confirm every ingredient with the vendor before consuming."
        )
    )
