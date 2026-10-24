"""app.py

Realbites - Indian Street Food Allergen & Ingredient Intelligence.
A modern Streamlit application utilizing Google Gemma 4 & Gemini multimodal AI
to analyze street food images, identify hidden allergens, and generate
culturally fluent vendor inquiry questions.
"""

import os
import streamlit as st
from PIL import Image, ImageOps
from dotenv import load_dotenv

from services.gemma_service import (
    DEFAULT_GEMMA_MODEL,
    FALLBACK_GEMINI_MODEL,
    FoodAnalysisResult,
    analyze_food_image,
    get_api_key,
)

# Load environment configuration
load_dotenv()

# ============================================================================
# Page Configuration & Modern Custom Styling
# ============================================================================

st.set_page_config(
    page_title="Realbites | Indian Street Food Allergen AI",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# High-impact, responsive CSS for street food aesthetics
st.markdown(
    """
    <style>
    /* Realbites Theme: Warm saffron, charcoal, and emerald accents */
    :root {
        --primary-saffron: #FF6B35;
        --secondary-gold: #F7B05B;
        --alert-crimson: #E63946;
        --warning-amber: #FB8500;
        --success-emerald: #2EC4B6;
        --card-bg: #FFFFFF;
        --text-dark: #1E293B;
    }

    .main-header {
        background: linear-gradient(135deg, #FF6B35 0%, #D84A1B 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(255, 107, 53, 0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF !important;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        font-size: 1.05rem;
        opacity: 0.95;
    }

    .safety-banner {
        background-color: #FEF2F2;
        border-left: 5px solid #E63946;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: #991B1B;
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .safety-banner strong {
        color: #7F1D1D;
    }

    .status-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .ingredient-tag {
        display: inline-block;
        background-color: #F1F5F9;
        color: #334155;
        padding: 0.35rem 0.75rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.88rem;
        border: 1px solid #CBD5E1;
    }
    .ingredient-tag.allergen {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        font-weight: 600;
    }

    .vendor-card {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.85rem;
    }
    .vendor-hindi {
        font-size: 1.15rem;
        font-weight: 700;
        color: #B45309;
        margin: 0.2rem 0;
    }
    .vendor-english {
        font-size: 0.95rem;
        color: #475569;
        font-style: italic;
    }
    .vendor-reason {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 0.3rem;
    }

    .uncertain-card {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Bilingual Content Dictionary
# ============================================================================

TRANSLATIONS = {
    "English": {
        "title": "Realbites",
        "subtitle": "Indian Street Food Allergen & Ingredient Intelligence with Gemma 4",
        "safety_disclaimer_title": "CRITICAL FOOD SAFETY NOTICE:",
        "safety_disclaimer_body": (
            "Food photographs CANNOT reveal invisible allergens, cooking oil purity (such as peanut or mustard oil), "
            "compounded asafoetida (hing) with wheat flour, or cross-contamination from shared street tawas and fryers. "
            "Never claim or assume food is safe based on an AI scan. Always speak directly with the vendor."
        ),
        "sidebar_config": "Settings & Preferences",
        "lang_select": "Choose Language / भाषा चुनें",
        "api_key_header": "Google GenAI API Key",
        "key_found": "API Key detected from environment (.env)",
        "key_missing": "API Key missing. Enter below or use Demo Mode:",
        "key_input_label": "Enter Gemini API Key",
        "key_help": "Get a free API key at https://aistudio.google.com/app/apikey",
        "model_label": "AI Model Selection",
        "model_help": (
            "Gemma 4 models (26B MoE & 31B Dense) provide cutting-edge open reasoning. "
            "If your key has not enabled Gemma 4 hosted endpoints, select Gemini 2.5 Flash as a verified option."
        ),
        "dietary_label": "Your Dietary Preferences",
        "allergen_label": "Known Allergies to Flag",
        "demo_toggle": "Run in Offline Demo Mode (No API key needed)",
        "upload_header": "1. Upload Food or Menu Photo",
        "upload_help": "Upload a clear photo of an Indian street food dish, snack plate, or vendor menu.",
        "camera_label": "Or take a live photo at the stall",
        "sample_btn": "Load Sample Image: Dahi Samosa Chaat",
        "clear_image_btn": "Clear Active Image",
        "analyze_btn": "Analyze Food with Gemma 4",
        "analyzing": "Gemma 4 is inspecting the dish, cross-referencing street recipes, and evaluating allergen risks...",
        "results_header": "Food Intelligence Report",
        "dish_identified": "Dish Identified",
        "regional_origin": "Cuisine / Region",
        "compliance_title": "Dietary Preference Evaluation",
        "ingredients_tab": "Identified Ingredients",
        "allergens_tab": "Allergen Warnings & Hidden Risks",
        "vendor_tab": "Ask The Bhaiya (Vendor Questions)",
        "uncertain_title": "Uncertain & Hidden Elements (Requires Verification)",
        "empty_state": "Upload a food image or choose a sample to begin your analysis.",
    },
    "Hindi": {
        "title": "Realbites (रियल-बाइट्स)",
        "subtitle": "भारतीय स्ट्रीट फूड सामग्री एवं एलर्जी जांच सहायक (Gemma 4 संचालित)",
        "safety_disclaimer_title": "महत्वपूर्ण खाद्य सुरक्षा सूचना:",
        "safety_disclaimer_body": (
            "तस्वीर देखकर कभी भी भोजन को पूरी तरह 'सुरक्षित' या 'एलर्जी-मुक्त' नहीं माना जा सकता। "
            "तस्वीरें छुपा हुआ मूंगफली का तेल, हींग में मिलाया गया गेहूं/मैदा, या साझा तवे पर होने वाले संदूषण को नहीं देख सकती हैं। "
            "खाने से पहले स्टॉल वाले भैया से सीधे बात जरूर करें।"
        ),
        "sidebar_config": "सेटिंग्स और प्राथमिकताएं",
        "lang_select": "Choose Language / भाषा चुनें",
        "api_key_header": "Google GenAI API Key",
        "key_found": "पर्यावरण (.env) से API Key प्राप्त हो गई",
        "key_missing": "API Key नहीं मिली। नीचे दर्ज करें या डेमो मोड चुनें:",
        "key_input_label": "Gemini API Key दर्ज करें",
        "key_help": "Google AI Studio से मुफ्त की प्राप्त करें: https://aistudio.google.com/app/apikey",
        "model_label": "AI मॉडल चयन",
        "model_help": "Gemma 4 मॉडल (26B MoE और 31B Dense) उपलब्ध हैं। यदि आवश्यक हो तो Gemini 2.5 Flash का उपयोग करें।",
        "dietary_label": "आपकी खान-पान प्राथमिकताएं",
        "allergen_label": "एलर्जी जिनसे बचना है",
        "demo_toggle": "ऑफलाइन डेमो मोड (बिना API Key के चलाएं)",
        "upload_header": "१. स्ट्रीट फूड या मेनू की फोटो अपलोड करें",
        "upload_help": "समोसा, चाट, डोसा, वड़ा पाव या किसी भी भारतीय व्यंजन की साफ तस्वीर अपलोड करें।",
        "camera_label": "या स्टॉल पर लाइव कैमरा से फोटो खींचें",
        "sample_btn": "नमूना तस्वीर लोड करें: दही समोसा चाट",
        "clear_image_btn": "तस्वीर हटाएं / रीसेट करें",
        "analyze_btn": "Gemma 4 द्वारा विश्लेषण करें",
        "analyzing": "Gemma 4 व्यंजन, सामग्री और एलर्जी संबंधी खतरों की जांच कर रहा है...",
        "results_header": "खाद्य विश्लेषण रिपोर्ट",
        "dish_identified": "पहचाना गया व्यंजन",
        "regional_origin": "क्षेत्रीय शैली / उद्गम",
        "compliance_title": "खान-पान उपयुक्तता मूल्यांकन",
        "ingredients_tab": "पहचानी गई सामग्री",
        "allergens_tab": "एलर्जी चेतावनियां व छिपे खतरे",
        "vendor_tab": "भैया से पूछें (दुकानदार से सवाल)",
        "uncertain_title": "अनिश्चित व अदृश्य सामग्री (सत्यापन आवश्यक)",
        "empty_state": "विश्लेषण शुरू करने के लिए तस्वीर अपलोड करें या नमूना लोड करें।",
    },
}

DIETARY_OPTIONS = [
    "Vegetarian (शाकाहारी)",
    "Vegan (वीगन - कोई डेयरी नहीं)",
    "Jain (जैन - बिना आलू, प्याज, लहसुन)",
    "Gluten-Free (ग्लूटन-मुक्त - हींग/मैदा से सावधान)",
    "Dairy-Free (डेयरी-मुक्त - मक्खन, घी, पनीर रहित)",
    "Nut-Free (मूंगफली व नट्स मुक्त)",
    "Halal (हलाल)",
]

ALLERGEN_OPTIONS = [
    "Peanuts / Groundnut (मूंगफली / सिंगदाना)",
    "Dairy / Ghee / Paneer / Butter (दूध / घी / पनीर / मक्खन)",
    "Gluten / Wheat / Maida / Hing (गेहूं / मैदा / हींग)",
    "Tree Nuts - Cashew / Almond (काजू / बादाम)",
    "Mustard / Sarson Oil (सरसों / सरसों का तेल)",
    "Sesame / Til (तिल)",
    "Soy / Soya (सोया)",
    "Eggs (अंडे)",
]


# ============================================================================
# Sidebar Configuration
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    # Language Selector
    selected_language = st.radio(
        "Language / भाषा",
        options=["English", "Hindi"],
        index=0,
        horizontal=True,
    )
    t = TRANSLATIONS[selected_language]

    st.markdown("---")
    st.subheader(f"🔑 {t['api_key_header']}")

    # Environment Key detection
    env_key = get_api_key()
    user_api_key = ""

    if env_key:
        st.success(f"✅ {t['key_found']}")
    else:
        st.warning(f"⚠️ {t['key_missing']}")
        user_api_key = st.text_input(
            t["key_input_label"],
            type="password",
            help=t["key_help"],
        )

    # Demo Mode Toggle
    demo_mode = st.checkbox(
        t["demo_toggle"],
        value=(not bool(env_key or user_api_key)),
        help="Use pre-packaged high-fidelity data to test the workflow without consuming API tokens.",
    )

    st.markdown("---")
    st.subheader("🧠 Model Architecture")

    # Model selector with Gemma 4 verified endpoints & Gemini fallback
    model_choices = [
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
        FALLBACK_GEMINI_MODEL,
        "gemini-1.5-flash",
    ]

    selected_model = st.selectbox(
        t["model_label"],
        options=model_choices,
        index=0,
        help=t["model_help"],
    )

    st.caption(
        "💡 **Gemma 4 Model Status**: Gemma 4 is Google's open multimodal family (released April 2026). "
        "Hosted Google AI Studio endpoints support direct vision input and Pydantic schemas. "
        "If your tier requires standard Gemini access, `gemini-2.5-flash` is available automatically."
    )

    st.markdown("---")
    st.subheader("🥗 Dietary Profile")

    selected_dietary = st.multiselect(
        t["dietary_label"],
        options=DIETARY_OPTIONS,
        default=["Vegetarian (शाकाहारी)"],
    )

    selected_allergens = st.multiselect(
        t["allergen_label"],
        options=ALLERGEN_OPTIONS,
        default=["Peanuts / Groundnut (मूंगफली / सिंगदाना)", "Gluten / Wheat / Maida / Hing (गेहूं / मैदा / हींग)"],
    )


# ============================================================================
# Main Application Header & Safety Alert
# ============================================================================

st.markdown(
    f"""
    <div class="main-header">
        <h1>🍛 {t['title']}</h1>
        <p>{t['subtitle']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mandatory Safety Disclaimer (Requirement 8)
st.markdown(
    f"""
    <div class="safety-banner">
        <strong>⚠️ {t['safety_disclaimer_title']}</strong><br/>
        {t['safety_disclaimer_body']}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Input Section: File Upload, Camera, or Sample
# ============================================================================

col_input, col_meta = st.columns([1.2, 0.8])

# Initialize session state for persistent image handling
if "active_image" not in st.session_state:
    st.session_state["active_image"] = None
if "active_source" not in st.session_state:
    st.session_state["active_source"] = None

with col_input:
    st.subheader(t["upload_header"])
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "webp"],
        help=t["upload_help"],
        label_visibility="collapsed",
    )

    camera_file = st.camera_input(t["camera_label"])

    # Update active image if user uploads a new file
    if uploaded_file is not None:
        if st.session_state.get("last_uploaded_name") != uploaded_file.name:
            opened_img = Image.open(uploaded_file)
            st.session_state["active_image"] = ImageOps.exif_transpose(opened_img)
            st.session_state["active_source"] = f"Uploaded file ({uploaded_file.name})"
            st.session_state["last_uploaded_name"] = uploaded_file.name
    elif camera_file is not None:
        if st.session_state.get("last_camera_id") != id(camera_file):
            opened_img = Image.open(camera_file)
            st.session_state["active_image"] = ImageOps.exif_transpose(opened_img)
            st.session_state["active_source"] = "Camera photo"
            st.session_state["last_camera_id"] = id(camera_file)

    # Action buttons: Load sample or clear current image
    sample_col, clear_col = st.columns([1, 1])
    with sample_col:
        if st.button(f"✨ {t['sample_btn']}", use_container_width=True):
            st.session_state["active_image"] = Image.new("RGB", (600, 450), color=(228, 140, 75))
            st.session_state["active_source"] = "Sample: Dahi Samosa Chaat"
            st.session_state["last_uploaded_name"] = None
            st.rerun()

    with clear_col:
        if st.session_state.get("active_image") is not None:
            if st.button(f"🗑️ {t['clear_image_btn']}", use_container_width=True):
                st.session_state["active_image"] = None
                st.session_state["active_source"] = None
                st.session_state["last_uploaded_name"] = None
                if "analysis_result" in st.session_state:
                    del st.session_state["analysis_result"]
                st.rerun()

with col_meta:
    st.subheader("🔍 Active Analysis Filters")
    st.markdown(
        f"""
        <div class="status-card">
            <strong>Language:</strong> {selected_language}<br/>
            <strong>Active Model:</strong> <code>{selected_model}</code><br/>
            <strong>Mode:</strong> {'<span style="color: green; font-weight: bold;">Demo Mode (Offline)</span>' if demo_mode else '<span style="color: blue; font-weight: bold;">Live Gemma 4 API</span>'}<br/>
            <hr style="margin: 0.5rem 0;"/>
            <strong>Dietary Rules:</strong> {', '.join(selected_dietary) if selected_dietary else 'None specified'}<br/>
            <strong>Allergen Filters:</strong> {', '.join(selected_allergens) if selected_allergens else 'None specified'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_img = st.session_state.get("active_image")
    current_source = st.session_state.get("active_source", "None")

    if current_img is not None:
        st.image(current_img, caption=f"Selected: {current_source}", use_container_width=True)


# ============================================================================
# Analysis Action Trigger
# ============================================================================

st.markdown("---")

current_img = st.session_state.get("active_image")
if current_img is None:
    st.info(f"💡 {t['empty_state']}")
else:
    analyze_click = st.button(f"🚀 {t['analyze_btn']}", type="primary", use_container_width=True)

    if analyze_click:
        with st.spinner(t["analyzing"]):
            try:
                result: FoodAnalysisResult = analyze_food_image(
                    image=current_img,
                    dietary_preferences=selected_dietary,
                    allergens=selected_allergens,
                    language=selected_language,
                    model_id=selected_model,
                    api_key=user_api_key or env_key,
                    demo_mode=demo_mode,
                )
                st.session_state["analysis_result"] = result
                st.session_state["analyzed_image"] = current_img
            except Exception as e:
                st.error(f"❌ Analysis encountered an issue: {e}")
                st.info(
                    "💡 Tip: If using live Gemma 4 API, make sure your GEMINI_API_KEY has Gemma 4 access, "
                    "or select 'gemini-2.5-flash' from the sidebar model dropdown."
                )

# ============================================================================
# Results Display
# ============================================================================

if "analysis_result" in st.session_state and st.session_state["analysis_result"]:
    res: FoodAnalysisResult = st.session_state["analysis_result"]

    st.markdown(f"## 📋 {t['results_header']}")

    # Top Metrics Banner
    m1, m2, m3 = st.columns([1, 1, 1.5])
    with m1:
        st.metric(label=t["dish_identified"], value=res.dish_name)
    with m2:
        st.metric(label=t["regional_origin"], value=res.cuisine_region)
    with m3:
        st.metric(label="Visual Confidence", value=res.confidence_summary[:30] + "...")

    # Dietary Compliance Status Card
    compliance_color = "#10B981" if "COMPLIANT" in res.dietary_compliance_assessment and "NON" not in res.dietary_compliance_assessment else "#EF4444"
    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border-left: 6px solid {compliance_color}; padding: 1rem 1.25rem; border-radius: 8px; margin: 1rem 0;">
            <strong style="color: #334155;">{t['compliance_title']}:</strong><br/>
            <span style="font-size: 1.05rem; font-weight: 600; color: {compliance_color};">{res.dietary_compliance_assessment}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Detail Tabs
    tab_ingredients, tab_allergens, tab_vendor, tab_uncertain = st.tabs([
        f"🥗 {t['ingredients_tab']}",
        f"⚠️ {t['allergens_tab']} ({len(res.allergen_warnings)})",
        f"🗣️ {t['vendor_tab']} ({len(res.vendor_questions)})",
        f"❓ {t['uncertain_title']}",
    ])

    # Tab 1: Identified Ingredients
    with tab_ingredients:
        st.write("Ingredients identified through visual cues and customary Indian street recipes:")
        for ing in res.identified_ingredients:
            badge_class = "ingredient-tag allergen" if ing.is_allergen else "ingredient-tag"
            st.markdown(
                f"""
                <div style="margin-bottom: 0.6rem;">
                    <span class="{badge_class}">
                        {'🚨 ' if ing.is_allergen else '🌱 '}{ing.name}
                    </span>
                    <span style="font-size: 0.8rem; color: #64748B; margin-left: 0.5rem;">
                        Confidence: <b>{ing.confidence}</b>
                    </span>
                    <div style="font-size: 0.85rem; color: #475569; margin-left: 1.8rem;">
                        {ing.notes or ''}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Tab 2: Allergen Alerts
    with tab_allergens:
        if not res.allergen_warnings:
            st.success("No high-risk user allergens visibly detected in this preparation.")
        else:
            for alert in res.allergen_warnings:
                risk_color = "#DC2626" if "High" in alert.risk_level else "#D97706"
                st.markdown(
                    f"""
                    <div style="background: #FEF2F2; border-left: 5px solid {risk_color}; padding: 0.9rem 1.2rem; border-radius: 8px; margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.05rem; font-weight: 700; color: #991B1B;">🚨 {alert.allergen}</span>
                            <span style="font-size: 0.8rem; background: {risk_color}; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: 600;">{alert.risk_level}</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #7F1D1D; margin-top: 0.35rem;">
                            {alert.reason}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Tab 3: Vendor Inquiries ("भैया से पूछें")
    with tab_vendor:
        st.info(
            "💡 **Pro-Tip for Indian Street Stalls**: Ask these exact questions before they prepare your plate. "
            "Pronounce the Hindi phrasing if the vendor prefers local dialect!"
        )

        for q in res.vendor_questions:
            st.markdown(
                f"""
                <div class="vendor-card">
                    <div class="vendor-hindi">🗣️ "{q.question_hindi}"</div>
                    <div class="vendor-english"><b>English:</b> {q.question_english}</div>
                    <div class="vendor-reason"><b>Why ask:</b> {q.reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Tab 4: Uncertain / Hidden Elements
    with tab_uncertain:
        st.warning(
            "⚠️ **The Hidden Street Food Reality**: The following elements CANNOT be judged from a photo. "
            "They represent the most frequent causes of accidental allergic reactions in street dining:"
        )
        for item in res.uncertain_ingredients:
            st.markdown(
                f"""
                <div class="uncertain-card">
                    <span style="font-weight: 600; color: #065F46;">🔍 {item}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Bottom Safety Confirmation
    st.markdown("---")
    st.caption(f"🛡️ **System Disclaimer**: {res.safety_disclaimer}")
