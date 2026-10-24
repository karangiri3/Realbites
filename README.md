# Realbites 🍛

**Indian Street Food Allergen & Ingredient Intelligence powered by Google Gemma 4 & Streamlit**

Realbites helps diners, travelers, and food lovers navigate the vibrant world of Indian street food safely. By combining multimodal vision analysis with deep culinary domain knowledge, Realbites identifies likely ingredients, flags hidden allergens (like compounded hing, peanut oil, or hidden dairy), and gives you ready-to-speak questions to ask the stall vendor.

---

## ⚠️ Critical Food Safety Directive

> **Visual AI analysis CANNOT guarantee that food is safe or allergen-free.**
> 
> In Indian street food preparation:
> - **Hidden cooking oils**: Deep fryers frequently use peanut/groundnut oil, mustard oil, or generic vegetable blends.
> - **Compounded Hing (Asafoetida)**: Hing used in dals, chaat, and samosa fillings is almost universally compounded with wheat flour (maida) as an anti-caking agent, posing serious risks for celiac or gluten-sensitive diners.
> - **Cross-contamination**: Vendors frequently use shared tawas (flat pans), deep-fry kadai, and ladles across vegetarian, dairy, and nut-containing dishes.
> - **Invisible traces**: Microscopic traces of allergens cannot be detected by any camera.
> 
> **Always communicate directly with the food stall vendor before eating.**

---

## 🧠 Gemma 4 Model Verification & Architecture

Before executing AI calls, Realbites adheres to the official Google GenAI model specifications:

1. **Official Gemma 4 Model Identifiers**:
   - `gemma-4-26b-a4b-it`: Gemma 4 Mixture-of-Experts (MoE) instruction-tuned model. Combines fast inference with multimodal vision and reasoning.
   - `gemma-4-31b-it`: Flagship 31B dense instruction-tuned multimodal model with deep culinary reasoning.
2. **Endpoint Compatibility**:
   - Gemma 4 models support native multimodal image inputs and Pydantic schemas via the official `google-genai` SDK (`config=GenerateContentConfig(response_mime_type="application/json", response_schema=FoodAnalysisResult)`).
   - *Compatibility Note*: In certain Google AI Studio tiers, hosted Gemma 4 endpoints may be in preview or require specific regional/allowlist enablement. If your API key returns an endpoint unavailable (404/Unsupported) error, Realbites automatically falls back to **`gemini-2.5-flash`**, ensuring zero downtime.
3. **Offline Demo Mode**:
   - If you don't have an API key yet, you can test the complete interface, interactive tabs, and vendor inquiries using the built-in **Offline Demo Mode**.

---

## 🚀 Quickstart Guide for Beginners

### 1. Prerequisites
You already have the Python virtual environment `.venv` configured in `C:\Realbites`.

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

Open `.env` in any text editor and add your Google Gemini API key:
```env
GEMINI_API_KEY=AIzaSy...your_actual_key_here
DEFAULT_MODEL=gemma-4-26b-a4b-it
```
*(Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey))*

### 3. Launch the Application
Run Streamlit using the virtual environment's Python:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Or, if you activate the virtual environment first:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Streamlit will automatically open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

| File | Purpose |
| :--- | :--- |
| `app.py` | Modern Streamlit web application with bilingual UI (English & Hindi), custom street-food CSS theme, image upload, camera capture, and interactive safety cards. |
| `services/gemma_service.py` | AI service layer. Invokes Gemma 4 / Gemini models using `google-genai`, supports `st.secrets` & `.env`, enforces Pydantic structured schemas, and manages endpoint fallbacks. |
| `services/__init__.py` | Package initialization exposing models and functions cleanly. |
| `.streamlit/config.toml` | Streamlit theme colors (saffron/charcoal) and headless server configuration. |
| `requirements.txt` | Project dependencies (`streamlit`, `google-genai`, `pillow`, `pydantic`, `python-dotenv`). |
| `.env.example` | Template for environment variables and model configuration. |
| `.gitignore` | Prevents `.venv`, `.env` secrets, and temporary cache files from being tracked. |
| `README.md` | Comprehensive documentation and beginner quickstart instructions. |

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

1. **Push your code to GitHub**:
   ```powershell
   git add .
   git commit -m "Initial commit of Realbites street food allergen app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/Realbites.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
   - Select your repository (`Realbites`), branch (`main`), and main file path (`app.py`).
   - Click **Advanced Settings** -> **Secrets**.
   - Add your API key in TOML format:
     ```toml
     GEMINI_API_KEY = "your_actual_api_key_here"
     DEFAULT_MODEL = "gemma-4-26b-a4b-it"
     ```
   - Click **Deploy**!


---

## 🌟 Key Features

- **Multimodal Visual Analysis**: Upload a photo or take a live picture at a street stall to inspect dishes like Samosa Chaat, Pav Bhaji, Pani Puri, Vada Pav, and Dosas.
- **Bilingual Interface**: Seamlessly switch between English and Hindi (हिंदी).
- **Personalized Dietary & Allergen Filters**:
  - Diets: Vegetarian, Vegan, Jain (no root vegetables/onion/garlic), Gluten-Free, Dairy-Free, Nut-Free, Halal.
  - Allergens: Peanuts, Dairy, Gluten/Hing, Tree Nuts, Mustard oil, Sesame, Soy, Eggs.
- **"Ask The Bhaiya" (भैया से पूछें)**:
  - Generates culturally fluent, polite questions formatted in both English and colloquial Hinglish/Hindi to speak directly to vendors.
- **Pydantic Validation**: Guarantees structured, reliable outputs that strictly adhere to schema definitions.
- **Uncertain Ingredients Card**: Specifically highlights hidden elements that cannot be judged by eyes alone (cooking oil type, compounded spices, shared frying surfaces).
