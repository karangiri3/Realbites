"""Services package for Realbites.

Contains AI model interaction, schema validation, and allergen reasoning logic.
"""

from .gemma_service import (
    AllergenAlert,
    FoodAnalysisResult,
    IngredientItem,
    VendorQuestion,
    analyze_food_image,
    get_sample_analysis,
)

__all__ = [
    "AllergenAlert",
    "FoodAnalysisResult",
    "IngredientItem",
    "VendorQuestion",
    "analyze_food_image",
    "get_sample_analysis",
]
