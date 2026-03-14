"""
DeceptGuard Taxonomy Module
12-category deception taxonomy for LLM agents.
"""

from .definitions import (
    DeceptionCategory,
    DeceptionMacroCategory,
    TAXONOMY,
    DECEPTION_CONSTITUTION,
    get_category,
    get_categories_by_macro,
)

__all__ = [
    "DeceptionCategory",
    "DeceptionMacroCategory",
    "TAXONOMY",
    "DECEPTION_CONSTITUTION",
    "get_category",
    "get_categories_by_macro",
]
