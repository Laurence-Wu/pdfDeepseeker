# Gemini Client Implementation - Part 2: Prompt Engine

## Overview
Advanced prompt engineering for optimal translation quality with constraints.

## Implementation

```python
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from dataclasses import dataclass

class DocumentType(Enum):
    """Document type enumeration"""
    SCIENTIFIC = "scientific"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    BUSINESS = "business"
    LITERARY = "literary"
    GENERAL = "general"

class PromptEngine:
    """
    Generate optimized prompts for translation via OpenRouter.
    Handles constraints, context, and document-specific requirements.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize prompt engine"""
        self.config = config or {}
        self.templates = self._load_templates()
        self.terminology_db = TerminologyDatabase()
        
    def generate_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        document_type: DocumentType,
        context: Optional[Dict] = None,
        constraints: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate optimized translation prompt.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            document_type: Type of document
            context: Translation context
            constraints: Translation constraints
            metadata: Additional metadata
            
        Returns:
            Optimized prompt string
        """
        # Build prompt components
        components = []
        
        # 1. Task definition
        components.append(self._generate_task_definition(
            source_lang, target_lang, document_type
        ))
        
        # 2. Constraints section
        if constraints:
            components.append(self._generate_constraints_section(constraints))
        
        # 3. Context section
        if context:
            components.append(self._generate_context_section(context, document_type))
        
        # 4. Terminology guidance
        if self.terminology_db.has_terms(document_type):
            components.append(self._generate_terminology_section(
                document_type, source_lang, target_lang
            ))
        
        # 5. Special instructions
        components.append(self._generate_special_instructions(
            document_type, metadata
        ))
        
        # 6. Text to translate
        components.append(f"TEXT TO TRANSLATE:\n{text}")
        
        # 7. Output format
        components.append(self._generate_output_format())
        
        return "\n\n".join(filter(None, components))
    
    def _generate_task_definition(
        self,
        source_lang: str,
        target_lang: str,
        document_type: DocumentType
    ) -> str:
        """Generate task definition section"""
        
        lang_names = {
            'en': 'English',
            'zh': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'ru': 'Russian',
            'pt': 'Portuguese'
        }
        
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        return (
            f"Translate the following {document_type.value} document content "
            f"from {source_name} to {target_name}."
        )
    
    def _generate_constraints_section(self, constraints: Dict) -> str:
        """Generate constraints section"""
        
        constraint_lines = ["CRITICAL CONSTRAINTS:"]
        
        # Length constraint
        if 'max_length' in constraints:
            constraint_lines.append(
                f"• Maximum length: {constraints['max_length']} characters "
                f"(current: {constraints.get('current_length', 'unknown')})"
            )
            constraint_lines.append(
                "• If translation exceeds limit, use abbreviations and concise phrasing"
            )
        
        # Layout constraints
        if 'preserve_lines' in constraints:
            constraint_lines.append(
                f"• Preserve line breaks at positions: {constraints['preserve_lines']}"
            )
        
        if 'bbox' in constraints:
            bbox = constraints['bbox']
            constraint_lines.append(
                f"• Text must fit within box: {bbox['width']}x{bbox['height']} pixels"
            )
        
        # Style constraints
        if 'preserve_formatting' in constraints:
            constraint_lines.append("• Maintain exact formatting (bold, italic, underline)")
        
        return "\n".join(constraint_lines)
    
    def _generate_context_section(
        self,
        context: Dict,
        document_type: DocumentType
    ) -> str:
        """Generate context section"""
        
        context_lines = ["CONTEXT:"]
        
        # Previous translations for consistency
        if 'previous_translations' in context:
            context_lines.append("Previously translated terms:")
            for term, translation in context['previous_translations'].items():
                context_lines.append(f"• {term} → {translation}")
        
        # Surrounding text
        if 'before_text' in context:
            context_lines.append(f"Previous text: ...{context['before_text'][-100:]}")
        
        if 'after_text' in context:
            context_lines.append(f"Following text: {context['after_text'][:100]}...")
        
        # Page context
        if 'page_number' in context:
            context_lines.append(f"Page {context['page_number']} of {context.get('total_pages', '?')}")
        
        return "\n".join(context_lines)
    
    def _generate_terminology_section(
        self,
        document_type: DocumentType,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Generate terminology guidance"""
        
        terms = self.terminology_db.get_terms(document_type, source_lang, target_lang)
        
        if not terms:
            return ""
        
        lines = ["TERMINOLOGY GUIDELINES:"]
        for category, term_list in terms.items():
            lines.append(f"\n{category}:")
            for source_term, target_term in term_list:
                lines.append(f"• {source_term} → {target_term}")
        
        return "\n".join(lines)
    
    def _generate_special_instructions(
        self,
        document_type: DocumentType,
        metadata: Optional[Dict]
    ) -> str:
        """Generate document-specific instructions"""
        
        instructions = {
            DocumentType.SCIENTIFIC: [
                "Preserve all formula references (e.g., Equation 1, Figure 2)",
                "Maintain scientific notation and units",
                "Keep citation formats unchanged [1], (Smith, 2023)"
            ],
            DocumentType.LEGAL: [
                "Use formal legal terminology",
                "Preserve section numbering (1.1, 1.2, etc.)",
                "Maintain party names exactly as written"
            ],
            DocumentType.TECHNICAL: [
                "Keep all product codes and model numbers unchanged",
                "Preserve measurement units and specifications",
                "Maintain technical abbreviations"
            ],
            DocumentType.MEDICAL: [
                "Use standard medical terminology (ICD-10 where applicable)",
                "Preserve drug names in original form",
                "Maintain dosage formats (mg, ml, etc.)"
            ],
            DocumentType.BUSINESS: [
                "Maintain professional tone",
                "Preserve company names and brands",
                "Keep financial figures and currencies as-is"
            ]
        }
        
        doc_instructions = instructions.get(document_type, [])
        
        if not doc_instructions:
            return ""
        
        lines = ["SPECIAL INSTRUCTIONS:"]
        lines.extend(f"• {inst}" for inst in doc_instructions)
        
        # Add metadata-specific instructions
        if metadata:
            if metadata.get('is_title'):
                lines.append("• This is a title - keep concise and impactful")
            if metadata.get('is_header'):
                lines.append("• This is a header - maintain brevity")
            if metadata.get('is_footer'):
                lines.append("• This is a footer - preserve page references")
        
        return "\n".join(lines)
    
    def _generate_output_format(self) -> str:
        """Generate output format instructions"""
        return (
            "OUTPUT FORMAT:\n"
            "Provide ONLY the translation without any explanations, notes, or markup.\n"
            "Do not include phrases like 'Here is the translation:' or similar."
        )
    
    def _load_templates(self) -> Dict:
        """Load prompt templates"""
        return {
            'length_critical': (
                "⚠️ LENGTH CRITICAL: The translation MUST be {max_chars} characters or less. "
                "Current text is {current_chars} characters. "
                "Use abbreviations if necessary to meet this requirement."
            ),
            'table_cell': (
                "Translate this table cell content. "
                "Keep extremely concise to fit table formatting."
            ),
            'formula_context': (
                "This text appears near a mathematical formula. "
                "Preserve any references to equations or variables."
            )
        }
    
    def generate_retry_prompt(
        self,
        original_text: str,
        failed_translation: str,
        reason: str,
        constraints: Dict
    ) -> str:
        """
        Generate retry prompt after translation failure.
        
        Args:
            original_text: Original source text
            failed_translation: Translation that failed constraints
            reason: Reason for failure
            constraints: Constraints that weren't met
            
        Returns:
            Retry prompt
        """
        return (
            f"The previous translation failed because: {reason}\n\n"
            f"Failed translation: {failed_translation}\n\n"
            f"Please retranslate with these requirements:\n"
            f"{self._generate_constraints_section(constraints)}\n\n"
            f"Original text: {original_text}\n\n"
            "Provide a shorter translation that meets ALL constraints."
        )


class TerminologyDatabase:
    """Manage domain-specific terminology"""
    
    def __init__(self):
        self.terms = self._load_terminology()
    
    def has_terms(self, document_type: DocumentType) -> bool:
        """Check if terminology exists for document type"""
        return document_type.value in self.terms
    
    def get_terms(
        self,
        document_type: DocumentType,
        source_lang: str,
        target_lang: str
    ) -> Dict:
        """Get terminology for translation pair"""
        
        key = f"{document_type.value}_{source_lang}_{target_lang}"
        return self.terms.get(key, {})
    
    def _load_terminology(self) -> Dict:
        """Load terminology database"""
        # This would typically load from a database or file
        return {
            'scientific_en_zh': {
                'Common Terms': [
                    ('hypothesis', '假设'),
                    ('methodology', '方法论'),
                    ('statistical significance', '统计显著性')
                ]
            },
            'technical_en_zh': {
                'Engineering': [
                    ('torque', '扭矩'),
                    ('specification', '规格'),
                    ('tolerance', '公差')
                ]
            }
        }
```

## Advanced Features

```python
class PromptOptimizer:
    """Optimize prompts based on model and context"""
    
    def optimize_for_model(
        self,
        prompt: str,
        model: str,
        token_limit: int = 2048
    ) -> str:
        """
        Optimize prompt for specific model characteristics.
        
        Args:
            prompt: Original prompt
            model: Model name (e.g., 'google/gemini-pro-1.5')
            token_limit: Maximum tokens
            
        Returns:
            Optimized prompt
        """
        # Model-specific optimizations
        if 'gemini' in model.lower():
            # Gemini responds well to structured prompts
            prompt = self._add_structure_markers(prompt)
        elif 'claude' in model.lower():
            # Claude prefers detailed context
            prompt = self._expand_context(prompt)
        elif 'llama' in model.lower():
            # Llama works better with examples
            prompt = self._add_examples(prompt)
        
        # Ensure within token limit
        prompt = self._trim_to_token_limit(prompt, token_limit)
        
        return prompt
    
    def _add_structure_markers(self, prompt: str) -> str:
        """Add structure markers for better parsing"""
        # Add clear section markers
        prompt = prompt.replace("CONSTRAINTS:", "=== CONSTRAINTS ===")
        prompt = prompt.replace("CONTEXT:", "=== CONTEXT ===")
        prompt = prompt.replace("TEXT TO TRANSLATE:", "=== TEXT TO TRANSLATE ===")
        return prompt
    
    def _trim_to_token_limit(self, prompt: str, limit: int) -> str:
        """Trim prompt to token limit"""
        # Rough estimation: 1 token ≈ 4 characters
        char_limit = limit * 4
        if len(prompt) > char_limit:
            # Preserve essential parts
            essential_end = prompt.rfind("TEXT TO TRANSLATE:")
            if essential_end > 0:
                prefix = prompt[:essential_end]
                text_section = prompt[essential_end:]
                available = char_limit - len(text_section) - 100  # Buffer
                prefix = prefix[:available] + "...\n\n"
                prompt = prefix + text_section
        return prompt
```

## Usage Example

```python
# Initialize engine
engine = PromptEngine()

# Generate prompt
prompt = engine.generate_prompt(
    text="本文档描述了系统架构设计。",
    source_lang="zh",
    target_lang="en",
    document_type=DocumentType.TECHNICAL,
    constraints={
        'max_length': 50,
        'current_length': 35
    },
    context={
        'page_number': 5,
        'previous_translations': {
            '系统': 'system',
            '架构': 'architecture'
        }
    }
)

print(prompt)
```
