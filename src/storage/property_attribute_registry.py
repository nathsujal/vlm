from typing import Dict, Optional
import numpy as np
import ollama
from functools import lru_cache
from toon import encode

from src.llm import LLM
from src.utils import get_logger

logger = get_logger(__name__)

class PropertyAttributeRegistry:
    """
    Central registry for property attributes.
    
    Uses embeddings to detect similar attribute names and LLM to reconcile
    conflicting values.
    """
    
    def __init__(self, llm: LLM, embedding_model: str = 'nomic-embed-text:v1.5', similarity_threshold: float = 0.80):
        """
        Initialize attribute registry.
        
        Args:
            llm: LLM instance for conflict resolution
            embedding_model: Ollama embedding model name
            similarity_threshold: Cosine similarity threshold (0-1)
        """
        self.registry: Dict[str, str] = {}  # attr_name -> value
        self.llm = llm
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"PropertyAttributeRegistry initialized | Model: {embedding_model} | Threshold: {similarity_threshold}")
    
    @lru_cache(maxsize=200)
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get cached embedding for text."""
        result = ollama.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        return np.array(result['embedding'])
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def _find_similar_attribute(self, attr_name: str) -> Optional[tuple[str, float]]:
        """
        Find most similar attribute in registry.
        
        Args:
            attr_name: Attribute name to search for
            
        Returns:
            (similar_name, similarity_score) or None
        """
        if not self.registry:
            return None
        
        query_emb = self._get_embedding(attr_name)
        
        best_match = None
        best_similarity = 0.0
        
        for existing_name in self.registry:
            existing_emb = self._get_embedding(existing_name)
            similarity = self._compute_similarity(query_emb, existing_emb)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = existing_name
        
        # Only return if above threshold
        if best_similarity >= self.similarity_threshold:
            return (best_match, best_similarity)
        
        return None
    
    def _reconcile_values(self, attr_name: str, old_value: str, new_value: str) -> str:
        """
        Use LLM to reconcile conflicting attribute values.
        
        Args:
            attr_name: Attribute name
            old_value: Existing value in registry
            new_value: New value from current frame
            
        Returns:
            Reconciled value
        """
        prompt = f"""
You are analyzing attribute values from drone security footage.

Attribute: {attr_name}
Existing value: "{old_value}"
New value: "{new_value}"

Task: Provide the single most accurate value that best represents both observations.

Rules:
- If values are equivalent, return one of them
- If one is more specific, choose the more specific
- If contradictory, choose the more informative one
- Return ONLY the final value, nothing else

Final value:"""

        reconciled = self.llm.invoke(prompt).strip().strip('"\'')
        
        logger.info(f"Reconciled '{attr_name}': '{old_value}' + '{new_value}' → '{reconciled}'")
        
        return reconciled
    
    def register(self, attributes: Dict[str, str]) -> None:
        """
        Register new attributes with intelligent deduplication.
        
        Logic:
        1. For each new attribute:
           - Find if similar attribute exists
           - If no: add directly
           - If yes: compare values
             - If values similar: keep existing
             - If values different: use LLM to reconcile
        
        Args:
            attributes: Dictionary of attribute_name -> value
        """
        if not self.registry:
            # First registration - add directly
            self.registry = attributes.copy()
            logger.info(f"Initialized registry with {len(attributes)} attributes")
            return
        
        # Process each new attribute
        for new_attr_name, new_value in attributes.items():
            # Step 1: Find similar attribute
            similar = self._find_similar_attribute(new_attr_name)
            
            if similar is None:
                # No similar attribute found - add directly
                self.registry[new_attr_name] = new_value
                logger.info(f"Added new attribute: {new_attr_name} = '{new_value}'")
            
            else:
                similar_name, similarity = similar
                logger.debug(f"Found similar: '{new_attr_name}' → '{similar_name}' (sim: {similarity:.2f})")
                
                # Get existing value
                old_value = self.registry[similar_name]
                
                # Step 2: Compare values
                if old_value.lower() == new_value.lower():
                    # Values are the same - keep existing
                    logger.debug(f"Same value for '{similar_name}', skipping")
                    continue
                
                else:
                    # Values differ - reconcile with LLM
                    reconciled = self._reconcile_values(similar_name, old_value, new_value)
                    
                    # Update registry with reconciled value
                    # Use the existing attribute name (not the new one)
                    self.registry[similar_name] = reconciled
    
    def get_all(self) -> Dict[str, str]:
        """Get all registered attributes."""
        return self.registry.copy()

    def get_all_toon(self) -> str:
        """Decode all registered attributes in toon format."""
        return encode(self.registry)
    
    def get(self, attr_name: str) -> Optional[str]:
        """Get specific attribute value."""
        return self.registry.get(attr_name)
    
    def __len__(self) -> int:
        """Number of registered attributes."""
        return len(self.registry)
    
    def __repr__(self) -> str:
        return f"PropertyAttributeRegistry({len(self.registry)} attributes)"