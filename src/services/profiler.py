import numpy as np
from typing import Tuple, Dict, Optional, Any

from config.prompts import ATTRIBUTE_SCHEMA_PROMPT, INTRINSIC_OBJECT_PROMPT, INTRINSIC_QUESTION_PROMPT

from src.llm import LLM
from src.utils import get_logger, retry
from src.models import Frame, Object, Attribute, AttributeSchema

logger = get_logger(__name__)


def generate_scene_attributes(
    description: str,
    llm: LLM
) -> Dict[str, Attribute]:
    """
    Generate attribute schema for a scene description.
    
    Args:
        description: Scene description from BLIP
        llm: LLM instance
        
    Returns:
        AttributeSchema with validated attributes
    """
    result: AttributeSchema = llm.invoke_structured(
        prompt=f"Analyze this scene and generate security attributes:\n\n{description}",
        system_message=ATTRIBUTE_SCHEMA_PROMPT,
        schema=AttributeSchema
    )
    
    return result.attributes


def generate_object_attributes(
    label: str,
    llm: LLM
) -> Dict[str, Attribute]:
    """
    Generate attribute schema for an object description.
    
    Args:
        label: Object label
        llm: LLM instance
        
    Returns:
        AttributeSchema with validated attributes
    """
    result: AttributeSchema = llm.invoke_structured(
        prompt=f"Analyze this object and generate security attributes:\n\n{label}",
        system_message=INTRINSIC_OBJECT_PROMPT,
        schema=AttributeSchema
    )
    
    return result.attributes

def interrogate_visual_attributes(
    image: np.ndarray,
    attributes: Dict[str, Attribute],
    llm: LLM,
    blip_engine: 'BLIPEngine',
    obj_bbox: Optional[Tuple[int, int, int, int]] = None
) -> Dict[str, str]:
    """
    Ask questions about the image based on the attribute schema.
    
    Args:
        obj_bbox: Object bounding box
        image: Image array
        attribute_schema: Attribute schema
        llm: LLM instance
        blip_engine: BLIP engine for captioning
        debug: Enable debug visualization and logging
        
    Returns:
        Dict of attribute values
    """
    attributes_dict = {}
    for attr_name, attr_data in attributes.items():
        prompt = (
          "Generate a question for this attribute:\n"
          f"Name: {attr_name}\n"
          f"Description: {attr_data.description}\n"
          f"Values: {attr_data.values}" if attr_data.values else ""
          f"Type: {attr_data.type}" if attr_data.type else ""
        )

        question = llm.invoke(
            prompt,
            system_message=INTRINSIC_QUESTION_PROMPT
        )

        answer = blip_engine.answer_question(
            image=image,
            question=question,
            bbox=obj_bbox
        )
        
        attributes_dict[attr_name] = answer
    
    return attributes_dict

