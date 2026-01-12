import numpy as np
from typing import List, Dict

from config.prompts import CONTEXTUAL_DESCRIPTION_PROMPT, BOUNDED_OBJECT_CONTEXT_PROMPT, BLIP_ANSWER_CONSENSUS_PROMPT
from src.models import Frame, Object, ContextualCaption
from src.llm import LLM
from src.vlm import BLIPEngine
from src.storage import FrameStore
from src.services import describer
from src.utils import get_logger, retry
from pydantic import BaseModel, Field

logger = get_logger(__name__)

class QuestionList(BaseModel):
    """Schema for VQA questions."""
    questions: List[str] = Field(description="A list of 3 descriptive questions")


def sample_object_frames(obj: Object, interval: int = 10) -> List[int]:
    """
    Sample frame IDs from object's lifetime at regular intervals.
    
    Args:
        obj: Object to sample frames from
        interval: Sampling interval (every nth frame)
    
    Returns:
        List of frame IDs to process
    """
    all_frame_ids = sorted(obj.bounding_boxes.keys())
    
    if not all_frame_ids:
        logger.warning(f"Object {obj.object_id} has no bounding boxes")
        return []
    
    first_frame = all_frame_ids[0]
    last_frame = all_frame_ids[-1]
    
    logger.debug(f"Object {obj.object_id}: frames {first_frame}-{last_frame}")
    
    # Sample every nth frame
    sampled = list(range(first_frame, last_frame + 1, interval))
    
    # Always include last frame
    if last_frame not in sampled:
        sampled.append(last_frame)
    
    # Only keep frames where object exists
    return [fid for fid in sampled if fid in all_frame_ids]

def generate_contextual_questions(obj_label: str, llm: LLM) -> List[str]:
    """
    Generate VQA questions to extract object's spatial context.
    
    Args:
        obj_label: Object category (e.g., "car", "person")
        llm: LLM instance for question generation
    
    Returns:
        List of 3 contextual questions
    """
    prompt = f"The object is: '{obj_label}'"
    response = llm.invoke_structured(
        prompt=prompt,
        system_message=BOUNDED_OBJECT_CONTEXT_PROMPT,
        schema=QuestionList
    )
    questions = response.questions
    logger.debug(f"Generated {len(questions)} questions for '{obj_label}'")
    for idx, q in enumerate(questions, 1):
        logger.debug(f"  Q{idx}: {q}")
    return questions


def describe(
    image: np.ndarray,
    blip_engine: BLIPEngine,
    question: str = "Describe this object's location and its relationship to the surrounding scene.",
    max_length: int = 200,
    num_beams: int = 5
) -> str:
    """
    Describe object using VQA with visual context.
    
    Args:
        frame: Frame containing object
        obj: Object to describe
        blip_engine: BLIPEngine instance
        question: Question to ask about object
        max_length: Maximum answer length
        num_beams: Beam search width
    
    Returns:
        Contextual description
    """
    answer = blip_engine.answer_question(
        image=image,
        question=question,
        max_length=max_length,
        num_beams=num_beams
    )
    
    if not answer or not answer.strip():
        raise ValueError("Empty answer from BLIP VQA")

    return answer

def ask_contextual_questions(
    frame: Frame,
    obj: Object,
    questions: List[str],
    llm: LLM,
    blip_engine: BLIPEngine
) -> List[str]:
    """
    Ask BLIP multiple contextual questions about an object.
    
    Args:
        frame: Frame containing the object
        obj: Object to analyze
        questions: List of questions to ask
        blip_engine: BLIP engine for VQA
    
    Returns:
        List of Q&A pairs
    """
    logger.debug(f"Asking BLIP {len(questions)} questions for object {obj.object_id}")

    bbox = obj.bounding_boxes[frame.frame_id]
    x1, y1, x2, y2 = map(int, bbox.to_list())
    crop = frame.image[y1:y2, x1:x2]

    # Pad the crop to match the original image height and stack horizontally
    padded_crop = np.pad(crop, ((0, frame.image.shape[0] - crop.shape[0]), (0, 0), (0, 0)))
    combined_image = np.hstack((frame.image, padded_crop))
    
    answers = []
    for idx, question in enumerate(questions, 1):
        answer = describe(
            image=combined_image,
            blip_engine=blip_engine,
            question=question,
            max_length=200,
            num_beams=5
        )
        answers.append(f"{question}\n{answer}")
    
    return answers

def synthesize_contextual_caption(
    obj_label: str,
    qa_pairs: List[str],
    llm: LLM
) -> str:
    """
    Synthesize multiple VQA answers into one coherent contextual caption.
    
    Args:
        obj_label: Object category
        qa_pairs: List of question-answer pairs
        llm: LLM for synthesis
    
    Returns:
        Final synthesized caption
    """
    logger.debug(f"Synthesizing {len(qa_pairs)} Q&A pairs for '{obj_label}'")
    
    caption = llm.invoke(
        system_message=CONTEXTUAL_DESCRIPTION_PROMPT,
        prompt=(
            f"The object is: '{obj_label}'.\n"
            f"Contextual captions: {chr(10).join(qa_pairs)}."
        )
    )
    
    logger.debug(f"Synthesized caption (len={len(caption)}): {caption[:100]}...")
    return caption

def describe_object_contextually(
    obj: Object,
    frame_store: FrameStore,
    llm: LLM,
    blip_engine: BLIPEngine,
    sample_interval: int = 10
) -> None:
    """
    Generate contextual captions for an object at sampled frames.
    
    This is the main entry point that orchestrates:
    1. Frame sampling
    2. Question generation
    3. VQA interrogation
    4. Caption synthesis
    
    Args:
        obj: Object to analyze (modified in-place)
        frame_store: Storage for retrieving frames
        llm: LLM for question generation and synthesis
        blip_engine: BLIP for VQA
        sample_interval: Sample every nth frame
    """
    # Get frame IDs to sample
    frame_ids = sample_object_frames(obj, sample_interval)
    
    if not frame_ids:
        return
    
    logger.debug(f"Object {obj.object_id}: sampling {len(frame_ids)} frames")
    
    for frame_id in frame_ids:
        # Get frame
        frame = frame_store.get(frame_id)
        if frame is None:
            logger.warning(f"Frame {frame_id} not found in store")
            continue
        
        logger.debug(f"Generating contextual caption for object {obj.object_id} at frame {frame_id}")
        
        # 1. Generate questions
        questions = generate_contextual_questions(obj.label, llm)
        
        # 2. Ask BLIP
        qa_pairs = ask_contextual_questions(frame, obj, questions, llm, blip_engine)
        
        # 3. Synthesize final caption
        caption = synthesize_contextual_caption(obj.label, qa_pairs, llm)
        
        # 4. Store
        obj.contextual_captions.append(ContextualCaption(
            frame_id=frame_id,
            timestamp_sec=round(frame.timestamp_sec, 1),
            caption=caption
        ))
