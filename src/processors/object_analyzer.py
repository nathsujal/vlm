from typing import List, Dict

from src.llm import LLM
from src.vlm import BLIPEngine
from src.storage import ObjectStore, FrameStore, ObjectsAttributeRegistry
from src.services import profiler, narrator
from src.utils import get_logger
from src.models import Object

logger = get_logger(__name__)

class ObjectAnalyzer:
    """
    Object analysis pipeline.
    
    Pipeline:
    """

    def __init__(
        self,
        llm: LLM,
        frame_store: FrameStore,
        blip_engine: BLIPEngine,

    ):
        self.llm = llm
        self.frame_store = frame_store
        self.blip_engine = blip_engine

        self.objects_attribute_registry = ObjectsAttributeRegistry()

        logger.info("ObjectAnalyzer initialized")

    def process(self, objects: ObjectStore):
        objects_list = objects.get_all_objects()
        logger.info(f"Processing {len(objects_list)} objects for attribute analysis")

        for idx, obj in enumerate(objects_list):
            try:
                logger.info(f"[{idx+1}/{len(objects_list)}] Processing object ID: {obj.object_id} ({obj.label})")

                # 1. Generate object attributes
                try:
                    attributes: Dict[str, Attribute] = profiler.generate_object_attributes(
                        label=obj.label,
                        llm=self.llm
                    )
                    logger.info(f"Generated {len(attributes)} attributes for object {obj.object_id}")
                except Exception as e:
                    logger.error(f"Failed to generate attributes for object {obj.object_id}: {e}")
                    attributes = {}

                # 2. Ask questions about the image based on the attribute schema
                if attributes:
                    try:
                        # Get first frame where object appears
                        first_frame_id = list(obj.bounding_boxes.keys())[0]
                        obj_bbox = obj.bounding_boxes.get(first_frame_id).to_list()
                        attributes: Dict[str, str] = profiler.interrogate_visual_attributes(
                            obj_bbox=obj_bbox,
                            image=self.frame_store.get(first_frame_id).image,
                            attributes=attributes,
                            llm=self.llm,
                            blip_engine=self.blip_engine
                        )
                        logger.info(f"Interrogated {len(attributes)} attributes for object {obj.object_id}")
                    except Exception as e:
                        logger.error(f"Failed to interrogate attributes for object {obj.object_id}: {e}")
                        attributes = {}

                # 3. Register attributes
                if attributes:
                    try:
                        self.objects_attribute_registry.register_attributes(obj.object_id, attributes)
                        logger.info(f"✓ Registered {len(attributes)} attributes for object {obj.object_id}")
                    except Exception as e:
                        logger.error(f"Failed to register attributes for object {obj.object_id}: {e}")

            except Exception as e:
                logger.error(f"Failed to process object {obj.object_id}: {e}", exc_info=True)
                continue

        for idx, obj in enumerate(objects_list):
            try:
                logger.info(f"[{idx+1}/{len(objects_list)}] Generating contextual captions for object {obj.object_id}")
                
                # 1. Generate contextual captions at sampled frames
                try:
                    narrator.describe_object_contextually(
                        obj=obj,
                        frame_store=self.frame_store,
                        llm=self.llm,
                        blip_engine=self.blip_engine,
                        sample_interval=10
                    )
                    contextual_captions = obj.contextual_captions
                    logger.info(f"Generated {len(contextual_captions)} contextual captions for object {obj.object_id}")
                except Exception as e:
                    logger.error(f"Failed to generate contextual captions for object {obj.object_id}: {e}")
                    contextual_captions = []

                # 2. Register events
                if contextual_captions:
                    try:
                        self.objects_attribute_registry.register_events(
                            object_id=obj.object_id,
                            events=contextual_captions
                        )
                        logger.info(f"✓ Registered {len(contextual_captions)} contextual snapshots for object {obj.object_id}")
                    except Exception as e:
                        logger.error(f"Failed to register events for object {obj.object_id}: {e}")

            except Exception as e:
                logger.error(f"Failed to process contextual analysis for object {obj.object_id}: {e}", exc_info=True)
                continue


