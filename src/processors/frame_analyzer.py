from typing import Iterator, List
from toon import encode

from src.llm import LLM
from src.vlm import BLIPEngine, CLIPEngine
from src.models import Frame, AttributeSchema, Attribute
from src.storage import FrameStore, FrameIndexer, PropertyAttributeRegistry
from src.services import embedder, indexer, searcher, describer, profiler
from src.utils import get_logger

logger = get_logger(__name__)


class FrameAnalyzer:
    """
    Frame analysis pipeline.
    
    Pipeline:
    1. Encode frames at intervals (CLIP)
    2. Index for similarity search (FAISS)
    3. Check novelty
    4. Describe novel frames (BLIP)
    5. Generate scene attributes (LLM)
    """
    
    def __init__(
        self,
        llm: LLM,
        blip_engine: BLIPEngine,
        clip_engine: CLIPEngine,
        frame_indexer: FrameIndexer,
        embedding_interval: int = 15,
        novelty_threshold: float = 0.70
    ):
        """
        Initialize Frame Analyzer.
        
        Args:
            llm: Language model for reasoning
            blip_engine: BLIP for captioning
            clip_engine: CLIP for embeddings
            frame_indexer: FAISS indexer
            embedding_interval: Generate embedding every N frames
            novelty_threshold: Similarity threshold for novelty
        """
        self.llm = llm
        self.blip_engine = blip_engine
        self.clip_engine = clip_engine
        self.indexer = frame_indexer
        
        self.embedding_interval = embedding_interval
        self.novelty_threshold = novelty_threshold

        # Attribute registry - for storing and reasoning about scene attributes
        self.attribute_registry = PropertyAttributeRegistry(llm)
        
        # Statistics
        self.frames_processed = 0
        self.embeddings_generated = 0
        self.frames_indexed = 0
        self.scenes_described = 0
        self.scenes_skipped = 0

        self.descriptions = []

        logger.info(
            f"FrameAnalyzer initialized | "
            f"Embedding interval: {embedding_interval} | "
            f"Novelty threshold: {novelty_threshold}"
        )
    
    def process(self, frames: List[Frame]) -> None:
        """
        Process frames through analysis pipeline.
        
        Pipeline:
        1. Filter frames for embedding (every N frames)
        2. Batch check novelty FIRST (before adding to index)
        3. Batch encode and index all target frames
        4. Batch describe novel frames
        5. Generate scene attributes for novel frames
        """
        if not frames:
            return
        
        # 1. Get frames for embedding (every N frames)
        target_frames = [
            f for f in frames 
            if f.frame_id % self.embedding_interval == 0
        ]
        
        if not target_frames:
            logger.debug("No frames to process in this batch")
            return

        # 2. Batch check novelty FIRST (before indexing new frames)
        novelty_results = searcher.batch_check_novelty(
            target_frames,
            self.clip_engine,
            self.indexer,
            threshold=self.novelty_threshold
        )

        # Separate novel and similar frames, and SET is_novel flag
        novel_frames = []
        similar_frames = []
        for frame, is_novel in zip(target_frames, novelty_results):
            frame.is_novel = is_novel  # Set the flag!
            if is_novel:
                novel_frames.append(frame)
            else:
                similar_frames.append(frame)

        logger.info(
            f"Novelty check: {len(novel_frames)} novel, "
            f"{len(similar_frames)} similar"
        )

        # 3. Batch encode and index ALL target frames (novel will be in index for next batch)
        indexer.batch_index_frames(
            target_frames,
            self.clip_engine, 
            self.indexer
        )

        self.embeddings_generated += len(target_frames)
        self.frames_indexed += len(target_frames)
        self.scenes_skipped += len(similar_frames)

        logger.info(
            f"Encoded and indexed {len(target_frames)} frames "
            f"(every {self.embedding_interval})"
        )

        # 4. Batch describe novel frames only
        if novel_frames:
            describer.batch_describe_scenes(
                novel_frames,
                self.blip_engine
            )
            
            self.scenes_described += len(novel_frames)
            
            logger.info(f"Described {len(novel_frames)} novel scenes")

        # 5. Generate scene attributes for novel frames
        for frame in novel_frames:
            frame_data = {
                "frame_id": frame.frame_id,
                "timestamp_sec": frame.timestamp_sec,
                "captured_at": frame.captured_at.isoformat(),
                "description": frame.caption if frame.caption else "",
                "attributes": {}
            }
            
            if frame.caption:
                attributes: Dict[str, Attribute] = profiler.generate_scene_attributes(
                    description=frame.caption,
                    llm=self.llm
                )
                
                # 6. Ask questions about the image based on the attribute schema
                attributes: Dict[str, str] = profiler.interrogate_visual_attributes(
                    image=frame.image,
                    attributes=attributes,
                    llm=self.llm,
                    blip_engine=self.blip_engine
                )
                
                # Store attributes in frame data
                frame_data["attributes"] = attributes

                # 7. Register attributes
                self.attribute_registry.register(attributes)
            
            # Add complete frame data to descriptions list
            self.descriptions.append(frame_data)
        
        self.frames_processed += len(frames)
    
    def process_single(self, frame: Frame) -> None:
        """
        Process single frame.
        
        Args:
            frame: Frame to process
        """
        # Check if should encode
        if frame.frame_id % self.embedding_interval != 0:
            logger.debug(f"Skipping frame {frame.frame_id} (not at interval)")
            return
        
        # Encode and index
        indexer.index_frame(frame, self.clip_engine, self.indexer)
        self.embeddings_generated += 1
        self.frames_indexed += 1
        
        # Check novelty
        is_novel = searcher.check_novelty(
            frame,
            self.clip_engine,
            self.indexer,
            threshold=self.novelty_threshold
        )
        
        # Describe if novel
        if is_novel:
            describer.describe_scene(frame, self.blip_engine)
            self.scenes_described += 1
            logger.info(f"Frame {frame.frame_id}: Novel - described")
        else:
            self.scenes_skipped += 1
            logger.debug(f"Frame {frame.frame_id}: Similar - skipped")
        
        self.frames_processed += 1

    def get_descriptions(self) -> str:
        """Get descriptions of novel scenes in toon format."""
        return encode(self.descriptions)

    def get_attributes(self) -> str:
        """Get attributes of novel scenes in toon format."""
        return self.attribute_registry.get_all_toon()
    
    def get_stats(self) -> dict:
        """Get analyzer statistics."""
        return {
            "frames_processed": self.frames_processed,
            "embeddings_generated": self.embeddings_generated,
            "frames_indexed": self.frames_indexed,
            "scenes_described": self.scenes_described,
            "scenes_skipped": self.scenes_skipped,
            "description_rate": (
                f"{self.scenes_described / max(1, self.frames_processed) * 100:.1f}%"
            ),
            "indexer_stats": indexer.get_index_stats(self.indexer),
            "clip_stats": self.clip_engine.get_stats(),
            "blip_stats": self.blip_engine.get_stats()
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.frames_processed = 0
        self.embeddings_generated = 0
        self.frames_indexed = 0
        self.scenes_described = 0
        self.scenes_skipped = 0
        logger.info("Statistics reset")