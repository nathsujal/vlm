"""Test script for running perception pipeline on a directory of frames."""

import json
import yaml
import argparse
import cv2
from toon import encode
from datetime import datetime, time
from pathlib import Path

from src.llm import LLM
from src.processors import Perciever, FrameAnalyzer, ObjectAnalyzer, Reporter
from src.data.loader import DroneFootageLoader
from src.detection import RTDETRDetector, VisualObjectTracker
from src.storage import FrameStore, FrameIndexer, ObjectStore
from src.vlm import BLIPEngine, CLIPEngine
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    """Run perception pipeline on image directory."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run perception pipeline on drone footage"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing image frames or video file"
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default="09:00:00",
        help="Recording start time in HH:MM:SS format (default: 09:00:00)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show real-time visualization"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return
    
    # Parse start time
    try:
        hour, minute, second = map(int, args.start_time.split(":"))
        start_time = datetime.now().replace(
            hour=hour,
            minute=minute,
            second=second,
            microsecond=0
        )
        logger.info(f"Recording start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    except ValueError:
        logger.error(f"Invalid time format: {args.start_time}. Use HH:MM:SS")
        return
    
    # Initialize components
    logger.info("PERCEPTION PIPELINE INITIALIZATION")

    # 0. Frame Store
    frame_store = FrameStore(persist_to_disk=True, cache_dir="data/frame_cache")
    
    # 1. Data Loader
    logger.info(f"Loading frames from: {input_path}")
    loader = DroneFootageLoader(
        source=str(input_path),
        record_start_time=start_time,
        frames=frame_store
    )
    
    # Show loader info
    info = loader.info()
    logger.info(f"Loader info: \n{yaml.dump(info, default_flow_style=False)}")
    
    # 2. Object Detector
    detector = RTDETRDetector()
    
    # 3. Visual Tracker
    tracker = VisualObjectTracker()
    
    # 3.5. Object Store for tracking
    object_store = ObjectStore()
    
    # 4. Perciever
    logger.info(f"Setting up perciever")
    perciever = Perciever(
        frames=frame_store,
        detector=detector,
        tracker=tracker,
        object_store=object_store,
        detect_interval=5
    )
    logger.info("STARTING PERCIEVER")
    
    # PROCESS ALL FRAMES
    logger.info("Processing all frames (detection + tracking)")
    logger.info("This may take a while...")
    
    try:
        for frame in perciever.process():
            frame_store.update(frame)
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise
    
    # Check if we have frames to visualize
    if not frame_store.get_all():
        logger.error("No frames were processed!")
        return
    
    logger.info(f"✓ Processed {len(frame_store.get_all())} frames successfully")

    # 5. Analyzer
    logger.info(f"Setting up analyzer")
    
    # Initialize components
    llm = LLM()
    blip_engine = BLIPEngine()
    clip_engine = CLIPEngine()
    frame_indexer = FrameIndexer(clip_engine=clip_engine)
    
    analyzer = FrameAnalyzer(
        llm=llm,
        blip_engine=blip_engine,
        clip_engine=clip_engine,
        frame_indexer=frame_indexer,
        embedding_interval=15,
        novelty_threshold=0.60
    )
    
    # PROCESS ALL FRAMES
    logger.info("Processing all frames (encoding + indexing)")
    logger.info("This may take a while...")
    
    # Get all frames from frame_store
    frames = list(frame_store.get_all())
    logger.info(f"Processing {len(frames)} frames")
    
    try:
        analyzer.process(frames)
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise

    descriptions = analyzer.get_descriptions()
    logger.info(f"\nDescriptions: \n{descriptions}")

    attributes = analyzer.get_attributes()
    logger.info(f"\nAttributes: \n{attributes}")
    
    # 6. Object Analyzer
    logger.info(f"Setting up object analyzer")
    
    object_analyzer = ObjectAnalyzer(
        llm=llm,
        frame_store=frame_store,
        blip_engine=blip_engine
    )
    
    logger.info("Processing objects...")
    object_analyzer.process(object_store)
    
    # Get and log object analysis results
    print("\n")
    for i in range(1, 3):
        object_attributes = object_analyzer.objects_attribute_registry.get_attributes(i, toon_encode=True)
        logger.info(f"\nObject Attributes: \n{object_attributes}")
    
        object_events = object_analyzer.objects_attribute_registry.get_events(i, toon_encode=True)
        logger.info(f"\nObject Events: \n{object_events}")

        print("\n\n")

    # 7. Analyst
    logger.info(f"Setting up analyst")
    
    reporter = Reporter(
        frame_store=frame_store,
        object_store=object_store,
        object_attribute_registry=object_analyzer.objects_attribute_registry,
        property_data=analyzer.descriptions  # Pass novel frame data
    )

    logger.info("Processing Reporter...")
    reporter.process()
    
    logger.info("PIPELINE COMPLETE")

if __name__ == "__main__":
    main()