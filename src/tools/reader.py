"""
Data reader module for surveillance analysis.

Provides cached, efficient access to timeline and object data.
LangChain @tool decorated functions for use in reasoning agents.
"""
import json
from toon import encode, decode
from typing import Optional, List, Dict, Callable
from datetime import datetime
from langchain_core.tools import tool

from config.settings import settings
from src.utils import get_logger

logger = get_logger(__name__)

# Caches
_timeline_cache: Optional[List[Dict]] = None
_objects_cache: Optional[List[Dict]] = None
_property_cache: Optional[List[Dict]] = None


def _load_timeline() -> List[Dict]:
    """Load timeline from JSON file (cached)."""
    global _timeline_cache
    if _timeline_cache is None:
        try:
            with open(settings.timeline_path, 'r') as f:
                _timeline_cache = json.load(f)
            logger.debug(f"Loaded {len(_timeline_cache)} timeline entries")
        except FileNotFoundError:
            logger.warning(f"Timeline file not found: {settings.timeline_path}")
            _timeline_cache = []
    return _timeline_cache


def _load_objects() -> List[Dict]:
    """Load objects report from JSON file (cached)."""
    global _objects_cache
    if _objects_cache is None:
        try:
            with open(settings.object_report_path, 'r') as f:
                _objects_cache = json.load(f)
            logger.debug(f"Loaded {len(_objects_cache)} object reports")
        except FileNotFoundError:
            logger.warning(f"Objects file not found: {settings.object_report_path}")
            _objects_cache = []
    return _objects_cache


def _load_property() -> List[Dict]:
    """Load property report from JSON file (cached)."""
    global _property_cache
    if _property_cache is None:
        try:
            with open(settings.property_report_path, 'r') as f:
                _property_cache = json.load(f)
            logger.debug(f"Loaded {len(_property_cache)} property reports")
        except FileNotFoundError:
            logger.warning(f"Property file not found: {settings.property_report_path}")
            _property_cache = []
    return _property_cache


@tool
def get_property() -> str:
    """
    Get the complete property report.
    
    Returns:
        Complete property report
    """
    logger.info("Tool called: get_property")
    return encode(_load_property())


@tool
def get_timeline_data() -> str:
    """
    Get the complete surveillance timeline with all events.
    Use this to understand what happened and when across all frames.
    
    Returns:
        Complete timeline data with all events
    """
    logger.info("Tool called: get_timeline_data")
    return encode(_load_timeline())


@tool
def get_timeline_by_time(start_sec: float, end_sec: float) -> str:
    """
    Get timeline events within a specific time range.
    Useful for investigating activity during specific periods.
    
    Args:
        start_sec: Start time in seconds
        end_sec: End time in seconds
        
    Returns:
        Filtered timeline data
    """
    logger.info(f"Tool called: get_timeline_by_time({start_sec}, {end_sec})")
    timeline = _load_timeline()
    
    if not timeline:
        return encode([])
    
    filtered = [
        entry for entry in timeline
        if start_sec <= entry['timestamp_sec'] <= end_sec
    ]
    
    return encode(filtered)


@tool  
def get_all_objects() -> str:
    """
    Get all tracked objects in the surveillance data.
    Use this to get overview of all activity.
    
    Returns:
        Complete list of all objects
    """
    logger.info("Tool called: get_all_objects")
    return encode(_load_objects())


@tool
def get_object_details(object_id: int) -> str:
    """
    Get complete details for a specific object including:
    - All attributes (color, type, etc.)
    - Full event timeline
    - Movement summary
    
    Args:
        object_id: ID of the object to investigate
        
    Returns:
        Detailed object report or error message
    """
    logger.info(f"Tool called: get_object_details({object_id})")
    objects = _load_objects()
    
    obj = next((o for o in objects if o['object_id'] == object_id), None)
    
    if obj is None:
        return f"Object {object_id} not found"
    
    return encode(obj)


@tool
def find_similar_objects_by_color(color: str) -> str:
    """
    Find all objects matching a specific color.
    Useful for identifying if same colored vehicle appeared multiple times.
    
    Args:
        color: Color to search for (e.g., 'blue', 'gray', 'red')
        
    Returns:
        List of objects with matching color
    """
    logger.info(f"Tool called: find_similar_objects_by_color({color})")
    objects = _load_objects()
    
    matching = [
        obj for obj in objects
        if obj.get('attributes', {}).get('color') == color
    ]
    
    return encode(matching)


@tool
def find_objects_by_vehicle_type(vehicle_type: str) -> str:
    """
    Find all objects of a specific vehicle type.
    
    Args:
        vehicle_type: Type to search (e.g., 'sedan', 'truck', 'suv')
        
    Returns:
        List of matching vehicles
    """
    logger.info(f"Tool called: find_objects_by_vehicle_type({vehicle_type})")
    objects = _load_objects()
    
    matching = [
        obj for obj in objects
        if obj.get('attributes', {}).get('vehicle_type') == vehicle_type
    ]
    
    return encode(matching)


@tool
def count_objects_by_type(object_type: str) -> str:
    """
    Count how many objects of a specific type were detected.
    Useful for understanding activity patterns.
    
    Args:
        object_type: Type to count (e.g., 'car', 'person')
        
    Returns:
        Count and list of matching objects
    """
    logger.info(f"Tool called: count_objects_by_type({object_type})")
    objects = _load_objects()
    
    matching = [obj for obj in objects if obj.get('label') == object_type]
    count = len(matching)
    
    return f"Found {count} {object_type} objects. Data: {encode(matching)}"


@tool
def find_long_duration_objects(min_seconds: float) -> str:
    """
    Find objects tracked for at least the specified duration.
    Useful for confirming loitering or investigating persistent presence.
    
    Args:
        min_seconds: Minimum tracking duration
        
    Returns:
        Objects meeting duration threshold
    """
    logger.info(f"Tool called: find_long_duration_objects({min_seconds})")
    objects = _load_objects()
    
    matching = [
        obj for obj in objects
        if obj.get('duration_sec', 0) >= min_seconds
    ]
    
    return encode(matching)


def read_property() -> str:
    """
    Read the complete property report.
    
    Returns:
        Complete property report
    """
    logger.info("Tool called: read_property")
    return encode(_load_property())


def read_timeline() -> str:
    """Read entire timeline (for programmatic use)."""
    return encode(_load_timeline())


def read_objects() -> str:
    """Read all objects (for programmatic use)."""
    return encode(_load_objects())


def find_object(object_id: int) -> str:
    """Find object by ID (for programmatic use)."""
    objects = _load_objects()
    obj = next((o for o in objects if o['object_id'] == object_id), None)
    return encode(obj)


def filter_objects_by_attribute(attribute_key: str, attribute_value: str) -> str:
    """Filter objects by attribute (for programmatic use)."""
    objects = _load_objects()
    matching = [
        obj for obj in objects
        if obj.get('attributes', {}).get(attribute_key) == attribute_value
    ]
    return encode(matching)


def filter_objects_by_label(label: str) -> str:
    """Filter objects by label (for programmatic use)."""
    objects = _load_objects()
    matching = [obj for obj in objects if obj.get('label') == label]
    return encode(matching)


def filter_objects_by_duration(
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None
) -> str:
    """Filter objects by duration range (for programmatic use)."""
    objects = _load_objects()
    
    results = []
    for obj in objects:
        duration = obj.get('duration_sec', 0)
        if min_duration is not None and duration < min_duration:
            continue
        if max_duration is not None and duration > max_duration:
            continue
        results.append(obj)
    
    return encode(results)


def filter_timeline_by_time(
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None
) -> str:
    """Filter timeline by time range (for programmatic use)."""
    timeline = _load_timeline()
    
    if not timeline:
        return encode([])
    
    # Default to full range
    if start_sec is None:
        start_sec = min(e['timestamp_sec'] for e in timeline)
    if end_sec is None:
        end_sec = max(e['timestamp_sec'] for e in timeline)
    
    filtered = [
        entry for entry in timeline
        if start_sec <= entry['timestamp_sec'] <= end_sec
    ]
    
    return encode(filtered)


def count_timeline_entries() -> int:
    """Get total number of timeline entries."""
    return len(_load_timeline())


def count_objects() -> int:
    """Get total number of tracked objects."""
    return len(_load_objects())


def get_object_labels() -> List[str]:
    """Get unique object labels in the dataset."""
    objects = _load_objects()
    return list(set(obj.get('label') for obj in objects if 'label' in obj))


def get_timeline_time_range() -> tuple[float, float]:
    """
    Get timeline time range.
    
    Returns:
        Tuple of (start_sec, end_sec)
    """
    timeline = _load_timeline()
    if not timeline:
        return (0.0, 0.0)
    
    timestamps = [e['timestamp_sec'] for e in timeline]
    return (min(timestamps), max(timestamps))
