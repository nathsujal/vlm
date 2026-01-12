from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime, time

from src.tools.reader import (
    read_property,
    read_timeline,
    read_objects,
    find_object,
    filter_objects_by_duration,
    filter_objects_by_label,
)
from src.llm import LLM
from src.models import Threat
from src.utils import get_logger
from toon import decode

logger = get_logger(__name__)


def detect_loitering(min_duration: float = 300.0) -> List[Threat]:
    """
    Detect objects that remain stationary for extended periods.
    
    Args:
        min_duration: Minimum duration in seconds to flag as loitering (default: 5 minutes)
        
    Returns:
        List of loitering threats
    """
    logger.info(f"Detecting loitering (threshold: {min_duration}s)")
    
    # Get all objects with duration >= threshold
    objects_data = decode(read_objects())
    long_duration_objects = [obj for obj in objects_data if obj.get('duration_sec', 0) >= min_duration]
    
    threats = []
    stationary_keywords = ["parked", "stationary", "waiting", "standing", "positioned", "stopped", "remains"]
    
    for obj in long_duration_objects:
        # Analyze event captions for stationary behavior
        events = obj.get('events', [])
        stationary_count = 0
        evidence = []
        
        for event in events:
            caption = event.get('caption', '').lower()
            if any(keyword in caption for keyword in stationary_keywords):
                stationary_count += 1
                evidence.append(f"Frame {event['frame_id']}: {event['caption'][:80]}")
        
        # If majority of events show stationary behavior
        if stationary_count >= len(events) * 0.5:
            severity = "MEDIUM" if obj['duration_sec'] < 600 else "HIGH"
            
            threats.append(Threat(
                threat_type="LOITERING",
                severity=severity,
                object_id=obj['object_id'],
                object_class=obj['label'],
                duration_sec=obj['duration_sec'],
                reasoning=f"{obj['label'].capitalize()} has been stationary for {obj['duration_sec']:.1f}s ({stationary_count}/{len(events)} events show no movement)",
                recommended_action="Investigate vehicle/person identity and purpose" if severity == "HIGH" else "Monitor for continued loitering",
                evidence=evidence[:3]  # Limit to first 3 pieces of evidence
            ))
    
    logger.info(f"Detected {len(threats)} loitering threats")
    return threats


def detect_off_hours_activity(
    business_hours_start: time = time(6, 0),
    business_hours_end: time = time(22, 0)
) -> List[Threat]:
    """
    Detect activity during off-hours (potential after-hours intrusion).
    
    Args:
        business_hours_start: Start of normal business hours
        business_hours_end: End of normal business hours
        
    Returns:
        List of off-hours threats
    """
    logger.info(f"Detecting off-hours activity (normal hours: {business_hours_start}-{business_hours_end})")
    
    timeline_data = decode(read_timeline())
    threats = []
    
    for entry in timeline_data:
        captured_dt = datetime.fromisoformat(entry['captured_at'])
        captured_time = captured_dt.time()
        
        # Check if outside business hours
        if captured_time < business_hours_start or captured_time > business_hours_end:
            for event in entry.get('events', []):
                threats.append(Threat(
                    threat_type="OFF_HOURS_ACTIVITY",
                    severity="MEDIUM",
                    object_id=event['object_id'],
                    object_class=event['object_class'],
                    reasoning=f"Activity detected at {captured_time.strftime('%H:%M:%S')} (outside {business_hours_start}-{business_hours_end})",
                    recommended_action="Review footage and verify authorization",
                    evidence=[f"Frame {entry['frame_id']}: {event['event'][:80]}"]
                ))
    
    logger.info(f"Detected {len(threats)} off-hours activity instances")
    return threats


def detect_coordinated_movement() -> List[Threat]:
    """
    Detect multiple objects appearing simultaneously (potential coordinated threat).
    
    Returns:
        List of coordinated movement threats
    """
    logger.info("Detecting coordinated movement patterns")
    
    timeline_data = decode(read_timeline())
    threats = []
    
    # Threshold: 3+ objects appearing in same frame
    coordination_threshold = 3
    
    for entry in timeline_data:
        events = entry.get('events', [])
        if len(events) >= coordination_threshold:
            object_ids = [e['object_id'] for e in events]
            object_classes = [e['object_class'] for e in events]
            
            threats.append(Threat(
                threat_type="COORDINATED_MOVEMENT",
                severity="MEDIUM",
                object_id=", ".join(map(str, object_ids)),
                object_class=", ".join(set(object_classes)),
                reasoning=f"{len(events)} objects detected simultaneously at frame {entry['frame_id']} - potential coordinated activity",
                recommended_action="Immediate review - possible organized threat",
                evidence=[f"Objects: {object_ids}", f"Time: {entry['captured_at']}"]
            ))
    
    logger.info(f"Detected {len(threats)} coordinated movement patterns")
    return threats


def analyze_suspicious_keywords(llm: LLM) -> List[Threat]:
    """
    Use LLM to analyze event captions for suspicious behavior keywords.
    
    Args:
        llm: LLM instance for semantic analysis
        
    Returns:
        List of threats detected via keyword analysis
    """
    logger.info("Analyzing event captions for suspicious keywords")
    
    objects_data = decode(read_objects())
    threats = []
    
    suspicious_keywords = [
        "abandoned", "unattended", "suspicious", "unusual", 
        "circling", "approaching", "observing", "hiding",
        "concealed", "obscured", "restricted", "unauthorized"
    ]
    
    for obj in objects_data:
        suspicious_events = []
        
        for event in obj.get('events', []):
            caption = event.get('caption', '').lower()
            matched_keywords = [kw for kw in suspicious_keywords if kw in caption]
            
            if matched_keywords:
                suspicious_events.append({
                    'frame_id': event['frame_id'],
                    'caption': event['caption'],
                    'keywords': matched_keywords
                })
        
        if suspicious_events:
            # Use LLM to assess overall suspicion level
            prompt = f"""
            Object: {obj['label']} (ID: {obj['object_id']})
            Duration: {obj['duration_sec']}s
            
            Suspicious events detected:
            {chr(10).join([f"- Frame {e['frame_id']}: {e['caption'][:100]}" for e in suspicious_events[:3]])}
            
            Assess if this behavior is genuinely suspicious or normal activity.
            Rate severity: LOW, MEDIUM, HIGH, or CRITICAL
            Explain reasoning in one sentence.
            """
            
            # For now, use rule-based (can enhance with LLM later)
            severity = "HIGH" if len(suspicious_events) > 2 else "MEDIUM"
            
            threats.append(Threat(
                threat_type="SUSPICIOUS_BEHAVIOR",
                severity=severity,
                object_id=obj['object_id'],
                object_class=obj['label'],
                duration_sec=obj['duration_sec'],
                reasoning=f"Detected {len(suspicious_events)} events with suspicious keywords: {set([kw for e in suspicious_events for kw in e['keywords']])}",
                recommended_action="Manual review required - behavior analysis needed",
                evidence=[f"Frame {e['frame_id']}: {', '.join(e['keywords'])}" for e in suspicious_events[:3]]
            ))
    
    logger.info(f"Detected {len(threats)} suspicious behavior patterns")
    return threats


def detect_property_anomalies() -> List[Threat]:
    """
    Detect threats based on property/environment conditions.
    
    Analyzes novel frame descriptions for:
    - Unusual lighting conditions (lights off during activity)
    - Restricted area access
    - Property condition anomalies
    - Environmental security indicators
    
    Returns:
        List of property-based threats
    """
    logger.info("Detecting property and environment-based anomalies")
    
    try:
        property_data = decode(read_property())
    except Exception as e:
        logger.warning(f"Could not load property data: {e}")
        return []
    
    if not property_data:
        logger.info("No property data available")
        return []
    
    threats = []
    
    # Suspicious environment keywords
    high_risk_indicators = [
        "low lighting", "dark", "dimly lit", "lights off",
        "restricted area", "unauthorized", "gate open", "door ajar",
        "fence breach", "perimeter breach", "broken window",
        "alarm", "motion detected"
    ]
    
    medium_risk_indicators = [
        "empty", "deserted", "unattended", "shadows",
        "obscured", "concealed", "hidden", "secluded"
    ]
    
    # Also get timeline to correlate time with property conditions
    timeline_data = decode(read_timeline())
    objects_data = decode(read_objects())
    
    for prop_frame in property_data:
        frame_id = prop_frame.get('frame_id')
        description = prop_frame.get('description', '').lower()
        attributes = prop_frame.get('attributes', {})
        
        # Check for high-risk indicators
        high_risk_matches = [kw for kw in high_risk_indicators if kw in description]
        medium_risk_matches = [kw for kw in medium_risk_indicators if kw in description]
        
        # Find objects present in this frame
        frame_objects = []
        for obj in objects_data:
            for event in obj.get('events', []):
                if event.get('frame_id') == frame_id:
                    frame_objects.append(obj)
                    break
        
        # HIGH severity: Suspicious environment + active objects
        if high_risk_matches and frame_objects:
            # Get time information
            timeline_entry = next((t for t in timeline_data if t['frame_id'] == frame_id), None)
            time_info = ""
            if timeline_entry:
                captured_dt = datetime.fromisoformat(timeline_entry['captured_at'])
                time_info = f" at {captured_dt.strftime('%H:%M:%S')}"
            
            object_ids = [obj['object_id'] for obj in frame_objects]
            object_classes = [obj['label'] for obj in frame_objects]
            
            threats.append(Threat(
                threat_type="PROPERTY_ANOMALY",
                severity="HIGH",
                object_id=", ".join(map(str, object_ids)),
                object_class=", ".join(set(object_classes)),
                reasoning=f"Suspicious environment detected{time_info}: {', '.join(high_risk_matches[:2])} with {len(frame_objects)} active object(s)",
                recommended_action="Immediate investigation required - environmental security breach",
                evidence=[
                    f"Frame {frame_id}: {description[:100]}",
                    f"Indicators: {', '.join(high_risk_matches)}",
                    f"Active objects: {object_ids}"
                ]
            ))
        
        # MEDIUM severity: Suspicious environment alone
        elif high_risk_matches or (medium_risk_matches and frame_objects):
            timeline_entry = next((t for t in timeline_data if t['frame_id'] == frame_id), None)
            time_info = ""
            if timeline_entry:
                captured_dt = datetime.fromisoformat(timeline_entry['captured_at'])
                time_info = f" at {captured_dt.strftime('%H:%M:%S')}"
            
            all_matches = high_risk_matches + medium_risk_matches
            
            threats.append(Threat(
                threat_type="PROPERTY_ANOMALY",
                severity="MEDIUM",
                object_id="property",
                object_class="environment",
                reasoning=f"Unusual property conditions detected{time_info}: {', '.join(all_matches[:2])}",
                recommended_action="Review property conditions and verify security status",
                evidence=[
                    f"Frame {frame_id}: {description[:100]}",
                    f"Indicators: {', '.join(all_matches)}"
                ]
            ))
    
    logger.info(f"Detected {len(threats)} property-based anomalies")
    return threats


class PatternDetector:
    """
    Main pattern detection engine.
    Coordinates all detection algorithms and outputs consolidated threat report.
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm
        logger.info("PatternDetector initialized")
    
    def detect_all_threats(self) -> Dict[str, List[Threat]]:
        """
        Run all detection algorithms and return consolidated threats.
        
        Returns:
            Dictionary mapping threat type to list of threats
        """
        logger.info("Running comprehensive threat detection")
        
        results = {
            "loitering": detect_loitering(min_duration=300),
            "off_hours": detect_off_hours_activity(),
            "coordinated": detect_coordinated_movement(),
            "property_anomaly": detect_property_anomalies(),
        }
        
        if self.llm:
            results["suspicious_behavior"] = analyze_suspicious_keywords(self.llm)
        
        # Calculate totals
        total_threats = sum(len(threats) for threats in results.values())
        logger.info(f"Detection complete: {total_threats} total threats found")
        
        return results
    
    def get_critical_threats(self, all_threats: Dict[str, List[Threat]]) -> List[Threat]:
        """Get only CRITICAL severity threats."""
        critical = []
        for threat_list in all_threats.values():
            critical.extend([t for t in threat_list if t.severity == "CRITICAL"])
        return critical
    
    def generate_summary_report(self, all_threats: Dict[str, List[Threat]]) -> str:
        """Generate human-readable summary report."""
        total = sum(len(threats) for threats in all_threats.values())
        
        report = [
            "=" * 80,
            "SECURITY THREAT DETECTION REPORT",
            "=" * 80,
            f"\nTotal Threats Detected: {total}\n"
        ]
        
        for threat_type, threats in all_threats.items():
            if threats:
                report.append(f"\n{threat_type.upper().replace('_', ' ')} ({len(threats)}):")
                for t in threats:
                    report.append(f"  • [{t.severity}] Object {t.object_id} ({t.object_class}): {t.reasoning}")
        
        return "\n".join(report)
