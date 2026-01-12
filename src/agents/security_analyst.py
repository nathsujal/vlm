from typing import Dict, List

from src.llm import LLM
from .pattern_detection import PatternDetector
from .reasoning_agent import ReasoningAgent
from .visualizer import Visualizer
from src.models import SecurityAlert, Threat
from src.storage import FrameStore
from src.utils import get_logger
from config.settings import settings

logger = get_logger(__name__)

class SecurityAnalyst:
    """
    Master security analyst that orchestrates the full analysis pipeline:
    1. Pattern Detection (rule-based threat finding)
    2. Agentic Reasoning (LLM investigation)
    3. Visualization (annotated frames)
    
    This is the main entry point for security analysis after preprocessing.
    """
    
    def __init__(self, frames: FrameStore, llm: LLM):
        """
        Initialize Security Analyst with specialist agents.
        
        Args:
            llm: Language model for reasoning
            frames: FrameStore for frame access
        """
        self.llm = llm
        self.frames = frames
        
        # Initialize specialist agents
        self.pattern_detector = PatternDetector(llm)
        self.reasoning_agent = ReasoningAgent(llm)
        self.visualizer = Visualizer(self.frames)
        
        # Results storage
        self.threats: Dict[str, List[Threat]] = {}
        self.alerts: List[SecurityAlert] = []
        self.visualization_results: Dict = {}
        
        logger.info("Security Analyst initialized")

    def analyze(self, visualize: bool = True) -> Dict:
        """
        Run the complete security analysis pipeline.
        
        Args:
            visualize: Whether to create visual outputs
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("=" * 80)
        logger.info("🚨 Starting Security Analysis Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Pattern Detection
        logger.info("\n🔍 Step 1: Running Pattern Detection...")
        logger.info("-" * 80)
        self.threats = self.pattern_detector.detect_all_threats()
        
        threat_count = sum(len(t_list) for t_list in self.threats.values())
        logger.info(f"✅ Pattern Detection Complete: {threat_count} threats found")
        
        # Log summary
        for threat_type, threat_list in self.threats.items():
            if threat_list:
                logger.info(f"  • {threat_type}: {len(threat_list)} threats")
        
        # Step 2: Agentic Reasoning
        logger.info("\n🤖 Step 2: Running Agentic Reasoning...")
        logger.info("-" * 80)
        self.alerts = self.reasoning_agent.evaluate_all_threats(self.threats)
        
        logger.info(f"✅ Reasoning Complete: {len(self.alerts)} alerts generated")
        
        # Log alert summary
        severity_counts = {}
        for alert in self.alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if severity in severity_counts:
                logger.info(f"  • {severity}: {severity_counts[severity]} alerts")
        
        # Step 3: Visualization (optional)
        if visualize and self.visualizer:
            logger.info("\n🎨 Step 3: Creating Threat Visualization...")
            logger.info("-" * 80)
            
            try:
                self.visualization_results = self.visualizer.visualize_threats(
                    self.alerts,
                    output_dir=settings.visualizer_output_dir
                )
                logger.info(f"✅ Visualization Complete: {self.visualization_results.get('annotated_frames', 0)} frames saved")
            except Exception as e:
                logger.error(f"❌ Visualization failed: {e}", exc_info=True)
                self.visualization_results = {'error': str(e)}
        
        # Step 4: Generate Executive Summary
        logger.info("\n📋 Generating Executive Summary...")
        summary = self.reasoning_agent.generate_executive_summary(self.alerts)
        
        logger.info("=" * 80)
        logger.info("✅ Security Analysis Pipeline Complete")
        logger.info("=" * 80)
        
        # Compile results
        results = {
            'threats': self.threats,
            'threat_count': threat_count,
            'alerts': self.alerts,
            'alert_count': len(self.alerts),
            'severity_breakdown': severity_counts,
            'visualization': self.visualization_results if visualize else None,
            'executive_summary': summary
        }
        
        return results
    
    def get_threat_summary(self) -> str:
        """Get a quick summary of detected threats."""
        if not self.threats:
            return "No analysis performed yet."
        
        threat_count = sum(len(t_list) for t_list in self.threats.values())
        alert_count = len(self.alerts)
        
        critical_count = sum(1 for a in self.alerts if a.severity == "CRITICAL")
        high_count = sum(1 for a in self.alerts if a.severity == "HIGH")
        
        summary = f"""
Security Analysis Summary:
--------------------------
Threats Detected: {threat_count}
Alerts Generated: {alert_count}
  • CRITICAL: {critical_count}
  • HIGH: {high_count}
  • Other: {alert_count - critical_count - high_count}
"""
        return summary
    
    def get_high_priority_alerts(self) -> List[SecurityAlert]:
        """Get CRITICAL and HIGH severity alerts only."""
        return [
            alert for alert in self.alerts 
            if alert.severity in ['CRITICAL', 'HIGH'] and not alert.is_false_positive
        ]
