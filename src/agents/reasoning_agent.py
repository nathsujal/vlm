from typing import List, Dict, Optional, Any, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field

from src.llm import LLM
from src.tools import (
    get_property,
    get_timeline_data,
    get_timeline_by_time,
    get_all_objects,
    get_object_details,
    find_similar_objects_by_color,
    find_objects_by_vehicle_type,
    count_objects_by_type,
    find_long_duration_objects,
)
from src.models import AgentState, SecurityAlert
from src.agents.pattern_detection import Threat
from src.utils import get_logger
from toon import decode

logger = get_logger(__name__)


class ReasoningAgent:
    """
    LLM-powered security analyst with agentic tool calling via LangGraph.
    
    The agent:
    1. Receives a threat from pattern detection
    2. Plans an investigation strategy
    3. Calls tools to gather evidence
    4. Synthesizes findings
    5. Generates a contextualized security alert
    """
    
    SYSTEM_PROMPT = """You are an expert security analyst investigating surveillance footage threats.

Your role is to:
1. Thoroughly investigate each detected threat using available tools
2. Gather evidence from multiple sources (timeline, object details, similar objects)
3. Cross-reference information to confirm or refute the threat
4. Distinguish between genuine security concerns and false positives
5. Generate specific, actionable security alerts

INVESTIGATION STRATEGY:
- Start by getting detailed information about the specific object(s) involved
- Check timeline context around the event
- Look for similar objects/patterns that might indicate recurring behavior
- Cross-reference multiple data sources before concluding
- Consider context: time of day, location, object behavior

SEVERITY GUIDELINES:
- CRITICAL: Active security breach, immediate threat (e.g., unauthorized entry, violent behavior)
- HIGH: Significant concern requiring urgent investigation (e.g., prolonged loitering at sensitive location)
- MEDIUM: Suspicious behavior worth monitoring (e.g., vehicle circling property)
- LOW: Minor anomaly, likely benign (e.g., brief stop by delivery vehicle)

FALSE POSITIVE INDICATORS:
- Brief, transient activity with logical explanation
- Normal business activity during business hours
- Single occurrence without suspicious context

When you have gathered sufficient evidence, use the "generate_alert" tool to create your final assessment.

IMPORTANT: Use tools strategically. Don't call every tool - only the ones relevant to investigating this specific threat."""

    def __init__(self, llm: LLM):
        """
        Initialize the reasoning agent.
        
        Args:
            llm: LLM instance for reasoning
        """
        self.llm = llm
        
        # Define tools available to the agent
        self.tools = [
            get_property,
            get_timeline_data,
            get_timeline_by_time,
            get_object_details,
            find_similar_objects_by_color,
            find_objects_by_vehicle_type,
            find_long_duration_objects,
            get_all_objects,
            count_objects_by_type,
        ]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build LangGraph workflow
        self.graph = self._build_graph()
        
        logger.info("ReasoningAgent initialized with agentic tool calling")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for agentic reasoning.
        
        Workflow:
        1. START → agent_think (agent decides what to do)
        2. agent_think → call_tools (if agent wants to use tools)
        3. call_tools → agent_think (loop back for more reasoning)
        4. agent_think → END (when investigation complete)
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent_think", self._agent_think_node)
        workflow.add_node("call_tools", ToolNode(self.tools))
        
        # Add edges
        workflow.add_edge(START, "agent_think")
        workflow.add_conditional_edges(
            "agent_think",
            self._should_continue,
            {
                "continue": "call_tools",
                "end": END
            }
        )
        workflow.add_edge("call_tools", "agent_think")
        
        return workflow.compile()
    
    def _agent_think_node(self, state: AgentState) -> AgentState:
        """
        Agent reasoning node - decides what to do next.
        
        The agent can:
        - Call tools to gather information
        - Conclude investigation and generate alert
        """
        messages = state["messages"]
        
        # Agent decides next action (tool call or finish)
        response = self.llm_with_tools.invoke(messages)
        
        return {
            **state,
            "messages": [response]
        }
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """
        Determine if agent should continue investigating or finish.
        
        Returns:
            "continue" if agent wants to call tools, "end" if done
        """
        last_message = state["messages"][-1]
        
        # If the last message has tool calls, continue to execute them
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        # Otherwise, investigation is complete
        return "end"
    
    def evaluate_threat(self, threat: Threat) -> SecurityAlert:
        """
        Evaluate a threat using agentic reasoning with tool calling.
        
        Args:
            threat: Detected threat from pattern detection
            
        Returns:
            SecurityAlert with investigation findings
        """
        logger.info(f"🔍 Starting agentic investigation: {threat.threat_type} for object {threat.object_id}")
        
        # Prepare initial message for the agent
        investigation_prompt = f"""Investigate this security threat:

THREAT TYPE: {threat.threat_type}
INITIAL SEVERITY: {threat.severity}
OBJECT ID: {threat.object_id}
OBJECT CLASS: {threat.object_class}
DURATION: {threat.duration_sec}s
REASONING: {threat.reasoning}
INITIAL EVIDENCE: {threat.evidence}

Your task:
1. Use available tools to thoroughly investigate this threat
2. Gather additional evidence (object details, timeline context, similar patterns)
3. Determine if this is a genuine threat or false positive
4. Assess final severity level
5. Provide specific recommended actions

Begin your investigation. Use tools strategically to gather relevant information."""

        # Initialize state
        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=investigation_prompt)
            ],
            "threat": threat,
            "final_alert": None,
            "investigation_complete": False
        }
        
        # Run the agentic workflow
        try:
            final_state = self.graph.invoke(initial_state)
            
            # Extract investigation results from conversation
            investigation_result = self._extract_investigation_result(final_state)
            
            # Generate security alert
            alert = self._create_alert_from_investigation(threat, investigation_result, final_state)
            
            logger.info(f"✅ Investigation complete: {alert.alert_id} [{alert.severity}]")
            return alert
            
        except Exception as e:
            logger.error(f"❌ Investigation failed: {e}", exc_info=True)
            
            # Fallback: create basic alert without investigation
            return self._create_fallback_alert(threat)
    
    def _extract_investigation_result(self, final_state: AgentState) -> str:
        """
        Extract the investigation conclusion from the agent's messages.
        
        Args:
            final_state: Final state after agent workflow
            
        Returns:
            Investigation summary text
        """
        messages = final_state["messages"]
        
        # Get the last AI message (the conclusion)
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        
        if ai_messages:
            last_ai_msg = ai_messages[-1]
            return last_ai_msg.content
        
        return "Investigation completed but no conclusion found."
    
    def _create_alert_from_investigation(
        self, 
        threat: Threat, 
        investigation_result: str,
        final_state: AgentState
    ) -> SecurityAlert:
        """
        Create SecurityAlert based on investigation findings.
        
        Args:
            threat: Original threat
            investigation_result: Agent's conclusion
            final_state: Final agent state with all messages
            
        Returns:
            SecurityAlert
        """
        # Parse investigation result to extract severity, confidence, etc.
        # For now, use LLM to structure the response
        
        structuring_prompt = f"""Based on this investigation, create a structured security alert.

ORIGINAL THREAT: {threat.threat_type}
INVESTIGATION FINDINGS:
{investigation_result}

Provide:
1. Final severity (CRITICAL/HIGH/MEDIUM/LOW)
2. Is this a false positive? (true/false)
3. Confidence score (0.0-1.0)
4. Specific recommended actions (list of 2-3 actions)
5. Brief summary (1-2 sentences)

Format as JSON:
{{
    "severity": "HIGH",
    "is_false_positive": false,
    "confidence": 0.85,
    "recommended_actions": ["action1", "action2"],
    "summary": "brief summary"
}}"""

        try:
            structured_response = self.llm.invoke(structuring_prompt)
            
            # Parse JSON (simplified - in production, use Pydantic for structured output)
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', structured_response, re.DOTALL)
            if json_match:
                alert_data = json.loads(json_match.group())
            else:
                # Fallback if parsing fails
                alert_data = {
                    "severity": threat.severity,
                    "is_false_positive": False,
                    "confidence": 0.7,
                    "recommended_actions": [threat.recommended_action],
                    "summary": investigation_result[:200]
                }
        except Exception as e:
            logger.warning(f"Failed to structure alert: {e}, using defaults")
            alert_data = {
                "severity": threat.severity,
                "is_false_positive": False,
                "confidence": 0.7,
                "recommended_actions": [threat.recommended_action],
                "summary": investigation_result[:200]
            }
        
        # Collect evidence from tool calls
        evidence = threat.evidence.copy()
        tool_messages = [msg for msg in final_state["messages"] if isinstance(msg, ToolMessage)]
        evidence.extend([f"Tool: {msg.name}" for msg in tool_messages[:3]])  # Add first 3 tools used
        
        # Parse object_id
        affected_objects = []
        if isinstance(threat.object_id, str):
            affected_objects = [int(x.strip()) for x in threat.object_id.split(',') if x.strip().isdigit()]
        else:
            affected_objects = [threat.object_id]
        
        return SecurityAlert(
            alert_id=f"{threat.threat_type}_{threat.object_id}",
            severity=alert_data.get("severity", threat.severity),
            title=f"{threat.threat_type.replace('_', ' ').title()}",
            description=alert_data.get("summary", investigation_result[:300]),
            affected_objects=affected_objects,
            evidence=evidence,
            recommended_actions=alert_data.get("recommended_actions", [threat.recommended_action]),
            is_false_positive=alert_data.get("is_false_positive", False),
            confidence=alert_data.get("confidence", 0.7),
            investigation_notes=investigation_result
        )
    
    def _create_fallback_alert(self, threat: Threat) -> SecurityAlert:
        """Create basic alert when investigation fails."""
        affected_objects = []
        if isinstance(threat.object_id, str):
            affected_objects = [int(x.strip()) for x in threat.object_id.split(',') if x.strip().isdigit()]
        else:
            affected_objects = [threat.object_id]
        
        return SecurityAlert(
            alert_id=f"{threat.threat_type}_{threat.object_id}",
            severity=threat.severity,
            title=f"{threat.threat_type.replace('_', ' ').title()}",
            description=threat.reasoning,
            affected_objects=affected_objects,
            evidence=threat.evidence,
            recommended_actions=[threat.recommended_action],
            is_false_positive=False,
            confidence=0.5,
            investigation_notes="Investigation failed - using basic threat data"
        )
    
    def evaluate_all_threats(
        self,
        pattern_results: Dict[str, List[Threat]]
    ) -> List[SecurityAlert]:
        """
        Evaluate all detected threats with agentic investigation.
        
        Args:
            pattern_results: Results from PatternDetector
            
        Returns:
            List of SecurityAlerts
        """
        logger.info("=" * 60)
        logger.info("Starting agentic evaluation of all threats")
        logger.info("=" * 60)
        
        alerts = []
        total_threats = sum(len(threats) for threats in pattern_results.values())
        current = 0
        
        for threat_type, threats in pattern_results.items():
            for threat in threats:
                current += 1
                logger.info(f"\n[{current}/{total_threats}] Investigating {threat_type}...")
                
                try:
                    alert = self.evaluate_threat(threat)
                    alerts.append(alert)
                except Exception as e:
                    logger.error(f"Failed to evaluate threat {threat.object_id}: {e}")
        
        logger.info("=" * 60)
        logger.info(f"Agentic evaluation complete: {len(alerts)} alerts generated")
        logger.info("=" * 60)
        
        return alerts
    
    def generate_executive_summary(self, alerts: List[SecurityAlert]) -> str:
        """
        Generate executive summary of security situation.
        
        Args:
            alerts: List of security alerts
            
        Returns:
            Executive summary text
        """
        if not alerts:
            return "✅ No security threats detected. All surveillance activity appears normal."
        
        # Count by severity
        critical = sum(1 for a in alerts if a.severity == "CRITICAL")
        high = sum(1 for a in alerts if a.severity == "HIGH")
        medium = sum(1 for a in alerts if a.severity == "MEDIUM")
        low = sum(1 for a in alerts if a.severity == "LOW")
        false_positives = sum(1 for a in alerts if a.is_false_positive)
        
        # Get top alerts
        genuine_alerts = [a for a in alerts if not a.is_false_positive]
        top_alerts = sorted(
            genuine_alerts,
            key=lambda a: (["CRITICAL", "HIGH", "MEDIUM", "LOW"].index(a.severity), -a.confidence)
        )[:5]
        
        summary_prompt = f"""Generate a concise executive security summary:

ALERT STATISTICS:
- Total Alerts: {len(alerts)}
- Critical: {critical}
- High: {high}
- Medium: {medium}
- Low: {low}
- False Positives: {false_positives}

TOP PRIORITY ALERTS:
{chr(10).join([f"- [{a.severity}] {a.title}: {a.description[:100]}" for a in top_alerts])}

Provide:
1. Overall security assessment (2-3 sentences)
2. Top 3 priority actions
3. Risk level summary

Be direct and actionable."""

        summary = self.llm.invoke(summary_prompt)
        return summary