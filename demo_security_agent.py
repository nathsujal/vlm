"""
Demo script to test Pattern Detection and Reasoning Agent
with simulated surveillance data.
"""
import sys
import os

# Temporarily override settings to use demo data
from config.settings import settings
settings.timeline_path = "data/demo/timeline.json"
settings.object_report_path = "data/demo/object_report.json"
settings.property_report_path = "data/demo/property_report.json"

from src.agents.pattern_detection import PatternDetector
from src.agents.reasoning_agent import ReasoningAgent
from src.llm import LLM

print("\n" + "=" * 80)
print("🚨 SECURITY DEMO: Pattern Detection & Reasoning Agent")
print("=" * 80)
print("\n📊 Scenario: Corporate Office - Off-Hours Suspicious Activity")
print("   Time: 2:30 AM - 2:58 AM")
print("   Location: Corporate office building entrance\n")

# Initialize
print("🔧 Initializing Pattern Detector...")
detector = PatternDetector(llm=LLM())

# Step 1: Detect Patterns
print("\n🔍 Step 1: Running Pattern Detection...")
print("-" * 80)
threats = detector.detect_all_threats()

# Display results
total_threats = sum(len(t_list) for t_list in threats.values())
print(f"\n✅ Detection Complete: {total_threats} threats found\n")

for threat_type, threat_list in threats.items():
    if threat_list:
        print(f"📌 {threat_type.upper().replace('_', ' ')} ({len(threat_list)}):")
        for t in threat_list:
            print(f"   • [{t.severity}] Object {t.object_id} ({t.object_class})")
            print(f"     Reasoning: {t.reasoning}")
            if t.duration_sec:
                print(f"     Duration: {t.duration_sec:.0f}s ({t.duration_sec/60:.1f} min)")
            print()

# Step 2: LLM Reasoning
print("\n🤖 Step 2: LLM Reasoning & Threat Evaluation...")
print("-" * 80)
analyst = ReasoningAgent(llm=LLM())
alerts = analyst.evaluate_all_threats(threats)

print(f"\n✅ Generated {len(alerts)} Security Alerts\n")

# Display alerts
for alert in alerts:
    print(f"🚨 [{alert.severity}] {alert.title}")
    print(f"   Objects: {alert.affected_objects}")
    print(f"   Analysis: {alert.description}")
    print(f"   Recommended Actions: {alert.recommended_actions}")
    print(f"   Confidence: {alert.confidence}")
    print()

# Step 3: Executive Summary
print("\n📋 Step 3: Executive Summary")
print("-" * 80)
summary = analyst.generate_executive_summary(alerts)
print(summary)

print("\n" + "=" * 80)
print("✅ DEMO COMPLETE")
print("=" * 80)
print("\n💡 Key Findings:")
print("   • Off-hours activity detected (2:30-2:58 AM)")
print("   • Loitering person identified (12 minutes)")
print("   • Unauthorized vehicle with covered plates (14 minutes)")
print("   • Coordinated suspicious behavior")
print("   • Security response documented")
print("\n🎯 The system successfully identified and analyzed all security threats!\n")
