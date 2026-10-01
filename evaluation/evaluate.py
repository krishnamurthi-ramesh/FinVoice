import json
from app.agent.graph import bfsi_agent
from app.agent.state import AgentState

TEST_CASES = [
    {
        "input_text": "What is the status of my loan?",
        "expected_intent": "loan_status",
        "should_escalate": False
    },
    {
        "input_text": "I want to speak to a human agent right now.",
        "expected_intent": "human_agent",
        "should_escalate": True
    },
    {
        "input_text": "What documents do I need for KYC?",
        "expected_intent": "policy_question",
        "should_escalate": False
    },
    {
        "input_text": "Calculate EMI for 500000 at 10 percent for 24 months.",
        "expected_intent": "calculate_emi",
        "should_escalate": False
    }
]

def evaluate_intents():
    print("Running Intent Evaluation...")
    passed = 0
    total = len(TEST_CASES)
    
    for case in TEST_CASES:
        state: AgentState = {
            "transcript": case["input_text"],
            "confidence": 0.9, # High confidence for testing intent logic
            "customer_id": "C1024"
        }
        
        # We invoke the graph but we only care about what happens after intent node
        result = bfsi_agent.invoke(state)
        intent = result.get("intent", "unknown")
        escalated = result.get("requires_human", False)
        
        if intent == case["expected_intent"] and escalated == case["should_escalate"]:
            passed += 1
            print(f"✅ Pass: '{case['input_text']}' -> {intent} (Escalate: {escalated})")
        else:
            print(f"❌ Fail: '{case['input_text']}'. Expected {case['expected_intent']}, got {intent}")
            
    print(f"\nEvaluation Complete. Score: {passed}/{total} ({passed/total*100}%)")

if __name__ == "__main__":
    evaluate_intents()
