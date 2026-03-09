from deepteam.guardrails.guards.prompt_injection_guard.prompt_injection_guard import PromptInjectionGuard
from deepteam.guardrails.guards.privacy_guard.privacy_guard import PrivacyGuard
from deepteam.guardrails.guards.topical_guard.topical_guard import TopicalGuard

prompt_injection_guard = PromptInjectionGuard()
privacy_guard = PrivacyGuard()
topical_guard = TopicalGuard(allowed_topics=["IT security policy", "data privacy", "compliance"])

def check_input(question: str) -> tuple[bool, str]:
    """
    Checks the input question against prompt injection and topical guardrails.
    Returns a tuple of (is_safe, message).
    """
    safety_assessment = prompt_injection_guard.guard_input(question)
    if "safe" not in safety_assessment.lower():
        return False, f"Warning: Prompt injection detected. Assessment: {safety_assessment}"

    topical_assessment = topical_guard.guard_input(question)
    if "safe" not in topical_assessment.lower():
        return False, f"Warning: The question is off-topic. Please ask about IT security policy, data privacy, or compliance. Assessment: {topical_assessment}"

    return True, "Input is safe."

def check_output(question: str, result: str) -> tuple[bool, str]:
    """
    Checks the output against the privacy guardrail.
    Returns a tuple of (is_safe, message).
    """
    privacy_assessment = privacy_guard.guard_output(input=question, output=result)
    if "safe" not in privacy_assessment.lower():
        return False, f"Warning: Sensitive information detected in the output. Assessment: {privacy_assessment}"
    
    return True, "Output is safe."
