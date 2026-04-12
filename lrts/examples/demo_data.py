"""Built-in demo scenario: Customer Support Bot v1 vs v2."""

V1_PROMPT = (
    "You are a customer support assistant. "
    "Answer user questions about our product. "
    "Be helpful and concise."
)

V2_PROMPT = (
    "You are a friendly customer support assistant for TechCo. "
    "Answer user questions about our SaaS product. "
    "Always greet the user, provide step-by-step instructions when relevant, "
    "and end with 'Is there anything else I can help with?'. "
    "Keep responses under 100 words."
)

DEMO_ITEMS = [
    {
        "input": "How do I reset my password?",
        "expected_output": None,
    },
    {
        "input": "What are your pricing plans?",
        "expected_output": None,
    },
    {
        "input": "I can't log into my account. It says invalid credentials.",
        "expected_output": None,
    },
    {
        "input": "How do I cancel my subscription?",
        "expected_output": None,
    },
    {
        "input": "Is there a free trial available?",
        "expected_output": None,
    },
    {
        "input": "How do I export my data?",
        "expected_output": None,
    },
    {
        "input": "What integrations do you support?",
        "expected_output": None,
    },
    {
        "input": "I'm getting a 500 error when uploading files.",
        "expected_output": None,
    },
    {
        "input": "Can I add more team members to my plan?",
        "expected_output": None,
    },
    {
        "input": "Do you have an API?",
        "expected_output": None,
    },
]
