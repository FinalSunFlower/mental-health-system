import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.cuspnet.layer3_llm import LazarusAppraisalChain

chain = LazarusAppraisalChain(model_name=r"D:\Models\huggingface\Qwen3.5-2B")

test_texts = [
    "I feel so hopeless, nothing matters anymore. I cant get out of bed and I dont see the point of living.",
    "Had a great day today! Went for a run and met friends for coffee. Feeling grateful.",
]

for i, text in enumerate(test_texts):
    print(f"=== Test {i+1} ===")
    print(f"Text: {text[:80]}...")

    prompt = (
        "You are a clinical psychologist. Rate the mental health threat in this text on a scale of 1-10. "
        'Respond ONLY with a JSON: {"threat_identified": true, "threat_type": "string", '
        '"threat_intensity": 8, "threat_narrative": "string", "primary_appraisal_score": 8}\n\n'
        f"Text: {text}\n\nJSON response:"
    )

    raw = chain._generate(prompt)
    print(f"Raw output (first 500 chars): [{raw[:500]}]")
    parsed = chain._parse_json(raw, {"primary_appraisal_score": 5})
    print(f"Parsed: {parsed}")
    print(f"primary_appraisal_score: {parsed.get('primary_appraisal_score', 'MISSING')}")
    print()
