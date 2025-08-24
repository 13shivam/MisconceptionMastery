import os
from typing import Dict, List, Optional


def generate_counterfactual_llm(
        stem: str,
        distractors: List[str],
        misconception_label: str,
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 250,
) -> Optional[Dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            prompt = f"""
You are generating a learning item variant to address a specific misconception.
Original stem: {stem}
Original options: {distractors}
Misconception to address: {misconception_label}

Task: Create a revised item that explicitly challenges the misconception without giving away the answer.
Return JSON with keys: stem, options (4), correct_index (0-based).
"""
            resp = client.chat.completions.create(
                model=chosen_model,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content
        except Exception:
            import openai
            openai.api_key = api_key
            chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            prompt = f"""
You are generating a learning item variant to address a specific misconception.
Original stem: {stem}
Original options: {distractors}
Misconception to address: {misconception_label}

Task: Create a revised item that explicitly challenges the misconception without giving away the answer.
Return JSON with keys: stem, options (4), correct_index (0-based).
"""
            resp = openai.ChatCompletion.create(
                model=chosen_model,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp["choices"][0]["message"]["content"]

        import json, re
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and "stem" in data and "options" in data and "correct_index" in data:
                return data
        return {"stem": text.strip(), "options": distractors, "correct_index": 0}
    except Exception:
        return None
