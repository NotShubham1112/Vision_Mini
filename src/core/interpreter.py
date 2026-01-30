
import json
import re
from typing import Optional, Dict, Any

class VisionCommandInterpreter:
    """
    ROLE: Multimodal Vision Command Interpreter
    Converts spoken instructions into structured JSON for real-time computer vision.
    """
    
    SYSTEM_PROMPT = """
ROLE: Multimodal Vision Command Interpreter
You are an AI module that converts spoken user commands into structured JSON instructions for a real-time computer vision system.

The vision system can see: People, Faces, Everyday objects (bottle, laptop, pen, phone, etc.)

The user may speak naturally and refer to people or objects using phrases like:
“this person”, “that guy”, “the man in front of me”, “this object”, “that bottle”, “the thing on the table”

🎯 POSSIBLE INTENTS:
"name_person", "name_object", "self_identification", "query_identity", "query_object", "none"

🧾 OUTPUT FORMAT (STRICT JSON):
{
  "intent": "one_of_the_intents",
  "target_description": "words describing who/what they refer to",
  "assigned_name": "name or label mentioned by the user",
  "confidence": 0.0_to_1.0
}
"""

    def __init__(self):
        print("[INTERPRETER] Initialized for Mistral Role Integration")

    def interpret(self, text: str) -> Dict[str, Any]:
        """
        Interprets text into the exact JSON structure provided by the user.
        """
        text_lower = text.lower().strip()
        
        # 🧠 INTERPRETATION RULES
        
        # “My name is X” → intent = self_identification, assigned_name = X, target_description = "speaker"
        if "my name is" in text_lower:
            name = self._extract_field(text_lower, r"my name is ([\w\s]+)")
            return self._format("self_identification", "speaker", name, 0.98)

        # “This person is X” → intent = name_person
        if "this person" in text_lower or "that person" in text_lower or "the guy" in text_lower:
            if " is " in text_lower:
                name = self._extract_field(text_lower, r"is ([\w\s]+)")
                desc = self._extract_desc(text_lower, ["this person", "that person", "this guy", "the guy"])
                return self._format("name_person", desc, name, 0.95)

        # “That is a bottle” → intent = name_object
        if "that is a" in text_lower or "this is a" in text_lower or "that bottle is" in text_lower:
            name = self._extract_field(text_lower, r"(?:a |is )([\w\s]+)")
            desc = self._extract_desc(text_lower, ["this object", "that object", "that bottle", "this bottle", "the thing"])
            return self._format("name_object", desc, name, 0.88)

        # “Who is this?” → intent = query_identity
        if "who is" in text_lower or "identify" in text_lower:
            desc = self._extract_desc(text_lower, ["this guy", "this person", "that guy"])
            return self._format("query_identity", desc, None, 0.93)

        # “What is that object?” → intent = query_object
        if "what is" in text_lower or "what's that" in text_lower:
            desc = self._extract_desc(text_lower, ["that object", "this object", "that thing"])
            return self._format("query_object", desc, None, 0.91)

        return self._format("none", None, None, 0.99)

    def _extract_field(self, text, pattern):
        match = re.search(pattern, text)
        return match.group(1).strip().title() if match else None

    def _extract_desc(self, text, options):
        for opt in options:
            if opt in text:
                return opt
        return text # Fallback to full text if none found

    def _format(self, intent, desc, name, conf):
        return {
            "intent": intent,
            "target_description": desc,
            "assigned_name": name,
            "confidence": conf
        }


if __name__ == "__main__":
    # Test cases from the User's prompt
    interp = VisionCommandInterpreter()
    print(json.dumps(interp.interpret("my name is shubham"), indent=2))
    print(json.dumps(interp.interpret("this person here is parth"), indent=2))
    print(json.dumps(interp.interpret("that bottle is mine"), indent=2))
    print(json.dumps(interp.interpret("who is this guy"), indent=2))
    print(json.dumps(interp.interpret("nice weather today"), indent=2))
