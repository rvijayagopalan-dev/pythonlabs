import re
from typing import List

BANNED = [r"(?i)password", r"(?i)credit\s*card", r"(?i)ssn"]


def basic_input_guard(text: str) -> List[str]:
    issues = []
    if len(text) > 8000:
        issues.append("input_too_long")
    for pat in BANNED:
        if re.search(pat, text):
            issues.append("contains_sensitive_marker")
            break
    return issues
