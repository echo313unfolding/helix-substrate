"""
Verifier v1 — rule-based verification for lobe scheduler.

Verification types:
    code_sanity          — syntax + unsafe pattern detection via ast
    shell_safety         — dangerous shell command pattern detection
    receipt_consistency  — structural/semantic route receipt validation
    symbolic_ir          — SymbolicIR dict or raw text validation

Each check returns a VerificationResult with:
    verification_type, target, result, issues, confidence, action_allowed

StrictVerificationMode is a context manager that raises RuntimeError
when a verification fails with action_allowed=False — prevents the
pipeline from silently passing bad output.

Capability manifest documents what each verification type checks and
whether it uses a model backend.

Work Orders: WO-VERIFIER-LOBE-V1, WO-SYMBOLIC-IR-V1
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Verification types
# ---------------------------------------------------------------------------

class VerificationType(Enum):
    CODE_SANITY = "code_sanity"
    SHELL_SAFETY = "shell_safety"
    RECEIPT_CONSISTENCY = "receipt_consistency"
    SYMBOLIC_IR = "symbolic_ir"


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Structured result from a verification check."""
    verification_type: VerificationType
    target: str               # human-readable description of what was verified
    result: str               # "pass", "fail", "warn"
    issues: List[str]
    confidence: float         # 0.0-1.0
    action_allowed: bool
    extra: dict = field(default_factory=dict)  # type-specific receipt fields

    def to_receipt(self) -> dict:
        """Serialize to receipt-compatible dict."""
        receipt = {
            "verification_type": self.verification_type.value,
            "verification_target": self.target,
            "verification_result": self.result,
            "issues_found": self.issues,
            "confidence": self.confidence,
            "action_allowed": self.action_allowed,
        }
        if self.extra:
            receipt.update(self.extra)
        return receipt


# ---------------------------------------------------------------------------
# Capability manifest
# ---------------------------------------------------------------------------

CAPABILITY_MANIFEST = {
    "code_sanity": {
        "description": "Python syntax validation and unsafe pattern detection",
        "checks": [
            "ast_parse", "eval_exec_usage", "subprocess_shell",
            "import_os_system", "dunder_import",
        ],
        "model_backed": True,
    },
    "shell_safety": {
        "description": "Dangerous shell command pattern detection",
        "checks": [
            "destructive_commands", "pipe_to_shell", "device_writes",
            "fork_bombs", "privilege_escalation",
        ],
        "model_backed": False,
    },
    "receipt_consistency": {
        "description": "Route receipt structural and semantic validation",
        "checks": [
            "schema_match", "required_fields", "lobe_registry",
            "step_count", "verification_flag",
        ],
        "model_backed": False,
    },
    "symbolic_ir": {
        "description": "SymbolicIR schema validation — required fields, known action_type/target, constraint consistency",
        "checks": [
            "schema_match", "required_fields", "known_action_type",
            "known_compiler_target", "constraint_types",
            "parse_status", "balanced_delimiters",
        ],
        "model_backed": False,
    },
}


# ---------------------------------------------------------------------------
# Shell safety patterns
# ---------------------------------------------------------------------------

_DANGEROUS_COMMANDS = [
    r"\brm\s+-rf\s+/",
    r"\brm\s+-rf\s+~",
    r"\brm\s+-rf\s+\*",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{",           # fork bomb
    r"\bchmod\s+(-R\s+)?777\s+/",
    r">\s*/dev/sd[a-z]",
    r"\bformat\s+[A-Za-z]:",
]

_PIPE_TO_SHELL = [
    r"curl\s+.*\|\s*(ba)?sh",
    r"wget\s+.*\|\s*(ba)?sh",
    r"curl\s+.*\|\s*sudo\s+(ba)?sh",
]

_PRIVILEGE_ESCALATION = [
    r"\bsudo\s+rm\b",
    r"\bsudo\s+chmod\b",
    r"\bsudo\s+chown\b",
    r"\bsudo\s+dd\b",
]


# ---------------------------------------------------------------------------
# Unsafe code patterns
# ---------------------------------------------------------------------------

_UNSAFE_CODE_PATTERNS = [
    (r"\beval\s*\(", "eval() usage detected"),
    (r"\bexec\s*\(", "exec() usage detected"),
    (r"\bos\.system\s*\(", "os.system() usage detected"),
    (r"subprocess\.call\s*\(.*shell\s*=\s*True", "subprocess.call with shell=True"),
    (r"subprocess\.Popen\s*\(.*shell\s*=\s*True", "subprocess.Popen with shell=True"),
    (r"__import__\s*\(", "__import__() usage detected"),
]

# Heuristic for "looks like Python code"
_PYTHON_KEYWORDS = ("def ", "class ", "import ", "from ", "if ", "for ", "while ", "return ")


# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------

def verify_code_sanity(code: str) -> VerificationResult:
    """Check Python code for syntax errors and unsafe patterns.

    Syntax check only runs when content looks like Python (has keywords).
    Unsafe pattern detection always runs.
    """
    issues: List[str] = []

    # Syntax check only if content appears to be Python
    looks_like_python = any(kw in code for kw in _PYTHON_KEYWORDS)
    if looks_like_python:
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"SyntaxError: {e.msg} (line {e.lineno})")

    # Unsafe pattern check — always runs
    for pattern, msg in _UNSAFE_CODE_PATTERNS:
        if re.search(pattern, code):
            issues.append(msg)

    if not issues:
        return VerificationResult(
            verification_type=VerificationType.CODE_SANITY,
            target="code_output",
            result="pass",
            issues=[],
            confidence=0.9,
            action_allowed=True,
        )

    has_syntax_error = any("SyntaxError" in i for i in issues)
    return VerificationResult(
        verification_type=VerificationType.CODE_SANITY,
        target="code_output",
        result="fail" if has_syntax_error else "warn",
        issues=issues,
        confidence=0.85,
        action_allowed=not has_syntax_error,
    )


def verify_shell_safety(command: str) -> VerificationResult:
    """Check a shell command for dangerous patterns."""
    issues: List[str] = []

    for pattern in _DANGEROUS_COMMANDS:
        if re.search(pattern, command, re.IGNORECASE):
            issues.append(f"dangerous command: {pattern}")

    for pattern in _PIPE_TO_SHELL:
        if re.search(pattern, command, re.IGNORECASE):
            issues.append(f"pipe to shell: {pattern}")

    for pattern in _PRIVILEGE_ESCALATION:
        if re.search(pattern, command, re.IGNORECASE):
            issues.append(f"privilege escalation: {pattern}")

    if not issues:
        return VerificationResult(
            verification_type=VerificationType.SHELL_SAFETY,
            target="shell_command",
            result="pass",
            issues=[],
            confidence=0.95,
            action_allowed=True,
        )

    return VerificationResult(
        verification_type=VerificationType.SHELL_SAFETY,
        target="shell_command",
        result="fail",
        issues=issues,
        confidence=0.95,
        action_allowed=False,
    )


def verify_receipt_consistency(receipt: dict) -> VerificationResult:
    """Check a route receipt for structural and semantic issues.

    Delegates to validate_route_receipt from lobe_scheduler.
    """
    from helix_substrate.lobe_scheduler import validate_route_receipt

    issues = validate_route_receipt(receipt)

    return VerificationResult(
        verification_type=VerificationType.RECEIPT_CONSISTENCY,
        target="route_receipt",
        result="pass" if not issues else "fail",
        issues=issues,
        confidence=1.0,
        action_allowed=len(issues) == 0,
    )


def verify_symbolic_ir(ir_input: str) -> VerificationResult:
    """Validate symbolic IR.

    Accepts either:
        - JSON string of a SymbolicIR dict → full schema validation
        - Raw text → basic structural checks (balanced delimiters)

    Full validation (JSON IR dict) checks:
        - Required fields present
        - Known action_type and compiler_target
        - Constraint type consistency
        - Parse status (error/ambiguous blocks execution)
        - Source text non-empty

    Receipt includes ir_schema, ir_valid, ir_issues, compiler_target,
    execution_allowed fields per WO-SYMBOLIC-IR-V1.
    """
    # Try JSON parse — full IR validation
    try:
        ir_dict = json.loads(ir_input)
        if isinstance(ir_dict, dict) and "ir_schema" in ir_dict:
            return _validate_symbolic_ir_dict(ir_dict)
    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to text-based checks
    return _validate_symbolic_ir_text(ir_input)


# Stems — matched with \b prefix only, so "drop" catches "dropping", "dropped", etc.
_DESTRUCTIVE_STEMS = ("drop", "delet", "destroy", "truncat", "wipe", "purg", "eras")


def _validate_symbolic_ir_dict(ir_dict: dict) -> VerificationResult:
    """Full SymbolicIR dict validation."""
    from helix_substrate.symbolic_ir import validate_symbolic_ir, SCHEMA_SYMBOLIC_IR

    issues = validate_symbolic_ir(ir_dict)
    is_valid = len(issues) == 0
    parse_ok = ir_dict.get("parse_status") == "ok"

    # execution_allowed: only True if validation passes AND parse succeeded
    execution_allowed = is_valid and parse_ok

    # Safety gate: detect destructive operations in target/intent.
    # Destructive keywords (drop, delete, destroy, ...) require explicit
    # authorization beyond allow_side_effects. This is a policy-level check:
    # "may write state" is different from "may destroy data."
    if execution_allowed:
        target = ir_dict.get("target", "").lower()
        intent = ir_dict.get("intent", "").lower()
        combined = f"{target} {intent}"
        found = [stem for stem in _DESTRUCTIVE_STEMS
                 if re.search(r"\b" + re.escape(stem), combined)]
        if found:
            issues.append(
                f"destructive operation detected ({', '.join(sorted(found))}) "
                f"without explicit authorization"
            )
            execution_allowed = False
            is_valid = False

    return VerificationResult(
        verification_type=VerificationType.SYMBOLIC_IR,
        target="symbolic_ir",
        result="pass" if is_valid else "fail",
        issues=issues,
        confidence=0.95,
        action_allowed=execution_allowed,
        extra={
            "ir_schema": ir_dict.get("ir_schema", SCHEMA_SYMBOLIC_IR),
            "ir_valid": is_valid,
            "ir_issues": issues,
            "compiler_target": ir_dict.get("compiler_target"),
            "execution_allowed": execution_allowed,
        },
    )


def _validate_symbolic_ir_text(ir_text: str) -> VerificationResult:
    """Text-based structural checks (balanced delimiters). Fallback path."""
    issues: List[str] = []

    if not ir_text or not ir_text.strip():
        return VerificationResult(
            verification_type=VerificationType.SYMBOLIC_IR,
            target="symbolic_ir",
            result="fail",
            issues=["empty IR text"],
            confidence=1.0,
            action_allowed=False,
        )

    # Balanced delimiters
    stack: List[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    for ch in ir_text:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack[-1] != ch:
                issues.append(f"unbalanced delimiter: '{ch}'")
                break
            stack.pop()
    if stack:
        issues.append(f"unclosed delimiters: {''.join(stack)}")

    return VerificationResult(
        verification_type=VerificationType.SYMBOLIC_IR,
        target="symbolic_ir",
        result="pass" if not issues else "fail",
        issues=issues,
        confidence=0.7,  # text-only check — lower confidence
        action_allowed=len(issues) == 0,
    )


# ---------------------------------------------------------------------------
# Dispatch: run the right verification for a given type string
# ---------------------------------------------------------------------------

def run_verification(content: str, verification_type: str) -> VerificationResult:
    """Dispatch to the appropriate verification function.

    Args:
        content: text to verify (code, shell command, JSON receipt, or IR)
        verification_type: VerificationType.value string

    Returns:
        VerificationResult
    """
    if verification_type == VerificationType.CODE_SANITY.value:
        return verify_code_sanity(content)
    elif verification_type == VerificationType.SHELL_SAFETY.value:
        return verify_shell_safety(content)
    elif verification_type == VerificationType.RECEIPT_CONSISTENCY.value:
        try:
            receipt = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return VerificationResult(
                verification_type=VerificationType.RECEIPT_CONSISTENCY,
                target="route_receipt",
                result="fail",
                issues=["content is not valid JSON"],
                confidence=1.0,
                action_allowed=False,
            )
        return verify_receipt_consistency(receipt)
    elif verification_type == VerificationType.SYMBOLIC_IR.value:
        return verify_symbolic_ir(content)
    else:
        return VerificationResult(
            verification_type=VerificationType.CODE_SANITY,
            target="unknown",
            result="warn",
            issues=[f"unknown verification type: {verification_type}"],
            confidence=0.0,
            action_allowed=True,
        )


# ---------------------------------------------------------------------------
# StrictVerificationMode
# ---------------------------------------------------------------------------

class StrictVerificationMode:
    """Context manager that blocks action on verification failure.

    When active, any VerificationResult with action_allowed=False
    raises RuntimeError. Modeled after StrictBenchmarkMode.

    Usage:
        with StrictVerificationMode():
            result = verify_code_sanity(code)
            StrictVerificationMode.enforce(result)  # raises if fail
    """
    _active = False

    def __enter__(self):
        StrictVerificationMode._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        StrictVerificationMode._active = False
        return False

    @classmethod
    def is_active(cls) -> bool:
        return cls._active

    @classmethod
    def enforce(cls, result: VerificationResult):
        """Raise RuntimeError if strict mode is active and verification failed."""
        if cls._active and not result.action_allowed:
            raise RuntimeError(
                f"StrictVerificationMode: verification failed — "
                f"type={result.verification_type.value}, "
                f"result={result.result}, "
                f"issues={result.issues}"
            )
