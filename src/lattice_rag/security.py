import logging
import os
import re
from pathlib import Path
from typing import Any

class SecretStr:
    """A wrapper around a string to mask secret values in logs and outputs."""
    
    def __init__(self, value: str):
        """
        Initialize the SecretStr. Works with msgspec as long as it handles
        single-argument string constructors.
        """
        self._value = value
    
    def __str__(self) -> str:
        return self._mask()

    def __repr__(self) -> str:
        return f"SecretStr('{self._mask()}')"

    def get_secret_value(self) -> str:
        """Returns the actual unmasked secret string."""
        return self._value

    def _mask(self) -> str:
        """Masks the secret if it matches common prefixes, otherwise returns [REDACTED]."""
        if not self._value:
            return "[REDACTED]"
        
        prefixes = ("sk-", "gsk_", "AIza")
        for prefix in prefixes:
            if self._value.startswith(prefix):
                if len(self._value) > len(prefix) + 4:
                    return f"{prefix}****{self._value[-4:]}"
                return f"{prefix}****"
        return "[REDACTED]"


class RedactingFilter(logging.Filter):
    """
    A logging filter that scans log records and redacts patterns matching API
    key prefixes (sk-, gsk_, AIza, Bearer, token=) using regex substitution.
    """
    
    # Matches common token patterns and URI credentials
    REDACT_PATTERN = re.compile(
        r"(?:sk-[a-zA-Z0-9_-]+|gsk_[a-zA-Z0-9_-]+|AIza[a-zA-Z0-9_-]+|Bearer\s+[a-zA-Z0-9_.-]+|token=[a-zA-Z0-9_.-]+|://[^:]+:[^@]+@)"
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.REDACT_PATTERN.sub("[REDACTED]", record.msg)
            
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    new_args.append(self.REDACT_PATTERN.sub("[REDACTED]", arg))
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)
            
        return True


def migrate_apikeys_to_dotenv(apikeys_path: Path, dotenv_path: Path) -> bool:
    """
    Reads apikeys.md, extracts lines matching known key patterns, 
    writes them to .env file, sets file permissions to 0o600, 
    and deletes the apikeys.md file.
    
    Returns:
        bool: True if migration occurred, False otherwise.
    """
    if not apikeys_path.exists():
        return False

    content = apikeys_path.read_text(encoding="utf-8")
    
    # Extract keys and their values
    pattern = re.compile(
        r"(TYPESAFE_API_KEY|GROQ_API_KEY|GEMINI_API_KEY|REDIS_URL)\s*[:=]\s*([^\s`]+)"
    )
    matches = pattern.findall(content)
    
    if not matches:
        # Still delete the file if it exists but no matches found?
        # Specification says "extracts... writes... sets permissions... and deletes"
        # We assume if the file exists we attempt migration.
        apikeys_path.unlink()
        return True

    env_lines = []
    for k, v in matches:
        env_lines.append(f"{k}={v}")

    # Write or append to dotenv
    with dotenv_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(env_lines) + "\n")
        
    # Set permissions to 0o600
    dotenv_path.chmod(0o600)
    
    # Delete the old file
    apikeys_path.unlink()
    
    return True


def sanitize_error_detail(detail: str) -> str:
    """
    Strips potential auth tokens/headers from error message strings.
    """
    if not detail:
        return detail
    pattern = re.compile(
        r"(?:sk-[a-zA-Z0-9_-]+|gsk_[a-zA-Z0-9_-]+|AIza[a-zA-Z0-9_-]+|Bearer\s+[a-zA-Z0-9_.-]+|token=[a-zA-Z0-9_.-]+|://[^:]+:[^@]+@)"
    )
    return pattern.sub("[REDACTED]", detail)
