from datetime import datetime
from typing import Optional


def unix_ms_to_datetime(unix_ms: int) -> datetime:
    return datetime.fromtimestamp(unix_ms / 1000)


def datetime_to_unix_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def iso_to_unix_ms(iso_str: Optional[str]) -> int:
    if iso_str is None:
        return 0

    # Strip the 'Z' and parse the datetime string
    dt: datetime = datetime.fromisoformat(iso_str.rstrip('Z'))

    # Convert datetime to Unix time in milliseconds
    unix_ms = int(dt.timestamp() * 1000)
    return unix_ms


# See https://api-docs.elevio.help/en/articles/25-setlanguage for list of valid language codes
VALID_LANGUAGE_CODES = {"ar", "bg", "cs", "da", "de", "de-at", "de-ch", "el", "en", "en-us", "en-gb", "es", "es-419",
                        "fa", "fi", "fr", "hu", "id", "it", "iw", "ja", "ko", "nl", "nl-be", "nn", "pl", "pt-br", "ro",
                        "ru", "sk", "sr", "sv", "th", "tr", "uk", "vi", "zh", "zh-hant", "en", "es", "fr", "de", "it",
                        "ja", "ko", "pt", "ru", "zh"}


def validate_language_code(language_code: str) -> bool:
    return language_code in VALID_LANGUAGE_CODES
