from google.api_core.exceptions import ResourceExhausted
from tenacity import RetryCallState
import re
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

def gemini_wait(retry_state: RetryCallState) -> float:
    """
    Tenacity wait strategy that respects the `retry_delay` Google returns
    inside a 429 ResourceExhausted error. Falls back to 24 s.
    """
    exc = retry_state.outcome.exception()
    # Default sleep
    delay_seconds = 24

    if isinstance(exc, ResourceExhausted):
        # 1. Native attribute (google-api-core >= 2.18)
        native = getattr(exc, "retry_delay", None)
        if native is not None:
            delay_seconds = getattr(native, "seconds", delay_seconds)
        else:
            # 2. Parse the serialized proto in the message
            m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(exc))
            if m:
                delay_seconds = int(m.group(1))

    logging.warning(
        "Gemini quota hit - sleeping %.0f s before retry (attempt %s).",
        delay_seconds,
        retry_state.attempt_number,
    )
    # Tenacity accepts a numeric return (seconds)
    return delay_seconds