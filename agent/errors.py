class BudgetExceeded(RuntimeError):
    """Raised when a deterministic task budget is exhausted."""


class RepositoryError(RuntimeError):
    """Raised when a repository operation is invalid or unsafe."""


class LLMProviderError(RuntimeError):
    """Base error for LLM provider failures."""


class LLMConnectionError(LLMProviderError):
    """Raised when connection to the LLM backend fails or times out."""


class LLMResponseError(LLMProviderError):
    """Raised when the LLM returns an HTTP error, malformed output, or invalid JSON."""
