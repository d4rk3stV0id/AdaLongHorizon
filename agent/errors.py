class BudgetExceeded(RuntimeError):
    """Raised when a deterministic task budget is exhausted."""


class RepositoryError(RuntimeError):
    """Raised when a repository operation is invalid or unsafe."""
