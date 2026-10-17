"""Explicit exception hierarchy for AML Sentinel.

Every error raised deliberately by the package derives from :class:`AmlSentinelError`,
so callers (CLI, API) can map them to safe user-facing messages without leaking
internals.

Security note:
    Exception messages must never contain secret values or raw sensitive field
    values. Include counts, column names or identifiers of *files*, not data.
"""


class AmlSentinelError(Exception):
    """Base class for all AML Sentinel errors."""


class ConfigurationError(AmlSentinelError):
    """Settings are missing, malformed or mutually inconsistent."""


class DataNotFoundError(AmlSentinelError):
    """An expected input file does not exist."""


class DataValidationError(AmlSentinelError):
    """Input data does not match the expected schema or value constraints."""


class FxEstimationError(AmlSentinelError):
    """Exchange rates could not be estimated reliably from the data."""


class LeakageError(AmlSentinelError):
    """An operation would let future information leak into the past."""


class IntegrityError(AmlSentinelError):
    """A file failed checksum verification."""
