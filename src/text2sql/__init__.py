"""Text-to-SQL agent: plain-English questions in, validated read-only SQL out."""

from text2sql.agent import Result, TextToSQLAgent
from text2sql.executor import QueryError, QueryResult, QueryTimeout, execute
from text2sql.guard import SQLRejected, validate

__all__ = [
    "Result",
    "TextToSQLAgent",
    "QueryError",
    "QueryResult",
    "QueryTimeout",
    "SQLRejected",
    "execute",
    "validate",
]
