"""Deterministic stand-in for a model.

Used by the tests and by ``eval/run_eval.py --fake`` so the whole pipeline can be
exercised without an API key. A scripted model also makes retry behaviour
assertable: each response is consumed in order and running out is an error, which
catches accidental extra model calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScriptedModel:
    responses: list[str]
    calls: list[list[dict[str, str]]] = field(default_factory=list)

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError(
                f"ScriptedModel received an unexpected call (#{len(self.calls)}); "
                "no scripted responses left"
            )
        return self.responses.pop(0)
