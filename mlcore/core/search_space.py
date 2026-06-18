"""
mlcore.core.search_space
========================

Search space abstraction used by optimization methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json


@dataclass(slots=True)
class SearchSpace:
    """
    Hyperparameter search space definition.
    """

    name: str

    parameters: dict[str, list]

    @classmethod
    def from_dict(
        cls,
        name: str,
        parameters: dict[str, list],
    ) -> "SearchSpace":
        """
        Build search space from dictionary.
        """

        return cls(
            name=name,
            parameters=parameters,
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "SearchSpace":
        """
        Load search space from JSON file.
        """

        with open(
            path,
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

        return cls(
            name=data["name"],
            parameters=data["parameters"],
        )

    def to_dict(
        self,
    ) -> dict:
        """
        Export search space as dictionary.
        """

        return {
            "name": self.name,
            "parameters": self.parameters,
        }

    def to_json(
        self,
        path: str | Path,
    ) -> None:
        """
        Export search space to JSON.
        """

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=4,
            )

    def get(
        self,
        parameter: str,
    ) -> list:
        """
        Retrieve parameter values.
        """

        return self.parameters[parameter]

    def exists(
        self,
        parameter: str,
    ) -> bool:
        """
        Check whether parameter exists.
        """

        return parameter in self.parameters

    def __len__(
        self,
    ) -> int:
        """
        Number of tunable parameters.
        """

        return len(
            self.parameters,
        )