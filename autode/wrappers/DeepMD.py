"""Wrapper for DeePMD-kit machine learning potentials."""
from __future__ import annotations

import numpy as np

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union, TYPE_CHECKING

from autode.log import logger
from autode.values import Gradient, PotentialEnergy
from autode.config import Config
from autode.wrappers.methods import Method
from autode.calculations.types import CalculationType as ct
from autode.exceptions import CalculationException, MethodUnavailable
from autode.wrappers.keywords import GradientKeywords, OptKeywords, OptTSKeywords
from autode.atoms import elements

try:
    from deepmd.infer import DeepPot  # type: ignore
except Exception:  # pragma: no cover - optional dependency handling
    try:  # deePMD < 2.0 exposes DeepPotential instead
        from deepmd.infer import DeepPotential as DeepPot  # type: ignore
    except Exception:  # pragma: no cover - optional dependency handling
        DeepPot = None  # type: ignore

if TYPE_CHECKING:
    from autode.calculations.executors import CalculationExecutor
    from autode.species.species import Species


class DeepMD(Method):
    """DeePMD-kit model that provides energies and forces from a graph file."""

    def __init__(self):
        super().__init__(
            name="deepmd",
            keywords_set=Config.DeepMD.keywords,
            doi_list=list(Config.DeepMD.doi_list),
        )

        model_path = Config.DeepMD.model_path
        self.model_path: Optional[Path] = (
            Path(model_path) if model_path is not None else None
        )
        self._type_lookup = self._build_type_lookup(Config.DeepMD.type_map)
        self._graph: Optional[DeepPot] = None

    def __repr__(self) -> str:
        return (
            "DeepMD(available="
            f"{self.is_available}, model_path={self.model_path})"
        )

    @property
    def uses_external_io(self) -> bool:
        return False

    def implements(self, calculation_type: "ct") -> bool:
        return calculation_type in (ct.energy, ct.gradient)

    @property
    def is_available(self) -> bool:
        if DeepPot is None:
            logger.debug("deepmd-kit is not installed; DeepMD unavailable")
            return False

        if self.model_path is None:
            logger.debug("No DeePMD model path configured; DeepMD unavailable")
            return False

        if not self.model_path.exists():
            logger.debug(
                "Configured DeePMD model path %s does not exist", self.model_path
            )
            return False

        if self._type_lookup is None:
            logger.debug("No DeePMD type map configured; DeepMD unavailable")
            return False

        return True

    def execute(self, calc: "CalculationExecutor") -> None:
        if not self.is_available:
            raise MethodUnavailable(
                "DeepMD is unavailable. Ensure deepmd-kit is installed, "
                "Config.DeepMD.model_path points to a DeePMD graph, and "
                "Config.DeepMD.type_map is populated."
            )

        graph = self._load_model()
        coords = np.asarray(calc.molecule.coordinates.to("ang"), dtype=float)
        atom_types = self._atomic_type_indices(calc.molecule)
        box = np.zeros((3, 3), dtype=float)

        need_gradients = self._needs_gradients(calc.input.keywords)

        try:
            energy = float(graph.eval(coords, box, atom_types))
            forces = (
                np.asarray(graph.eval_force(coords, box, atom_types), dtype=float)
                if need_gradients
                else None
            )
        except Exception as exc:  # pragma: no cover - runtime failure handling
            raise CalculationException("DeePMD evaluation failed") from exc

        calc.molecule.energy = PotentialEnergy(
            energy, units="eV", method=self, keywords=calc.input.keywords
        )

        if need_gradients and forces is not None:
            gradient = Gradient(-forces, units="eV/ang").to("Ha/ang")
            calc.molecule.gradient = gradient
        else:
            calc.molecule.gradient = None

        calc.molecule.hessian = None
        return None

    @staticmethod
    def _needs_gradients(keywords) -> bool:
        return isinstance(
            keywords, (GradientKeywords, OptKeywords, OptTSKeywords)
        )

    def _load_model(self) -> "DeepPot":
        if self._graph is None:
            assert DeepPot is not None
            self._graph = DeepPot(str(self.model_path))
        return self._graph

    def _atomic_type_indices(self, species: "Species") -> np.ndarray:
        if self._type_lookup is None:
            raise MethodUnavailable("Config.DeepMD.type_map is not set")

        try:
            return np.asarray(
                [self._type_lookup[self._symbol(atom)] for atom in species.atoms],
                dtype=int,
            )
        except KeyError as exc:
            missing = str(exc.args[0])
            raise CalculationException(
                "Encountered element %s that is missing from the DeePMD type map"
                % missing
            ) from exc

    @staticmethod
    def _symbol(atom) -> str:
        return atom.atomic_symbol

    @staticmethod
    def _build_type_lookup(
        raw_map: Optional[Union[Sequence[Union[str, int]], Mapping[Union[str, int], int]]]
    ) -> Optional[Dict[str, int]]:
        if raw_map is None:
            return None

        if isinstance(raw_map, Mapping):
            lookup = {
                DeepMD._normalise_symbol(symbol): int(index)
                for symbol, index in raw_map.items()
            }
            return lookup or None

        if isinstance(raw_map, Sequence) and not isinstance(raw_map, str):
            lookup: Dict[str, int] = {}
            for index, symbol in enumerate(raw_map):
                lookup[DeepMD._normalise_symbol(symbol)] = int(index)
            return lookup or None

        raise TypeError(
            "Config.DeepMD.type_map must be a mapping or a sequence of element labels"
        )

    @staticmethod
    def _normalise_symbol(symbol: Union[str, int]) -> str:
        if isinstance(symbol, str):
            stripped = symbol.strip()
            if not stripped:
                raise ValueError("Empty element label in DeePMD type map")
            return stripped[0].upper() + stripped[1:].lower()

        if isinstance(symbol, int):
            if symbol <= 0 or symbol > len(elements):
                raise ValueError(
                    f"Atomic number {symbol} is outside the supported range"
                )
            return elements[symbol - 1]

        raise TypeError(
            "Element identifiers in DeePMD type map must be strings or integers"
        )


__all__ = ["DeepMD"]
