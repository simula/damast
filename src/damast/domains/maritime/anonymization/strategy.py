"""
The interface every anonymization strategy for AIS data implements.
"""
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.transformations import PipelineElement
from damast.domains.maritime.anonymization.owner_risk import DropHighOwnerRisk, OwnerRisk
from damast.domains.maritime.transformers.mmsi_pattern_classifier import MMSIPatternClassifier

__all__ = [
    "AISColumns",
    "AnonymizationStrategy",
    "Step",
    "owner_risk_steps",
]


@dataclass(frozen=True)
class AISColumns:
    """
    Names of the AIS columns in the data to anonymize - the defaults are those of the
    Norwegian Coastal Administration's AIS feed.
    """
    mmsi: str = "mmsi"
    timestamp: str = "msgtime"
    lat: str = "latitude"
    lon: str = "longitude"
    speed: str = "speedOverGround"
    course: str = "courseOverGround"
    status: str = "navigationalStatus"
    ship_type: str = "shipType"
    imo: str = "imoNumber"
    #: AIS transponder class, "A" or "B" - None if the data has no such column
    report_class: str | None = "reportClass"


class Step(NamedTuple):
    """
    A pipeline step, as the arguments of `DataProcessingPipeline.add`.
    """
    name: str
    transformer: PipelineElement
    name_mappings: dict[str, str]


def owner_risk_steps(columns: AISColumns) -> list[Step]:
    """
    The steps that drop the vessels likely to identify a private person - see `OwnerRisk`.

    Args:
        columns: Names of the AIS columns in the data

    Returns:
        `MMSIPatternClassifier`, `OwnerRisk` and `DropHighOwnerRisk`
    """
    category = f"{columns.mmsi}_category"
    return [
        Step("classify mmsi", MMSIPatternClassifier(), {"x": columns.mmsi}),
        Step("assess owner risk", OwnerRisk(report_class=columns.report_class),
             {"mmsi": columns.mmsi, "mmsi_category": category, "ship_type": columns.ship_type,
              "imo": columns.imo}),
        Step("drop high owner risk", DropHighOwnerRisk(), {"owner_risk": "owner_risk"}),
    ]


class AnonymizationStrategy(DataProcessingPipeline, ABC):
    """
    A way to anonymize AIS data - a pipeline made of the steps the strategy defines.

    It runs on its own, or is chained after other steps with `DataProcessingPipeline.extend`.
    Either way it saves as a plain pipeline that lists each step with its parameters, so it can
    be loaded without the strategy class.

    A new strategy implements `steps`, and sets its own parameters before calling this
    constructor, which builds the steps. Its last step should keep only the columns that may be
    released, see `PseudonymizeVoyages` for an example.

    Example:

    ```python
    from damast.core.dataprocessing import DataProcessingPipeline
    from damast.domains.maritime.anonymization import VoyageKAnonymity

    anonymize = VoyageKAnonymity(key_env="AIS_PSEUDONYM_KEY", k=5)
    released = anonymize.transform(df=adf)

    pipeline = DataProcessingPipeline(name="ais_public_release", base_dir=".")
    pipeline.add("drop missing mmsi", DropMissingOrNan(), name_mappings={"x": "mmsi"})
    pipeline.extend(anonymize)
    ```

    Args:
        columns: Names of the AIS columns in the data
        name: Name of the pipeline, also the prefix of its step names, "<name>: <step>"
        base_dir: Base directory of the pipeline, see `DataProcessingPipeline`
    """

    def __init__(self, *, columns: AISColumns = AISColumns(), name: str = "anonymize",
                 base_dir: str | Path = tempfile.gettempdir()):
        super().__init__(name=name, description=self.__class__.__doc__.strip().splitlines()[0],
                         base_dir=base_dir)
        self.columns = columns
        for step in self.steps(columns):
            self.add(f"{name}: {step.name}", step.transformer, name_mappings=step.name_mappings)

    @abstractmethod
    def steps(self, columns: AISColumns) -> list[Step]:
        """
        The steps that anonymize data with the given columns, in order.

        Args:
            columns: Names of the AIS columns in the data

        Returns:
            The pipeline steps, in the order they run
        """
