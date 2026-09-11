"""
Shared per-node "what should a renderer show" facts for a `DataProcessingPipeline` - factored
out of `damast.viz.svg_export.SvgExporter`/`damast.viz.mermaid_export.MermaidExporter`, which
each walk `pipeline.processing_graph` themselves and call these to turn a `Node` into
display-ready facts (required/input/output columns, description), instead of independently
re-deriving them from `input_metadata`/`input_specs`/`output_specs`/`@describe`.

`ProcessingGraph` already is the shared graph - its `.nodes()`/`.edges(data=True)` give the
traversal and join-slot info directly. What was actually duplicated between renderers was this
per-node fact extraction, so that's the only thing pulled out here; there's no separate graph
structure to keep in sync with it.

`ColumnInfo`/`DataSourceFacts`/`StepFacts` hold only strings and lists thereof - no
`damast.core` types - so a renderer needs no knowledge of `Node`/`PipelineElement` internals
beyond calling `datasource_facts`/`step_facts` on the node it's currently visiting.
"""
from __future__ import annotations

from dataclasses import dataclass

from damast.core.constants import DECORATED_DESCRIPTION
from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.metadata import DataSpecification
from damast.core.processing_graph import Node

__all__ = [
    "ColumnInfo",
    "DataSourceFacts",
    "PipelineExporter",
    "StepFacts",
]


@dataclass(frozen=True)
class ColumnInfo:
    """One column, as declared by a step's `@input`/`@output` decorator or a datasource's
    `input_metadata` - name plus the two optional facts worth showing about it."""
    name: str
    unit: str | None = None
    description: str | None = None
    representation_type: str | None = None


@dataclass(frozen=True)
class DataSourceFacts:
    """Display facts for a datasource node."""
    name: str
    class_name: str
    description: str | None
    required_columns: list[ColumnInfo]


@dataclass(frozen=True)
class StepFacts:
    """Display facts for a processing-step node."""
    name: str
    class_name: str
    description: str | None
    input_slots: dict[str, list[ColumnInfo]]
    output_columns: list[ColumnInfo]


class PipelineExporter:
    """
    Base class for `damast.viz` renderers - holds the `pipeline` being visualized, and the
    per-node display facts (`datasource_facts`/`step_facts`/`output_columns`) shared by every
    renderer built on top of it (`damast.viz.svg_export.SvgExporter`,
    `damast.viz.mermaid_export.MermaidExporter`).

    Args:
        pipeline: The pipeline to visualize
    """
    _pipeline: DataProcessingPipeline

    def __init__(self, pipeline: DataProcessingPipeline):
        self._pipeline = pipeline

    def supported_filetypes(self) -> list[str]:
        """The file extensions (including the leading dot) this exporter can write, e.g. ``[".svg"]``."""
        raise NotImplementedError(f"{self.__class__.__name__}.supported_filetypes has not been implemented")

    @classmethod
    def _column_info(cls, spec: DataSpecification) -> ColumnInfo:
        """Extract the renderer-relevant, already-display-ready facts from one `DataSpecification`."""
        return ColumnInfo(
            name=spec.name,
            unit=spec.unit.to_string() if spec.unit is not None else None,
            representation_type=str(spec.representation_type) if spec.representation_type is not None else "undefined",
            description=spec.description or None,
        )

    @classmethod
    def _description(cls, node: Node) -> str | None:
        """The node's `@describe` text, if any."""
        if hasattr(node.transformer.transform, DECORATED_DESCRIPTION):
            return getattr(node.transformer.transform, DECORATED_DESCRIPTION)
        return None

    def datasource_facts(self, node: Node) -> DataSourceFacts:
        """
        Display facts for a datasource node: the columns the pipeline requires from it, since a
        `DataSource` transformer's own declared input/output is always empty.

        Example:

        ```python
        exporter = PipelineExporter(pipeline)
        for node in pipeline.processing_graph.nodes():
            if node.is_datasource():
                facts = exporter.datasource_facts(node)
        ```

        Args:
            node: A datasource node (`node.is_datasource()`) from this exporter's pipeline

        Returns:
            `DataSourceFacts` for `node`

        Raises:
            RuntimeError: See `DataProcessingPipeline._declared_interface`
        """
        required = self._pipeline.input_metadata(node.name).columns
        return DataSourceFacts(
            name=node.name,
            class_name=type(node.transformer).__name__,
            description=self._description(node),
            required_columns=[self._column_info(spec) for spec in required],
        )

    @classmethod
    def step_facts(cls, node: Node) -> StepFacts:
        """
        Display facts for a processing-step node: its input columns per slot (more than one slot
        only for a join), and its declared output columns.

        Example:

        ```python
        exporter = PipelineExporter(pipeline)
        for node in pipeline.processing_graph.nodes():
            if not node.is_datasource():
                facts = exporter.step_facts(node)
        ```

        Args:
            node: A processing-step node from `pipeline.processing_graph`

        Returns:
            `StepFacts` for `node`
        """
        input_slots: dict[str, list[ColumnInfo]] = {
            label: [cls._column_info(spec) for spec in specs]
            for label, specs in node.transformer.input_specs.items()
        }
        return StepFacts(
            name=node.name,
            class_name=type(node.transformer).__name__,
            description=cls._description(node),
            input_slots=input_slots,
            output_columns=[cls._column_info(spec) for spec in node.transformer.output_specs],
        )

    def output_columns(self) -> list[ColumnInfo]:
        """
        The pipeline's overall guaranteed output columns - the synthetic "pipeline output"
        box/cluster both renderers draw after the last step.

        Example:

        ```python
        exporter = PipelineExporter(pipeline)
        columns = exporter.output_columns()
        ```

        Returns:
            `ColumnInfo` for each of `pipeline`'s guaranteed output columns

        Raises:
            RuntimeError: See `DataProcessingPipeline._declared_interface`
        """
        return [self._column_info(spec) for spec in self._pipeline.output_metadata().columns]
