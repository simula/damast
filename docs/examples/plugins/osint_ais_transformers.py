import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.transformations import PipelineElement


class JoinByTimestamp(PipelineElement):
    """Join two datasources on a shared timestamp column - a minimal join operator."""

    @damast.core.describe("Join by timestamp")
    @damast.core.input({"timestamp": {}, "lon": {}, "lat": {}})
    @damast.core.input({"timestamp": {}, "lat": {}, "lon": {}}, label="other")
    @damast.core.output({
        "{{other:event_type}}": {"representation_type": str},
        "{{other:lat}}": {},
        "{{other:lon}}": {},
    })
    def transform(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame) -> AnnotatedDataFrame:
        other_timestamp = self.get_name("timestamp", datasource="other")
        df_timestamp = self.get_name("timestamp")
        df.lazyframe = df.join(other.lazyframe, left_on=df_timestamp, right_on=other_timestamp)
        df._metadata = df._metadata.merge(other._metadata).drop(other_timestamp)
        return df
