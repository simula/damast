import pytest

from damast.core.annotations import Annotation


def test_annotation():
    with pytest.raises(ValueError) as e:
        Annotation(name=Annotation.Key.Comment)

    with pytest.raises(ValueError) as e:
        Annotation(name=Annotation.Key.License)

    a = Annotation(name=Annotation.Key.Comment,
                   value="test-comment")

    assert a.name == Annotation.Key.Comment
    assert a.value == "test-comment"


