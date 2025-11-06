import datetime as dt

import pytest

from damast.core.annotations import Annotation, Change, History


def test_annotation():
    with pytest.raises(ValueError) as _:
        Annotation(name=Annotation.Key.Comment)

    with pytest.raises(ValueError) as _:
        Annotation(name=Annotation.Key.License)

    a = Annotation(name=Annotation.Key.Comment,
                   value="test-comment")

    assert a.name == Annotation.Key.Comment
    assert a.value == "test-comment"

    equal_a = Annotation(name=a.name,
                         value=a.value)
    b = Annotation(name=Annotation.Key.Comment,
                         value="test-comment-b")

    assert a != 10
    assert a == equal_a
    assert a != b
    assert a != Annotation(name="test-name",
                           value=a.value)

def test_change():
    with pytest.raises(ValueError, match="timezone-aware"):
        c = Change(title="a-change-name",
                   description="a-change-description",
                   timestamp=dt.datetime.now()
                   )

    created = dt.datetime.fromisoformat("2024-01-01 00:00:01")
    created = created.replace(tzinfo=dt.timezone.utc)
    timestamp_txt = created.strftime(Change.TIMESTAMP_FORMAT)
    timestamp = dt.datetime.strptime(timestamp_txt, Change.TIMESTAMP_FORMAT)

    c = Change(title="a-change-name",
               description="a-change-description",
               timestamp=created
               )

    equal_c = Change(title="a-change-name",
               description="a-change-description",
               timestamp=created
               )

    assert c.timestamp == timestamp
    assert c == equal_c

    assert timestamp_txt in str(c)

    assert c != 0
    assert c != Change(title="new-title", description=c.description, timestamp=created)
    assert c != Change(title=c.title, description="other-description", timestamp=created)

    assert c != Change(title=c.title, description=c.description)


def test_history():
    history = History()

    assert history.name == Annotation.Key.History.value
    a_change = Change(title="a-change", description="a-change-description")
    b_change = Change(title="b-change", description="b-change-description")

    history.add_change(a_change)
    history.add_change(b_change)

    assert a_change in history.changes
    assert b_change in history.changes

    assert history != 0
    assert history != History()
    assert history != History(changes=[a_change])
    print(history.changes)
    equal_history = History(changes=[b_change, a_change])
    print(equal_history.changes)
    assert history == equal_history
