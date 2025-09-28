import sys

if sys.version_info < (3,11):
    import iso8601
    def fromisoformat(iso_string: str):
        return iso8601.parse_date(iso_string)
else:
    import datetime as dt
    def fromisoformat(iso_string: str):
        return dt.datetime.fromisoformat(iso_string)
