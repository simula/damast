import sys
import importlib

if sys.version_info < (3,11):
    import iso8601
    def fromisoformat(iso_string: str):
        return iso8601.parse_date(iso_string)
else:
    import datetime as dt
    def fromisoformat(iso_string: str):
        return dt.datetime.fromisoformat(iso_string)


def ensure_packages(pkgs: list[str], required_for: str, hint: str | None = None,
                    install: dict[str, str] = {}):
    requirements = []
    for pkg in pkgs:
        if pkg in install:
            requirements.append(install[pkg])
        else:
            requirements.append(pkg)

        msg = f"{required_for} requires to 'pip install {' '.join(requirements)}'"

        if hint is not None:
            msg += f", {hint}"

        if not importlib.util.find_spec(pkg):
            raise RuntimeError(msg)
