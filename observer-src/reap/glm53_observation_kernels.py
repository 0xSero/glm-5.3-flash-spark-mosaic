"""Fail-closed optional kernel selection and checkpoint identity."""
import importlib
import inspect
import os


def prepare_kda_import():
    backend = os.environ.get("GLM53_OBSERVER_KDA", "auto")
    if backend not in ("auto", "fla"):
        raise ValueError("GLM53_OBSERVER_KDA must be auto or fla")
    if backend == "fla":
        # Import before the Transformers model: circular optional imports can
        # otherwise silently bind its reference implementation.
        importlib.import_module("fla.ops.kda")
    return backend


def selected_kda_identity(function, requested):
    seen, pending = set(), [function]
    while pending:
        current = pending.pop()
        if id(current) in seen or not inspect.isfunction(current):
            continue
        seen.add(id(current))
        values = inspect.getclosurevars(current).nonlocals
        implementation = values.get("implementation")
        if inspect.isfunction(implementation):
            module = implementation.__module__
            if requested == "fla" and not module.startswith("fla.ops.kda"):
                raise RuntimeError("requested FLA kernel silently fell back")
            version = None
            if module.startswith("fla.ops.kda"):
                from importlib.metadata import version as package_version
                version = package_version("fla-core")
            return {"module": module, "function": implementation.__qualname__,
                    "package_version": version}
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)
    raise RuntimeError("cannot establish the selected KDA implementation")
