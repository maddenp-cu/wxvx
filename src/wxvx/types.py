from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import cached_property
from pathlib import Path
from typing import Any, cast

from uwtools.api.config import get_yaml_config, validate

from wxvx.util import LINETYPE, WXVXError, expand, resource_path, to_datetime, to_timedelta

_DatetimeT = str | datetime
_TimedeltaT = str | int

Source = Enum(
    "Source",
    [
        ("BASELINE", auto()),
        ("FORECAST", auto()),
    ],
)

ToGridVal = Enum(
    "ToGridVal",
    [
        ("FCST", auto()),
        ("OBS", auto()),
    ],
)

VxType = Enum(
    "VxType",
    [
        ("GRID", auto()),
        ("POINT", auto()),
    ],
)


def validated_config(config_path: Path) -> Config:
    yc = get_yaml_config(config_path)
    yc.dereference()
    if not validate(schema_file=resource_path("config.jsonschema"), config_data=yc.data):
        msg = "Config failed schema validation"
        raise WXVXError(msg)
    return Config(yc.data)


# Below, assert statements relate to config requirements that should have been enforced by a prior
# schema check. If an assertion is triggered, it's a wxvx bug, not a user issue.


@dataclass(frozen=True)
class Baseline:
    compare: bool
    name: str
    url: str
    type: VxType

    def __post_init__(self):
        keys = ["grid", "point"]
        assert self.type in keys
        newval = dict(zip(keys, [VxType.GRID, VxType.POINT]))
        _force(self, "type", newval.get(str(self.type), self.type))


class Config:
    def __init__(self, raw: dict):
        paths = raw["paths"]
        grids = paths["grids"]
        self.baseline = Baseline(**raw["baseline"])
        self.cycles = Cycles(raw["cycles"])
        self.forecast = Forecast(**raw["forecast"])
        self.leadtimes = Leadtimes(raw["leadtimes"])
        self.paths = Paths(
            grids.get("baseline"), grids.get("forecast"), paths.get("obs"), paths["run"]
        )
        self.raw = raw
        self.regrid = Regrid(**raw.get("regrid", {}))
        self.variables = raw["variables"]
        self._validate()

    KEYS = ("baseline", "cycles", "forecast", "leadtimes", "paths", "variables")

    def __eq__(self, other):
        return all(getattr(self, k) == getattr(other, k) for k in self.KEYS)

    def __hash__(self):
        return _hash(self)

    def __repr__(self):
        parts = ["%s=%s" % (x, getattr(self, x)) for x in self.KEYS]
        return "%s(%s)" % (self.__class__.__name__, ", ".join(parts))

    def _validate(self) -> None:
        # Validation tests involving disparate config subtrees, which are awkward to define in JSON
        # Schema (or that give poor user feedback when so defined) are performed here.
        if self.baseline.type == VxType.GRID and not self.paths.grids_baseline:
            msg = "Specify path.grids.baseline when baseline.type is 'grid'"
            raise WXVXError(msg)
        if self.baseline.type == VxType.POINT and not self.paths.obs:
            msg = "Specify path.obs when baseline.type is 'point'"
            raise WXVXError(msg)
        if self.regrid.to == "OBS":
            msg = "Cannot regrid to observations per regrid.to config value"
            raise WXVXError(msg)


@dataclass(frozen=True)
class Coords:
    latitude: str
    level: str
    longitude: str
    time: Time

    KEYS = ("latitude", "level", "longitude", "time")

    def __hash__(self):
        return _hash(self)

    def __post_init__(self):
        if isinstance(self.time, dict):
            _force(self, "time", Time(**self.time))


class Cycles:
    def __init__(self, raw: dict[str, str | int | datetime] | list[str | datetime]):
        self.raw = raw

    def __eq__(self, other):
        return self.values == other.values

    def __hash__(self):
        return hash(tuple(self.values))

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.raw)

    @cached_property
    def values(self) -> list[datetime]:
        if isinstance(self.raw, dict):
            dt_start, dt_stop = [
                to_datetime(cast(_DatetimeT, self.raw[x])) for x in ("start", "stop")
            ]
            td_step = to_timedelta(cast(_TimedeltaT, self.raw["step"]))
            return expand(dt_start, td_step, dt_stop)
        return sorted(map(to_datetime, self.raw))


class Forecast:
    KEYS = (
        "coords",
        "mask",
        "name",
        "path",
        "_projection",  # use '_projection' (not 'projection') to avoid triggering the property.
    )

    def __init__(
        self,
        name: str,
        path: str,
        coords: Coords | dict | None = None,
        mask: list[list[float]] | None = None,
        projection: dict | None = None,
    ):
        self._name = name
        self._path = path
        self._coords = coords
        self._mask = mask
        self._projection = projection

    def __eq__(self, other):
        return all(getattr(self, k) == getattr(other, k) for k in self.KEYS)

    def __hash__(self):
        return _hash(self)

    def __repr__(self):
        parts = ["%s=%s" % (x, getattr(self, x)) for x in self.KEYS]
        return "%s(%s)" % (self.__class__.__name__, ", ".join(parts))

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def coords(self) -> Coords | None:
        if isinstance(self._coords, dict):
            self._coords = Coords(**self._coords)
        return self._coords

    @property
    def mask(self) -> list[list[float]] | None:
        return self._mask

    @property
    def projection(self) -> dict:
        if self._projection is None:
            logging.info("No forecast projection specified, defaulting to latlon")
            self._projection = {"proj": "latlon"}
        return self._projection


class Leadtimes:
    def __init__(self, raw: dict[str, str | int] | list[str | int]):
        self.raw = raw

    def __eq__(self, other):
        return self.values == other.values

    def __hash__(self):
        return hash(tuple(self.values))

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.raw)

    @cached_property
    def values(self) -> list[timedelta]:
        if isinstance(self.raw, dict):
            td_start, td_step, td_stop = [
                to_timedelta(cast(_TimedeltaT, self.raw[x])) for x in ("start", "step", "stop")
            ]
            return expand(td_start, td_step, td_stop)
        return sorted(map(to_timedelta, self.raw))


@dataclass(frozen=True)
class Paths:
    grids_baseline: Path
    grids_forecast: Path
    obs: Path
    run: Path

    def __post_init__(self):
        for key in ["grids_baseline", "grids_forecast", "obs", "run"]:
            if val := getattr(self, key):
                _force(self, key, Path(val))


@dataclass(frozen=True)
class Regrid:
    # See https://metplus.readthedocs.io/projects/met/en/main_v11.0/Users_Guide/appendixB.html#grids
    # for information on the "GNNN" grid names accepted as regrid-to values.

    method: str = "NEAREST"
    to: ToGrid | None = None

    def __post_init__(self):
        _force(self, "to", ToGrid("forecast" if self.to is None else str(self.to)))
        assert self.to is not None


@dataclass(frozen=True)
class Time:
    inittime: str
    leadtime: str | None = None
    validtime: str | None = None

    def __post_init__(self):
        assert self.leadtime is not None or self.validtime is not None


class ToGrid:
    def __init__(self, val: str):
        self.val: str | ToGridVal = val
        mapping = {"baseline": ToGridVal.OBS, "forecast": ToGridVal.FCST}
        self.val = mapping.get(val, val)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        if isinstance(self.val, ToGridVal):
            return self.val.name
        return self.val


@dataclass(frozen=True)
class VarMeta:
    cf_standard_name: str
    description: str
    level_type: str
    met_stats: list[str]
    name: str
    units: str
    # Optional:
    cat_thresh: list[str] | None = None
    cnt_thresh: list[str] | None = None
    nbrhd_shape: str | None = None
    nbrhd_width: list[int] | None = None

    def __post_init__(self):
        for k, v in vars(self).items():
            match k:
                case "cat_thresh":
                    assert v is None or (v and all(isinstance(x, str) for x in v))
                case "cf_standard_name":
                    assert v
                case "cnt_thresh":
                    assert v is None or (v and all(isinstance(x, str) for x in v))
                case "description":
                    assert v
                case "level_type":
                    assert v in ("atmosphere", "heightAboveGround", "isobaricInhPa", "surface")
                case "met_stats":
                    assert v
                    assert all(x in LINETYPE for x in v)
                case "name":
                    assert v
                case "nbrhd_shape":
                    assert v is None or v in ("CIRCLE", "SQUARE")
                case "nbrhd_width":
                    assert v is None or (v and all(isinstance(x, int) for x in v))
                case "units":
                    assert v


# Helpers


def _force(obj: Any, name: str, val: Any) -> None:
    object.__setattr__(obj, name, val)


def _hash(obj: Any) -> int:
    h = None
    for k in obj.KEYS:
        x = getattr(obj, k)
        try:
            h = hash((h, hash(x)))
        except TypeError:
            h = hash((h, json.dumps(x, sort_keys=True)))
    assert h is not None
    return h
