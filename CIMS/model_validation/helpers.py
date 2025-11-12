from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Optional, Callable

@dataclass
class ValidationResult:
    errors: Dict[str, List[str]] = field(default_factory=dict)
    warnings: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(self.errors.values())

class ValidationError(Exception):
    pass

@dataclass(frozen=True)
class ValidatorConfig:
    sector_list: Optional[Iterable[str]] = None
    strict_schema: bool = True
    verbose: bool = True
    raise_on_error: bool = False

