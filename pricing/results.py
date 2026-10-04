from dataclasses import dataclass, field


@dataclass(frozen=True)
class PricingResult:
    method: str
    price: float
    delta: float
    gamma: float
    elapsed: float = 0.0
    price_stderr: float | None = None
    delta_stderr: float | None = None
    gamma_stderr: float | None = None
    extra: dict = field(default_factory=dict)
