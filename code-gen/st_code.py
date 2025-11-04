from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class STCode:
    """Represents a piece of Structured Text code"""

    lines: Tuple[str, ...]

    def __add__(self, other: "STCode") -> "STCode":
        """Combine two code blocks"""
        return STCode(self.lines + other.lines)

    def indent(self, level: int = 1) -> "STCode":
        """Return indented version of code"""
        indent_str = "    " * level
        return STCode(tuple(indent_str + line if line else line for line in self.lines))

    def to_string(self) -> str:
        """Convert to string"""
        return "\n".join(self.lines)

    @staticmethod
    def from_lines(*lines: str) -> "STCode":
        """Create from individual lines"""
        return STCode(tuple(lines))

    @staticmethod
    def empty() -> "STCode":
        """Create empty code block"""
        return STCode(())

    @staticmethod
    def blank_line() -> "STCode":
        """Create blank line"""
        return STCode(("",))
