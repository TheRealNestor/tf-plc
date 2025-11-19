"""
Helpers to represent and manipulate Structured Text code snippets.
"""


from dataclasses import dataclass, field
from typing import Tuple
from contextlib import contextmanager


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


@dataclass
class STCodeBuilder:
    """Helper for building ST code with automatic indentation tracking."""

    _code: STCode = field(default_factory=STCode.empty)
    _indent_level: int = 0

    def add_line(self, line: str = "") -> "STCodeBuilder":
        """Add a single indented line and return self for chaining."""
        if line:
            self._code += STCode.from_lines("    " * self._indent_level + line)
        else:
            self._code += STCode.blank_line()
        return self

    def add_lines(self, *lines: str) -> "STCodeBuilder":
        """Add multiple indented lines and return self for chaining."""
        for line in lines:
            self.add_line(line)
        return self

    def add_code(self, code: STCode) -> "STCodeBuilder":
        """Add pre-built code block with current indentation and return self for chaining."""
        self._code += code.indent(self._indent_level)
        return self

    def __iadd__(self, code: STCode) -> "STCodeBuilder":
        """Support += operator for adding STCode directly."""
        return self.add_code(code)

    @contextmanager
    def indent(self):
        """Context manager for indented blocks."""
        self._indent_level += 1
        try:
            yield self
        finally:
            self._indent_level -= 1

    def build(self) -> STCode:
        """Get the final STCode."""
        return self._code
