"""
Module for building Python code programmatically using builder pattern.
"""

from dataclasses import dataclass, field
from typing import Tuple, List
from contextlib import contextmanager


@dataclass(frozen=True)
class PyCode:
    """Represents a piece of Python code"""

    lines: Tuple[str, ...]

    def __str__(self) -> str:
        """Return string-formatted code"""
        return "\n".join(self.lines)

    def __repr__(self):
        """Return repr for debugging"""
        return f"PyCode({len(self.lines)})"

    def __add__(self, other: "PyCode") -> "PyCode":
        """Combine two PyCode instances."""
        if not isinstance(other, PyCode):
            return NotImplemented
        return PyCode(self.lines + other.lines)

    def indent(self, level: int = 1) -> "PyCode":
        """Return indented version of code"""
        indent_str = "  " * level
        return PyCode(tuple(indent_str + line if line else line for line in self.lines))

    def to_string(self) -> str:
        """Convert to string"""
        return "\n".join(self.lines)

    @staticmethod
    def from_lines(*lines: str) -> "PyCode":
        """Create from individual lines"""
        return PyCode(tuple(lines))

    @staticmethod
    def empty() -> "PyCode":
        """Create empty code block"""
        return PyCode(())

    @staticmethod
    def blank_line() -> "PyCode":
        """Create blank line"""
        return PyCode(("",))


@dataclass
class PyCodeBuilder:
    _code: PyCode = field(default_factory=PyCode.empty)
    _indent_level: int = 0
    _imports: List[str] = field(default_factory=list)

    def add_import(self, module: str, items: List[str] = None) -> "PyCodeBuilder":
        """Add an import statement"""
        if items:
            import_stmt = f"from {module} import {', '.join(items)}"
        else:
            import_stmt = f"import {module}"
        if import_stmt not in self._imports:
            self._imports.append(import_stmt)
        return self

    def add_line(self, line: str = "") -> "PyCodeBuilder":
        """Add a single indented line and return self for chaining."""
        if line:
            self._code += PyCode.from_lines("    " * self._indent_level + line)
        else:
            self._code += PyCode.blank_line()
        return self

    def add_lines(self, *lines: str) -> "PyCodeBuilder":
        """Add multiple indented lines and return self for chaining."""
        for line in lines:
            self.add_line(line)
        return self

    def add_code(self, code: PyCode) -> "PyCodeBuilder":
        """Add pre-built code block with current indentation and return self for chaining."""
        self._code += code.indent(self._indent_level)
        return self

    def add_comment(self, text: str) -> "PyCodeBuilder":
        """Add a comment line."""
        return self.add_line(f"# {text}")

    def __iadd__(self, code: PyCode) -> "PyCodeBuilder":
        """Support += operator for adding PyCode directly."""
        return self.add_code(code)

    @contextmanager
    def indent(self):
        """Context manager for indented blocks."""
        self._indent_level += 1
        try:
            yield self
        finally:
            self._indent_level -= 1

    @contextmanager
    def function(self, name: str, params: List[str] = None):
        """Context manager for function definition."""
        params_str = ", ".join(params) if params else ""
        self.add_line(f"def {name}({params_str}):")
        with self.indent():
            yield self

    @contextmanager
    def for_loop(self, var: str, start: int, end: int):
        """Context manager for for loop."""
        self.add_line(f"for {var} in range({start}, {end} + 1):")
        with self.indent():
            yield self

    @contextmanager
    def if_block(self, condition: str):
        """Context manager for if statement."""
        self.add_line(f"if {condition}:")
        with self.indent():
            yield self

    def build(self) -> PyCode:
        """Get the final PyCode with imports at top."""
        if self._imports:
            import_code = PyCode.from_lines(*self._imports, "")
            return import_code + self._code
        return self._code
