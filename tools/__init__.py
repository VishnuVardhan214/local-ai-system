"""Importing this package registers every tool. To add a tool, create a module here,
decorate a function with @tool(...), and add the module name to the import line below."""
from tools.base import REGISTRY, ToolContext, ToolError, tool  # noqa: F401
from tools import fs, personal, rag_tools  # noqa: F401
