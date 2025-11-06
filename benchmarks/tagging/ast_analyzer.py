"""
Automatic tagging system for TritonBench.
Traces code on AST level.
Assume all dependent sources are import-able but do not require them to run.
Assume kernels are tracable through function calls (no class methods involved).
"""

import ast
import importlib
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass
from os.path import abspath, exists
from typing import Any, Dict, List, Optional, Optional, Set, Tuple


def setup_tritonbench_cwd():
    original_dir = abspath(os.getcwd())

    for tritonbench_dir in (
        ".",
        "../../../tritonbench",
    ):
        if exists(tritonbench_dir):
            break

    if exists(tritonbench_dir):
        tritonbench_dir = abspath(tritonbench_dir)
        os.chdir(tritonbench_dir)
        sys.path.append(tritonbench_dir)
    return original_dir


setup_tritonbench_cwd()

from tritonbench.utils.env_utils import is_fbcode


@dataclass(frozen=True)
class Site:
    filename: str
    lineno: int
    col: int


@dataclass
class FuncDescriptor:
    name: str
    decorators: List[str]
    site: Site


@dataclass
class Edge:
    caller: str
    callee: str
    site: Site
    call_type: str
    callee_descriptor: FuncDescriptor


class CallGraph(ast.NodeVisitor):
    def __init__(
        self,
        filename: str = "<string>",
        module_name: str = "<module>",
        include_decorators: bool = False,
        backends: Optional[List[str]] = None,
    ):
        self.filename = filename
        self.include_decorators = include_decorators

        self.edges: List[Edge] = []
        self.decorator_edges: List[Edge] = []
        assert backends, "Backends must nto be none"
        self.backends: Dict[str, List[Any]] = {}
        for backend in backends:
            self.backends[backend] = []

        self.scope_stack: List[str] = []
        self.module_name = module_name

        self.bindings_stack: List[Dict[str, str]] = [dict()]
        self.local_functions: Set[str] = set()

        # lambda node -> synthetic id (stable within this pass)
        self._lambda_ids: Dict[ast.Lambda, str] = {}

    # ---------- helpers ----------
    def _cur_scope(self) -> str:
        return ".".join([self.module_name] + self.scope_stack).strip(".")

    def _push_scope(self, name: str):
        self.scope_stack.append(name)
        self.bindings_stack.append(dict())

    def _pop_scope(self):
        self.scope_stack.pop()
        self.bindings_stack.pop()

    def _bind(self, name: str, target: str):
        self.bindings_stack[-1][name] = target

    def _bind_func_descriptor(self, node, decorators: List[str]):
        name = node.name
        site = Site(
            self.filename, getattr(node, "lineno", -1), getattr(node, "col_offset", -1)
        )
        self.bindings_stack[-1][f"__{name}_descriptor__"] = FuncDescriptor(
            name, decorators, site
        )

    def _resolve_name(self, id_: str) -> str:
        for env in reversed(self.bindings_stack):
            if id_ in env:
                return env[id_]
        return id_

    def _resolve_func_descriptor(self, id_: str) -> List[str]:
        for env in reversed(self.bindings_stack):
            decorator_constant = f"__{id_}_descriptor__"
            if decorator_constant in env:
                return env[decorator_constant]
        return None

    def _resolve_attr(self, node: ast.AST) -> str:
        parts: List[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            head = self._resolve_name(cur.id)
        else:
            return "<dynamic_attr>"
        return ".".join([head] + list(reversed(parts)))

    def _lambda_id(self, node: ast.Lambda) -> str:
        lid = self._lambda_ids.get(node)
        if lid is None:
            scope = self._cur_scope() or "<module>"
            lid = f"{scope}.<lambda>@{getattr(node,'lineno',-1)}:{getattr(node,'col_offset',-1)}"
            self._lambda_ids[node] = lid
        return lid

    def _record_assignment(self, target, rhs, node: ast.AST):
        site = Site(
            self.filename, getattr(node, "lineno", -1), getattr(node, "col_offset", -1)
        )
        self.edges.append(
            Edge(target, rhs, site=site, callee_descriptor=None, call_type="assignment")
        )

    def _record_call(
        self, callee: str, node: ast.AST, maybe_triton: bool = False, caller=None
    ):
        if caller is None:
            caller = self._cur_scope() or "<module>"
        site = Site(
            self.filename, getattr(node, "lineno", -1), getattr(node, "col_offset", -1)
        )
        # trace the backend call in tritonbench
        if (
            "tritonbench.operators." in caller
            and any([f"Operator.{backend}" in caller for backend in self.backends])
            and not callee == "tritonbench.utils.triton_op.register_benchmark"
        ):
            if callee.startswith("self."):
                return
            if is_fbcode() and callee.startswith("liger_kernel."):
                return
            # identify this call belongs to which backend
            for backend in self.backends:
                if (
                    caller.endswith(f"Operator.{backend}")
                    or f"Operator.{backend}." in caller
                ):
                    self.backends[backend].append(self._resolve_name(callee))
            self.edges.append(
                Edge(
                    caller,
                    callee,
                    callee_descriptor=self._resolve_func_descriptor(callee),
                    site=site,
                    call_type="regular",
                )
            )
        elif any([backend in caller for backend in self.backends]):
            # "torch.ops" is a binary custom ops
            if callee.startswith("self."):
                return
            if callee.startswith("torch.") and not callee.startswith("torch.ops."):
                return
            # we are sure there is no kernel defined in this package ;-)
            if callee.startswith("tritonbench.utils."):
                return
            callee_descriptor = self._resolve_func_descriptor(callee)
            # heuristic that this maybe a triton kernel call
            # TODO: this is not a good heuristic, but we couldn't find a better way to identify triton kernel calls
            if maybe_triton and callee_descriptor == None:
                callee_descriptor = FuncDescriptor(
                    callee,
                    ["triton.jit"],
                    Site(
                        self.filename,
                        getattr(node, "lineno", -1),
                        getattr(node, "col_offset", -1),
                    ),
                )
            self.edges.append(
                Edge(
                    caller,
                    callee,
                    callee_descriptor=callee_descriptor,
                    site=site,
                    call_type="regular",
                )
            )

    def _record_decorator_edge(self, callee: str, node: ast.AST):
        caller = self._cur_scope() or "<module>"
        site = Site(
            self.filename, getattr(node, "lineno", -1), getattr(node, "col_offset", -1)
        )
        self.decorator_edges.append(
            Edge(
                caller, callee, callee_descriptor=None, site=site, call_type="decorator"
            )
        )

    # ---------- imports / aliases ----------
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self._bind(name, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        for alias in node.names:
            local = alias.asname or alias.name
            target = f"{mod}.{alias.name}" if mod else alias.name
            self._bind(local, target)
        self.generic_visit(node)

    # ---------- defs ----------
    def visit_FunctionDef(self, node: ast.FunctionDef):
        return self._visit_function_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self._visit_function_like(node)

    def _visit_function_like(self, node):
        qual = (self._cur_scope() + "." if self._cur_scope() else "") + node.name
        self._bind(node.name, qual)
        self.local_functions.add(qual)

        decorators = []
        if node.decorator_list:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    callee = self._resolve_name(dec.id)
                elif isinstance(dec, ast.Attribute):
                    callee = self._resolve_attr(dec)
                elif isinstance(dec, ast.Call):
                    # best effort to guess the structure of the decorator
                    if isinstance(dec.func, ast.Name):
                        callee = f"<dynamic_decorator_{dec.func.id}>"
                    elif isinstance(dec.func, ast.Attribute):
                        if isinstance(dec.func.value, ast.Name):
                            callee = f"<dynamic_decorator_{dec.func.value.id}.{dec.func.attr}>"
                        elif isinstance(dec.func.value, ast.Attribute):
                            callee = f"<dynamic_decorator_{dec.func.value.value.id}.{dec.func.attr}>"
                        else:
                            callee = f"<dynamic_decorator>"
                    else:
                        callee = "<dynamic_decorator>"
                else:
                    callee = "<dynamic_decorator>"
                decorators.append(callee)

        self._bind_func_descriptor(node, decorators)

        self._push_scope(node.name)
        if node.name in self.backends:
            self._record_call(node.name, node, maybe_triton=False, caller=node.name)
        self.generic_visit(node)
        self._pop_scope()

    def visit_ClassDef(self, node: ast.ClassDef):
        qual = (self._cur_scope() + "." if self._cur_scope() else "") + node.name
        self._bind(node.name, qual)
        self._push_scope(node.name)
        self.generic_visit(node)
        self._pop_scope()

    # ---------- lambda support ----------
    def visit_Lambda(self, node: ast.Lambda):
        """
        Give each lambda a synthetic qualified name and traverse its body in that scope
        so we can record calls made inside lambda bodies.
        """
        lid = self._lambda_id(node)
        # Enter a readable, stable scope name
        scope_name = lid.split(".")[-1]  # "<lambda>@line:col"
        self._push_scope(scope_name)
        # The lambda body is a single expression; visit it so nested Calls are captured
        self.visit(node.body)
        self._pop_scope()
        # Do not call generic_visit (we already visited body)

    def visit_Assign(self, node: ast.Assign):
        def rhs_symbol(n: ast.AST) -> Optional[str]:
            if isinstance(n, ast.Name):
                return self._resolve_name(n.id)
            if isinstance(n, ast.Attribute):
                return self._resolve_attr(n)
            if isinstance(n, ast.Lambda):
                return self._lambda_id(n)
            return None

        sym = rhs_symbol(node.value)
        if sym:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self._bind(t.id, sym)
                    if any([t.id == backend for backend in self.backends.keys()]):
                        print(
                            f"recording assignment for backend: {self.backends}, tid: {t.id}"
                        )
                        self._record_assignment(t.id, sym, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # a: T = lambda ...
        if node.value is not None:
            if isinstance(node.target, ast.Name):
                if isinstance(node.value, ast.Lambda):
                    self._bind(node.target.id, self._lambda_id(node.value))
                elif isinstance(node.value, ast.Name):
                    self._bind(node.target.id, self._resolve_name(node.value.id))
                elif isinstance(node.value, ast.Attribute):
                    self._bind(node.target.id, self._resolve_attr(node.value))
        self.generic_visit(node)

    # ---------- call sites ----------
    def visit_Call(self, node: ast.Call):
        fn = node.func
        maybe_triton = False
        if isinstance(fn, ast.Name):
            callee = self._resolve_name(fn.id)
        elif isinstance(fn, ast.Attribute):
            callee = self._resolve_attr(fn)
        elif isinstance(fn, ast.Lambda):
            callee = self._lambda_id(fn)  # inline IIFE-style lambda
        elif isinstance(
            fn, ast.Subscript
        ):  # highly likely to be triton - met a function call with subscript
            if isinstance(fn.value, ast.Name):
                callee = fn.value.id
            elif isinstance(fn.value, ast.Attribute):
                callee = fn.value.value.id
            else:
                callee = "<dynamic_call>"
            maybe_triton = True  # FIXME: this could also be cute, see blackwell_attentions cute dsl
        else:
            callee = "<dynamic_call>"

        self._record_call(callee, node, maybe_triton=maybe_triton)
        self.generic_visit(node)


def validate_edges(edges) -> Dict[str, str]:
    result_tags = {}
    result_tags["tags"] = []
    result_tags["kernels"] = []
    for edge in edges:
        if edge.callee_descriptor and (
            "triton.jit" in edge.callee_descriptor.decorators
            or "<dynamic_decorator_triton.jit>" in edge.callee_descriptor.decorators
        ):
            result_tags["tags"].append("triton")
            result_tags["kernels"].append(edge.callee)
        if edge.callee == "cutlass.cute.compile":
            result_tags["tags"].append("cutedsl")
            result_tags["kernels"].append(edge.caller)
        if edge.callee.startswith("torch.ops."):
            result_tags["tags"].append("native_custom_ops")
            # definition is in cpp, so we don't have the definition site
            result_tags["kernels"].append(edge.callee)
        if edge.callee.startswith("triton.experimental.gluon"):
            result_tags["tags"].append("gluon")
        if edge.callee.startswith("torch.nn."):
            result_tags["tags"].append("aten")
            result_tags["kernels"].append(edge.callee)
        if edge.callee.startswith("tilelang.compile"):
            result_tags["tags"].append("tilelang")
            result_tags["kernels"].append(edge.caller)
    # remove duplicates
    result_tags["tags"] = list(set(result_tags["tags"]))
    if not result_tags["kernels"] and not result_tags["tags"]:
        return None
    return result_tags


def gen_static_extension_tags(callee: str) -> Dict[str, str]:
    result_tags = {}
    result_tags["tags"] = {"native_extension"}
    result_tags["kernels"] = [callee]
    return result_tags


def trace_callees(callees_with_module: List[Tuple[str, str]], depth=8):
    """Bread-first search, maximum depth 10"""
    queue = [
        {"callee": callee[0], "module": callee[1], "depth": 1}
        for callee in callees_with_module
    ]
    seen = set()
    while len(queue):
        cur = queue.pop(0)
        if cur["depth"] > depth:
            break
        callee = cur["callee"]
        module_name = cur["module"]
        if (callee, module_name) in seen:
            continue
        else:
            seen.add((callee, module_name))
        # hack: change .apply to .forward function for Autograd
        if callee.endswith(".apply"):
            callee = callee[: callee.rfind(".apply")] + ".forward"

        callee_module = callee[: callee.rfind(".")] if "." in callee else None
        callee_name = callee[callee.rfind(".") + 1 :] if "." in callee else callee
        maybe_callee_module = (
            callee_module[: callee_module.rfind(".")]
            if callee_module and "." in callee_module
            else None
        )
        maybe_callee_class = (
            callee_module[callee_module.rfind(".") + 1 :]
            if callee_module and "." in callee_module
            else None
        )
        # best effort to find and import the module
        # print(f"callee: {callee}")
        # print(f"callee module: {callee_module}")
        # print(f"callee name: {callee_name}")
        # print(f"module name: {module_name}")
        # print(f"maybe callee module: {maybe_callee_module}")
        # print(f"maybe callee class: {maybe_callee_class}")
        if not callee_module and not maybe_callee_module:
            continue
        try:
            module = importlib.import_module(callee_module)
            source_file = inspect.getfile(module)
        except (ModuleNotFoundError, TypeError):
            try:
                # try with relative import
                module = importlib.import_module(f"{module_name}.{callee_module}")
                source_file = inspect.getfile(module)
            except (ModuleNotFoundError, TypeError):
                if maybe_callee_module == None:
                    continue
                try:
                    module = importlib.import_module(maybe_callee_module)
                    source_file = inspect.getfile(module)
                    callee_name = f"{maybe_callee_class}.{callee_name}"
                except (ModuleNotFoundError, TypeError):
                    try:
                        module = importlib.import_module(
                            f"{module_name}.{maybe_callee_module}"
                        )
                        source_file = inspect.getfile(module)
                        callee_name = f"{maybe_callee_class}.{callee_name}"
                    except Exception:
                        # give up
                        print(
                            f"Failed to load module {maybe_callee_module} from entity {callee}"
                        )
                        continue
        if not module:
            print(f"Failed to find {callee} at module {callee_module} ")
            continue
        print(
            f"Found entity {callee} at module {module.__name__}. Searching callee {callee_name}"
        )
        if source_file == "static-extension":
            return gen_static_extension_tags(callee)
        if source_file.endswith(".so"):
            continue
        real_module_name = module.__name__
        print(f"=============== SEARCHING FILE {source_file} ===============")
        with open(source_file, "r") as fp:
            source = fp.read()
        tree = ast.parse(source, filename=source_file, mode="exec")
        cg = CallGraph(
            filename=os.path.basename(source_file),
            module_name=real_module_name,
            include_decorators=True,
            backends=[callee_name],
        )
        cg.visit(tree)
        # get next level callees
        print(cg.edges)
        # validate the edges: if any of them uses triton, apply triton tag and stop searching
        tags = validate_edges(cg.edges)
        if tags:
            return tags
        next_level_callees = [edge.callee for edge in cg.edges]
        for next_level_callee in next_level_callees:
            queue.append(
                {
                    "callee": next_level_callee,
                    "module": real_module_name,
                    "depth": cur["depth"] + 1,
                }
            )
    # No valid tags are found, return None
    return None


def build_backend_callees(
    source: str,
    filename: str = "<string>",
    module_name: str = "",
    backends: Optional[List[str]] = None,
) -> Tuple[Dict[str, Set[str]], List[Edge]]:
    tree = ast.parse(source, filename=filename, mode="exec")
    cg = CallGraph(
        filename=filename,
        module_name=module_name,
        include_decorators=True,
        backends=backends,
    )
    cg.visit(tree)
    print(cg.backends)
    return cg.backends
