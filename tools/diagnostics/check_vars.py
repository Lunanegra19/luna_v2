import ast
import sys
import builtins

class VariableChecker(ast.NodeVisitor):
    def __init__(self):
        self.scopes = [set(dir(builtins))]
        self.errors = []
        self.imported = set()

    def visit_Import(self, node):
        for alias in node.names:
            self.scopes[-1].add(alias.asname or alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.scopes[-1].add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.scopes[-1].add(node.name) # Function name is in outer scope
        
        inner_scope = set()
        for arg in node.args.args + getattr(node.args, 'kwonlyargs', []):
            inner_scope.add(arg.arg)
        if node.args.vararg:
            inner_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            inner_scope.add(node.args.kwarg.arg)
            
        self.scopes.append(inner_scope)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_ClassDef(self, node):
        self.scopes[-1].add(node.name)
        self.scopes.append(set())
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Assign(self, node):
        self.generic_visit(node) # visit right side first
        for t in node.targets:
            self._add_to_scope(t)

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        self._add_to_scope(node.target)

    def visit_For(self, node):
        self.generic_visit(node.iter)
        self._add_to_scope(node.target)
        for stmt in node.body + node.orelse:
            self.visit(stmt)

    def visit_comprehension(self, node):
        self._add_to_scope(node.target)
        self.generic_visit(node)

    def visit_withitem(self, node):
        self.generic_visit(node.context_expr)
        if node.optional_vars:
            self._add_to_scope(node.optional_vars)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.scopes[-1].add(node.name)
        self.generic_visit(node)

    def _add_to_scope(self, target):
        if isinstance(target, ast.Name):
            self.scopes[-1].add(target.id)
        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
            for elt in target.elts:
                self._add_to_scope(elt)

    def visit_Name(self, node):
        # Ignore common implicit globals that might be imported inside functions or conditionally
        ignores = ['pd', 'np', 'logger', 'cfg', 'Path', 'json', 'sys', 'time', 'os', 'math']
        if isinstance(node.ctx, ast.Load) and node.id not in ignores:
            if not any(node.id in s for s in self.scopes):
                self.errors.append((node.lineno, node.id))
        self.generic_visit(node)

for f in sys.argv[1:]:
    try:
        code = open(f, encoding='utf-8').read()
        tree = ast.parse(code)
        checker = VariableChecker()
        checker.visit(tree)
        
        # Deduplicate errors
        errs = list(set(checker.errors))
        errs.sort(key=lambda x: x[0])
        
        for line, var in errs:
            print(f"{f}:{line} Potential NameError: {var}")
    except Exception as e:
        print(f"{f}: Error parsing: {e}")
