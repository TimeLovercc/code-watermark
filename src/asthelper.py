import ipdb
import ast
import random
import string
import copy
import re
import logging
from typing import List, Optional, Union, Dict

logger = logging.getLogger(__name__)


"""
Supported code transformations:

1. Variable renaming
   - Randomly renames variables while preserving program semantics
   - Maintains consistent renaming within scope

2. Loop structure transformations  
   - Converts while loops to for loops with iterators
   - Transforms for loops to while loops with counters
   - Unrolls simple loops into repeated statements

3. Expression rewrites
   - Simplifies arithmetic expressions
   - Breaks down complex expressions into simpler ones
   - Combines simple expressions into compound ones
   - Reorders commutative operations

4. Boolean logic transformations
   - Applies De Morgan's laws
   - Converts AND to OR with negation
   - Converts OR to AND with negation 
   - Simplifies boolean expressions

5. Comprehension transformations
   - Converts list comprehensions to map/filter
   - Transforms dict/set comprehensions to loops
   - Converts generator expressions to loops
   - Combines multiple comprehensions

6. Unary operation transformations
   - Simplifies double negations
   - Converts comparison operators to opposites
   - Transforms is/is not operations
   - Converts in/not in operations
"""
def fix_missing_locations(node):
    """Recursively sets missing line numbers and column offsets in AST nodes"""
    def _fix(node, lineno, col_offset):
        if 'lineno' not in node._attributes:
            node._attributes = node._attributes + ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')
        
        if not hasattr(node, 'lineno'):
            node.lineno = lineno
        if not hasattr(node, 'col_offset'):
            node.col_offset = col_offset
        if not hasattr(node, 'end_lineno'):
            node.end_lineno = lineno
        if not hasattr(node, 'end_col_offset'):
            node.end_col_offset = col_offset + 1

        for child in ast.iter_child_nodes(node):
            _fix(child, node.lineno, node.col_offset)
        return node

    return _fix(node, 1, 0)

class BaseTransformer(ast.NodeTransformer):
    """Base class for all transformers to handle location fixing"""
    def generic_visit(self, node):
        result = super().generic_visit(node)
        if isinstance(result, (ast.AST, list)):
            if isinstance(result, list):
                for n in result:
                    if isinstance(n, ast.AST):
                        fix_missing_locations(n)
            else:
                fix_missing_locations(result)
        return result

class VariableRenamer(BaseTransformer):
    """AST transformer for renaming variables with different naming styles"""
    def __init__(self):
        self.variable_map = {}
        self.counter = 0
        # Common prefixes for different variable types
        self.prefixes = {
            'i': ['index', 'idx', 'iter', 'count'],  # Loop variables
            'n': ['num', 'size', 'length', 'total'],  # Numeric counts
            'str': ['text', 'name', 'label', 'key'],  # String variables
            'lst': ['list', 'array', 'items', 'elements'],  # Lists/Collections
            'dict': ['map', 'dict', 'table', 'cache'],  # Dictionaries
            'tmp': ['temp', 'tmp', 'aux', 'buf'],  # Temporary variables
            'res': ['result', 'output', 'value', 'ret']  # Results
        }
        self.naming_styles = [
            'camelCase',     # e.g., myVariable 
            'PascalCase',    # e.g., MyVariable
            'snake_case',    # e.g., my_variable
            'underscore_init'  # e.g., _myVariable
        ]
        
    def _convert_to_case(self, name: str, style: str) -> str:
        """Convert a name to the specified case style"""
        words = name.replace('_', ' ').split()
        
        if style == 'camelCase':
            return words[0].lower() + ''.join(w.capitalize() for w in words[1:])
            
        elif style == 'PascalCase':
            return ''.join(w.capitalize() for w in words)
            
        elif style == 'snake_case':
            return '_'.join(w.lower() for w in words)
            
        elif style == 'underscore_init':
            camel_case = words[0].lower() + ''.join(w.capitalize() for w in words[1:])
            return f"_{camel_case}"
            
        return name

    def _infer_variable_type(self, node) -> str:
        """Try to infer the variable type from context"""
        try:
            if isinstance(node.parent, ast.For):
                return 'i'  # Loop variable
            elif isinstance(node.parent, ast.Dict):
                return 'dict'
            elif isinstance(node.parent, ast.List):
                return 'lst'
            elif isinstance(node.parent, ast.Assign):
                # Try to infer from the value being assigned
                if isinstance(node.parent.value, ast.Num):
                    return 'n'
                elif isinstance(node.parent.value, ast.Str):
                    return 'str'
                elif isinstance(node.parent.value, (ast.List, ast.ListComp)):
                    return 'lst'
                elif isinstance(node.parent.value, (ast.Dict, ast.DictComp)):
                    return 'dict'
            elif isinstance(node.parent, ast.FunctionDef):
                return 'res'  # Function parameters often lead to results
        except AttributeError:
            pass
        return 'tmp'  # Default to temporary variable

    def _generate_new_name(self, old_name: str = None, var_type: str = 'tmp') -> str:
        """Generate a meaningful new variable name with varying styles"""
        # Try to preserve some meaning from the original name
        if old_name:
            # Keep same prefix if it's a common one
            for prefix, options in self.prefixes.items():
                if any(old_name.lower().startswith(opt.lower()) for opt in options):
                    var_type = prefix
                    break
        
        # Get list of possible name bases for this type
        base_names = self.prefixes.get(var_type, self.prefixes['tmp'])
        base_name = random.choice(base_names)
        
        # Add a numeric suffix if needed
        self.counter += 1
        if self.counter > 1:
            base_name = f"{base_name}{self.counter}"
        
        # Randomly choose a naming style
        style = random.choice(self.naming_styles)
        return self._convert_to_case(base_name, style)

    def visit_Name(self, node):
        """Visit and potentially rename a variable"""
        if isinstance(node.ctx, ast.Store):
            if node.id not in self.variable_map:
                var_type = self._infer_variable_type(node)
                self.variable_map[node.id] = self._generate_new_name(node.id, var_type)
        if node.id in self.variable_map:
            node.id = self.variable_map[node.id]
        return node
    
class UnaryTransformer(ast.NodeTransformer):
    """Transform unary operations and negations"""
    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and random.random() < 0.5:
            if isinstance(node.operand, ast.Compare):
                return self._transform_comparison(node.operand)
        return node
    
    def _transform_comparison(self, node):
        """Transform comparison to its logical opposite"""
        op_map = {
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
            ast.Lt: ast.GtE,
            ast.LtE: ast.Gt,
            ast.Gt: ast.LtE,
            ast.GtE: ast.Lt,
            ast.Is: ast.IsNot,
            ast.IsNot: ast.Is,
            ast.In: ast.NotIn,
            ast.NotIn: ast.In
        }
        if len(node.ops) == 1 and isinstance(node.ops[0].__class__, type):
            op_type = type(node.ops[0])
            if op_type in op_map:
                return ast.Compare(
                    left=node.left,
                    ops=[op_map[op_type]()],
                    comparators=node.comparators
                )
        return node

class LoopTransformer(BaseTransformer):
    """Transform loop structures"""
    def visit_For(self, node):
        self.generic_visit(node)
        if random.random() < 0.3:
            # Transform for loop to while loop
            target = node.target
            iter_var = self._generate_temp_var()
            iter_setup = ast.Assign(
                targets=[ast.Name(id=iter_var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='iter', ctx=ast.Load()),
                    args=[node.iter],
                    keywords=[]
                )
            )
            
            while_test = ast.Compare(
                left=ast.Constant(value=True),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=True)]
            )
            
            try_block = ast.Try(
                body=[
                    ast.Assign(
                        targets=[target],
                        value=ast.Call(
                            func=ast.Name(id='next', ctx=ast.Load()),
                            args=[ast.Name(id=iter_var, ctx=ast.Load())],
                            keywords=[]
                        )
                    )
                ] + node.body,
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id='StopIteration', ctx=ast.Load()),
                        body=[ast.Break()]
                    )
                ],
                orelse=[],
                finalbody=[]
            )
            
            return [
                iter_setup,
                ast.While(
                    test=while_test,
                    body=[try_block],
                    orelse=node.orelse
                )
            ]
        return node
    
    def _generate_temp_var(self):
        return f"_iter_{random.randint(1000, 9999)}"

class ExpressionTransformer(BaseTransformer):
    """Transform expressions and operations"""
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if random.random() < 0.4:
            if isinstance(node.op, ast.Add):
                # Transform a + b to -(-a - b)
                return ast.UnaryOp(
                    op=ast.USub(),
                    operand=ast.BinOp(
                        left=ast.UnaryOp(op=ast.USub(), operand=node.left),
                        op=ast.Sub(),
                        right=node.right
                    )
                )
            elif isinstance(node.op, ast.Mult):
                # Transform a * b to -(-a * b)
                return ast.UnaryOp(
                    op=ast.USub(),
                    operand=ast.BinOp(
                        left=ast.UnaryOp(op=ast.USub(), operand=node.left),
                        op=ast.Mult(),
                        right=node.right
                    )
                )
        return node

class BooleanTransformer(BaseTransformer):
    """Transform boolean operations and conditions"""
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.And) and random.random() < 0.4:
            # Transform (a and b) to not (not a or not b)
            return ast.UnaryOp(
                op=ast.Not(),
                operand=ast.BoolOp(
                    op=ast.Or(),
                    values=[ast.UnaryOp(op=ast.Not(), operand=value) 
                           for value in node.values]
                )
            )
        elif isinstance(node.op, ast.Or) and random.random() < 0.4:
            # Transform (a or b) to not (not a and not b)
            return ast.UnaryOp(
                op=ast.Not(),
                operand=ast.BoolOp(
                    op=ast.And(),
                    values=[ast.UnaryOp(op=ast.Not(), operand=value) 
                           for value in node.values]
                )
            )
        return node

class ComprehensionTransformer(BaseTransformer):
    """Transform list/dict/set comprehensions"""
    
    def _create_lambda_args(self, target):
        """Create lambda arguments handling both simple variables and tuples"""
        if isinstance(target, ast.Name):
            return [ast.arg(arg=target.id)]
        elif isinstance(target, ast.Tuple):
            return [ast.arg(arg=elt.id) for elt in target.elts]
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")

    def _create_lambda_param(self, target):
        """Create the lambda parameter expression matching the target structure"""
        if isinstance(target, ast.Name):
            return ast.Name(id=target.id, ctx=ast.Load())
        elif isinstance(target, ast.Tuple):
            return ast.Tuple(
                elts=[ast.Name(id=elt.id, ctx=ast.Load()) for elt in target.elts],
                ctx=ast.Load()
            )
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")

    def visit_ListComp(self, node):
        self.generic_visit(node)
        if random.random() < 0.3:
            try:
                # Create appropriate lambda arguments based on target type
                lambda_args = self._create_lambda_args(node.generators[0].target)
                
                # Create the lambda body using the appropriate parameter structure
                lambda_param = self._create_lambda_param(node.generators[0].target)
                
                # Transform [x for x in lst] to list(map(lambda x: x, lst))
                # or [(x,y) for x,y in pairs] to list(map(lambda x,y: (x,y), pairs))
                return ast.Call(
                    func=ast.Name(id='list', ctx=ast.Load()),
                    args=[
                        ast.Call(
                            func=ast.Name(id='map', ctx=ast.Load()),
                            args=[
                                ast.Lambda(
                                    args=ast.arguments(
                                        posonlyargs=[],
                                        args=lambda_args,
                                        kwonlyargs=[],
                                        kw_defaults=[],
                                        defaults=[]
                                    ),
                                    body=node.elt
                                ),
                                node.generators[0].iter
                            ],
                            keywords=[]
                        )
                    ],
                    keywords=[]
                )
            except (ValueError, AttributeError):
                # If we encounter any issues with the transformation, return original node
                return node
        return node

class CodeAugmenter:
    """Main class for applying code augmentations with robust error handling"""
    def __init__(self):
        self.transformers = [
            VariableRenamer(),
            UnaryTransformer(),
            LoopTransformer(),
            ExpressionTransformer(),
            BooleanTransformer(),
            ComprehensionTransformer()
        ]
    
    def clean_code(self, code: str) -> str:
        """Pre-process code to fix common issues"""
        # Remove any invalid UTF-8 characters
        code = code.encode('ascii', 'ignore').decode()
        
        # Remove null bytes
        code = code.replace('\x00', '')
        
        # Try parsing without the last line if needed
        lines = code.splitlines()
        try:
            ast.parse(code)
        except SyntaxError:
            if lines:
                code = '\n'.join(lines[:-1])
        
        # Fix indentation
        lines = code.splitlines()
        cleaned_lines = []
        for line in lines:
            # Replace tabs with spaces
            cleaned_line = line.replace('\t', '    ')
            # Fix mixed indentation
            indent_match = re.match(r'^(\s*)', cleaned_line)
            if indent_match:
                indent = indent_match.group(1)
                spaces_count = len(indent)
                if spaces_count % 4 != 0:
                    spaces_count = (spaces_count // 4) * 4
                    cleaned_line = ' ' * spaces_count + cleaned_line.lstrip()
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def validate_syntax(self, code: str) -> Optional[str]:
        """Validate and potentially fix syntax issues"""
        try:
            ast.parse(code)
            return code
        except SyntaxError as e:
            # Try to fix common syntax issues
            if 'EOF in multi-line statement' in str(e):
                # Add missing closing brackets/parentheses
                for bracket in ['()', '[]', '{}']:
                    opening_count = code.count(bracket[0]) - code.count(bracket[1])
                    if opening_count > 0:
                        code += bracket[1] * opening_count
                
            try:
                ast.parse(code)
                return code
            except SyntaxError:
                return None
    
    def augment(self, code: str, num_variations: int = 20) -> List[str]:
        """Generate augmented versions of the input code"""
        # Clean and validate the code first
        cleaned_code = self.clean_code(code)
        valid_code = self.validate_syntax(cleaned_code)
        
        if valid_code is None:
            raise ValueError("Could not fix syntax errors in the input code")
        
        try:
            tree = ast.parse(valid_code)
        except Exception as e:
            raise ValueError(f"Failed to parse code: {str(e)}")
        
        augmented_versions = []
        
        # Generate exactly num_variations attempts
        for _ in range(num_variations):
            try:
                new_tree = copy.deepcopy(tree)
                num_transformers = random.randint(1, len(self.transformers))
                selected_transformers = random.sample(self.transformers, k=num_transformers)
                
                for transformer in selected_transformers:
                    new_tree = transformer.visit(new_tree)
                
                augmented_code = ast.unparse(new_tree)
                
                # Add if it's valid and different from original
                if (augmented_code != valid_code and 
                    augmented_code not in augmented_versions and 
                    self.validate_syntax(augmented_code)):
                    augmented_versions.append(augmented_code)
                    
            except Exception:
                continue
        
        return augmented_versions

    def augment_safely(self, code: str, num_variations: int = 20) -> tuple[List[str], List[str]]:
        """Wrapper method that returns both successful augmentations and error messages"""
        try:
            augmented = self.augment(code, num_variations)
            return augmented, []
        except Exception as e:
            return [], [str(e)]

def augment_code_samples(samples: List[str], num_variations: int = 2) -> List[str]:
    """Utility function to augment multiple code samples"""
    augmenter = CodeAugmenter()
    augmented_samples = []
    
    for sample in samples:
        augmented_versions = augmenter.augment(sample, num_variations)
        augmented_samples.extend(augmented_versions)
        
    return augmented_samples