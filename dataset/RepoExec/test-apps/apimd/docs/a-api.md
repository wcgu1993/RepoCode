## a API

### Module `apimd`
<a id="apimd"></a>

A Python API compiler for universal Markdown syntax.

#### gen_api()

*Full name:* `apimd.gen_api`
<a id="apimd-gen_api"></a>

| root_names | pwd | * | prefix | link | level | toc | dry | return |
|:----------:|:---:|:---:|:------:|:----:|:-----:|:---:|:---:|:------:|
| `dict[str, str]` | <code>str &#124; None</code> |   | `str` | `bool` | `int` | `bool` | `bool` | `collections.abc.Sequence[str]` |
|   | `None` |   | `'docs'` | `True` | `1` | `False` | `False` |   |

Generate API. All rules are listed in the readme.

The path `pwd` is the current path that provided to `pkgutil`,
which allows the "site-packages" directory to be used.

### Module `apimd.__main__`
<a id="apimd-__main__"></a>

The command line launcher of apimd.

#### main()

*Full name:* `apimd.__main__.main`
<a id="apimd-__main__-main"></a>

| return |
|:------:|
| `None` |

Main function.

### Module `apimd.loader`
<a id="apimd-loader"></a>

| Constants | Type |
|:---------:|:----:|
| `PEP561_SUFFIX` | `str` |

Compiler functions.

#### loader()

*Full name:* `apimd.loader.loader`
<a id="apimd-loader-loader"></a>

| root | pwd | link | level | toc | return |
|:----:|:---:|:----:|:-----:|:---:|:------:|
| `str` | `str` | `bool` | `int` | `bool` | `str` |

Package searching algorithm.

#### walk_packages()

*Full name:* `apimd.loader.walk_packages`
<a id="apimd-loader-walk_packages"></a>

| name | path | return |
|:----:|:----:|:------:|
| `str` | `str` | `collections.abc.Iterator[tuple[str, str]]` |

Walk packages without import them.

### Module `apimd.parser`
<a id="apimd-parser"></a>

| Constants | Type |
|:---------:|:----:|
| `ANY` | `str` |

Data structures.

#### code()

*Full name:* `apimd.parser.code`
<a id="apimd-parser-code"></a>

| doc | return |
|:---:|:------:|
| `str` | `str` |

Escape Markdown charters from inline code.

#### const_type()

*Full name:* `apimd.parser.const_type`
<a id="apimd-parser-const_type"></a>

| node | return |
|:----:|:------:|
| `ast.expr` | `str` |

Constant type inference.

#### doctest()

*Full name:* `apimd.parser.doctest`
<a id="apimd-parser-doctest"></a>

| doc | return |
|:---:|:------:|
| `str` | `str` |

Wrap doctest as markdown Python code.

#### esc_underscore()

*Full name:* `apimd.parser.esc_underscore`
<a id="apimd-parser-esc_underscore"></a>

| doc | return |
|:---:|:------:|
| `str` | `str` |

Escape underscore in names.

#### is_magic()

*Full name:* `apimd.parser.is_magic`
<a id="apimd-parser-is_magic"></a>

| name | return |
|:----:|:------:|
| `str` | `bool` |

Check magic name.

#### is\_public\_family()

*Full name:* `apimd.parser.is_public_family`
<a id="apimd-parser-is_public_family"></a>

| name | return |
|:----:|:------:|
| `str` | `bool` |

Check the name is come from public modules or not.

#### parent()

*Full name:* `apimd.parser.parent`
<a id="apimd-parser-parent"></a>

| name | * | level | return |
|:----:|:---:|:-----:|:------:|
| `str` |   | `int` | `str` |
|   |   | `1` |   |

Get parent name with level.

#### class Parser

*Full name:* `apimd.parser.Parser`
<a id="apimd-parser-parser"></a>

| Decorators |
|:----------:|
| `@dataclasses.dataclass` |

| Members | Type |
|:-------:|:----:|
| `alias` | `dict[str, str]` |
| `b_level` | `int` |
| `const` | `dict[str, str]` |
| `doc` | `dict[str, str]` |
| `docstring` | `dict[str, str]` |
| `imp` | `dict[str, set[str]]` |
| `level` | `dict[str, int]` |
| `link` | `bool` |
| `root` | `dict[str, str]` |
| `toc` | `bool` |

AST parser.

Usage:
```python
>>> p = Parser()
>>> with open("pkg_path", 'r') as f:
>>>     p.parse('pkg_name', f.read())
>>> s = p.compile()
```

Or create with parameters:
```python
>>> p = Parser.new(link=True, level=1)
```

##### Parser.api()

*Full name:* `apimd.parser.Parser.api`
<a id="apimd-parser-parser-api"></a>

| self | root | node | * | prefix | return |
|:----:|:----:|:----:|:---:|:------:|:------:|
| `Self` | `str` | <code>ast.FunctionDef &#124; ast.AsyncFunctionDef &#124; ast.ClassDef</code> |   | `str` | `None` |
|   |   |   |   | `''` |   |

Create API doc for only functions and classes.
Where `name` is the full name.

##### Parser.class_api()

*Full name:* `apimd.parser.Parser.class_api`
<a id="apimd-parser-parser-class_api"></a>

| self | root | name | bases | body | return |
|:----:|:----:|:----:|:-----:|:----:|:------:|
| `Self` | `str` | `str` | `list[ast.expr]` | `list[ast.stmt]` | `None` |

Create class API.

##### Parser.compile()

*Full name:* `apimd.parser.Parser.compile`
<a id="apimd-parser-parser-compile"></a>

| self | return |
|:----:|:------:|
| `Self` | `str` |

Compile documentation.

##### Parser.func_ann()

*Full name:* `apimd.parser.Parser.func_ann`
<a id="apimd-parser-parser-func_ann"></a>

| self | root | args | * | has_self | cls_method | return |
|:----:|:----:|:----:|:---:|:--------:|:----------:|:------:|
| `Self` | `str` | `collections.abc.Sequence[ast.arg]` |   | `bool` | `bool` | `collections.abc.Iterator[str]` |

Function annotation table.

##### Parser.func_api()

*Full name:* `apimd.parser.Parser.func_api`
<a id="apimd-parser-parser-func_api"></a>

| self | root | name | node | returns | * | has_self | cls_method | return |
|:----:|:----:|:----:|:----:|:-------:|:---:|:--------:|:----------:|:------:|
| `Self` | `str` | `str` | `ast.arguments` | <code>ast.expr &#124; None</code> |   | `bool` | `bool` | `None` |

Create function API.

##### Parser.globals()

*Full name:* `apimd.parser.Parser.globals`
<a id="apimd-parser-parser-globals"></a>

| self | root | node | return |
|:----:|:----:|:----:|:------:|
| `Self` | `str` | <code>ast.Assign &#124; ast.AnnAssign</code> | `None` |

Set up globals:

+ Type alias
+ Constants
+ `__all__` filter

##### Parser.imports()

*Full name:* `apimd.parser.Parser.imports`
<a id="apimd-parser-parser-imports"></a>

| self | root | node | return |
|:----:|:----:|:----:|:------:|
| `Self` | `str` | <code>ast.Import &#124; ast.ImportFrom</code> | `None` |

Save import names.

##### Parser.is_public()

*Full name:* `apimd.parser.Parser.is_public`
<a id="apimd-parser-parser-is_public"></a>

| self | s | return |
|:----:|:---:|:------:|
| `Self` | `str` | `bool` |

Check the name is public style or listed in `__all__`.

##### Parser.load_docstring()

*Full name:* `apimd.parser.Parser.load_docstring`
<a id="apimd-parser-parser-load_docstring"></a>

| self | root | m | return |
|:----:|:----:|:---:|:------:|
| `Self` | `str` | `types.ModuleType` | `None` |

Load docstring from the module.

##### Parser.new()

*Full name:* `apimd.parser.Parser.new`
<a id="apimd-parser-parser-new"></a>

| Decorators |
|:----------:|
| `@classmethod` |

| cls | link | level | toc | return |
|:---:|:----:|:-----:|:---:|:------:|
| `type[Self]` | `bool` | `int` | `bool` | `Self` |

Create a parser by options.

##### Parser.parse()

*Full name:* `apimd.parser.Parser.parse`
<a id="apimd-parser-parser-parse"></a>

| self | root | script | return |
|:----:|:----:|:------:|:------:|
| `Self` | `str` | `str` | `None` |

Main parser of the entire module.

##### Parser.resolve()

*Full name:* `apimd.parser.Parser.resolve`
<a id="apimd-parser-parser-resolve"></a>

| self | root | node | self_ty | return |
|:----:|:----:|:----:|:-------:|:------:|
| `Self` | `str` | `ast.expr` | `str` | `str` |
|   |   |   | `''` |   |   |

Search and resolve global names in annotation.

#### class Resolver

*Full name:* `apimd.parser.Resolver`
<a id="apimd-parser-resolver"></a>

| Bases |
|:-----:|
| `ast.NodeTransformer` |

Annotation resolver.

##### Resolver.\_\_init\_\_()

*Full name:* `apimd.parser.Resolver.__init__`
<a id="apimd-parser-resolver-__init__"></a>

| self | root | alias | self_ty | return |
|:----:|:----:|:-----:|:-------:|:------:|
| `Self` | `str` | `dict[str, str]` | `str` | `Any` |
|   |   |   | `''` |   |   |

Set root module, alias and generic self name.

##### Resolver.visit_Attribute()

*Full name:* `apimd.parser.Resolver.visit_Attribute`
<a id="apimd-parser-resolver-visit_attribute"></a>

| self | node | return |
|:----:|:----:|:------:|
| `Self` | `ast.Attribute` | `ast.AST` |

Remove `typing.*` prefix of annotation.

##### Resolver.visit_Constant()

*Full name:* `apimd.parser.Resolver.visit_Constant`
<a id="apimd-parser-resolver-visit_constant"></a>

| self | node | return |
|:----:|:----:|:------:|
| `Self` | `ast.Constant` | `ast.AST` |

Check string is a name.

##### Resolver.visit_Name()

*Full name:* `apimd.parser.Resolver.visit_Name`
<a id="apimd-parser-resolver-visit_name"></a>

| self | node | return |
|:----:|:----:|:------:|
| `Self` | `ast.Name` | `ast.AST` |

Replace global names with its expression recursively.

##### Resolver.visit_Subscript()

*Full name:* `apimd.parser.Resolver.visit_Subscript`
<a id="apimd-parser-resolver-visit_subscript"></a>

| self | node | return |
|:----:|:----:|:------:|
| `Self` | `ast.Subscript` | `ast.AST` |

Implementation of PEP585 and PEP604.

#### table()

*Full name:* `apimd.parser.table`
<a id="apimd-parser-table"></a>

| *titles | items | return |
|:-------:|:-----:|:------:|
| `str` | <code>collections.abc.Iterable[str &#124; Iterable[str]]</code> | `str` |

Create multi-column table with the titles.

Usage:
```python
>>> table('a', 'b', [['c', 'd'], ['e', 'f']])
```
| a | b |
|:---:|:---:|
| c | d |
| e | f |

#### walk_body()

*Full name:* `apimd.parser.walk_body`
<a id="apimd-parser-walk_body"></a>

| body | return |
|:----:|:------:|
| `collections.abc.Sequence[ast.stmt]` | `collections.abc.Iterator[ast.stmt]` |

Traverse around body and its simple definition scope.

### Module `apimd.pep585`
<a id="apimd-pep585"></a>

| Constants | Type |
|:---------:|:----:|
| `PEP585` | `dict[str, str]` |

Implementation of PEP585 deprecated name alias.

### Module `apimd.test_gen_api`
<a id="apimd-test_gen_api"></a>

| Constants | Type |
|:---------:|:----:|
| `LOCAL_TIMEOUT` | `int` |

#### test_0()

*Full name:* `apimd.test_gen_api.test_0`
<a id="apimd-test_gen_api-test_0"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_1()

*Full name:* `apimd.test_gen_api.test_1`
<a id="apimd-test_gen_api-test_1"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_14()

*Full name:* `apimd.test_gen_api.test_14`
<a id="apimd-test_gen_api-test_14"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_17()

*Full name:* `apimd.test_gen_api.test_17`
<a id="apimd-test_gen_api-test_17"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_18()

*Full name:* `apimd.test_gen_api.test_18`
<a id="apimd-test_gen_api-test_18"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_2()

*Full name:* `apimd.test_gen_api.test_2`
<a id="apimd-test_gen_api-test_2"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_20()

*Full name:* `apimd.test_gen_api.test_20`
<a id="apimd-test_gen_api-test_20"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_22()

*Full name:* `apimd.test_gen_api.test_22`
<a id="apimd-test_gen_api-test_22"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_23()

*Full name:* `apimd.test_gen_api.test_23`
<a id="apimd-test_gen_api-test_23"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_24()

*Full name:* `apimd.test_gen_api.test_24`
<a id="apimd-test_gen_api-test_24"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_25()

*Full name:* `apimd.test_gen_api.test_25`
<a id="apimd-test_gen_api-test_25"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_26()

*Full name:* `apimd.test_gen_api.test_26`
<a id="apimd-test_gen_api-test_26"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_28()

*Full name:* `apimd.test_gen_api.test_28`
<a id="apimd-test_gen_api-test_28"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_3()

*Full name:* `apimd.test_gen_api.test_3`
<a id="apimd-test_gen_api-test_3"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_30()

*Full name:* `apimd.test_gen_api.test_30`
<a id="apimd-test_gen_api-test_30"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_32()

*Full name:* `apimd.test_gen_api.test_32`
<a id="apimd-test_gen_api-test_32"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_33()

*Full name:* `apimd.test_gen_api.test_33`
<a id="apimd-test_gen_api-test_33"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_36()

*Full name:* `apimd.test_gen_api.test_36`
<a id="apimd-test_gen_api-test_36"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_37()

*Full name:* `apimd.test_gen_api.test_37`
<a id="apimd-test_gen_api-test_37"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_39()

*Full name:* `apimd.test_gen_api.test_39`
<a id="apimd-test_gen_api-test_39"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_4()

*Full name:* `apimd.test_gen_api.test_4`
<a id="apimd-test_gen_api-test_4"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_6()

*Full name:* `apimd.test_gen_api.test_6`
<a id="apimd-test_gen_api-test_6"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_8()

*Full name:* `apimd.test_gen_api.test_8`
<a id="apimd-test_gen_api-test_8"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |

#### test_9()

*Full name:* `apimd.test_gen_api.test_9`
<a id="apimd-test_gen_api-test_9"></a>

| Decorators |
|:----------:|
| `@timeout_decorator.timeout(5)` |

| return |
|:------:|
| `Any` |
