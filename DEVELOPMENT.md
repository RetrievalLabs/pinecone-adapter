# Development Guide

## Setup

### Prerequisites
- Python 3.10+
- uv package manager
- Pinecone account and API key
- rag_control 0.1.3

### Installation

1. Clone the repository
```bash
git clone https://github.com/RetrievalLabs/pinecone-adapter.git
cd pinecone-adapter
```

2. Create virtual environment
```bash
make venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
make install-dev
```

## Development Workflow

### Running Tests

Run all tests:
```bash
make test
```

Run tests with coverage report:
```bash
make coverage
```

Coverage reports are generated in:
- Terminal output with missing lines
- `htmlcov/index.html` for interactive HTML report
- `coverage.xml` for CI/CD integration

### Code Quality

Format code with Black and Ruff:
```bash
make format
```

Lint with Ruff:
```bash
make lint
```

Type check with mypy:
```bash
make typecheck
```

## Project Structure

```
pinecone-adapter/
├── pinecone_adopter/
│   ├── __init__.py                 # Package exports
│   ├── version.py                  # Version info
│   └── vector_store/
│       ├── __init__.py             # Exports PineconeVectorStoreAdapter
│       └── adapter.py              # Main adapter implementation
├── tests/
│   ├── __init__.py
│   └── test_adapter.py             # Adapter tests
├── examples/                        # Usage examples
├── pyproject.toml                  # Project configuration
├── Makefile                        # Development commands
├── README.md                        # User documentation
└── DEVELOPMENT.md                  # This file
```

## Key Classes

### PineconeVectorStoreAdapter

Located in `pinecone_adopter/vector_store/adapter.py`

Implements `rag_control.adapters.VectorStore` interface with:
- `embedding_model` property: Returns the embedding model identifier
- `search()` method: Performs vector similarity search with optional filtering and namespace isolation

## Understanding the Code

### Adapter Initialization

The adapter:
1. Initializes Pinecone client with API key
2. Verifies index exists via `describe_index()`
3. Gets Index instance for queries
4. Raises `VectorStoreAdapterError` on initialization failure

### Search Flow

1. Extract namespace from `user_context.attributes["namespace"]` (defaults to "")
2. Convert rag_control Filter to Pinecone filter format
3. Query Pinecone with embedding, top_k, namespace, and filter
4. Convert Pinecone results to `VectorStoreRecord` objects
5. Return `VectorStoreSearchResponse` with records and metadata

### Filter Conversion

The adapter maps rag_control filter operators to Pinecone operators:
- `equals` → `$eq`
- `in` → `$in`
- `lt` → `$lt`
- `lte` → `$lte`
- `gt` → `$gt`
- `gte` → `$gte`
- `exists` → `$exists`
- `intersects` → `$in` (approximation)

Compound filters with `and_`/`or_` are recursively converted to `$and`/`$or`.

## Testing

### Test Structure

Tests are in `tests/test_adapter.py` and cover:
- Adapter initialization success and error cases
- Search with various parameters
- Filter conversion
- Result conversion
- Error handling

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage report
make coverage
```

### Mocking

Tests use `unittest.mock` to mock:
- Pinecone client
- Index instance
- API responses

Example:
```python
from unittest.mock import MagicMock, patch

@patch('pinecone_adopter.vector_store.adapter.Pinecone')
def test_search(mock_pinecone_class):
    mock_client = MagicMock()
    mock_pinecone_class.return_value = mock_client
    # ... test code
```

## Type Checking

The project uses strict mypy configuration. Run type checks:

```bash
make typecheck
```

### Type Hints

All public functions and methods should have complete type hints:
```python
def search(
    self,
    embedding: list[float],
    top_k: int = 5,
    user_context: UserContext | None = None,
    filter: Filter | None = None,
) -> VectorStoreSearchResponse:
```

## Dependencies

### Core Dependencies
- `pinecone>=3.0.0` - Vector database client
- `rag-control==0.1.3` - RAG governance framework

### Dev Dependencies
- `black>=25.11.0` - Code formatter
- `mypy>=1.19.1` - Type checker
- `pytest>=8.4.2` - Test framework
- `pytest-cov>=7.0.0` - Coverage plugin
- `ruff>=0.15.1` - Linter

See `pyproject.toml` for dependency groups.

## Making Changes

### Adding a Feature

1. Create feature branch
```bash
git checkout -b feature/my-feature
```

2. Write tests first (TDD approach recommended)
3. Implement the feature
4. Run tests and type checks
```bash
make test
make typecheck
```

5. Format code
```bash
make format
```

6. Commit with clear message
```bash
git commit -m "Add feature description"
```

### Fixing a Bug

1. Create bug fix branch
```bash
git checkout -b fix/bug-description
```

2. Write a failing test that reproduces the bug
3. Fix the bug
4. Verify test passes
5. Run full test suite
```bash
make test
```

6. Format and commit

## Error Handling

All errors should raise `VectorStoreAdapterError` from `rag_control.exceptions`:

```python
from rag_control.exceptions import VectorStoreAdapterError

try:
    # ... code
except Exception as e:
    raise VectorStoreAdapterError(f"Descriptive error message: {str(e)}") from e
```

## Documentation

### Code Comments

Use docstrings for all public classes and methods:

```python
def search(
    self,
    embedding: list[float],
    top_k: int = 5,
    user_context: UserContext | None = None,
    filter: Filter | None = None,
) -> VectorStoreSearchResponse:
    """
    Search Pinecone index using vector similarity.

    Args:
        embedding: Query embedding vector
        top_k: Number of top results to return
        user_context: Optional user context for access control
        filter: Optional metadata filter

    Returns:
        VectorStoreSearchResponse containing ranked records and metadata

    Raises:
        VectorStoreAdapterError: If search fails
    """
```

### README Updates

Update README.md when:
- Adding new public methods or parameters
- Changing behavior
- Adding significant features

Update this DEVELOPMENT.md when:
- Changing directory structure
- Adding new classes or modules
- Changing development workflow

## Continuous Integration

GitHub Actions workflows run on every push:
- `tests.yml` - Run pytest via `make test`
- `lint.yml` - Run ruff via `make lint`
- `typecheck.yml` - Run mypy via `make typecheck`
- `coverage.yml` - Generate coverage report via `make coverage`

Check `.github/workflows/` for details.

## Versioning

Version is defined in `pinecone_adopter/version.py` and should follow [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

Update version when releasing:
```python
# pinecone_adopter/version.py
__version__ = "0.2.0"
```

## Troubleshooting

### Pinecone Connection Issues

- Verify API key is correct
- Check index name spelling
- Ensure index exists in Pinecone console
- Verify network connectivity

### Filter Not Working

- Check field names match metadata keys
- Verify operator is supported (see filter conversion section)
- Ensure value type matches expected type
- Test with simpler filters first

### Type Checking Failures

- Add explicit type hints
- Use `Union` or `|` for optional types
- Cast results if needed: `cast(VectorStoreRecord, value)`
- Check rag_control type definitions

### Test Failures

Run with verbose output to diagnose:
```bash
make test  # Run all tests
```

Check coverage report for untested code paths:
```bash
make coverage
# Open htmlcov/index.html to see detailed coverage
```

## Make Commands Reference

| Command | Purpose |
|---------|---------|
| `make venv` | Create virtual environment |
| `make activate` | Show activation command |
| `make install` | Install core dependencies |
| `make install-dev` | Install all dependencies including dev |
| `make test` | Run all tests |
| `make coverage` | Run tests with coverage report |
| `make lint` | Check code with ruff |
| `make format` | Format code with ruff and black |
| `make typecheck` | Type check with mypy |

## Contributing

For contribution guidelines, see the main repository CONTRIBUTING.md.

## Support

For questions or issues:
- Open GitHub issue
- Check existing issues and discussions
- Contact support@retrievallabs.com
