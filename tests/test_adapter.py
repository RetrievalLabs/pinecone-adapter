"""
Copyright (c) 2026 RetrievalLabs Co. All rights reserved.
Licensed under the Apache License, Version 2.0.
"""

from unittest.mock import MagicMock, patch

import pytest
from rag_control.exceptions import VectorStoreAdapterError
from rag_control.models import (
    Filter,
    FilterCondition,
    UserContext,
    VectorStoreRecord,
)

from pinecone_adapter.vector_store import PineconeVectorStoreAdapter


class TestPineconeVectorStoreAdapterInit:
    """Test initialization of PineconeVectorStoreAdapter."""

    @patch("pinecone_adapter.vector_store.adapter.Pinecone")
    def test_init_success(self, mock_pinecone_class):
        """Test successful initialization."""
        # Setup mocks
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index

        # Initialize adapter
        adapter = PineconeVectorStoreAdapter(
            api_key="test-key",
            index_name="test-index",
            embedding_model="text-embedding-3-small",
        )

        # Verify client was created with correct API key
        mock_pinecone_class.assert_called_once_with(api_key="test-key")

        # Verify index existence was checked
        mock_client.describe_index.assert_called_once_with("test-index")

        # Verify index instance was created
        mock_client.Index.assert_called_once_with(name="test-index")

        # Verify adapter properties
        assert adapter.embedding_model == "text-embedding-3-small"

    @patch("pinecone_adapter.vector_store.adapter.Pinecone")
    def test_init_invalid_api_key(self, mock_pinecone_class):
        """Test initialization with invalid API key."""
        mock_pinecone_class.side_effect = Exception("Invalid API key")

        with pytest.raises(VectorStoreAdapterError) as exc_info:
            PineconeVectorStoreAdapter(
                api_key="invalid-key",
                index_name="test-index",
                embedding_model="text-embedding-3-small",
            )

        assert "Failed to initialize Pinecone adapter" in str(exc_info.value)

    @patch("pinecone_adapter.vector_store.adapter.Pinecone")
    def test_init_index_not_found(self, mock_pinecone_class):
        """Test initialization when index does not exist."""
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.describe_index.side_effect = Exception("Index not found")

        with pytest.raises(VectorStoreAdapterError) as exc_info:
            PineconeVectorStoreAdapter(
                api_key="test-key",
                index_name="nonexistent-index",
                embedding_model="text-embedding-3-small",
            )

        assert "Failed to initialize Pinecone adapter" in str(exc_info.value)


class TestPineconeVectorStoreAdapterSearch:
    """Test search functionality of PineconeVectorStoreAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create a mock adapter for testing."""
        with patch("pinecone_adapter.vector_store.adapter.Pinecone"):
            adapter = PineconeVectorStoreAdapter(
                api_key="test-key",
                index_name="test-index",
                embedding_model="text-embedding-3-small",
            )
            adapter._index = MagicMock()
            adapter._index.name = "test-index"
            return adapter

    def test_search_success(self, adapter):
        """Test successful search."""
        # Setup mock results
        mock_results = {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "metadata": {"text": "sample document 1"},
                }
            ]
        }
        adapter._index.query.return_value = mock_results

        # Perform search
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        response = adapter.search(embedding=embedding, top_k=5)

        # Verify query was called correctly
        adapter._index.query.assert_called_once_with(
            vector=embedding,
            top_k=5,
            namespace="",
            filter=None,
            include_metadata=True,
            include_values=False,
        )

        # Verify response
        assert len(response.records) == 1
        assert response.records[0].id == "doc1"
        assert response.records[0].score == 0.95
        assert response.records[0].content == "sample document 1"
        assert response.metadata.provider == "pinecone"
        assert response.metadata.index == "test-index"
        assert response.metadata.top_k == 5
        assert response.metadata.returned == 1

    def test_search_with_user_context(self, adapter):
        """Test search with user context containing namespace."""
        mock_results = {"matches": []}
        adapter._index.query.return_value = mock_results

        # Create user context with namespace
        user_context = UserContext(
            user_id="user123", org_id="org123", attributes={"namespace": "user-ns"}
        )

        adapter.search(
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
            user_context=user_context,
        )

        # Verify namespace was passed to query
        call_args = adapter._index.query.call_args
        assert call_args.kwargs["namespace"] == "user-ns"

    def test_search_with_filter(self, adapter):
        """Test search with metadata filter."""
        mock_results = {"matches": []}
        adapter._index.query.return_value = mock_results

        # Create simple filter
        condition = FilterCondition(field="category", operator="equals", value="news")
        filter_obj = Filter(name="category_filter", condition=condition)

        adapter.search(
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
            filter=filter_obj,
        )

        # Verify filter was passed to query
        call_args = adapter._index.query.call_args
        assert call_args.kwargs["filter"] == {"category": {"$eq": "news"}}

    def test_search_multiple_results(self, adapter):
        """Test search returning multiple results."""
        mock_results = {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "metadata": {"text": "first result"},
                },
                {
                    "id": "doc2",
                    "score": 0.87,
                    "metadata": {"text": "second result"},
                },
                {
                    "id": "doc3",
                    "score": 0.76,
                    "metadata": {"text": "third result"},
                },
            ]
        }
        adapter._index.query.return_value = mock_results

        response = adapter.search(embedding=[0.1, 0.2, 0.3], top_k=3)

        assert len(response.records) == 3
        assert response.records[0].id == "doc1"
        assert response.records[1].id == "doc2"
        assert response.records[2].id == "doc3"
        assert response.metadata.returned == 3

    def test_search_no_results(self, adapter):
        """Test search with no results."""
        mock_results = {"matches": []}
        adapter._index.query.return_value = mock_results

        response = adapter.search(embedding=[0.1, 0.2, 0.3], top_k=5)

        assert len(response.records) == 0
        assert response.metadata.returned == 0

    def test_search_empty_results_key(self, adapter):
        """Test search when results dict has no 'matches' key."""
        mock_results = {}
        adapter._index.query.return_value = mock_results

        response = adapter.search(embedding=[0.1, 0.2, 0.3], top_k=5)

        assert len(response.records) == 0

    def test_search_failure(self, adapter):
        """Test search failure."""
        adapter._index.query.side_effect = Exception("Search failed")

        with pytest.raises(VectorStoreAdapterError) as exc_info:
            adapter.search(embedding=[0.1, 0.2, 0.3], top_k=5)

        assert "Pinecone search failed" in str(exc_info.value)

    def test_search_metadata_enrichment(self, adapter):
        """Test that search response metadata is properly enriched."""
        mock_results = {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "metadata": {"text": "content", "source": "web"},
                }
            ]
        }
        adapter._index.query.return_value = mock_results

        response = adapter.search(embedding=[0.1, 0.2, 0.3], top_k=5)

        # Verify metadata contains namespace info
        assert "namespace" in response.metadata.raw
        assert response.metadata.raw["namespace"] == ""
        assert "matches" in response.metadata.raw
        assert response.metadata.raw["matches"] == 1


class TestFilterBuilding:
    """Test filter building functionality."""

    @pytest.fixture
    def adapter(self):
        """Create a mock adapter for testing."""
        with patch("pinecone_adapter.vector_store.adapter.Pinecone"):
            adapter = PineconeVectorStoreAdapter(
                api_key="test-key",
                index_name="test-index",
                embedding_model="text-embedding-3-small",
            )
            adapter._index = MagicMock()
            return adapter

    def test_build_filter_none(self, adapter):
        """Test building filter from None."""
        result = adapter._build_pinecone_filter(None)
        assert result is None

    @pytest.mark.parametrize(
        "field,operator,value,expected",
        [
            ("status", "equals", "active", {"status": {"$eq": "active"}}),
            ("category", "in", ["news", "blog"], {"category": {"$in": ["news", "blog"]}}),
            ("price", "lt", 100, {"price": {"$lt": 100}}),
            ("score", "lte", 80, {"score": {"$lte": 80}}),
            ("views", "gt", 1000, {"views": {"$gt": 1000}}),
            ("rating", "gte", 4, {"rating": {"$gte": 4}}),
            ("author", "exists", True, {"author": {"$exists": True}}),
            ("tags", "intersects", ["python", "api"], {"tags": {"$in": ["python", "api"]}}),
        ],
    )
    def test_condition_operators(self, adapter, field, operator, value, expected):
        """Test filter condition operators."""
        condition = FilterCondition(field=field, operator=operator, value=value)
        result = adapter._condition_to_pinecone(condition)
        assert result == expected

    def test_condition_intersects_none(self, adapter):
        """Test intersects operator with None value."""
        condition = FilterCondition(field="tags", operator="intersects", value=None)
        result = adapter._condition_to_pinecone(condition)
        assert result is None

    def test_condition_handles_all_valid_operators(self, adapter):
        """Test that all valid operators are handled."""
        valid_operators = ["equals", "in", "lt", "lte", "gt", "gte", "exists"]
        for operator in valid_operators:
            value = (
                ["a", "b"]
                if operator == "in"
                else (1 if operator in ["lt", "lte", "gt", "gte"] else "value")
            )
            condition = FilterCondition(field="test_field", operator=operator, value=value)
            result = adapter._condition_to_pinecone(condition)
            assert result is not None

    def test_condition_unsupported_operator_raises_error(self, adapter):
        """Test that unsupported operators are rejected."""
        # Create a mock condition with an invalid operator to test error handling
        condition = MagicMock()
        condition.field = "test"
        condition.operator = "unsupported_op"
        condition.value = "value"

        with pytest.raises(VectorStoreAdapterError) as exc_info:
            adapter._condition_to_pinecone(condition)
        assert "Unsupported filter operator" in str(exc_info.value)

    def test_filter_with_single_condition(self, adapter):
        """Test filter with single condition."""
        condition = FilterCondition(field="status", operator="equals", value="active")
        filter_obj = Filter(name="status_filter", condition=condition)
        result = adapter._build_pinecone_filter(filter_obj)
        assert result == {"status": {"$eq": "active"}}

    def test_filter_with_and_conditions(self, adapter):
        """Test filter with AND logic."""
        condition1 = FilterCondition(field="status", operator="equals", value="active")
        condition2 = FilterCondition(field="type", operator="equals", value="article")
        filter1 = Filter(name="filter1", condition=condition1)
        filter2 = Filter(name="filter2", condition=condition2)
        filter_obj = Filter(name="and_filter", and_=[filter1, filter2])

        result = adapter._build_pinecone_filter(filter_obj)
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_filter_with_or_conditions(self, adapter):
        """Test filter with OR logic."""
        condition1 = FilterCondition(field="status", operator="equals", value="active")
        condition2 = FilterCondition(field="status", operator="equals", value="pending")
        filter1 = Filter(name="filter1", condition=condition1)
        filter2 = Filter(name="filter2", condition=condition2)
        filter_obj = Filter(name="or_filter", or_=[filter1, filter2])

        result = adapter._build_pinecone_filter(filter_obj)
        assert "$or" in result
        assert len(result["$or"]) == 2

    def test_nested_filters_and_or(self, adapter):
        """Test nested filters with mixed AND/OR logic."""
        # Create nested structure: (status=active AND type=article) OR (status=draft)
        cond1 = FilterCondition(field="status", operator="equals", value="active")
        cond2 = FilterCondition(field="type", operator="equals", value="article")
        filter1 = Filter(name="filter1", condition=cond1)
        filter2 = Filter(name="filter2", condition=cond2)
        and_filter = Filter(name="and_filter", and_=[filter1, filter2])

        cond3 = FilterCondition(field="status", operator="equals", value="draft")
        filter3 = Filter(name="filter3", condition=cond3)

        or_filter = Filter(name="or_filter", or_=[and_filter, filter3])
        result = adapter._build_pinecone_filter(or_filter)

        assert "$or" in result
        assert len(result["$or"]) == 2

    def test_filter_empty_and_list(self, adapter):
        """Test filter with empty and_ list returns None."""
        filter_obj = Filter(name="empty_and", and_=[])
        result = adapter._build_pinecone_filter(filter_obj)
        assert result is None

    def test_filter_empty_or_list(self, adapter):
        """Test filter with empty or_ list returns None."""
        filter_obj = Filter(name="empty_or", or_=[])
        result = adapter._build_pinecone_filter(filter_obj)
        assert result is None


class TestResultConversion:
    """Test conversion of Pinecone results to rag_control format."""

    @pytest.fixture
    def adapter(self):
        """Create a mock adapter for testing."""
        with patch("pinecone_adapter.vector_store.adapter.Pinecone"):
            adapter = PineconeVectorStoreAdapter(
                api_key="test-key",
                index_name="test-index",
                embedding_model="text-embedding-3-small",
            )
            return adapter

    @pytest.mark.parametrize(
        "num_results",
        [1, 2, 5],
    )
    def test_convert_multiple_results(self, adapter, num_results):
        """Test converting multiple Pinecone results."""
        matches = [
            {
                "id": f"doc{i}",
                "score": 0.95 - (i * 0.05),
                "metadata": {"text": f"result {i}"},
            }
            for i in range(num_results)
        ]
        results = {"matches": matches}

        records = adapter._convert_results_to_records(results)

        assert len(records) == num_results
        for i, record in enumerate(records):
            assert isinstance(record, VectorStoreRecord)
            assert record.id == f"doc{i}"
            assert record.score == pytest.approx(0.95 - (i * 0.05), abs=0.01)
            assert record.content == f"result {i}"

    def test_convert_result_no_metadata(self, adapter):
        """Test converting result with no metadata."""
        results = {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.95,
                }
            ]
        }

        records = adapter._convert_results_to_records(results)

        assert len(records) == 1
        assert records[0].id == "doc1"
        assert records[0].content == ""
        assert records[0].metadata == {}

    def test_convert_result_missing_text_metadata(self, adapter):
        """Test converting result with metadata but no text field."""
        results = {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "metadata": {"source": "web"},
                }
            ]
        }

        records = adapter._convert_results_to_records(results)

        assert len(records) == 1
        assert records[0].content == ""
        assert records[0].metadata == {"source": "web"}

    def test_convert_empty_results(self, adapter):
        """Test converting empty results."""
        results = {"matches": []}
        records = adapter._convert_results_to_records(results)
        assert records == []

    def test_convert_results_missing_id(self, adapter):
        """Test converting result with missing id field."""
        results = {
            "matches": [
                {
                    "score": 0.95,
                    "metadata": {"text": "content"},
                }
            ]
        }

        records = adapter._convert_results_to_records(results)

        assert len(records) == 1
        assert records[0].id == ""

    def test_convert_results_missing_score(self, adapter):
        """Test converting result with missing score field."""
        results = {
            "matches": [
                {
                    "id": "doc1",
                    "metadata": {"text": "content"},
                }
            ]
        }

        records = adapter._convert_results_to_records(results)

        assert len(records) == 1
        assert records[0].score == 0.0


class TestEmbeddingModelProperty:
    """Test embedding_model property."""

    @patch("pinecone_adapter.vector_store.adapter.Pinecone")
    @pytest.mark.parametrize(
        "embedding_model",
        [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "custom-model-v1",
        ],
    )
    def test_embedding_model_property(self, mock_pinecone_class, embedding_model):
        """Test that embedding_model property returns the correct value."""
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client

        adapter = PineconeVectorStoreAdapter(
            api_key="test-key",
            index_name="test-index",
            embedding_model=embedding_model,
        )

        assert adapter.embedding_model == embedding_model
