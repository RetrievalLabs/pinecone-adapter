"""
Copyright (c) 2026 RetrievalLabs Co. All rights reserved.
Licensed under the Apache License, Version 2.0.
"""

from datetime import datetime
from typing import Any

from pinecone import Pinecone  # type: ignore[import-untyped]
from rag_control.adapters import VectorStore
from rag_control.exceptions import VectorStoreAdapterError
from rag_control.models import (
    FilterCondition,
    Filter,
    UserContext,
    VectorStoreRecord,
    VectorStoreSearchMetadata,
    VectorStoreSearchResponse,
)


class PineconeVectorStoreAdapter(VectorStore):
    """
    Pinecone adapter implementing the rag_control VectorStore interface.

    This adapter integrates with Pinecone vector database to enable
    semantic search with rag_control governance and access control.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedding_model: str,
    ):
        """
        Initialize the Pinecone VectorStore adapter.

        Args:
            api_key: Pinecone API key for authentication
            index_name: Name of the Pinecone index to query
            embedding_model: The embedding model identifier (e.g., "text-embedding-3-small")

        Raises:
            VectorStoreAdapterError: If index does not exist or API key is invalid
        """
        try:
            # Initialize Pinecone client
            client = Pinecone(api_key=api_key)

            # Verify index exists by describing it
            client.describe_index(index_name)

            # Get the index instance
            self._index = client.Index(name=index_name)
            self._embedding_model = embedding_model
        except Exception as e:
            raise VectorStoreAdapterError(f"Failed to initialize Pinecone adapter: {str(e)}") from e

    @property
    def embedding_model(self) -> str:
        """Return the embedding model identifier."""
        return self._embedding_model

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
        try:
            start_time = datetime.now()

            # Extract namespace from user_context attributes if available
            namespace = ""
            if user_context:
                namespace = user_context.attributes.get("namespace", "")

            # Build filter dict from rag_control Filter object
            pinecone_filter = self._build_pinecone_filter(filter) if filter else None

            # Query Pinecone
            results = self._index.query(
                vector=embedding,
                top_k=top_k,
                namespace=namespace,
                filter=pinecone_filter,
                include_metadata=True,
                include_values=False,
            )

            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000

            # Convert Pinecone results to rag_control format
            records = self._convert_results_to_records(results)

            metadata = VectorStoreSearchMetadata(
                provider="pinecone",
                index=self._index.name,
                latency_ms=latency_ms,
                top_k=top_k,
                returned=len(records),
                timestamp=end_time,
                raw={
                    "namespace": namespace,
                    "matches": len(results.get("matches", [])) if results else 0,
                },
            )

            return VectorStoreSearchResponse(records=records, metadata=metadata)

        except Exception as e:
            raise VectorStoreAdapterError(f"Pinecone search failed: {str(e)}") from e

    def _build_pinecone_filter(self, filter: Filter) -> dict[str, Any] | None:
        """
        Convert rag_control Filter to Pinecone filter format.

        Args:
            filter: rag_control Filter object

        Returns:
            Pinecone filter dict or None
        """
        if not filter:
            return None

        # Handle simple condition
        if filter.condition:
            return self._condition_to_pinecone(filter.condition)

        # Handle compound filters (and/or)
        filters = []

        if filter.and_:
            for sub_filter in filter.and_:
                sub_pinecone_filter = self._build_pinecone_filter(sub_filter)
                if sub_pinecone_filter:
                    filters.append(sub_pinecone_filter)
            if filters:
                return {"$and": filters}

        if filter.or_:
            for sub_filter in filter.or_:
                sub_pinecone_filter = self._build_pinecone_filter(sub_filter)
                if sub_pinecone_filter:
                    filters.append(sub_pinecone_filter)
            if filters:
                return {"$or": filters}

        return None

    def _condition_to_pinecone(self, condition: FilterCondition) -> dict[str, Any] | None:
        """
        Convert rag_control Condition to Pinecone filter format.

        Args:
            condition: rag_control Condition object

        Returns:
            Pinecone filter dict or None
        """
        field = condition.field
        operator = condition.operator
        value = condition.value

        # Map rag_control operators to Pinecone operators
        if operator == "equals":
            return {field: {"$eq": value}}
        elif operator == "in":
            return {field: {"$in": value}}
        elif operator == "lt":
            return {field: {"$lt": value}}
        elif operator == "lte":
            return {field: {"$lte": value}}
        elif operator == "gt":
            return {field: {"$gt": value}}
        elif operator == "gte":
            return {field: {"$gte": value}}
        elif operator == "exists":
            return {field: {"$exists": value}}
        elif operator == "intersects":
            # Pinecone doesn't have native intersect, use 'in' as approximation
            return {field: {"$in": value}} if value else None
        else:
            raise VectorStoreAdapterError(f"Unsupported filter operator: {operator}")

    def _convert_results_to_records(self, results: dict[str, Any]) -> list[VectorStoreRecord]:
        """
        Convert Pinecone query results to VectorStoreRecord objects.

        Args:
            results: Raw Pinecone query results

        Returns:
            List of VectorStoreRecord objects
        """
        records = []
        matches = results.get("matches", [])

        for match in matches:
            record = VectorStoreRecord(
                id=match.get("id", ""),
                content=match.get("metadata", {}).get("text", "") if match.get("metadata") else "",
                score=match.get("score", 0.0),
                metadata=match.get("metadata", {}),
            )
            records.append(record)

        return records
