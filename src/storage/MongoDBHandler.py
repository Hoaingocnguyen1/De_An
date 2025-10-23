"""
src/storage/mongodb_handler.py
Enhanced MongoDB handler with optimized vector search for multimodal RAG
"""

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure
from typing import List, Dict, Any, Optional, Literal
from bson import ObjectId
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MongoDBHandler:
    """Enhanced MongoDB handler with vector search support"""
    
    def __init__(
        self, 
        connection_string: str, 
        database_name: str = "multimodal_rag"
    ):
        """Initialize MongoDB connection"""
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        
        # Collections
        self.knowledge_units = self.db["knowledge_units"]
        self.sources = self.db["sources"]
        
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes for performance"""
        try:
            # KU indexes
            self.knowledge_units.create_index([("source_id", ASCENDING)])
            self.knowledge_units.create_index([("ku_id", ASCENDING)], unique=True)
            self.knowledge_units.create_index([("ku_type", ASCENDING)])
            self.knowledge_units.create_index([("source_type", ASCENDING)])
            
            # Source indexes
            self.sources.create_index([("source_uri", ASCENDING)], unique=True)
            self.sources.create_index([("status", ASCENDING)])
            
            logger.info("✓ MongoDB indexes created")
            logger.info("⚠️  Remember to create Atlas Vector Search indexes manually:")
            logger.info("    - 'vector_index' on 'embeddings.vector' field")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def test_connection(self) -> bool:
        """Test MongoDB connection"""
        try:
            self.client.admin.command('ping')
            logger.info("✓ MongoDB connection successful")
            return True
        except ConnectionFailure:
            logger.error("✗ MongoDB connection failed")
            return False
    
    # ==================== SOURCE OPERATIONS ====================
    
    def create_source(self, source_data: Dict[str, Any]) -> ObjectId:
        """Create a new source document"""
        source_doc = {
            "source_type": source_data.get("source_type"),
            "source_uri": source_data.get("source_uri"),
            "original_filename": source_data.get("original_filename"),
            "status": source_data.get("status", "pending"),
            "error_message": None,
            "total_kus": 0,
            "processing_start": source_data.get("processing_start"),
            "processing_end": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = self.sources.insert_one(source_doc)
        logger.info(f"Created source document: {result.inserted_id}")
        return result.inserted_id
    
    def update_source_status(
        self, 
        source_id: ObjectId, 
        status: str,
        error_message: Optional[str] = None,
        total_kus: Optional[int] = None
    ):
        """Update source processing status"""
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        
        if error_message:
            update_data["error_message"] = error_message
        
        if status == "completed":
            update_data["processing_end"] = datetime.utcnow()
        
        if total_kus is not None:
            update_data["total_kus"] = total_kus
        
        self.sources.update_one(
            {"_id": source_id},
            {"$set": update_data}
        )
        logger.info(f"Updated source {source_id} status to {status}")
    
    def get_source(self, source_id: ObjectId) -> Optional[Dict]:
        """Retrieve source document"""
        return self.sources.find_one({"_id": source_id})
    
    def list_sources(
        self, 
        status: Optional[str] = None,
        source_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """List sources with optional filters"""
        query = {}
        if status:
            query["status"] = status
        if source_type:
            query["source_type"] = source_type
        
        return list(self.sources.find(query).limit(limit))
    
    # ==================== KNOWLEDGE UNIT OPERATIONS ====================
    
    def insert_knowledge_units(self, kus: List[Dict]) -> List[ObjectId]:
        """Batch insert knowledge units"""
        if not kus:
            return []
        
        # Add timestamps
        for ku in kus:
            if "created_at" not in ku:
                ku["created_at"] = datetime.utcnow()
        
        result = self.knowledge_units.insert_many(kus)
        logger.info(f"✓ Inserted {len(result.inserted_ids)} knowledge units")
        return result.inserted_ids
    
    def get_ku_by_id(self, ku_id: str) -> Optional[Dict]:
        """Retrieve KU by its readable ID"""
        return self.knowledge_units.find_one({"ku_id": ku_id})
    
    def get_kus_by_source(
        self, 
        source_id: ObjectId,
        ku_type: Optional[str] = None
    ) -> List[Dict]:
        """Get all KUs for a source"""
        query = {"source_id": source_id}
        if ku_type:
            query["ku_type"] = ku_type
        
        return list(self.knowledge_units.find(query))
    
    def count_kus(
        self,
        source_id: Optional[ObjectId] = None,
        ku_type: Optional[str] = None
    ) -> int:
        """Count knowledge units"""
        query = {}
        if source_id:
            query["source_id"] = source_id
        if ku_type:
            query["ku_type"] = ku_type
        
        return self.knowledge_units.count_documents(query)
    
    # ==================== VECTOR SEARCH OPERATIONS ====================
    
    def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        ku_types: Optional[List[str]] = None,
        source_types: Optional[List[str]] = None,
        index_name: str = "vector_index"
    ) -> List[Dict]:
        """
        Perform vector similarity search using Atlas Vector Search
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            ku_types: Filter by KU types (e.g., ['text_chunk', 'figure'])
            source_types: Filter by source types (e.g., ['pdf', 'youtube'])
            index_name: Atlas Search index name
        
        Returns:
            List of KU documents with scores
        
        Note: Requires Atlas Search index on 'embeddings.vector' field
        """
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embeddings.vector",
                    "queryVector": query_vector,
                    "numCandidates": top_k * 10,
                    "limit": top_k
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        # Add filters if specified
        match_conditions = {}
        if ku_types:
            match_conditions["ku_type"] = {"$in": ku_types}
        if source_types:
            match_conditions["source_type"] = {"$in": source_types}
        
        if match_conditions:
            # Insert match stage after vectorSearch
            pipeline.insert(1, {"$match": match_conditions})
        
        try:
            results = list(self.knowledge_units.aggregate(pipeline))
            logger.info(f"Vector search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            logger.error("Make sure you have created the Atlas Vector Search index!")
            return []
    
    def hybrid_search(
        self,
        query_vector: List[float],
        text_query: Optional[str] = None,
        top_k: int = 10,
        vector_weight: float = 0.7
    ) -> List[Dict]:
        """
        Hybrid search combining vector similarity and text search
        
        Args:
            query_vector: Query embedding
            text_query: Optional text query for keyword matching
            top_k: Number of results
            vector_weight: Weight for vector score (0-1)
        
        Returns:
            List of KU documents with hybrid scores
        """
        # Get vector search results
        vector_results = self.vector_search(query_vector, top_k=top_k*2)
        
        if not text_query:
            return vector_results[:top_k]
        
        # Get text search results
        text_results = list(self.knowledge_units.find(
            {
                "$text": {"$search": text_query}
            },
            {
                "text_score": {"$meta": "textScore"}
            }
        ).limit(top_k*2))
        
        # Merge and rerank
        seen_ids = set()
        merged_results = []
        
        # Normalize and combine scores
        for result in vector_results:
            ku_id = result['ku_id']
            if ku_id not in seen_ids:
                result['hybrid_score'] = result.get('score', 0) * vector_weight
                merged_results.append(result)
                seen_ids.add(ku_id)
        
        for result in text_results:
            ku_id = result['ku_id']
            text_score = result.get('text_score', 0) / 10  # Normalize
            
            if ku_id not in seen_ids:
                result['hybrid_score'] = text_score * (1 - vector_weight)
                merged_results.append(result)
                seen_ids.add(ku_id)
            else:
                # Update existing entry
                for r in merged_results:
                    if r['ku_id'] == ku_id:
                        r['hybrid_score'] += text_score * (1 - vector_weight)
        
        # Sort by hybrid score
        merged_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        return merged_results[:top_k]
    
    # ==================== UTILITY OPERATIONS ====================
    
    def delete_source_and_kus(self, source_id: ObjectId) -> Dict[str, int]:
        """Delete a source and all its knowledge units"""
        # Delete KUs
        ku_result = self.knowledge_units.delete_many({"source_id": source_id})
        
        # Delete source
        source_result = self.sources.delete_one({"_id": source_id})
        
        logger.info(f"Deleted source {source_id} and {ku_result.deleted_count} KUs")
        
        return {
            "deleted_kus": ku_result.deleted_count,
            "deleted_source": source_result.deleted_count
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            "total_sources": self.sources.count_documents({}),
            "total_kus": self.knowledge_units.count_documents({}),
            "sources_by_status": {},
            "kus_by_type": {},
            "kus_by_source_type": {}
        }
        
        # Sources by status
        for status in ["pending", "processing", "completed", "failed"]:
            count = self.sources.count_documents({"status": status})
            stats["sources_by_status"][status] = count
        
        # KUs by type
        pipeline = [
            {"$group": {"_id": "$ku_type", "count": {"$sum": 1}}}
        ]
        for result in self.knowledge_units.aggregate(pipeline):
            stats["kus_by_type"][result["_id"]] = result["count"]
        
        # KUs by source type
        pipeline = [
            {"$group": {"_id": "$source_type", "count": {"$sum": 1}}}
        ]
        for result in self.knowledge_units.aggregate(pipeline):
            stats["kus_by_source_type"][result["_id"]] = result["count"]
        
        return stats
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        logger.info("MongoDB connection closed")