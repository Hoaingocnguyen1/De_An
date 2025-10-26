import logging
from typing import List, Dict, Any, Optional
import voyageai

from .storage.MongoDBHandler import MongoDBHandler
from .embedder import TextEmbedder, MultimodalEmbedder
from .enrichment.clients.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

class VoyageReranker:
    """Reranking service using Voyage Rerank 2.5"""
    
    def __init__(self, api_key: str, model_name: str = "rerank-2.5"):
        voyageai.api_key = api_key
        self.model_name = model_name
        logger.info(f"VoyageReranker initialized with model: {self.model_name}")
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query
        
        Returns:
            List of dicts with 'index' and 'relevance_score'
        """
        try:
            result = voyageai.rerank(
                query=query,
                documents=documents,
                model=self.model_name,
                top_k=top_k
            )
            return result.results
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return []


class EnhancedQueryEngine:
    """Enhanced query engine with reranking and multimodal support"""
    
    def __init__(
        self,
        mongo_handler: MongoDBHandler,
        text_embedder: TextEmbedder,
        multimodal_embedder: MultimodalEmbedder,
        reranker: VoyageReranker,
        synthesis_llm: BaseLLMClient,
        use_query_cache: bool = True
    ):
        self.db = mongo_handler
        self.text_embedder = text_embedder
        self.multimodal_embedder = multimodal_embedder
        self.reranker = reranker
        self.synthesis_llm = synthesis_llm
        self.use_query_cache = use_query_cache
        self.query_cache = {}
        logger.info("✓ Enhanced Query Engine ready with reranking")

    def _prepare_documents_for_reranking(
        self, 
        contexts: List[Dict] # This is the list that contains None values
    ) -> tuple[List[str], Dict[int, Dict]]:
        """
        Prepare documents for reranking and maintain index mapping
        
        Returns:
            (document_texts, index_to_context_mapping)
        """
        documents = []
        index_mapping = {}
        
        # The 'contexts' list is what you are iterating over. It contains a None.
        for idx, ctx in enumerate(contexts):
            
            # --- FIX: ADD THIS LINE TO SKIP NONE VALUES ---
            if ctx is None:
                continue
            # ----------------------------------------------
            
            # The rest of your code will now only run when 'ctx' is a valid dictionary.
            raw_content = ctx.get('raw_content', {})
            
            # Priority: enriched summary > source text > raw text
            text = (
                ctx.get('enriched_content', {}).get('summary', '') or
                raw_content.get('source_text_for_embedding', '') or
                raw_content.get('text', '') or
                f"[{ctx.get('ku_type', 'unknown')} content]"
            )
            
            # Add metadata for better reranking
            ku_type = ctx.get('ku_type', 'unknown')
            text_with_metadata = f"[Type: {ku_type}] {text}"
            
            documents.append(text_with_metadata)
            index_mapping[idx] = ctx
        
        return documents, index_mapping

    def _format_context_for_synthesis(self, contexts: List[Dict]) -> str:
        """Create a concise, readable context string for LLM synthesis"""
        context_parts = []
        
        for idx, ctx in enumerate(contexts, 1):
            ku_type = ctx.get('ku_type', 'unknown')
            
            # Get content based on type
            raw_content = ctx.get('raw_content', {})
            enriched = ctx.get('enriched_content', {})
            
            # Build context section
            header = f"[Source {idx}] Type: {ku_type.upper()}"
            
            # Add page/time context if available
            context_meta = ctx.get('context', {})
            if page := context_meta.get('page_number'):
                header += f" | Page: {page}"
            if start_time := context_meta.get('start_time_seconds'):
                header += f" | Time: {start_time:.1f}s"
            
            # Add enriched summary if available
            content = enriched.get('summary', '')
            if not content:
                content = raw_content.get('source_text_for_embedding', '')[:500]
            
            # Add keywords if available
            keywords = enriched.get('keywords', [])
            keywords_str = f"\nKeywords: {', '.join(keywords)}" if keywords else ""
            
            context_parts.append(f"{header}\n{content}{keywords_str}")
        
        return "\n\n---\n\n".join(context_parts)

    def _format_sources_for_response(self, contexts: List[Dict]) -> List[Dict]:
        """Format sources for end-user response"""
        sources = []
        
        for idx, ctx in enumerate(contexts, 1):
            source = {
                "id": idx,
                "ku_id": ctx.get("ku_id"),
                "source_uri": ctx.get("source_uri"),
                "ku_type": ctx.get("ku_type"),
                "score": ctx.get("score", 0.0),
                "rerank_score": ctx.get("rerank_score")
            }
            
            # Add context metadata
            context_meta = ctx.get('context', {})
            if page := context_meta.get('page_number'):
                source['page'] = page
            if start_time := context_meta.get('start_time_seconds'):
                source['timestamp'] = f"{int(start_time // 60)}:{int(start_time % 60):02d}"
            
            # Add preview
            raw_content = ctx.get('raw_content', {})
            enriched = ctx.get('enriched_content', {})
            
            preview = (
                enriched.get('summary', '') or
                raw_content.get('source_text_for_embedding', '')
            )[:150]
            source['preview'] = preview + "..." if len(preview) == 150 else preview
            
            sources.append(source)
        
        return sources

    async def query(
        self, 
        question: str, 
        initial_k: int = 20,
        final_k: int = 5,
        include_multimodal: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete RAG query with reranking
        
        Args:
            question: User query
            initial_k: Number of candidates to retrieve initially
            final_k: Number of documents after reranking
            include_multimodal: Whether to search multimodal content
        
        Returns:
            Dict with 'answer' and 'sources'
        """
        # Check cache
        cache_key = f"{question}_{initial_k}_{final_k}"
        if self.use_query_cache and cache_key in self.query_cache:
            logger.info("✓ Query cache HIT!")
            return self.query_cache[cache_key]
        
        logger.info("Query cache MISS. Processing query...")
        
        # Step 1: Embed query for text search
        logger.info(f"  [1/4] Embedding query with {self.text_embedder.model_name}")
        query_vector = self.text_embedder.embed(question, input_type="query")
        if not query_vector:
            raise ValueError("Failed to embed the query")

        # Step 2: Initial retrieval
        logger.info(f"  [2/4] Retrieving top-{initial_k} candidates...")
        retrieved_chunks = self.db.vector_search(
            query_vector, 
            top_k=initial_k
        )
        
        if not retrieved_chunks:
            return {
                "answer": "I could not find any relevant information in the knowledge base.",
                "sources": []
            }

        # Step 3: Rerank with Voyage Rerank 2.5
        logger.info(f"  [3/4] Reranking with {self.reranker.model_name}...")
        documents, index_mapping = self._prepare_documents_for_reranking(retrieved_chunks)
        
        rerank_results = self.reranker.rerank(
            query=question,
            documents=documents,
            top_k=final_k
        )
        
        # Map reranked results back to contexts
        reranked_contexts = []
        for result in rerank_results:
            original_ctx = index_mapping[result['index']]
            original_ctx['rerank_score'] = result['relevance_score']
            reranked_contexts.append(original_ctx)
        
        logger.info(f"  ✓ Reranked to top-{len(reranked_contexts)} results")

        # Step 4: Synthesize answer using Gemini
        logger.info(f"  [4/4] Synthesizing answer with {self.synthesis_llm.__class__.__name__}")
        context_str = self._format_context_for_synthesis(reranked_contexts)
        
        # Create synthesis prompt
        synthesis_prompt = f"""Based on the following sources, provide a comprehensive answer to the question.

Question: {question}

Sources:
{context_str}

Instructions:
- Provide a clear, accurate answer based on the sources
- Reference specific sources when making claims (e.g., "According to Source 1...")
- If information is conflicting, mention it
- If sources don't fully answer the question, acknowledge limitations
- Be concise but thorough

Answer:"""
        
        answer = await self.synthesis_llm.synthesize_answer(
            question=synthesis_prompt,
            context=""  # Context already in prompt
        )
        
        # Format final result
        formatted_sources = self._format_sources_for_response(reranked_contexts)
        result = {
            "answer": answer,
            "sources": formatted_sources,
            "retrieval_stats": {
                "initial_candidates": initial_k,
                "reranked_results": len(reranked_contexts),
                "vector_search_model": self.text_embedder.model_name,
                "rerank_model": self.reranker.model_name,
                "synthesis_model": self.synthesis_llm.__class__.__name__
            }
        }
        
        # Cache result
        if self.use_query_cache:
            self.query_cache[cache_key] = result
        
        logger.info("✓ Query completed successfully")
        return result

    async def multimodal_query(
        self,
        question: str,
        image_query: Optional[Any] = None,
        initial_k: int = 20,
        final_k: int = 5
    ) -> Dict[str, Any]:
        """
        Execute multimodal query (text + optional image)
        Useful for queries like "Find images similar to this" or "What charts show X"
        """
        # TODO: Implement hybrid search combining text and image embeddings
        # This would use self.multimodal_embedder for image query
        pass