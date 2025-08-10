from typing import Optional, List, Dict, Any, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from ollama import chat, ChatResponse
import json
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from agents.llm.llm import LLMClient
import pickle
import os
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    answer: str
    confidence: float
    sources: List[str]
    related_questions: List[str]
    answer_type: str  # direct, synthesized, fallback
    processing_time: float


@dataclass
class FAQEntry:
    question: str
    answer: str
    category: str
    keywords: List[str]
    popularity: int = 0
    last_updated: str = ""
    relevance_score: float = 0.0


class ProductKnowledgeBase:
    """Enhanced RAG system with intelligent retrieval, context awareness, and adaptive learning."""

    def __init__(
        self,
        llm: LLMClient,
        faq_file_path: str = "agents/rag/data/QA.json",
        cache_dir: str = "cache",
    ):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.faqs: List[FAQEntry] = []
        self.cache_dir = cache_dir
        self.faq_file_path = faq_file_path
        self.llm = LLMClient()

        # Query analytics and learning
        self.query_history = []
        self.query_patterns = defaultdict(int)
        self.answer_feedback = {}

        # Context and conversation awareness
        self.conversation_context = {}
        self.user_preferences = {}

        # Performance tracking
        self.performance_metrics = {
            "total_queries": 0,
            "successful_answers": 0,
            "average_confidence": 0.0,
            "common_failures": Counter(),
        }

        self._ensure_cache_dir()
        self.initialize_knowledge_base()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def initialize_knowledge_base(self):
        """Initialize FAISS index with enhanced FAQ processing."""
        start_time = datetime.now()

        try:
            # Load FAQ data
            with open(self.faq_file_path, "r", encoding="utf-8") as f:
                raw_faqs = json.load(f)

            # Process and enhance FAQ entries
            self.faqs = []
            questions_for_embedding = []

            for idx, faq_data in enumerate(raw_faqs):
                # Extract keywords and categorize
                keywords = self._extract_keywords(faq_data.get("question", ""))
                category = self._categorize_question(faq_data.get("question", ""))

                faq_entry = FAQEntry(
                    question=faq_data.get("question", ""),
                    answer=faq_data.get("answer", ""),
                    category=category,
                    keywords=keywords,
                    last_updated=datetime.now().isoformat(),
                    relevance_score=1.0,  # Default relevance
                )

                self.faqs.append(faq_entry)

                # Prepare text for embedding (question + keywords for better matching)
                embedding_text = f"{faq_entry.question} {' '.join(faq_entry.keywords)}"
                questions_for_embedding.append(embedding_text)

            # Generate embeddings
            if questions_for_embedding:
                logger.info(
                    f"Generating embeddings for {len(questions_for_embedding)} FAQs..."
                )
                embeddings = self.embedding_model.encode(
                    questions_for_embedding,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )

                # Add to FAISS index
                self.index.add(embeddings.astype(np.float32))  # type: ignore

                # Save embeddings for future use
                cache_path = os.path.join(self.cache_dir, "faq_embeddings.pkl")
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "embeddings": embeddings,
                            "faqs": self.faqs,
                            "created_at": datetime.now().isoformat(),
                        },
                        f,
                    )

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Initialized knowledge base with {len(self.faqs)} FAQs in {processing_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}", exc_info=True)
            raise

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction - can be enhanced with NLP libraries
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out common words and keep important terms
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "must",
            "shall",
            "how",
            "what",
            "where",
            "when",
            "why",
            "who",
            "which",
        }

        keywords = [word for word in words if len(word) > 2 and word not in stopwords]
        return list(set(keywords))  # Remove duplicates

    def _categorize_question(self, question: str) -> str:
        """Categorize questions into relevant topics."""
        question_lower = question.lower()

        # Category keywords mapping
        categories = {
            "shipping": [
                "ship",
                "delivery",
                "shipping",
                "deliver",
                "tracking",
                "track",
                "when",
                "arrive",
            ],
            "returns": [
                "return",
                "refund",
                "exchange",
                "warranty",
                "defective",
                "broken",
                "damaged",
            ],
            "payment": [
                "payment",
                "pay",
                "credit",
                "card",
                "billing",
                "charge",
                "cost",
                "price",
            ],
            "product": [
                "product",
                "item",
                "feature",
                "specification",
                "compatible",
                "work",
                "use",
            ],
            "account": ["account", "login", "password", "profile", "register", "sign"],
            "support": [
                "help",
                "support",
                "contact",
                "customer",
                "service",
                "issue",
                "problem",
            ],
        }

        # Count category matches
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            category_scores[category] = score

        # Return category with highest score, default to 'general'
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            return best_category[0] if best_category[1] > 0 else "general"

        return "general"

    def query(self, question: str, user_id: str = None, conversation_context: Dict = None, k: int = 3) -> QueryResult:  # type: ignore
        """Enhanced query processing with context awareness and intelligent retrieval."""
        start_time = datetime.now()

        try:
            # Update performance metrics
            self.performance_metrics["total_queries"] += 1

            # Store conversation context
            if user_id and conversation_context:
                self.conversation_context[user_id] = conversation_context

            # Preprocess query
            processed_query = self._preprocess_query(question, conversation_context)

            # Multi-stage retrieval
            retrieval_results = self._multi_stage_retrieval(processed_query, k)

            if not retrieval_results["contexts"]:
                logger.warning(f"No relevant contexts found for query: {question}")
                fallback_result = self._generate_fallback_response(question)
                return QueryResult(
                    answer=fallback_result,
                    confidence=0.3,
                    sources=[],
                    related_questions=[],
                    answer_type="fallback",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                )

            # Generate enhanced answer
            answer_result = self._generate_enhanced_answer(
                question, retrieval_results, conversation_context
            )

            # Generate related questions
            related_questions = self._generate_related_questions(
                question, retrieval_results["contexts"]
            )

            # Calculate confidence score
            confidence = self._calculate_confidence(
                question, retrieval_results, answer_result
            )

            # Update learning metrics
            self._update_learning_metrics(question, answer_result, confidence)

            processing_time = (datetime.now() - start_time).total_seconds()

            result = QueryResult(
                answer=answer_result,
                confidence=confidence,
                sources=retrieval_results["sources"],
                related_questions=related_questions,
                answer_type="direct" if confidence > 0.7 else "synthesized",
                processing_time=processing_time,
            )

            # Log successful query
            self.performance_metrics["successful_answers"] += 1
            logger.info(
                f"Query answered successfully in {processing_time:.3f}s with confidence {confidence:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing query '{question}': {e}", exc_info=True)
            self.performance_metrics["common_failures"][str(type(e).__name__)] += 1

            return QueryResult(
                answer="I'm sorry, I'm having trouble processing your question right now. Please try rephrasing or contact support for assistance.",
                confidence=0.1,
                sources=[],
                related_questions=[],
                answer_type="error",
                processing_time=(datetime.now() - start_time).total_seconds(),
            )

    def _preprocess_query(self, question: str, context: Dict = None) -> str:  # type: ignore
        """Preprocess query with context awareness and query expansion."""
        processed_query = question.strip()

        # Add context information for better retrieval
        if context:
            # Add previous product mentions
            if "product_interests" in context:
                products = context["product_interests"][:2]  # Last 2 products
                processed_query += f" related to {' '.join(products)}"

            # Add user preferences
            if "preferences" in context:
                preferences = context["preferences"]
                if "category" in preferences:
                    processed_query += f" {preferences['category']}"

        # Query expansion with synonyms
        processed_query = self._expand_query_with_synonyms(processed_query)

        return processed_query

    def _expand_query_with_synonyms(self, query: str) -> str:
        """Expand query with synonyms for better matching."""
        synonyms = {
            "buy": ["purchase", "order", "get"],
            "broken": ["damaged", "defective", "not working"],
            "fast": ["quick", "rapid", "speedy"],
            "cheap": ["affordable", "inexpensive", "budget"],
            "good": ["quality", "excellent", "best"],
        }

        words = query.lower().split()
        expanded_words = []

        for word in words:
            expanded_words.append(word)
            if word in synonyms:
                expanded_words.extend(synonyms[word])

        return " ".join(expanded_words)

    def _multi_stage_retrieval(self, query: str, k: int) -> Dict[str, Any]:
        """Multi-stage retrieval: semantic + keyword + category matching."""

        # Stage 1: Semantic similarity search
        semantic_results = self._semantic_search(query, k * 2)

        # Stage 2: Keyword matching
        keyword_results = self._keyword_search(query, k)

        # Stage 3: Category-based filtering
        category_results = self._category_search(query, k)

        # Combine and rank results
        combined_results = self._combine_retrieval_results(
            semantic_results, keyword_results, category_results, k
        )

        return combined_results

    def _semantic_search(self, query: str, k: int) -> List[Tuple[int, float, FAQEntry]]:
        """Perform semantic similarity search using FAISS."""
        if not self.faqs:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding.astype(np.float32), min(k, len(self.faqs)))  # type: ignore

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.faqs):
                # Convert distance to similarity score
                similarity = 1.0 / (1.0 + dist)
                results.append((idx, similarity, self.faqs[idx]))

        return results

    def _keyword_search(self, query: str, k: int) -> List[Tuple[int, float, FAQEntry]]:
        """Perform keyword-based search."""
        query_keywords = set(self._extract_keywords(query))

        results = []
        for idx, faq in enumerate(self.faqs):
            # Calculate keyword overlap
            faq_keywords = set(faq.keywords)
            overlap = len(query_keywords.intersection(faq_keywords))

            if overlap > 0:
                score = overlap / len(query_keywords.union(faq_keywords))
                results.append((idx, score, faq))

        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _category_search(self, query: str, k: int) -> List[Tuple[int, float, FAQEntry]]:
        """Search within the same category as the query."""
        query_category = self._categorize_question(query)

        results = []
        for idx, faq in enumerate(self.faqs):
            if faq.category == query_category:
                # Use relevance score and popularity
                score = faq.relevance_score * (1 + faq.popularity * 0.1)
                results.append((idx, score, faq))

        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _combine_retrieval_results(
        self, semantic: List, keyword: List, category: List, k: int
    ) -> Dict[str, Any]:
        """Combine different retrieval results with weighted scoring."""

        # Weight different retrieval methods
        weights = {"semantic": 0.5, "keyword": 0.3, "category": 0.2}

        # Combine scores for each FAQ
        combined_scores = defaultdict(float)
        faq_map = {}

        # Add semantic results
        for idx, score, faq in semantic:
            combined_scores[idx] += score * weights["semantic"]
            faq_map[idx] = faq

        # Add keyword results
        for idx, score, faq in keyword:
            combined_scores[idx] += score * weights["keyword"]
            faq_map[idx] = faq

        # Add category results
        for idx, score, faq in category:
            combined_scores[idx] += score * weights["category"]
            faq_map[idx] = faq

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Prepare final results
        contexts = []
        sources = []

        for idx, score in sorted_results[:k]:
            faq = faq_map[idx]
            contexts.append(f"Q: {faq.question}\nA: {faq.answer}")
            sources.append(f"FAQ #{idx}: {faq.question}")

        return {
            "contexts": contexts,
            "sources": sources,
            "scores": [score for idx, score in sorted_results[:k]],
        }

    def _generate_enhanced_answer(self, question: str, retrieval_results: Dict, context: Dict = None) -> str:  # type: ignore
        """Generate enhanced answer using retrieved contexts and conversation awareness."""

        contexts_text = "\n\n".join(retrieval_results["contexts"])

        # Build context-aware prompt
        context_info = ""
        if context:
            if "product_interests" in context:
                context_info += f"\nUser is interested in: {', '.join(context['product_interests'])}"
            if "last_action" in context:
                context_info += f"\nPrevious user action: {context['last_action']}"
            if "user_sentiment" in context:
                context_info += f"\nUser sentiment: {context['user_sentiment']}"

        try:

            answer = self.llm.chat(
                prompt_name="RAG_Prompt",
                question=question,
                contexts_text=contexts_text,
                context_info=context_info,
            )
            # answer = response["message"]["content"]
            logger.debug(f"Generated enhanced answer for: {question}")
            return answer

        except Exception as e:
            logger.error(f"Error generating enhanced answer: {e}")
            # Fallback to simple context-based answer
            return self._simple_context_answer(question, contexts_text)

    def _simple_context_answer(self, question: str, contexts: str) -> str:
        """Simple fallback answer generation."""
        if contexts:
            # Extract the most relevant answer from contexts
            context_lines = contexts.split("\n")
            answer_lines = [line for line in context_lines if line.startswith("A:")]
            if answer_lines:
                return answer_lines[0][2:].strip()  # Remove 'A:' prefix

        return "I found some information that might help, but I'm unable to provide a specific answer right now. Please contact customer support for detailed assistance."

    def _generate_related_questions(
        self, question: str, contexts: List[str]
    ) -> List[str]:
        """Generate related questions that might interest the user."""

        if not contexts:
            return []

        contexts_text = "\n".join(contexts[:2])  # Use top 2 contexts

        prompt = f"""
        Based on this user question and related FAQ contexts, suggest 2-3 related questions that the user might also want to know.
        
        User question: {question}
        
        Related contexts:
        {contexts_text}
        
        Generate questions that are:
        1. Naturally related to the original question
        2. Helpful for the user's likely needs
        3. Answerable from our product knowledge
        4. Different from the original question
        
        Return as JSON array: ["Question 1?", "Question 2?", "Question 3?"]
        """

        try:
            response: ChatResponse = chat(
                model="llama3.1:latest",
                messages=[
                    {
                        "role": "system",
                        "content": "You are helpful at suggesting related questions. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            content = response["message"]["content"]
            # Clean JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()

            related_questions = json.loads(content)
            return related_questions if isinstance(related_questions, list) else []

        except Exception as e:
            logger.error(f"Error generating related questions: {e}")
            return []

    def _calculate_confidence(
        self, question: str, retrieval_results: Dict, answer: str
    ) -> float:
        """Calculate confidence score for the generated answer."""

        # Base confidence from retrieval scores
        if retrieval_results["scores"]:
            avg_retrieval_score = sum(retrieval_results["scores"]) / len(
                retrieval_results["scores"]
            )
        else:
            avg_retrieval_score = 0.0

        # Context coverage score
        contexts = retrieval_results["contexts"]
        context_coverage = min(len(contexts) / 3, 1.0)  # Ideal is 3 contexts

        # Answer length and quality heuristics
        answer_length_score = min(len(answer) / 200, 1.0)  # Ideal is ~200 characters

        # Keyword overlap between question and answer
        question_keywords = set(self._extract_keywords(question))
        answer_keywords = set(self._extract_keywords(answer))
        keyword_overlap = len(question_keywords.intersection(answer_keywords)) / max(
            len(question_keywords), 1
        )

        # Combine factors
        confidence = (
            avg_retrieval_score * 0.4
            + context_coverage * 0.3
            + answer_length_score * 0.2
            + keyword_overlap * 0.1
        )

        return min(confidence, 1.0)

    def _generate_fallback_response(self, question: str) -> str:
        """Generate fallback response when no relevant contexts are found."""

        # Analyze the question to provide helpful guidance
        question_category = self._categorize_question(question)

        category_responses = {
            "shipping": "I don't have specific shipping information for your question, but I can help you track an order if you provide an order ID, or you can contact our shipping department for detailed delivery information.",
            "returns": "For return and refund questions, I recommend contacting our customer service team who can review your specific situation and guide you through the return process.",
            "payment": "For payment-related questions, please contact our billing department who can securely access your account and help resolve any payment issues.",
            "product": "I don't have detailed information about that specific product, but you can browse our product catalog or speak with a product specialist for detailed specifications.",
            "account": "For account-related issues, please visit our help center or contact customer support who can assist with account access and settings.",
            "support": "I want to help, but I need more information about your specific situation. Please contact our customer support team who can provide personalized assistance.",
        }

        return category_responses.get(
            question_category,
            "I don't have specific information about that topic, but our customer support team is available to help with any questions you might have.",
        )

    def _update_learning_metrics(self, question: str, answer: str, confidence: float):
        """Update learning metrics and patterns."""

        # Store query for analytics
        self.query_history.append(
            {
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence,
                "category": self._categorize_question(question),
            }
        )

        # Update pattern tracking
        question_category = self._categorize_question(question)
        self.query_patterns[question_category] += 1

        # Update average confidence
        total_confidence = self.performance_metrics["average_confidence"] * (
            self.performance_metrics["successful_answers"] - 1
        )
        self.performance_metrics["average_confidence"] = (
            total_confidence + confidence
        ) / self.performance_metrics["successful_answers"]

    def get_analytics(self) -> Dict[str, Any]:
        """Get knowledge base analytics and performance metrics."""

        # Recent query analysis
        recent_queries = (
            self.query_history[-100:]
            if len(self.query_history) > 100
            else self.query_history
        )

        # Category distribution
        category_distribution = dict(self.query_patterns)

        # Confidence trends
        confidence_trend = []
        if recent_queries:
            for i in range(0, len(recent_queries), 10):
                batch = recent_queries[i : i + 10]
                avg_confidence = sum(q["confidence"] for q in batch) / len(batch)
                confidence_trend.append(avg_confidence)

        return {
            "performance_metrics": self.performance_metrics,
            "total_faqs": len(self.faqs),
            "query_patterns": category_distribution,
            "recent_queries_count": len(recent_queries),
            "confidence_trend": confidence_trend,
            "common_categories": sorted(
                category_distribution.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "knowledge_base_health": (
                "healthy"
                if self.performance_metrics["average_confidence"] > 0.6
                else "needs_improvement"
            ),
        }

    def add_faq(self, question: str, answer: str, category: str = None) -> bool:  # type: ignore
        """Dynamically add new FAQ entry."""
        try:
            keywords = self._extract_keywords(question)
            category = category or self._categorize_question(question)

            new_faq = FAQEntry(
                question=question,
                answer=answer,
                category=category,
                keywords=keywords,
                last_updated=datetime.now().isoformat(),
                relevance_score=1.0,
            )

            # Add to FAQs list
            self.faqs.append(new_faq)

            # Generate embedding and add to index
            embedding_text = f"{question} {' '.join(keywords)}"
            embedding = self.embedding_model.encode(
                [embedding_text], convert_to_numpy=True
            )
            self.index.add(embedding.astype(np.float32))  # type: ignore

            logger.info(f"Added new FAQ: {question}")
            return True

        except Exception as e:
            logger.error(f"Error adding FAQ: {e}")
            return False
