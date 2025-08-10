from typing import Dict, Any

# from agents.rag.ProductKnowledgeBase import ProductKnowledgeBase
from agents.rag.KnowledgeBase import ProductKnowledgeBase
from agents.dataclass.dataclass import ChatAgentResponse
import logging

logger = logging.getLogger(__name__)


class ProductInfoAgent:
    def __init__(self, knowledge_base: ProductKnowledgeBase):
        self.knowledge_base = knowledge_base

    def answer_query(self, question: str) -> ChatAgentResponse:
        """Answer product-related queries using the knowledge base."""
        try:
            logger.info(f"Processing product query: {question}")
            query_result = self.knowledge_base.query(question)

            logger.info(f" Knowledge Base Results: {query_result}")
            if query_result and query_result.answer:
                logger.info(f"Product query answered successfully")
                # Format the response using the QueryResult's answer
                response_text = query_result.answer

                return ChatAgentResponse(
                    response=response_text,
                    agent="ProductInfoAgent",
                    tool_calls=[],
                    handover="ProductInfoAgent → OrchestratorAgent",
                )
            else:
                logger.info(f"No answer found for product query: {question}")
                return ChatAgentResponse(
                    response="I don't have information about that product. Could you please provide more details or try rephrasing your question?",
                    agent="ProductInfoAgent",
                    tool_calls=[],
                    handover="ProductInfoAgent → OrchestratorAgent",
                )

        except Exception as e:
            logger.error(f"Error in answer_query: {str(e)}", exc_info=True)
            return ChatAgentResponse(
                response="I'm having trouble accessing product information right now. Please try again later.",
                agent="ProductInfoAgent",
                tool_calls=[],
                handover="ProductInfoAgent → OrchestratorAgent",
            )
