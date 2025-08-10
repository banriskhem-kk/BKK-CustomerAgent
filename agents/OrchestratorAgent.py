from agents.CancellationAgent import OrderCancellationAgent
from agents.ProductInfoAgent import ProductInfoAgent
from agents.TrackingAgent import OrderTrackingAgent
from agents.utils.Id_validator import OrderValidator
from agents.utils.sessionmanager import Memory
from agents.rag.KnowledgeBase import ProductKnowledgeBase
from agents.api.OrderDataAPI import OrdersAPI
from agents.llm.llm import LLMClient
from agents.api.RedisClient import client
import re
import logging
import json
from typing import Dict, Any
from ollama import chat, ChatResponse
from agents.dataclass.dataclass import ChatAgentResponse


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CoreSystemOrchestrator:
    def __init__(self):
        self.validator = OrderValidator()
        self.api = OrdersAPI()
        self.llm = LLMClient()
        self.session_manager = Memory(redis_client=client)
        self.tracking_agent = OrderTrackingAgent(self.api, self.validator, self.llm)
        self.cancellation_agent = OrderCancellationAgent(
            self.api, self.validator, self.llm
        )
        self.product_info_agent = ProductInfoAgent(ProductKnowledgeBase(self.llm))

    def _detect_intent_and_extract(self, message: str, context) -> Dict[str, Any]:
        """Uses OLLAMA to detect intent and extract relevant entities like order_id."""
        try:
            content = self.llm.chat(
                prompt_name="OrchestratorPrompt", message=message, context=context
            )
            logger.info(f"OLLAMA Response: {content}")

            # Try to extract JSON from the response
            try:
                # Remove any markdown formatting
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                parsed = json.loads(content)

                # Validate the parsed response
                if "intent" not in parsed:
                    raise ValueError("Missing intent field")

                # Ensure valid intent
                valid_intents = ["cancel_order", "track_order", "product_query"]
                if parsed["intent"] not in valid_intents:
                    parsed["intent"] = "product_query"

                logger.info(f"Parsed intent: {parsed}")
                return parsed

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Error parsing OLLAMA response: {e}")
                # Fallback intent detection using keywords
                return self._fallback_intent_detection(message)

        except Exception as e:
            logger.error(f"Error in intent detection: {e}", exc_info=True)
            return self._fallback_intent_detection(message)

    def _fallback_intent_detection(self, message: str) -> Dict[str, Any]:
        """Fallback intent detection using keyword matching."""
        message_lower = message.lower()

        # Extract order ID using regex
        order_id_match = re.search(r"ORD-\d{4}", message, re.IGNORECASE)
        order_id = order_id_match.group() if order_id_match else None

        # Intent detection based on keywords
        if any(
            word in message_lower
            for word in ["cancel", "cancellation", "delete", "remove"]
        ):
            return {"intent": "cancel_order", "order_id": order_id}
        elif any(
            word in message_lower
            for word in ["track", "tracking", "status", "where", "delivery"]
        ):
            return {"intent": "track_order", "order_id": order_id}
        else:
            return {"intent": "product_query", "order_id": order_id}

    def _update_session_context(
        self, session_id: str, context_updates: Dict[str, Any]
    ) -> None:
        """Safely update session context."""
        try:
            self.session_manager.update_context(session_id, context_updates)
            logger.info(f"Updated session {session_id} context: {context_updates}")
        except Exception as e:
            logger.error(f"Error updating session context: {e}")

    def _get_session_safely(self, session_id: str) -> Dict[str, Any]:
        """Safely retrieve session data."""
        try:
            session = self.session_manager.get_session(session_id)
            return session if session else {"context": {}}
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return {"context": {}}

    async def process_request(self, session_id: str, message: str) -> ChatAgentResponse:
        """Process incoming requests, prioritizing session context and using LLM for interpretation."""
        logger.info(f"Processing request for session {session_id}: {message}")

        try:
            # Get session data safely
            session = self._get_session_safely(session_id)
            context = session.get("context", {})

            # Extract context variables
            last_action = context.get("last_action")
            last_order_id = context.get("last_order_id")
            last_product_query = context.get("last_product_query")
            last_result = context.get("last_result")

            # Prepare context for LLM
            context_summary = (
                (
                    f"Last intent: {last_action}, "
                    f"Last order ID: {last_order_id}, "
                    f"Last product query: {last_product_query}, "
                    f"Last result: {last_result}"
                )
                if last_action
                else "No previous context available."
            )

            # Use LLM to determine if message continues previous context or requires new intent
            content = self.llm.chat(
                prompt_name="GeneralPrompt",
                message=message,
                context_summary=context_summary,
            )
            logger.info(f"LLM response: {content}")

            try:
                # Remove any markdown formatting
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                parsed = json.loads(content)

                # Validate the parsed response
                if "intent" not in parsed:
                    raise ValueError("Missing intent field")

                # Ensure valid intent
                valid_intents = ["cancel_order", "track_order", "product_query"]
                if parsed["intent"] not in valid_intents:
                    parsed["intent"] = "Null"

                logger.info(f"Parsed intent: {parsed}")
                decision = parsed

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Error parsing LLM response: {e}")
                decision = {
                    "is_continuation": False,
                    "action": None,
                    "order_id": None,
                    "product_query": None,
                    "intent": "product_query",
                }

            # Parse LLM response (assuming it returns valid JSON)
            is_continuation = decision.get("is_continuation", False)
            action = decision.get("action")
            order_id = decision.get("order_id")
            product_query = decision.get("product_query")
            intent = decision.get("intent")

            # Handle continuation of previous conversation
            if is_continuation and action:
                logger.info(f"Continuing previous action: {action}")

                # Continue cancellation
                if action == "cancel_order" and last_order_id:
                    if (
                        order_id
                        or "confirm" in message.lower()
                        or "cancel" in message.lower()
                    ):
                        tracking_result = await self.tracking_agent.track_order(
                            last_order_id
                        )
                        result = await self.cancellation_agent.cancel_order(
                            last_order_id
                        )
                        if (
                            result.tool_calls
                            and result.tool_calls[0].result.get("status") == "cancelled"
                        ):
                            self._update_session_context(
                                session_id,
                                {
                                    "last_order_id": last_order_id,
                                    "last_action": "cancel_order",
                                    "last_result": result.response,
                                    "tracking_details": tracking_result,
                                },
                            )
                        return result
                    else:
                        return ChatAgentResponse(
                            response=f"Do you want to proceed with canceling order {last_order_id}? Please confirm.",
                            agent="OrchestratorAgent",
                            tool_calls=[],
                            handover="OrchestratorAgent",
                        )

                # Continue tracking
                elif action == "track_order" and last_order_id:
                    if (
                        order_id
                        or "track" in message.lower()
                        or "status" in message.lower()
                    ):
                        result = await self.tracking_agent.track_order(last_order_id)
                        if (
                            result.tool_calls
                            and result.tool_calls[0].result.get("status") != "error"
                        ):
                            self._update_session_context(
                                session_id,
                                {
                                    "last_order_id": last_order_id,
                                    "last_action": "track_order",
                                    "last_result": result.response,
                                },
                            )
                        return result
                    else:
                        return ChatAgentResponse(
                            response=f"Do you want to check the status of order {last_order_id} again?",
                            agent="OrchestratorAgent",
                            tool_calls=[],
                            handover="OrchestratorAgent",
                        )

                # Continue product query
                elif action == "product_query" and last_product_query:
                    if product_query or last_product_query in message.lower():
                        result = self.product_info_agent.answer_query(message)
                        self._update_session_context(
                            session_id,
                            {
                                "last_product_query": last_product_query,
                                "last_action": "product_query",
                                "last_result": result.response,
                            },
                        )
                        return result
                    else:
                        return ChatAgentResponse(
                            response=f"Would you like more information about {last_product_query}?",
                            agent="OrchestratorAgent",
                            tool_calls=[],
                            handover="OrchestratorAgent",
                        )

            # Handle new intent from LLM
            logger.info(f"Processing new intent: {intent}")
            if intent == "cancel_order":
                if order_id:
                    tracking_result = await self.tracking_agent.track_order(order_id)
                    result = await self.cancellation_agent.cancel_order(order_id)
                    if (
                        result.tool_calls
                        and result.tool_calls[0].result.get("status") == "cancelled"
                    ):
                        self._update_session_context(
                            session_id,
                            {
                                "last_order_id": order_id,
                                "last_action": "cancel_order",
                                "last_result": result.response,
                                "tracking_details": tracking_result,
                            },
                        )
                    return result
                else:
                    return ChatAgentResponse(
                        response="Please provide an order ID (e.g., ORD-1234) to cancel your order.",
                        agent="OrchestratorAgent",
                        tool_calls=[],
                        handover="OrchestratorAgent",
                    )

            elif intent == "track_order":
                if order_id:
                    result = await self.tracking_agent.track_order(order_id)
                    if (
                        result.tool_calls
                        and result.tool_calls[0].result.get("status") != "error"
                    ):
                        self._update_session_context(
                            session_id,
                            {
                                "last_order_id": order_id,
                                "last_action": "track_order",
                                "last_result": result.response,
                            },
                        )
                    return result
                else:
                    return ChatAgentResponse(
                        response="Please provide an order ID (e.g., ORD-1234) to track your order.",
                        agent="OrchestratorAgent",
                        tool_calls=[],
                        handover="OrchestratorAgent",
                    )

            elif intent == "product_query":
                result = self.product_info_agent.answer_query(message)
                product_keywords = [
                    "bluetooth headphones",
                    "laptop",
                    "smartphone",
                    "tablet",
                ]
                detected_keyword = next(
                    (kw for kw in product_keywords if kw in message.lower()), None
                )
                if detected_keyword:
                    self._update_session_context(
                        session_id,
                        {
                            "last_product_query": detected_keyword,
                            "last_action": "product_query",
                            "last_result": result.response,
                        },
                    )
                return result

            # Fallback for unrecognized intents
            else:
                return ChatAgentResponse(
                    response="I'm not sure how to help with that. You can ask me to:\n"
                    "• Cancel an order (provide order ID like ORD-1234)\n"
                    "• Track an order status\n"
                    "• Get information about our products",
                    agent="OrchestratorAgent",
                    tool_calls=[],
                    handover="OrchestratorAgent",
                )

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return ChatAgentResponse(
                response="Sorry, something went wrong. Please try again later.",
                agent="OrchestratorAgent",
                tool_calls=[],
                handover="OrchestratorAgent",
            )

    async def shutdown(self):
        """Cleanup resources."""
        try:
            await self.api.close()
            logger.info("Orchestrator shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
