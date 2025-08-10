from typing import Dict, Any
from agents.utils.Id_validator import OrderValidator
from agents.api.OrderDataAPI import OrdersAPI
from agents.dataclass.dataclass import ChatAgentResponse, ToolCall
from agents.llm.llm import LLMClient
import logging

logger = logging.getLogger(__name__)


class OrderTrackingAgent:
    """Handles order tracking requests via the Beeceptor API."""

    def __init__(self, api: OrdersAPI, validator: OrderValidator, llm: LLMClient):
        self.api = api
        self.validator = validator
        self.llm = llm

    async def track_order(self, order_id: str) -> ChatAgentResponse:
        """Validates order ID and retrieves order status from Beeceptor."""
        try:
            if not self.validator.is_valid_order_id(order_id):
                logger.warning(f"Invalid order ID {order_id} for tracking")
                return ChatAgentResponse(
                    response="Sorry, the order ID is invalid. It should be in the format ORD-XXXX.",
                    agent="OrderTrackingAgent",
                    tool_calls=[],
                    handover="OrderTrackingAgent → OrchestratorAgent",
                )

            result = await self.api.track_order(order_id)
            logger.info(f"Tracking API Result: {result}")

            tool_call = ToolCall(
                tool="OrderTrackingAPI",
                input={"orderId": order_id},
                result=result,
            )

            if result.get("status") and result.get("status") != "error":
                status = result.get("status", "Unknown")
                delivery = result.get("estimated_delivery", "Not available")
                response_text = f"Your order {order_id} is currently {status}. Estimated delivery: {delivery}."
                logger.info(f"Order {order_id} tracked successfully: {status}")
            else:
                error_msg = result.get(
                    "message", "Unable to retrieve order information"
                )
                response_text = f"Failed to track order {order_id}: {error_msg}"
                logger.error(f"Failed to track order {order_id}: {error_msg}")

            return ChatAgentResponse(
                response=response_text,
                agent="OrderTrackingAgent",
                tool_calls=[tool_call],
                handover="OrderTrackingAgent → OrchestratorAgent",
            )

        except Exception as e:
            logger.error(f"Error in track_order: {str(e)}", exc_info=True)
            return ChatAgentResponse(
                response=f"An error occurred while tracking order {order_id}. Please try again later.",
                agent="OrderTrackingAgent",
                tool_calls=[],
                handover="OrderTrackingAgent → OrchestratorAgent",
            )
