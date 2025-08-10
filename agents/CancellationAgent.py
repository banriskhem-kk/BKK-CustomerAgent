from agents.utils.Id_validator import OrderValidator
from typing import Dict, Any
from agents.api.OrderDataAPI import OrdersAPI
from agents.llm.llm import LLMClient
from agents.dataclass.dataclass import ChatAgentResponse, ToolCall
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrderCancellationAgent:
    def __init__(self, api: OrdersAPI, validator: OrderValidator, llm: LLMClient):
        self.api = api
        self.llm = llm
        self.validator = validator

    async def cancel_order(self, order_id: str) -> ChatAgentResponse:
        """Validates order ID and sends cancellation request to Beeceptor."""
        try:
            if not self.validator.is_valid_order_id(order_id):
                logger.warning(f"Invalid order ID {order_id} for cancellation")
                return ChatAgentResponse(
                    response="Sorry, the order ID is invalid. It should be in the format ORD-XXXX.",
                    agent="OrderCancellationAgent",
                    tool_calls=[],
                    handover="OrderCancellationAgent → OrchestratorAgent",
                )

            result = await self.api.cancel_order(order_id)
            logger.info(f"API Result: {result}")

            tool_call = ToolCall(
                tool="OrderCancellationAPI",
                input={"orderId": order_id},
                result=result,
            )

            if result.get("status") == "cancelled":
                response_text = f"Your order {order_id} has been successfully canceled."
                logger.info(f"Order {order_id} canceled successfully")
            else:
                error_msg = result.get("message", "Unknown error occurred")
                response_text = f"Failed to cancel order {order_id}: {error_msg}"
                logger.error(f"Failed to cancel order {order_id}: {error_msg}")

            return ChatAgentResponse(
                response=response_text,
                agent="OrderCancellationAgent",
                tool_calls=[tool_call],
                handover="OrderCancellationAgent → OrchestratorAgent",
            )

        except Exception as e:
            logger.error(f"Error in cancel_order: {str(e)}", exc_info=True)
            return ChatAgentResponse(
                response=f"An error occurred while canceling order {order_id}. Please try again later.",
                agent="OrderCancellationAgent",
                tool_calls=[],
                handover="OrderCancellationAgent → OrchestratorAgent",
            )
