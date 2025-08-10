# Fixed BeeCeptorAPI.py
from typing import Dict, Any, Optional
import httpx
import logging
from dotenv import load_dotenv
import os
from datetime import datetime, timezone
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrdersAPI:
    def __init__(self, base_url: str = None, timeout: float = 30.0, max_retries: int = 3):  # type: ignore
        """
        Initialize Beeceptor API client.

        Args:
            base_url: API base URL (from env if not provided)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.base_url = base_url or os.getenv("BASE_URL")
        if not self.base_url:
            raise ValueError(
                "BASE_URL must be provided either as parameter or environment variable"
            )

        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
        logger.info(f"BeeceptorOrderAPI initialized with base_url: {self.base_url}")

    def _normalize_response_status(
        self, api_response: Dict[str, Any], operation: str
    ) -> Dict[str, Any]:
        """Normalize API response status for consistency."""
        # If the response already has our standard format, return it
        if "status" in api_response and api_response["status"] in [
            "success",
            "error",
            "cancelled",
            "shipped",
            "processing",
            "delivered",
        ]:
            return api_response

        # Handle different response formats from Beeceptor
        response = api_response.copy()

        # For cancellation responses
        if operation == "cancel":
            if (
                response.get("cancelled") is True
                or response.get("status") == "cancelled"
            ):
                response["status"] = "cancelled"
                response["message"] = response.get(
                    "message", "Order cancelled successfully"
                )
            elif "error" in response or response.get("status") == "error":
                response["status"] = "error"
                response["message"] = response.get(
                    "message", response.get("error", "Cancellation failed")
                )
            else:
                # Assume success if no error indicators
                response["status"] = "cancelled"
                response["message"] = response.get(
                    "message", "Order cancelled successfully"
                )

        # For tracking responses
        elif operation == "track":
            if "status" not in response:
                # Try to infer status from common fields
                if response.get("cancelled"):
                    response["status"] = "cancelled"
                elif response.get("shipped"):
                    response["status"] = "shipped"
                elif response.get("delivered"):
                    response["status"] = "delivered"
                else:
                    response["status"] = "processing"  # Default status

            # Ensure estimated_delivery is present
            if "estimated_delivery" not in response:
                response["estimated_delivery"] = response.get(
                    "delivery_date", "Not available"
                )

        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    )
    async def _make_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request with retry logic."""
        logger.debug(f"Making {method} request to {url}")
        response = await self.client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order with comprehensive error handling and validation."""
        logger.info(f"Attempting to cancel order {order_id}")

        if not order_id or not order_id.strip():
            return {
                "status": "error",
                "message": "Order ID cannot be empty",
                "order_id": order_id,
            }

        try:
            # Step 1: Get order details first
            logger.debug(f"Fetching order details for {order_id}")
            get_url = f"{self.base_url}/orders/{order_id}"

            try:
                response = await self._make_request("GET", get_url)
                order_data = response.json()
                logger.debug(f"Order details for {order_id}: {order_data}")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning(f"Order {order_id} not found")
                    return {
                        "status": "error",
                        "message": f"Order {order_id} not found",
                        "order_id": order_id,
                    }
                elif e.response.status_code == 403:
                    return {
                        "status": "error",
                        "message": "Access denied to order information",
                        "order_id": order_id,
                    }
                else:
                    logger.error(
                        f"HTTP error fetching order {order_id}: {e.response.status_code}"
                    )
                    return {
                        "status": "error",
                        "message": f"Could not retrieve order information (HTTP {e.response.status_code})",
                        "order_id": order_id,
                    }

            # Step 2: Validate order timing
            order_time_str = (
                order_data.get("placed_at")
                or order_data.get("created_at")
                or order_data.get("order_date")
            )

            if order_time_str:
                try:
                    # Handle various datetime formats
                    if order_time_str.endswith("Z"):
                        order_time_str = order_time_str[:-1] + "+00:00"
                    elif "+" not in order_time_str and "T" in order_time_str:
                        order_time_str += "+00:00"

                    order_time = datetime.fromisoformat(order_time_str)
                    current_time = datetime.now(timezone.utc)
                    time_diff = current_time - order_time
                    total_hours = time_diff.total_seconds() / 3600

                    logger.info(
                        f"Order {order_id} was placed {time_diff} ago ({total_hours:.2f} hours)"
                    )

                    # Check 24-hour cancellation window
                    if total_hours > 24:
                        logger.warning(
                            f"Order {order_id} is too old to cancel ({total_hours:.2f} hours ago)"
                        )
                        return {
                            "status": "error",
                            "message": f"Order can only be cancelled within 24 hours. This order was placed {int(total_hours)} hours ago.",
                            "order_id": order_id,
                            "hours_since_order": round(total_hours, 2),
                        }

                except ValueError as e:
                    logger.warning(
                        f"Could not parse order timestamp '{order_time_str}' for {order_id}: {e}"
                    )
                    # Continue with cancellation attempt despite timestamp parsing error
            else:
                logger.warning(
                    f"No timestamp found for order {order_id}, proceeding with cancellation attempt"
                )

            # Step 3: Check if order is already cancelled or completed
            current_status = order_data.get("status", "").lower()
            if current_status in ["cancelled", "canceled"]:
                return {
                    "status": "error",
                    "message": f"Order {order_id} is already cancelled",
                    "order_id": order_id,
                }
            elif current_status in ["shipped", "delivered", "completed"]:
                return {
                    "status": "error",
                    "message": f"Order {order_id} cannot be cancelled as it has been {current_status}",
                    "order_id": order_id,
                }

            # Step 4: Attempt cancellation
            logger.info(f"Proceeding with cancellation of order {order_id}")
            cancel_url = f"{self.base_url}/orders/{order_id}"

            try:
                response = await self._make_request("DELETE", cancel_url)
                result = response.json()
                logger.info(f"Cancellation response for {order_id}: {result}")

                # Normalize the response
                normalized_result = self._normalize_response_status(result, "cancel")
                normalized_result["order_id"] = order_id

                if normalized_result.get("status") == "cancelled":
                    logger.info(f"Successfully cancelled order {order_id}")
                else:
                    logger.warning(
                        f"Cancellation may have failed for {order_id}: {normalized_result}"
                    )

                return normalized_result

            except httpx.HTTPStatusError as e:
                error_msg = f"Cancellation failed with HTTP {e.response.status_code}"
                if e.response.status_code == 409:
                    error_msg = "Order cannot be cancelled due to its current status"
                elif e.response.status_code == 404:
                    error_msg = f"Order {order_id} not found for cancellation"

                logger.error(f"HTTP error cancelling order {order_id}: {error_msg}")
                return {
                    "status": "error",
                    "message": error_msg,
                    "order_id": order_id,
                    "http_status": e.response.status_code,
                }

        except httpx.RequestError as e:
            logger.error(f"Network error processing order {order_id}: {e}")
            return {
                "status": "error",
                "message": "Could not connect to order service. Please try again later.",
                "order_id": order_id,
                "error_type": "network_error",
            }
        except Exception as e:
            logger.exception(f"Unexpected error cancelling order {order_id}")
            return {
                "status": "error",
                "message": "An unexpected error occurred. Please contact support.",
                "order_id": order_id,
                "error_type": "internal_error",
            }

    async def track_order(self, order_id: str) -> Dict[str, Any]:
        """Track an order with comprehensive error handling."""
        logger.info(f"Tracking order {order_id}")

        if not order_id or not order_id.strip():
            return {
                "status": "error",
                "message": "Order ID cannot be empty",
                "order_id": order_id,
            }

        try:
            track_url = f"{self.base_url}/orders/{order_id}"

            try:
                response = await self._make_request("GET", track_url)
                result = response.json()
                logger.info(f"Tracking response for {order_id}: {result}")

                # Normalize the response
                normalized_result = self._normalize_response_status(result, "track")
                normalized_result["order_id"] = order_id

                # Validate required fields
                if "status" not in normalized_result:
                    normalized_result["status"] = "unknown"
                if "estimated_delivery" not in normalized_result:
                    normalized_result["estimated_delivery"] = "Not available"

                logger.info(
                    f"Successfully tracked order {order_id}: status = {normalized_result.get('status')}"
                )
                return normalized_result

            except httpx.HTTPStatusError as e:
                error_msg = f"Could not retrieve order information"

                if e.response.status_code == 404:
                    error_msg = f"Order {order_id} not found"
                elif e.response.status_code == 403:
                    error_msg = "Access denied to order information"
                elif e.response.status_code == 500:
                    error_msg = "Order service is currently unavailable"

                logger.error(
                    f"HTTP error tracking order {order_id}: {error_msg} (Status: {e.response.status_code})"
                )
                return {
                    "status": "error",
                    "message": error_msg,
                    "order_id": order_id,
                    "http_status": e.response.status_code,
                }

        except httpx.RequestError as e:
            logger.error(f"Network error tracking order {order_id}: {e}")
            return {
                "status": "error",
                "message": "Could not connect to order service. Please try again later.",
                "order_id": order_id,
                "error_type": "network_error",
            }
        except Exception as e:
            logger.exception(f"Unexpected error tracking order {order_id}")
            return {
                "status": "error",
                "message": "An unexpected error occurred while tracking your order.",
                "order_id": order_id,
                "error_type": "internal_error",
            }

    async def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            health_url = f"{self.base_url}/health"
            response = await self._make_request("GET", health_url)
            logger.info("API health check passed")
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "base_url": self.base_url,
            }
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "base_url": self.base_url,
            }

    async def close(self):
        """Clean up HTTP client resources."""
        try:
            if self.client:
                await self.client.aclose()
                logger.info("BeeceptorOrderAPI client closed successfully")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {e}")

    def __del__(self):
        """Ensure client is closed on garbage collection."""
        if hasattr(self, "client") and self.client and not self.client.is_closed:
            try:
                # Create a new event loop if none exists (for cleanup)
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if not loop.is_closed():
                    loop.run_until_complete(self.close())
            except Exception as e:
                logger.warning(f"Could not properly close client in destructor: {e}")
