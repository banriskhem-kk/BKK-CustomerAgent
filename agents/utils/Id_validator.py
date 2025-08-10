import re
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrderValidator:
    @staticmethod
    def is_valid_order_id(order_id: str) -> bool:
        is_valid = bool(re.match(r"^ORD-\d{4}$", order_id))
        logger.info(
            f"Validating order ID {order_id}: {'Valid' if is_valid else 'Invalid'}"
        )
        return is_valid
