from typing import Dict, Any
from datetime import datetime
import logging
from typing import Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Memory:
    def __init__(self, redis_client, default_ttl: int = 3600):
        """
        Initialize memory manager with Redis client.

        Args:
            redis_client: Redis client instance
            default_ttl: Default time-to-live for sessions in seconds (1 hour)
        """
        self.redis_client = redis_client
        self.default_ttl = default_ttl

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data for a given session ID."""
        try:
            session_key = f"session:{session_id}"
            session_data = self.redis_client.get(session_key)

            if session_data:
                parsed_data = json.loads(session_data)
                logger.info(f"Retrieved session {session_id}")
                logger.info(f"Retrieved session data: {parsed_data}")

                return parsed_data
            else:
                # Create new session if it doesn't exist
                new_session = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "context": {},
                }
                self._save_session(session_id, new_session)
                logger.info(f"Created new session {session_id}")
                return new_session

        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None

    def update_context(self, session_id: str, context_updates: Dict[str, Any]) -> bool:
        """Update the context for a session."""
        try:
            session = self.get_session(session_id)
            if session:
                session["context"].update(context_updates)
                session["last_updated"] = datetime.now().isoformat()
                return self._save_session(session_id, session)
            return False

        except Exception as e:
            logger.error(f"Error updating context for session {session_id}: {e}")
            return False

    def _save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Save session data to Redis."""
        try:
            session_key = f"session:{session_id}"
            serialized_data = json.dumps(session_data, default=str)
            self.redis_client.setex(session_key, self.default_ttl, serialized_data)
            logger.debug(f"Saved session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            session_key = f"session:{session_id}"
            result = self.redis_client.delete(session_key)
            logger.info(f"Deleted session {session_id}")
            return bool(result)

        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False

    def extend_session(self, session_id: str, additional_ttl: int = None) -> bool:  # type: ignore
        """Extend the TTL of a session."""
        try:
            session_key = f"session:{session_id}"
            ttl = additional_ttl or self.default_ttl
            result = self.redis_client.expire(session_key, ttl)
            logger.debug(f"Extended TTL for session {session_id}")
            return bool(result)

        except Exception as e:
            logger.error(f"Error extending session {session_id}: {e}")
            return False
