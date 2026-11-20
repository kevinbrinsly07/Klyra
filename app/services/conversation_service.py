import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ConversationService:
    def __init__(self, db_path: str = "data/conversations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_database(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
            }

            if "title" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN title TEXT")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id)
                        ON DELETE CASCADE
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, id)
            """)

    def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (conversation_id, timestamp, timestamp),
            )

        return conversation_id

    def conversation_exists(self, conversation_id: str) -> bool:
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

        return result is not None

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, role, content, timestamp),
            )

            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (timestamp, conversation_id),
            )

    def get_history(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> list[dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()

        rows.reverse()

        return [
            {
                "role": role,
                "content": content,
                "created_at": created_at,
            }
            for role, content, created_at in rows
        ]

    def list_conversations(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.id,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.id) AS message_count,
                    (
                        SELECT content
                        FROM messages
                        WHERE conversation_id = c.id
                        ORDER BY id DESC
                        LIMIT 1
                    ) AS last_message
                FROM conversations c
                LEFT JOIN messages m
                    ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        return [
            {
                "id": row[0],
                "created_at": row[1],
                "updated_at": row[2],
                "message_count": row[3],
                "last_message": row[4],
            }
            for row in rows
        ]
