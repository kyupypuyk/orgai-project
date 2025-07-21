from __future__ import annotations
import json
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_neo4j import Neo4jGraph

class OriginalNeo4jChatMessageHistory(BaseChatMessageHistory):
    """
    思考プロセスを永続化する機能を追加した、独自のNeo4jチャット履歴クラス。
    """
    def __init__(
        self,
        *,
        graph: Neo4jGraph,
        session_id: str,
        node_label: str = "Session",
    ):
        if not session_id:
            raise ValueError("Session ID cannot be empty or None.")
        
        self.graph = graph
        self.session_id = session_id
        self.node_label = node_label
        
        self.graph.query(
            f"MERGE (s:{self.node_label} {{id: $session_id}})",
            {"session_id": self.session_id},
        )
        
    @property
    def messages(self) -> List[BaseMessage]:
        """
        メッセージ履歴と、関連する思考プロセスを取得する。
        """
        # (この部分は変更なし)
        query = """
        MATCH (s:Session {id: $session_id})
        OPTIONAL MATCH path = (s)-[:HAS_MESSAGE]->(m:Message)<-[:HAS_NEXT_MESSAGE*0..]-(all_messages:Message)
        WITH nodes(path) as message_nodes
        UNWIND message_nodes as message_node
        WITH DISTINCT message_node
        ORDER BY message_node.timestamp ASC
        RETURN message_node.type AS type, 
               message_node.content AS content,
               message_node.thinking_process AS thinking_process
        """
        results = self.graph.query(query, {"session_id": self.session_id})
        items = []
        for record in results:
            if not record or not record.get("type"):
                continue
            additional_kwargs = {}
            thinking_process_str = record.get("thinking_process")
            if thinking_process_str:
                try:
                    additional_kwargs["thinking_process"] = json.loads(thinking_process_str)
                except json.JSONDecodeError:
                    additional_kwargs["thinking_process"] = {"error": "Failed to parse thinking process."}
            items.append({
                "type": record["type"], 
                "data": { "content": record["content"], "additional_kwargs": additional_kwargs }
            })
        return messages_from_dict(items)

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        self.clear()
        for message in messages:
            self.add_message(message)

    def add_message(self, message: BaseMessage) -> None:
        """
        メッセージを追加する。AIのメッセージの場合、思考プロセスも一緒に保存する。
        """
        # (この部分は変更なし)
        thinking_process_json = None
        if message.type == "ai" and message.additional_kwargs.get("thinking_process"):
            thinking_process_json = json.dumps(message.additional_kwargs["thinking_process"])
        query = """
        MATCH (s:Session {id: $session_id})
        OPTIONAL MATCH (s)-[r_old:HAS_MESSAGE]->(last_message:Message)
        CREATE (new_message:Message)
        SET new_message.type = $type,
            new_message.content = $content,
            new_message.timestamp = timestamp()
        FOREACH (tp IN CASE WHEN $thinking_process_json IS NOT NULL THEN [$thinking_process_json] ELSE [] END |
            SET new_message.thinking_process = tp
        )
        CREATE (s)-[:HAS_MESSAGE]->(new_message)
        FOREACH (ignored IN CASE WHEN last_message IS NOT NULL THEN [1] ELSE [] END |
            DELETE r_old
            CREATE (last_message)-[:HAS_NEXT_MESSAGE]->(new_message)
        )
        """
        self.graph.query(
            query,
            {
                "session_id": self.session_id,
                "type": message.type,
                "content": message.content,
                "thinking_process_json": thinking_process_json
            },
        )

    # ===== ここに不足していたメソッドを追記します =====
    def update_session_title(self, title: str):
        """現在のセッションにタイトルを設定（または更新）する。"""
        if not self.session_id or not title:
            return
        self.graph.query(
            f"""
            MATCH (s:{self.node_label} {{id: $session_id}})
            SET s.title = $title
            """,
            {"session_id": self.session_id, "title": title}
        )
    # ===== 追記ここまで =====

    def clear(self) -> None:
        """このセッションに関連する全てのメッセージノードとリレーションを削除する。"""
        # (この部分は変更なし)
        query = """
        MATCH (s:Session {id: $session_id})
        OPTIONAL MATCH (s)-[r_has:HAS_MESSAGE]->(m:Message)
        OPTIONAL MATCH (m)<-[:HAS_NEXT_MESSAGE*0..]-(all_messages)
        DETACH DELETE all_messages
        """
        self.graph.query(query, {"session_id": self.session_id})