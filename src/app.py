import streamlit as st
import uuid
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# rag.py から必要なコンポーネントをインポート
from rag import (
    graph,
    llm,
    initial_retrieval,
    cypher_planning,
    graph_execution,
    final_answer_prompt
)
from custom_history import OriginalNeo4jChatMessageHistory

# --- ヘルパー関数 ---
@st.cache_data(show_spinner=False)
def generate_chat_title(_question: str) -> str:
    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate.from_template("以下のユーザーからの質問を、チャット履歴のタイトルとして表示するための、ごく短い要約（最大20文字程度）にしてください。\n\n質問: {question}\n\n要約タイトル:")
    chain = prompt | llm | StrOutputParser()
    try:
        title = chain.invoke({"question": _question})
        return title.strip().replace('"', '').replace("'", "").replace("\n", " ")
    except Exception as e:
        print(f"タイトル生成エラー: {e}")
        return _question[:30] + "..."

def get_all_sessions():
    query = "MATCH (s:Session)-[:HAS_MESSAGE]->(latest_message:Message) ORDER BY latest_message.timestamp DESC RETURN s.id AS session_id, s.title AS title"
    try:
        results = graph.query(query)
        return [{"id": r['session_id'], "title": r.get('title') or f"会話-{r['session_id'][:8]}"} for r in results]
    except Exception as e:
        st.error(f"セッションの読み込みに失敗しました: {e}")
        return []

def create_new_session_id():
    return str(uuid.uuid4())

def display_thinking_process(tp_data):
    """思考プロセスの辞書データを見やすく表示する関数"""
    st.markdown("##### 1. 初期探索 (Hybrid Search)")
    st.code(tp_data.get("initial_context", "情報なし"), language="text")
    st.markdown("##### 2. 探索計画 (Cypher Generation)")
    st.code(tp_data.get("cypher_query", "生成されませんでした"), language="cypher")
    st.markdown("##### 3. グラフ探索結果 (JSON)")
    graph_context_str = tp_data.get("graph_context", "{}")
    try:
        # 整形されたJSON文字列を再度整形
        graph_context_formatted = json.dumps(json.loads(graph_context_str), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        graph_context_formatted = graph_context_str
    st.code(graph_context_formatted, language="json")

# --- Streamlit UI設定 ---
st.set_page_config(page_title="Graph RAG Chat", layout="wide")

# --- セッション管理 ---
if "session_id" not in st.session_state:
    all_sessions = get_all_sessions()
    st.session_state.session_id = all_sessions[0]['id'] if all_sessions else create_new_session_id()

# --- サイドバーUI ---
with st.sidebar:
    st.title("📄 チャットセッション")
    if st.button("➕ 新規チャット", use_container_width=True):
        st.session_state.session_id = create_new_session_id()
        st.rerun()
    st.divider()
    all_sessions = get_all_sessions()
    session_map = {s['id']: s['title'] for s in all_sessions}
    session_ids = [s['id'] for s in all_sessions]
    if st.session_state.session_id not in session_ids:
        session_ids.insert(0, st.session_state.session_id)
        session_map[st.session_state.session_id] = "新しいチャット"
    try:
        current_session_index = session_ids.index(st.session_state.session_id)
    except ValueError:
        current_session_index = 0
    selected_session = st.selectbox("チャット履歴", options=session_ids, format_func=lambda session_id: session_map.get(session_id, session_id), index=current_session_index)
    if selected_session != st.session_state.session_id:
        st.session_state.session_id = selected_session
        st.rerun()

# --- メインチャット画面 ---
st.title("Graph RAG Chatbot")
current_title = session_map.get(st.session_state.session_id, "新しいチャット")
st.caption(f"現在のチャット: {current_title}")

history = OriginalNeo4jChatMessageHistory(graph=graph, session_id=st.session_state.session_id)

# ===== チャット履歴の表示ロジックを修正 =====
for msg in history.messages:
    with st.chat_message(msg.type):
        # AIのメッセージで、思考プロセスが保存されている場合
        if msg.type == 'ai' and "thinking_process" in msg.additional_kwargs:
            with st.expander("思考プロセスを見る"):
                display_thinking_process(msg.additional_kwargs["thinking_process"])
        
        # 最終的な回答を表示
        st.markdown(msg.content)

# ===== ユーザー入力処理ブロックを全面的に修正 =====
if prompt := st.chat_input("質問を入力してください..."):
    st.chat_message("user").markdown(prompt)

    state = {"question": prompt, "chat_history": history.messages}
    thinking_process_data = {}

    with st.chat_message("assistant"):
        # 思考プロセスはst.status内で実行し、完了後自動で閉じる
        with st.status("思考中...") as status:
            try:
                # 各ステップを実行し、結果をローカル変数に保存
                step1_result = initial_retrieval(state) #type: ignore
                status.write("✅ 初期探索完了")
                step2_result = cypher_planning({**state, **step1_result}) #type: ignore
                status.write("✅ 探索計画完了")
                step3_result = graph_execution({**state, **step1_result, **step2_result}) #type: ignore
                status.write("✅ グラフ探索完了")

                # 全ての思考プロセスデータを一つの辞書にまとめる
                thinking_process_data = {**step1_result, **step2_result, **step3_result}
                status.update(label="思考完了！", state="complete")

            except Exception as e:
                status.update(label="エラーが発生しました", state="error")
                st.error(f"処理中にエラーが発生しました: {e}")
                st.stop()
        
        # 思考プロセスが完了した後、永続的なUIとして表示
        with st.expander("思考プロセスを見る"):
            display_thinking_process(thinking_process_data)

        # 最終回答をストリーミング表示
        final_answer_chain = final_answer_prompt | llm | StrOutputParser()
        history_str = "\n".join([f"  - {msg.type}: {msg.content}" for msg in state.get("chat_history", [])])
        chain_input = {**state, **thinking_process_data, "chat_history": history_str}
        full_response = st.write_stream(final_answer_chain.stream(chain_input))

    # --- 会話履歴の保存とUI更新 ---
    is_first_message = not history.messages
    history.add_user_message(prompt)
    
    # 思考プロセスを含んだAIメッセージを作成して保存
    ai_message_with_thinking = AIMessage(
        content=full_response,
        additional_kwargs={"thinking_process": thinking_process_data}
    )
    history.add_message(ai_message_with_thinking)

    if is_first_message:
        title = generate_chat_title(prompt)
        history.update_session_title(title)
        st.rerun()