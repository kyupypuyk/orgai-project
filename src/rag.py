import os
import re
import json
from typing import List, Dict, TypedDict
from datetime import date, datetime

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph
from custom_history import OriginalNeo4jChatMessageHistory

from langgraph.graph import StateGraph, END

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "None")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "None")

if NEO4J_PASSWORD == "None" or GOOGLE_API_KEY == "None":
    raise ValueError("'.env'ファイルにNEO4J_PASSWORDとGOOGLE_API_KEYを設定してください。")

llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# グラフスキーマ定義
GRAPH_SCHEMA = """
# ノード ラベルと主要なプロパティ
- Team(id, title, contents): 組織やチームを表す。
- Member(id, title, grade, skills): メンバーを表す。titleは氏名。
- Project(id, title, contents, status): プロジェクトを表す。
- Task(id, title, contents, progress, due_date): タスクを表す。
- Meeting(id, title, contents, date): ミーティング議事録を表す。

# Teamからのリレーション
(:Team)-[:HAS_REPRESENTATIVE]->(:Member)
(:Team)-[:HAS_OPERATOR]->(:Member)
(:Team)-[:HAS_MEMBER]->(:Member)
(:Team)-[:HAS_SUB_TEAM]->(:Team)
(:Team)-[:RELATED_TO_MEETING]->(:Meeting)
(:Team)-[:RELATED_TO_PROJECT]->(:Project)
(:Team)-[:CHILD_OF]->(:Team)
# Memberからのリレーション
(:Member)-[:BELONGS_TO]->(:Team)
(:Member)-[:OPERATES]->(:Team)
(:Member)-[:REPRESENTS]->(:Team)
(:Member)-[:PARTICIPATES_IN]->(:Project)
(:Member)-[:REPRESENTS]->(:Project)
(:Member)-[:ATTENDS]->(:Meeting)
# Projectからのリレーション
(:Project)-[:HAS_OWNER]->(:Member)
(:Project)-[:HAS_MEMBER]->(:Member)
(:Project)-[:RELATED_TO]->(:Team)
(:Project)-[:RELATED_TO]->(:Task)
(:Project)-[:RELATED_TO]->(:Meeting)
(:Project)-[:HAS_SUB_PROJECT]->(:Project)
(:Project)-[:CHILD_OF]->(:Project)
# Taskからのリレーション
(:Task)-[:ASSIGNED_TO]->(:Member)
(:Task)-[:PART_OF]->(:Project)
# Meetingからのリレーション
(:Meeting)-[:ATTENDED_BY]->(:Member)
(:Meeting)-[:RELATED_TO]->(:Team)
(:Meeting)-[:DISCUSSED]->(:Project)
"""

# Cypher生成のためのFew-Shotプロンプト
CYPHER_FEW_SHOT_EXAMPLES = """
# 質問: 広報部に所属しているメンバーは誰ですか？
# 思考: Teamノードを'広報部'というtitleで探し、そこからBELONGS_TOリレーションで繋がるMemberノードを見つける。
MATCH (:Team {title: '広報部'})<-[:BELONGS_TO]-(m:Member)
RETURN m.title AS `メンバー名`, m.grade AS `学年`

# 質問: Class-Navi開発プロジェクトに参加しているメンバーが持つスキルを教えて。
# 思考: Projectノードを長いtitleで正確に探し、PARTICIPATES_INリレーションで繋がるMemberノードのスキルを返す。
MATCH (:Project {title: '学内講義レビューアプリ「Class-Navi」開発'})<-[:PARTICIPATES_IN]-(m:Member)
RETURN m.title AS `メンバー名`, m.skills AS `スキル`

# 質問: Pythonのスキルを持つメンバーを教えてください。
# 思考: Memberノードのskillsプロパティに 'Python' という文字列が含まれているかを CONTAINS で検索する。
MATCH (m:Member)
WHERE m.skills CONTAINS 'Python'
RETURN m.title AS `メンバー名`, m.grade AS `学年`

# 質問: 佐藤さんが所属しているチームを教えて。
# 思考: ユーザーは苗字だけで質問している可能性が高い。Memberノードのtitleプロパティに '佐藤' という文字列が含まれるかを CONTAINS で検索し、そのメンバーが所属するチームを調べる。
MATCH (m:Member)-[:BELONGS_TO]->(t:Team)
WHERE m.title CONTAINS '佐藤'
RETURN m.title AS `メンバー名`, t.title AS `所属チーム`

# 質問: 鈴木さんについて詳しく教えて
# 思考: ユーザーは苗字だけで人物に関する包括的な情報を求めている。まず CONTAINS でMemberノードを特定する。その人物を説明するのに適したプロパティ（氏名、学年、詳細な自己紹介、スキルなど）を複数返してあげるのが親切だろう。
MATCH (m:Member)
WHERE m.title CONTAINS '鈴木'
RETURN m.title AS `氏名`, m.grade AS `学年`, m.contents AS `詳細`, m.skills AS `スキル`

# 質問: 各チームのメンバー数を教えてください。
# 思考: 全てのTeamノードを取得し、OPTIONAL MATCHで所属メンバーを探す。メンバーがいないチームも0人と表示するためにOPTIONALを使い、チームごとにメンバーをcountで集計する。
MATCH (t:Team)
OPTIONAL MATCH (t)<-[:BELONGS_TO]-(m:Member)
RETURN t.title AS `チーム名`, count(m) AS `メンバー数`
ORDER BY `メンバー数` DESC

# 質問: アプリ部のメンバーを教えて
# 思考: 'アプリ部'は正式名称ではないかもしれない。チームの正式名称(title)だけでなく、説明文(contents)にその名前が含まれている可能性も考慮して検索するべきだ。
MATCH (t:Team)<-[:BELONGS_TO]-(m:Member)
WHERE t.title CONTAINS 'アプリ' OR t.contents CONTAINS 'アプリ'
RETURN t.title AS `チーム名`, m.title AS `メンバー名`

# 質問: Class-Naviプロジェクトに関する直近のミーティングで何が決まりましたか？
# 思考: Projectノード 'Class-Navi' を起点に、DISCUSSEDリレーションで繋がるMeetingノードを探す。最新の情報を得るため、dateで降順に並び替え、LIMIT 1で1件に絞る。
MATCH (p:Project {title: '学内講義レビューアプリ「Class-Navi」開発'})<-[:DISCUSSED]-(m:Meeting)
RETURN m.title AS `ミーティング名`, m.date AS `日付`, m.contents AS `議事録内容`
ORDER BY m.date DESC
LIMIT 1

# 質問: 新歓ハッカソンの企画はどのチームが関連していて、その担当者は誰ですか？
# 思考: Project '新入生歓迎ハッカソン2025企画' を起点に、まずRELATED_TOリレーションで関連チームを探す。次に、そのプロジェクトのHAS_OWNERリレーションで担当者(オーナー)を探す。WITH句で中間結果を整理する。
MATCH (p:Project {title: '新入生歓迎ハッカソン2025企画'})
MATCH (p)-[:RELATED_TO_TEAM]->(team:Team)
MATCH (p)-[:HAS_OWNER]->(owner:Member)
RETURN p.title AS `プロジェクト名`, collect(DISTINCT team.title) AS `関連チーム`, owner.title AS `責任者`

# 質問: 誰も担当者が決まっていないタスクはありますか？
# 思考: 全てのTaskノードを探し、そのタスクからMemberへのASSIGNED_TOリレーションが存在しない(NOT EXISTS)ものをフィルタリングする。
MATCH (t:Task)
WHERE NOT EXISTS ((t)-[:ASSIGNED_TO]->(:Member))
RETURN t.title AS `未割り当てタスク名`, t.priority AS `優先度`, t.due_date AS `期限`

# 質問: 鈴木さんが参加しているプロジェクトと、彼に割り当てられているタスクをすべて教えて。
# 思考: Member '鈴木 太郎' を起点に、PARTICIPATES_INリレーションでProjectを、ASSIGNED_TOリレーションでTaskをそれぞれ探す。両方の情報を集約して返すために、OPTIONAL MATCHとcollect関数を利用する。
MATCH (m:Member {title: '鈴木 太郎'})
OPTIONAL MATCH (m)-[:PARTICIPATES_IN]->(p:Project)
OPTIONAL MATCH (m)<-[:ASSIGNED_TO]-(t:Task)
RETURN m.title AS `メンバー名`, collect(DISTINCT p.title) AS `参加プロジェクト`, collect(DISTINCT t.title) AS `担当タスク`
"""

# Cypher生成用プロンプトテンプレート
cypher_generation_prompt = PromptTemplate(
    template="""あなたは、人間の質問を構造化されたNeo4j Cypherクエリに変換するエキスパートです。
あなたの仕事は、提供された情報のみに基づき、ユーザーの質問に答えるための単一の、正確かつ効率的なCypherクエリを生成することです。

**指示:**
1.  **思考の連鎖(Chain-of-Thought):** まず、ユーザーの質問を小さなステップに分解してください。どのノードラベルが必要か、それらをどのリレーションシップで繋ぐべきか、どのプロパティでフィルタリングする必要があるかを頭の中で組み立てます。
2.  **スキーマの厳守:** 提供されたグラフスキーマに厳密に従ってください。スキーマにないラベル、リレーションシップ、プロパティは絶対に使用しないでください。
3.  **コンテキストの活用:** `initial_context` は、質問に関連する可能性が高いエンティティのリストです。クエリを作成する際の出発点や、WHERE句の条件を特定するために積極的に活用してください。特に `title` プロパティでの完全一致検索の参考にしてください。
4.  **柔軟な文字列検索:** ユーザーの質問が曖昧な場合や、スキル名・キーワードを含むエンティティを探す場合は、完全一致（`=`）だけでなく `CONTAINS` 演算子を効果的に使用してください。
5.  **複雑なクエリの構築:** 複数の情報を組み合わせる必要がある場合は、`MATCH` を複数回使用したり、`WITH` 句で中間結果を次の句に渡したり、`OPTIONAL MATCH` で関連情報がない場合も結果に含めたりするなど、高度なクエリ技術を駆使してください。集計（`count`, `collect`など）も必要に応じて使用してください。
6.  **可読性の高い出力:** ユーザーが結果を理解しやすいように、`AS` を使って列名に意味のある日本語エイリアスを付けてください（例: `m.title AS メンバー名`）。
7.  **出力はクエリのみ:** 生成するCypherクエリのみを返してください。説明、前置き、コードブロックのマークダウン（```cypher）は一切含めないでください。

**グラフスキーマ:**
{schema}

**良いCypherクエリの例 (思考プロセス付き):**
{examples}

**初期コンテキスト (関連ノード):**
{initial_context}

**ユーザーの質問:**
{question}

**生成するCypherクエリ:**
""",
    input_variables=["schema", "question", "initial_context", "examples"],
)

# 最終回答生成用プロンプトテンプレート
final_answer_prompt = ChatPromptTemplate.from_template(
    """あなたは親切なAIアシスタントです。提供されたコンテキスト情報を使って、ユーザーの質問に包括的かつ分かりやすく回答してください。

**指示:**
1.  提供された「グラフからの検索結果」を最優先の拠り所として回答を生成します。
2.  検索結果がJSON形式の場合、その構造を理解し、人間が読みやすい文章に変換してください。
3.  回答は、質問に対して直接的かつ具体的に答えるようにしてください。
4.  情報が見つからなかった場合や、コンテキストが質問と関連性が低い場合は、正直に「関連する情報が見つかりませんでした。」と伝えてください。
5.  過去の会話履歴を参考に、文脈に沿った回答を心がけてください。

**過去の会話履歴:**
{chat_history}

**実行されたCypherクエリ:**
```cypher
{cypher_query}

**グラフからの検索結果:**
{graph_context}

**ユーザーの質問:**
{question}

**回答:**
"""
)

class AgentState(TypedDict):
    """ワークフローの状態を管理するクラス"""
    question: str
    initial_context: str
    cypher_query: str
    graph_context: str
    answer: str
    chat_history: List[BaseMessage]

def custom_hybrid_search(question: str, top_k: int = 5) -> str:
    """
    ベクトル検索とキーワード検索を組み合わせたハイブリッド検索。

    Args:
        question (str): ユーザーからの質問文。
        top_k (int): 各検索方法で取得する上位件数。

    Returns:
        str: LLMのコンテキストとして利用可能な形式の文字列。
    """
    print(f"\n--- ハイブリッド検索を開始 ---")
    print(f"質問: {question}")

    vector_index_name = "entity_embedding_index"
    fulltext_index_name = "entity_fulltext_index"

    base_query = """
    YIELD node, score
    RETURN elementId(node) AS element_id, properties(node) AS properties, labels(node) AS labels, score
    """
    vector_query = f"CALL db.index.vector.queryNodes($index_name, $top_k, $question_vector) {base_query}"
    keyword_query = f"CALL db.index.fulltext.queryNodes($index_name, $question_text, {{limit: $top_k}}) {base_query}"

    print(f"\n1. ベクトル検索を実行中 (Index: {vector_index_name})...")
    question_vector = embeddings.embed_query(question)
    try:
        vector_results = graph.query(vector_query, params={
            "index_name": vector_index_name,
            "top_k": top_k,
            "question_vector": question_vector
        })
    except Exception as e:
        print(f"  [エラー] ベクトル検索に失敗しました: {e}")
        vector_results = []
    print(f"  -> {len(vector_results)} 件の結果を取得しました。")

    print(f"\n2. キーワード検索を実行中 (Index: {fulltext_index_name})...")
    try:
        keyword_results = graph.query(keyword_query, params={
            "index_name": fulltext_index_name,
            "question_text": question,
            "top_k": top_k
        })
    except Exception as e:
        print(f"  [エラー] キーワード検索に失敗しました: {e}")
        keyword_results = []
    print(f"  -> {len(keyword_results)} 件の結果を取得しました。")
    
    print("\n3. 検索結果をマージしています...")
    merged_results = {}
    all_results = vector_results + keyword_results
    
    for result in all_results:
        node_id = result['element_id']
        if node_id not in merged_results:
            merged_results[node_id] = result

    print(f"  -> 重複排除後、{len(merged_results)} 件のユニークなノードを取得しました。")

    context_list = []
    for node_id, result_data in merged_results.items():
        labels = result_data.get('labels', [])
        properties = result_data.get('properties', {})
        
        labels_str = ":".join(labels) if labels else "Entity"
        title = properties.get('title', 'N/A')
        
        context_list.append(f"Node(labels=:{labels_str}, title='{title}')")

    initial_context = "\n".join(context_list)
    return initial_context

def initial_retrieval(state: AgentState) -> Dict[str, str]:
    """ハイブリッド検索で初期ノード群を取得する"""
    print("---ステップ1: 初期探索 (カスタムハイブリッド検索)---")
    question = state["question"]
    initial_context = custom_hybrid_search(question, top_k=5)
    print(f"\n生成された初期コンテキスト:\n{initial_context}")
    
    if not initial_context:
        return {"initial_context": "初期探索では関連情報が見つかりませんでした。"}
    return {"initial_context": initial_context}

def cypher_planning(state: AgentState) -> Dict[str, str]:
    """LLMが探索計画としてのCypherクエリを生成する"""
    print("---ステップ2: 探索計画 (Cypher生成)---")
    
    cypher_generation_chain = (
        {
            "schema": lambda x: GRAPH_SCHEMA,
            "question": lambda x: x["question"],
            "initial_context": lambda x: x["initial_context"],
            "examples": lambda x: CYPHER_FEW_SHOT_EXAMPLES,
        }
        | cypher_generation_prompt
        | llm
        | StrOutputParser()
    )
    
    cypher_query = cypher_generation_chain.invoke({
        "question": state["question"],
        "initial_context": state["initial_context"]
    })
    
    # LLMが生成するコードブロック記法を削除
    cypher_query = re.sub(r"```cypher\n|```", "", cypher_query).strip()
    print(f"生成されたCypherクエリ:\n{cypher_query}")
    return {"cypher_query": cypher_query}

def json_serial(obj):
    """JSON-serializableでないオブジェクト（datetimeなど）を処理する"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError (f"Type {type(obj)} not serializable")

def graph_execution(state: AgentState) -> Dict[str, str]:
    """生成されたCypherクエリを実行し、エラー時はフォールバック"""
    print("---ステップ3: グラフ探索実行---")
    cypher_query = state["cypher_query"]
    
    try:
        graph_data = graph.query(cypher_query)
        if not graph_data:
            graph_context = json.dumps({"result": "クエリは成功しましたが、結果は空でした。"}, ensure_ascii=False, indent=2, default=json_serial)
        else:
            graph_context = json.dumps(graph_data, ensure_ascii=False, indent=2, default=json_serial)
    except Exception as e:
        print(f"Cypherクエリの実行エラー: {e}")
        # エラー発生時は、より多くの情報をLLMに渡す
        graph_context = json.dumps({
            "error": f"Cypherクエリの実行に失敗しました: {e}",
            "failed_query": cypher_query,
            "fallback_hint": "クエリが失敗したため、代わりに初期コンテキストの情報を参考に回答してください。",
            "fallback_data": state["initial_context"]
        }, ensure_ascii=False, indent=2)
        
    print(f"グラフからの検索結果:\n{graph_context}")
    return {"graph_context": graph_context}

def final_answer_generation(state: AgentState) -> Dict[str, str]:
    """グラフ探索結果を基に最終回答を生成する"""
    print("---ステップ4: 最終回答生成---")
    
    final_answer_chain = final_answer_prompt | llm | StrOutputParser()
    
    # chat_historyのフォーマットを整える
    history_str = "\n".join(
        [f"  - Human: {msg.content}" if isinstance(msg, HumanMessage) else f"  - AI: {msg.content}"
         for msg in state.get("chat_history", [])]
    )

    answer = final_answer_chain.invoke({
        "question": state["question"],
        "cypher_query": state["cypher_query"],
        "graph_context": state["graph_context"],
        "chat_history": history_str if history_str else "（なし）"
    })
    
    return {"answer": answer}

workflow = StateGraph(AgentState)

workflow.add_node("initial_retriever", initial_retrieval)
workflow.add_node("cypher_planner", cypher_planning)
workflow.add_node("graph_executor", graph_execution)
workflow.add_node("answer_generator", final_answer_generation)

workflow.set_entry_point("initial_retriever")
workflow.add_edge("initial_retriever", "cypher_planner")
workflow.add_edge("cypher_planner", "graph_executor")
workflow.add_edge("graph_executor", "answer_generator")
workflow.add_edge("answer_generator", END)

graph_agent = workflow.compile()

def run_chat(session_id: str):
    """コマンドラインで対話を実行し、Neo4jに会話履歴を保存する関数"""
    history = OriginalNeo4jChatMessageHistory(
        graph=graph,
        session_id=session_id
    )
    print("チャットを開始します。終了するには 'exit' と入力してください。")
    while True:
        question = input("\nあなた: ")
        if question.lower() in ['exit', '終了']:
            print("チャットを終了します。")
            break
        if not question.strip():
            continue
        try:
            inputs = {
                "question": question,
                "chat_history": history.messages
            }
            result = graph_agent.invoke(inputs) # type: ignore
            answer = result.get("answer", "申し訳ありません、回答を生成できませんでした。")
            print(f"\nAI: {answer}")
            # 会話履歴を更新
            history.add_user_message(question)
            history.add_ai_message(answer)
        except Exception as e:
            print(f"\n予期せぬエラーが発生しました: {e}")

# app.pyから各ステップを直接呼び出せるように、コンポーネントをエクスポート
__all__ = [
    "graph",
    "llm",
    "graph_agent",
    "initial_retrieval",
    "cypher_planning",
    "graph_execution",
    "final_answer_prompt",
    "OriginalNeo4jChatMessageHistory",
]

if __name__ == "__main__":
    session_id = "session1"  # セッションIDは適宜変更
    run_chat(session_id)