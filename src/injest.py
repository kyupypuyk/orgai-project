import os
import yaml
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jVector
from pydantic import SecretStr

# .envファイルから環境変数を読み込む
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "None")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "None")

# 必須の環境変数が設定されているか確認
if NEO4J_PASSWORD == "None" or GOOGLE_API_KEY == "None":
    raise ValueError("'.env'ファイルにNEO4J_PASSWORDとGOOGLE_API_KEYを設定してください。")


def get_graph_connection():
    """Neo4jグラフへの接続を確立して返します。"""
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

def clear_database(graph: Neo4jGraph):
    """Neo4jデータベース全体をクリアします。"""
    print("データベースをクリアしています...")
    graph.query("MATCH (n) DETACH DELETE n")
    try:
        # 既存のインデックスを削除
        graph.query("DROP INDEX entity_embedding_index IF EXISTS")
    except Exception as e:
        # 初回実行時など、インデックスが存在しない場合はエラーが出ることがあるが問題ない
        print(f"インデックスを削除できませんでした（初回実行では正常です）: {e}")
    print("データベースがクリアされました。")

def create_constraints(graph: Neo4jGraph):
    """パフォーマンス向上とデータ整合性のために、ノードIDにユニーク制約を作成します。"""
    print("制約を作成しています...")
    graph.query("CREATE CONSTRAINT team_id IF NOT EXISTS FOR (n:Team) REQUIRE n.id IS UNIQUE")
    graph.query("CREATE CONSTRAINT member_id IF NOT EXISTS FOR (n:Member) REQUIRE n.id IS UNIQUE")
    graph.query("CREATE CONSTRAINT project_id IF NOT EXISTS FOR (n:Project) REQUIRE n.id IS UNIQUE")
    graph.query("CREATE CONSTRAINT task_id IF NOT EXISTS FOR (n:Task) REQUIRE n.id IS UNIQUE")
    graph.query("CREATE CONSTRAINT meeting_id IF NOT EXISTS FOR (n:Meeting) REQUIRE n.id IS UNIQUE")
    graph.query("CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE")
    print("制約が作成されました。")

def process_yaml_files(graph: Neo4jGraph, data_path: str = "./data"):
    """dataディレクトリを探索し、各YAMLファイルを処理します。"""
    print("YAMLファイルの処理を開始します...")
    
    print("ステップ1: 全ノードの作成...")
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".yml") or file.endswith(".yaml"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if not data or 'id' not in data:
                        continue
                    entity_type = os.path.basename(root)
                    create_node(graph, entity_type, data)

    print("ステップ2: 全リレーションシップの作成...")
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".yml") or file.endswith(".yaml"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if not data or 'id' not in data:
                        continue
                    entity_type = os.path.basename(root)
                    create_relationships(graph, entity_type, data)

    print("すべてのYAMLファイルの処理が完了しました。")

def create_node(graph: Neo4jGraph, entity_type: str, data: dict):
    """単一のノードを作成または更新します。リスト形式のプロパティを文字列に変換します。"""
    node_id = data.get('id')
    
    # 元の辞書を変更しないようにコピーを作成
    props = data.copy()

    # 値がリストである全てのプロパティをカンマ区切りの文字列に変換する
    for key, value in props.items():
        if isinstance(value, list):
            props[key] = ", ".join(map(str, value))

    # ノードのプロパティからリレーションシップ定義用のキーを削除
    relationship_keys = [
        'representative', 'operators', 'members', 'sub_items', 'related_meetings', 'parent_item',
        'team_belongs_to', 'team_operates', 'team_represents', 'project_participates_in',
        'project_represents', 'tasks_assigned', 'meetings_attends', 'owner', 'related_teams',
        'assignee', 'related_project', 'attendees', 'related_projects'
    ]
    final_props = {k: v for k, v in props.items() if k not in relationship_keys}

    label = entity_type.rstrip('s').capitalize()
    
    graph.query(
        f"""
        MERGE (n:{label}:Entity {{id: $id}})
        SET n += $props
        """,
        params={'id': node_id, 'props': final_props}
    )

def create_relationships(graph: Neo4jGraph, entity_type: str, data: dict):
    """データに基づいてリレーションシップを作成します。"""
    source_id = data.get('id')
    source_label = entity_type.rstrip('s').capitalize()

    def link(target_label, target_ids, rel_type):
        """target_idsがリスト、単一の値(文字列/数値)のいずれであっても対応できるようにリレーションを張る"""
        if not target_ids:
            return

        ids = []
        if isinstance(target_ids, list):
            # 値がリストの場合 (例: ['MEM-001', 'MEM-002'])
            ids = [str(item) for item in target_ids]
        else:
            # 値が単一の文字列や数値の場合 (例: 'MEM-001')
            ids = [str(target_ids)]

        # 空のIDリストになった場合は何もしない
        if not ids or (len(ids) == 1 and not ids[0]):
            return

        graph.query(
            f"""
            MATCH (source:{source_label} {{id: $source_id}})
            UNWIND $target_ids AS target_id
            MATCH (target:{target_label} {{id: target_id}})
            MERGE (source)-[:{rel_type}]->(target)
            """,
            params={'source_id': source_id, 'target_ids': ids}
        )

    # 各ラベルに応じてリレーションシップを作成
    if source_label == 'Team':
        link('Member', data.get('representative'), 'HAS_REPRESENTATIVE')
        link('Member', data.get('operators'), 'HAS_OPERATOR')
        link('Member', data.get('members'), 'HAS_MEMBER')
        link('Team', data.get('sub_items'), 'HAS_SUB_TEAM')
        link('Meeting', data.get('related_meetings'), 'RELATED_TO_MEETING')
        link('Project', data.get('related_projects'), 'RELATED_TO_PROJECT')
        if data.get('parent_item'):
            link('Team', data.get('parent_item'), 'CHILD_OF')
    elif source_label == 'Member':
        link('Team', data.get('team_belongs_to'), 'BELONGS_TO')
        link('Team', data.get('team_operates'), 'OPERATES')
        link('Team', data.get('team_represents'), 'REPRESENTS')
        link('Project', data.get('project_participates_in'), 'PARTICIPATES_IN')
        link('Project', data.get('project_represents'), 'REPRESENTS')
        link('Meeting', data.get('meetings_attends'), 'ATTENDS')
    elif source_label == 'Project':
        link('Member', data.get('owner'), 'HAS_OWNER')
        link('Member', data.get('members'), 'HAS_MEMBER')
        link('Team', data.get('related_teams'), 'RELATED_TO')
        link('Task', data.get('related_tasks'), 'RELATED_TO')
        link('Meeting', data.get('related_meetings'), 'RELATED_TO')
        link('Project', data.get('sub_items'), 'HAS_SUB_PROJECT')
        if data.get('parent_item'):
            link('Project', data.get('parent_item'), 'CHILD_OF')
    elif source_label == 'Task':
        link('Member', data.get('assignee'), 'ASSIGNED_TO')
        link('Project', data.get('related_project'), 'PART_OF')
    elif source_label == 'Meeting':
        link('Member', data.get('attendees'), 'ATTENDED_BY')
        link('Team', data.get('related_teams'), 'RELATED_TO')
        link('Project', data.get('related_projects'), 'DISCUSSED')


def create_vector_index(graph: Neo4jGraph):
    """Neo4jにスーパーラベル'Entity'に対する単一のベクトルインデックスを作成します。"""
    print("ステップ3: ベクトルインデックスの作成...")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(GOOGLE_API_KEY))
    
    try:
        Neo4jVector.from_existing_graph(
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name="entity_embedding_index",
            node_label="Entity",
            text_node_properties=["title", "contents", "skills", "grade"],
            embedding_node_property="embedding",
        )
        print("'entity_embedding_index'が正常に作成・生成されました。")
    except Exception as e:
        print(f"ベクトルインデックスの作成中にエラーが発生しました: {e}")
        raise

def create_fulltext_index(graph: Neo4jGraph):
    """キーワード検索のパフォーマンス向上のために、全文検索インデックスを作成します。"""
    print("ステップ4: 全文検索インデックスを作成しています...")
    # 検索対象としたいプロパティをリストアップ
    properties_to_index = ["title", "contents", "name", "skills"] 
    
    # 全てのEntityサブタイプに共通のインデックスを作成
    graph.query(
        f"""
        CREATE FULLTEXT INDEX entity_fulltext_index IF NOT EXISTS
        FOR (n:Entity)
        ON EACH [{', '.join(f'n.{prop}' for prop in properties_to_index)}]
        """
    )
    print("全文検索インデックスが作成されました。")    

if __name__ == "__main__":
    graph = get_graph_connection()
    clear_database(graph)
    create_constraints(graph)
    process_yaml_files(graph)
    create_vector_index(graph)
    create_fulltext_index(graph) 

    print("\n--- インジェスト処理完了 ---")
    print("Neo4jデータベースに構造化データと単一のベクトルインデックスと全文検索インデックスが生成されました。")