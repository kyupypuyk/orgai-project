# orgai-project

組織内に散在するドキュメントや情報をナレッジグラフとして構造化し、自然言語で対話的に検索・活用するためのGraph RAG（Retrieval-Augmented Generation）アプリケーションです。

このプロジェクトは、メンバー、チーム、プロジェクト、タスク、議事録などの情報をYAMLファイルで管理し、Neo4jグラフデータベースにインポートします。ユーザーはStreamlit製のチャットインターフェースを通じて質問を投げかけると、LangGraphで構築されたエージェントがグラフ構造を自律的に探索し、最適な回答を生成します。

## 主な機能

*   **データインジェスト**: `data`ディレクトリ内のYAMLファイルを解析し、自動的にNeo4jグラフデータベースにノードとリレーションシップを構築します。
*   **Graph RAGパイプライン**: LangGraphを利用して、複雑な質問にも対応可能な高度な検索・応答プロセスを実現します。
    *   **ハイブリッド検索**: ベクトル検索とキーワード検索を組み合わせ、質問に最も関連性の高い情報を初期コンテキストとして特定します。
    *   **動的Cypher生成**: LLMが質問内容とグラフスキーマを理解し、その場で最適なCypherクエリを生成します。
    *   **グラフ探索と回答生成**: 生成されたクエリを実行してグラフから正確な情報を抽出し、LLMが人間にとって分かりやすい文章で回答します。
*   **Webインターフェース**: Streamlitによる直感的なチャットUIを提供します。
*   **思考プロセスの可視化**: AIがどのように考え、どのデータを参照して回答を生成したか、その思考プロセス（初期探索、Cypherクエリ、グラフ探索結果）をUI上で確認できます。
*   **チャット履歴の永続化**: 全ての会話履歴とAIの思考プロセスはNeo4jデータベースに保存され、いつでも過去の対話を振り返ることができます。

## 技術スタック

*   **LLM / RAG Framework**: LangChain, LangGraph, Google Gemini
*   **Database**: Neo4j (Graph Database)
*   **Frontend**: Streamlit
*   **Data Handling**: Python, PyYAML
*   **Environment**: Docker, Docker Compose

## セットアップと実行方法

### 1. 前提条件

*   Git
*   Docker および Docker Compose
*   Python 3.9以上

### 2. 環境設定

まず、このリポジトリをクローンし、プロジェクトディレクトリに移動します。

```bash
git clone https://github.com/kyupypuyk/orgai-project.git
cd orgai-project
```

次に、環境変数ファイル `.env.example` をコピーして `.env` ファイルを作成します。

```bash
cp .env.example .env
```

作成した `.env` ファイルを開き、お使いのAPIキーとパスワードを設定してください。

```dotenv
# .env

# Neo4jデータベースのパスワード（任意の値を設定してください）
NEO4J_PASSWORD="your_strong_password"

# Google AI Studioで取得したAPIキー
GOOGLE_API_KEY="your_google_api_key"
```

### 3. Neo4jの起動

Docker Composeを使用してNeo4jデータベースを起動します。

```bash
docker compose up -d
```

コンテナが正常に起動したか確認してください。Neo4j Browserは `http://localhost:7474` でアクセスできます。（ユーザー名: `neo4j`, パスワード: `.env`で設定した値）

### 4. 依存関係のインストール

Pythonの仮想環境を作成し、必要なライブラリをインストールします。

```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合は `venv\Scripts\activate`
pip install -r requirements.txt
```

### 5. データインジェスト

`data`ディレクトリ内のサンプルYAMLデータをNeo4jデータベースにインポートします。このスクリプトは、データベースの初期化、制約の作成、ノードとリレーションの登録、そして検索用インデックスの構築までを自動的に行います。

```bash
python src/injest.py
```

### 6. アプリケーションの起動

準備が整いました。以下のコマンドでStreamlitアプリケーションを起動します。

```bash
streamlit run src/app.py
```

ブラウザで表示されたURL（通常は `http://localhost:8501`）にアクセスすると、チャット画面が表示されます。

---

## 開発者向け情報 (For Developers)

このプロジェクトへのコントリビュートを歓迎します。以下に内部の技術的な詳細を記します。

### プロジェクト構成

```
orgai-project/
├── data/              # 構造化データ(YAML)を格納
│   ├── meetings/
│   ├── members/
│   ├── projects/
│   ├── tasks/
│   └── teams/
├── neo4j/             # Neo4jの永続化データ (Docker Volume)
├── src/               # ソースコード
│   ├── app.py         # Streamlitアプリケーション本体
│   ├── custom_history.py # 思考プロセスを保存するカスタムチャット履歴クラス
│   ├── injest.py      # データインジェスト用スクリプト
│   └── rag.py         # Graph RAGエージェントの定義 (LangGraph)
├── .env               # 環境変数ファイル
├── docker-compose.yml # Docker Compose設定
└── requirements.txt   # Python依存ライブラリ
```

### データモデル (グラフスキーマ)

本プロジェクトのナレッジグラフは、以下のノードとリレーションシップで構成されています。データは `injest.py` によってYAMLからグラフに変換されます。

*   **ノードラベル**: `Team`, `Member`, `Project`, `Task`, `Meeting`
    *   全てのノードには共通のスーパーラベル `Entity` が付与され、検索インデックスの対象となります。
*   **リレーションシップ**:
    *   `(:Team)-[:HAS_MEMBER]->(:Member)`
    *   `(:Member)-[:BELONGS_TO]->(:Team)`
    *   `(:Project)-[:HAS_MEMBER]->(:Member)`
    *   `(:Member)-[:PARTICIPATES_IN]->(:Project)`
    *   `(:Task)-[:ASSIGNED_TO]->(:Member)`
    *   `(:Meeting)-[:ATTENDED_BY]->(:Member)`
    *   など、詳細は `rag.py` 内の `GRAPH_SCHEMA` を参照してください。

### Graph RAG パイプライン

中核となるRAGエージェントは `src/rag.py` で定義されており、LangGraphによって構築されたステートマシンとして動作します。

**`AgentState`**: ワークフローの状態を一元管理するクラス。質問、中間生成物（コンテキスト、Cypherクエリ）、最終的な回答などを保持します。

**ワークフローの各ステップ:**

1.  **`initial_retrieval` (初期探索)**:
    *   `custom_hybrid_search` 関数を実行し、ユーザーの質問に基づいて関連性の高いエンティティ（ノード）の候補をリストアップします。
    *   この検索は、意味的な類似性で検索する**ベクトル検索** (`entity_embedding_index`)と、キーワードで検索する**全文検索** (`entity_fulltext_index`)を組み合わせたハイブリッド方式です。
    *   ここで得られたコンテキストは、次のステップでLLMがクエリを生成する際の重要なヒントとなります。

2.  **`cypher_planning` (探索計画)**:
    *   LLM（Gemini）が、ユーザーの質問、グラフスキーマ、Few-shotサンプル、そして前のステップで得られた初期コンテキストを基に、Neo4jから直接的な答えを引き出すための**Cypherクエリ**を動的に生成します。
    *   プロンプト (`cypher_generation_prompt`) には、曖昧な質問への対応方法や、複雑なリレーションを辿るためのクエリの組み立て方など、質の高いクエリを生成するための様々な指示が含まれています。

3.  **`graph_execution` (グラフ探索)**:
    *   生成されたCypherクエリをNeo4jデータベースで実行します。
    *   クエリが失敗した場合でも処理が停止しないようエラーハンドリングが実装されており、失敗した旨とフォールバック用の情報をLLMに渡します。
    *   実行結果はJSON形式で整形され、次のステップに渡されます。

4.  **`final_answer_generation` (最終回答生成)**:
    *   LLMが、元の質問、実行されたCypherクエリ、そしてグラフから得られたJSON形式の結果を総合的に解釈し、ユーザーにとって自然で分かりやすい文章の最終回答を生成します。

### データインジェストプロセス (`injest.py`)

このスクリプトは、ローカルのYAMLデータをNeo4jに投入するための重要な役割を担います。

1.  **データベースのクリア**: 冪等性を担保するため、既存のデータを全て削除します。
2.  **制約の作成**: 各ノードの `id` プロパティにユニーク制約を付与し、データの整合性とパフォーマンスを向上させます。
3.  **ノードの作成**: `data` ディレクトリを再帰的に探索し、全てのYAMLファイルからノードを作成します。この際、全ノードに `Entity` ラベルが付与されます。
4.  **リレーションシップの作成**: 再度YAMLファイルを読み込み、ファイル内に記述された関連ID（例: `members: [MEM-001, MEM-002]`）に基づいてノード間のリレーションシップを構築します。
5.  **ベクトルインデックスの作成**: `Entity` ラベルを持つ全てのノードを対象に、`title`, `contents` などのテキストプロパティからベクトル埋め込みを生成し、`entity_embedding_index` という単一のベクトルインデックスを作成します。
6.  **全文検索インデックスの作成**: 同様に `Entity` ラベルを対象に、キーワード検索のパフォーマンスを向上させるための `entity_fulltext_index` を作成します。

### チャット履歴の永続化 (`custom_history.py`)

標準のチャット履歴クラスを拡張した `OriginalNeo4jChatMessageHistory` を実装しています。

*   ユーザーとAIのメッセージは、`Session` ノードに紐づく `Message` ノードとして時系列でNeo4jに保存されます。
*   特筆すべき点として、AIの回答 (`AIMessage`) には、その回答を生成するために経由した**思考プロセス** (`thinking_process` データ)がJSON形式のプロパティとして一緒に保存されます。これにより、`app.py` は過去の会話の回答だけでなく、その根拠となった思考の過程も再現・表示することが可能です。

## コントリビューション (Contributing)

バグ報告、機能改善の提案、プルリクエストを歓迎します。コントリビューションを行う際は、まずIssueを立てて提案内容について議論してください。

改善案の例:
*   Cypher生成プロンプトの改善（Few-shotサンプルの追加や指示の精緻化）
*   データモデルの拡張（新しいノードやリレーションの追加）
*   UI/UXの改善
*   テストコードの追加

## 今後の展望：より賢く、使いやすいツールへの進化

このコードは、より大きな構想を実現するための第一歩です。将来的には、現在のコア機能をさらに磨き上げ、**あなたの活動をより深く理解し、知的作業をスムーズに手助けしてくれるパートナー**へと進化させていきます。

以下に、計画している主な改善・拡張の方向性を示します。

### 1. AIの連携強化と役割分担（マルチエージェント化）

現在の仕組みを発展させ、AIたちがチームのように連携して、より複雑なリクエストに応えられるアーキテクチャを目指します。

*   **司令塔となるAI**: ユーザーの最終的な目的を理解し、タスクを分解して、それぞれの処理に最適な専門AIに動的に指示を出します。
*   **専門家AI**:
    *   **ハイブリッド検索AI**: まず関連情報を素早く見つけ出します。
    *   **グラフ探索AI**: 情報の繋がりを深く掘り下げて調査します。
    *   **Web検索AI**: 外部の公開情報を参照します。
    *   **アクション実行AI**: 他のツールでの作業を代行します。

これにより、「〇〇について調査して要約し、関係者に共有した上で、話し合う日程を調整して」といった、複数のステップからなる作業を一度の指示で完結できるようになります。

### 2. アクション実行能力の獲得（各種ツール連携）

対話を通じて情報を検索するだけでなく、ユーザーの指示に基づいて、普段使っているツールで具体的なアクションを実行する能力を獲得します。

*   **タスク管理ツールとの連携**: 会話内容からタスクを自動で起票・更新します。
*   **ドキュメント作成ツールとの連携**: 議事録や報告書などのドキュメントをテンプレートから自動で生成します。
*   **コミュニケーションツールとの連携**: 要約した情報や決定事項を、適切なチャンネルやグループに自動で投稿します。
*   **カレンダーツールとの連携**: 関係者の空き時間を見つけて、イベントや会議を自動で設定します。

### 3. データソースの拡張と継続的な同期

手動でのデータ投入だけでなく、様々な外部サービスと連携し、ナレッジグラフが常に最新の状態に保たれる仕組みを構築します。

*   **自動インジェスターの開発**: 各種クラウドサービスと連携し、新しい情報や変更を自動で検知してグラフに反映させることで、メンテナンスの手間を大幅に削減します。
*   **データモデルの柔軟な拡張**: 活動の変化に合わせて、「スキルの熟練度」や「プロジェクト間の依存関係」といった、より詳細で豊かな情報を表現できるよう、グラフの構造を継続的に改善していきます。

### 4. 能動的な情報提供機能（プロアクティブなアシスト）

ユーザーからの質問を待つだけでなく、システム側から主体的に役立つ情報を提案する機能を追加します。

*   **リスクや機会の検知**: タスクの遅延、重要な情報の見落としといったリスクの予兆を検知して知らせたり、個人のスキルが活かせそうな新しい機会を提案したりします。
*   **コンテキストに応じた情報推薦**: ユーザーが今見ているドキュメントやプロジェクトに対し、関連する過去の資料や、知見を持つ可能性のある人物などを推薦します。

### 5. ユーザーインターフェースの進化

高度な機能を誰もが直感的に使えるよう、より洗練されたWebアプリケーションを開発します。

*   **インタラクティブなグラフ可視化**: AIの回答が、グラフ上のどの情報に基づいているのかを視覚的に表示し、ユーザー自身が情報の繋がりを直感的に探索できるようにします。
*   **AIの思考プロセスの可視化**: AIがどのような手順で答えを導き出したのかをリアルタイムで表示することで、システムの透明性と信頼性を高めます。

## ライセンス (License)

このプロジェクトはMITライセンスの下で公開されています。
