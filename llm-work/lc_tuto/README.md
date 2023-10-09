# LangChainとは？

[公式ドキュメント](https://docs.langchain.com/docs/)

22/10誕生、23/4リリース？、23/6バズ。[盛り上がりの推移。](https://trends.google.co.jp/trends/explore?geo=JP&q=langchain&hl=ja)


# フロー

1. LLM : 言語モデルによる推論の実行。
2. プロンプトテンプレート : ユーザー入力からのプロンプトの生成。
3. チェーン : 複数のLLMやプロンプトの入出力を繋げる。
4. エージェント : ユーザー入力を元にしたチェーンの動的呼び出し。
5. メモリ : チェーンやエージェントで状態を保持。

## 1. [テンプレートについて](https://di-acc2.com/programming/python/26764/)

指示文を指定
* 言語モデルへの指示文章
* 入力変数
* 言語モデルがより良い応答を生成するのに役立ついくつかの教師データ

ExampleやOutput perserがある（！）

代表的な３種類
LLM Prompt Templates: LLMのプロンプト表示方法を示したテンプレート
Chat Prompt Templates: チャットモデルのプロンプト表示方法を示したテンプレート
FewShot PromptTemplate: LLMのプロンプト表示方法に加え、「教師データをどのようなフォーマットで学習させるのか」も指定したテンプレート

## エージェント

・エージェント : 実行するアクションと順番を決定
・アクション : ツールを実行してその出力を観察 or ユーザーに戻る
・ツール : 特定の機能


* [クイックスタート]()
* [openaiパイプ](https://qiita.com/syoyo/items/d0fb68d5fe1127276e2a)
* [ローカルのモデルを使う](https://qiita.com/syoyo/items/d0fb68d5fe1127276e2a)