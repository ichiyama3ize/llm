# llm
otamesi LLM

LLMのossライブラリLangChainを使ってみた。
> LangChainは、大規模言語モデル（LLM）の機能を拡張できるライブラリです。LLM と外部リソース（データソース、言語処理系）を組み合わせて、より高度なアプリケーションやサービスの開発をサポートすることを目的としています。


OpenAIのapi key取得と、pythonのライブラリインストールで動きます。

dockerコンテナ立ち上げてますが大したことしてません。
以下のpython環境/ライブラリverで動作確認しました。

```
python = ">=3.9,<3.13"
langchain = "^0.0.310"
openai = "^0.28.1"
tiktoken = "^0.5.1"
numpy = "^1.26.0"
```
