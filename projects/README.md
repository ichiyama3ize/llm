# 使い方

* OpenAI　API
* python環境

## Open AI API
[Open AI](https://openai.com/)にサインインし、APIを登録。
APIの使用には決済情報の登録が必要。
おそらく初回だと無料である程度使えます。

モデルの呼び出し（正しくは"トークン数"）で使用料が変わる。
試す程度ならそんなにかからない。

## python環境
`./lc_tuto/pyproject.toml`の`[tool.poetry.dependencies]`に記載されているverで動かせます。


# LangChainについて

./lc_tuto/README.md参照
<!-- @import "./lc_tuto/README.md" -->


---------------------------------------------------------
# 構成の動機

docker 環境ごと共有するため。
pyenv  機械学習で、pythonのverで動く・動かないを嫌って。
        (developの段階のため、入れています。本来ならdockerで十分)
poetry pythonプロジェクトレベルの環境を(少ないファイルで)共有・管理するため。
    管理は.toml、共有は.lockで行われる。

[why poetry > pipenv](https://vaaaaaanquish.hatenablog.com/entry/2021/03/29/221715)

# poetry

仮想環境としてのメリット
対抗馬のpipenvとは異なり、メタデータもtomlファイルに記述される

## 2.1.0 project

### 新規作成
new project_name
(project_nameはライブラリと被らないように！)

### 既存のtomlがある場合
cd .tomlのあるディレクトリ
poetry install

## python -V の切り替え
```
// python -Vの固定化
// インストールしたpyenv pythonにpyenv(local/global)
// で有効化しておく必要がある。
cd pyproject
pyenv install 3.10
pyenv local 3.10
poetry env use 3.10

// deactivate
pyenv local system
poetry env use system

// list up
poetry env list
```


## 参考
* [一通り入門](https://qiita.com/ksato9700/items/b893cf1db83605898d8a)
* [使い方](https://zenn.dev/canonrock/articles/poetry_basics)
