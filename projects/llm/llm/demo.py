from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent

import openai
# openai.log = "debug"

def demo() -> str:
    """使うだけ

    Returns:
        str: llmはstrを返す
    """
    prompt_template = PromptTemplate(
                            input_variables   = ["subject", "question"],
                            template          = "{subject}の{question}は何ですか？",
                            validate_template = True,  # 入力変数とテンプレートの検証有無
                        )
    format = prompt_template.format(subject="りんご", question="色")
    llm = OpenAI(temperature=0.9, n=1)
    return llm(format)

def culc_token(text: str, model_name='gpt-3.5-turbo'):
    """入力したいtextの、token数を計算する。

    Args:
        text (str): _description_
        model_name (str, optional): in [gpt2, gpt-3.5-turbo, ] and so on. Defaults to 'gpt-3.5-turbo'.

    Returns:
        len_token: トークン数。apiの料金にかかわる数字。
        tokens: エンコーディングしたトークンのIDのリスト。元のtextの長さとは一般的に関係ない。
    """
    import tiktoken

    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    return len(tokens), tokens


"""
1. テンプレートについて
LLMの入力となる自然言語のテンプレート。
https://di-acc2.com/programming/python/26764/
"""
def how_template_1_basic():
    """
    1. 基本的なプロンプト
    """
    from langchain.prompts import PromptTemplate

    prompt_template = PromptTemplate(
                        input_variables   = ["subject", "question"],
                        template          = "{subject}の{question}は何？",
                        validate_template = True,  # 入力変数とテンプレートの検証有無
                    )

    format = prompt_template.format(subject="りんご", question="色")
    # prompt_template.save("./template.json")
    # prompt_template = load_template("./template.json")
    return format

def how_template_2_chat():
    """
    2. chat プロンプト
    """
    from langchain.prompts import (
        PromptTemplate, load_prompt,
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )

    from langchain.chat_models import ChatOpenAI

    system_template = "あなたは{input_language}を{output_language}に翻訳するアシスタントです。"
    human_template  = "{input_text}"

    # プロンプト作成
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt  = HumanMessagePromptTemplate.from_template(human_template)

    # 上記テンプレートを統合しチャットプロンプトテンプレート作成
    chat_prompt = ChatPromptTemplate.from_messages([
                        system_message_prompt, 
                        human_message_prompt
                        ])

    chat_prompt_format = chat_prompt.format_prompt(
                              input_language  = "日本語", 
                              output_language = "英語", 
                              input_text      = "これはペンです。"
                         ).to_messages()

    # chat_model = ChatOpenAI(temperature=1, n=1)
    # response = chat_model(chat_prompt_format)

    return chat_prompt_format
    
def how_template_3_example(input_animal: str):
    """
    3. Example template

    教師データとして使用
    """
    from langchain.prompts.example_selector.base import BaseExampleSelector
    #from langchain import FewShotPromptTemplate
    from langchain.prompts.few_shot import FewShotPromptTemplate

    from typing import Dict, List
    import numpy as np

    # 教師ありリスト
    # examples = [
    #     {"country": "Japan",   "capital": "Tokyo"},
    #     {"country": "France",  "capital": "Paris"},
    #     {"country": "UK",      "capital": "London"},
    #     ]
    examples =[
        {"animal": "dog",     "familia": "イヌ科"},
        {"animal": "wolf",    "familia": "イヌ科"},
        {"animal": "cat",     "familia": "ネコ科"},
        {"animal": "sparrow", "familia": "スズメ科"},
        {"animal": "airplane","familia": "動物ではありません"},
    ]

    # 教師ありリストをフォーマットするためにテンプレートに指定
    example_template = \
        """
        動物名: {animal} 科: {familia}
        """

    # 教師データの選択
    class CustomExampleSelector(BaseExampleSelector):
        
        def __init__(self, examples: List[Dict[str, str]]):
            self.examples = examples
        
        # 教師データを追加する関数
        def add_example(self, example: Dict[str, str]) -> None:
            self.examples.append(example)

        # 教師データを選択するための関数
        def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
            return np.random.choice(self.examples, size=2, replace=False)  


    # 教師ありリストを解釈するプロンプトテンプレート
    example_prompt_template = PromptTemplate(
                            input_variables   = ["animal", "familia"],
                            template          = example_template,
                            validate_template = True,                 # 入力変数とテンプレートの検証有無
                            )

    # CustomExampleSelectorに教師リストを登録
    example_selector = CustomExampleSelector(examples)

    fewshot_prompt = FewShotPromptTemplate(
            example_selector  = example_selector,
            example_prompt    = example_prompt_template,
            prefix            = "Tell me the familia of the animal in Japanese.", # プロンプト内の教師ありリストの前に配置する接頭辞
            suffix            = "動物名: {input} \n 科: ",                         # プロンプト内の教師ありリストの前に配置する接尾辞　通常、ユーザーの入力が入る
            input_variables   = ["input"],
            example_separator = "\n",                                            # 接頭辞、教師ありリスト、接尾辞の結合に使用する文字列
    )

    # FewShotPromptTemplateにフォーマット適用
    fewshot_prompt_format = fewshot_prompt.format(input=input_animal)
    
    return fewshot_prompt_format

def how_template_4_output():
    """
    4. 出力の形式を指定
    
    具体的には、回答を複数得たい時、など。
    """
    from langchain.output_parsers import PydanticOutputParser

    from pydantic import BaseModel, Field, validator
    from typing import List

    # 変数:型 = Field(説明)
    class output_format(BaseModel):
        name:            str = Field(description = "国名")
        food_list: List[str] = Field(description = "国の美味しい料理名リスト")

    parser = PydanticOutputParser(pydantic_object = output_format)

    template = "{query}\n\n{format_instructions}\n"
    prompt_template = PromptTemplate(
                    input_variables   = ["query"], 
                    template          = template, 
                    validate_template = True,
                    partial_variables = {"format_instructions": parser.get_format_instructions()} # 出力形式
                       )

    # プロンプトテンプレートにフォーマット適用
    prompt_format = prompt_template.format_prompt(query="日本のおすすめ料理を教えて")

    # llm = OpenAI(temperature=0,n=1)
    # response = llm(prompt_format.to_string())
    # parsed   = parser.parse(response)

    return prompt_format, parser

"""
2. チェーンについて
"""
def how_chain():
    llm = OpenAI(temperature=0.9)

    # プロンプトテンプレート
    prompt = PromptTemplate(
        input_variables=["company", "product"],
        template="{product}を作る日本の{company}の名前を1つ提案してください",
        )

    # チェーン
    chain = LLMChain(llm=llm, prompt=prompt)
    run = chain.run({
        'company': "スタートアップの会社",
        'product': "家庭用ロボット"
        })
    print(run)


    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
    )
    human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="What is a good name for a company that makes {product}?",
                input_variables=["product"],
            )
        )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chat = ChatOpenAI(temperature=0.9)
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)
    print(chain.run("colorful socks"))

"""
3. エージェントについて
"""
def how_agent():

    # # ツールの準備
    # tools = load_tools(["llm-math"], llm=llm)

    # # エージェントの準備
    # mathmatic_agent = initialize_agent(
    #     tools, 
    #     llm, 
    #     agent="zero-shot-react-description", 
    #     verbose=True)

    # from langchain import OpenAI, ConversationChain

    # # ConversationChainの準備
    # chain = ConversationChain(
    #     llm=OpenAI(temperature=0), 
    #     verbose=True
    # )

    # # 会話の実行
    # chain.predict(input="こんにちは！")

    return None


if __name__ == '__main__':

    # print(demo())

    llm = OpenAI(temperature=1, n=1)
    format = how_template_1_basic()
    print(llm(format))

    # llm = ChatOpenAI(temperature = 1, n=1)
    # format = how_template_2_chat()
    # print(llm(format))

    # llm = OpenAI(temperature=0, n=1)
    # format = how_template_3_example(input_animal="pig")
    # answer = "イノシシ科"
    # print(llm(format))

    # llm = OpenAI(temperature=0, n=1)
    # format, parser = how_template_4_output()
    # response = llm(format.to_string())
    # parsed   = parser.parse(response)
    # print(parsed)