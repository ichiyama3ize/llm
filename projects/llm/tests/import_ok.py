from langchain.llms import LlamaCpp
from transformers import pipeline
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate

model_path = "../models/ELYZA-japanese-Llama-2-7b-fast-q2_K.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=128,
    temperature=0,
    max_tokens=64,
    verbose=False,
    streaming=False
)
print("loaded ", type(llm))

prompt_template = PromptTemplate(
    input_variables   = ["subject", "question"],
    template          = "{subject}の{question}は何ですか？",
    validate_template = True,
    )
format = prompt_template.format(subject="りんご", question="色")

output = llm(format)
print("output")
print(output)

"""
▅ rinko りんごの色は、赤です。 Unterscheidung zwischen einem und einem ist in der Regel nicht möglich. 2019/12/31 - Pinterest で 478 人のユーザーがフォローしています。
「クリスマスリース
"""

# from langchain import PromptTemplate, LLMChain
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# template = """Question: {question}
# Answer: Let's think step by step."""

# prompt = PromptTemplate(template=template, input_variables=["question"])


# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# llm = LlamaCpp(
#     model_path="ggml-alpaca-7b-q4.bin", callback_manager=callback_manager
# )

# llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
# llm_chain.run(question)

# from llama_cpp import Llama
# # プロンプトを記入
# prompt = """[INST] <<SYS>>
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.If you don't know the answer to a question, please don't share false information.
# <</SYS>>
# Write a story about llamas.Please answer in Japanese.[/INST]"""
# # ダウンロードしたModelをセット.
# llm = Llama(model_path="./models/llama-2-13b-chat.Q5_K_M.gguf", n_gpu_layers=20)
# # 生成実行
# output = llm(
#     prompt,max_tokens=500,stop=["System:", "User:", "Assistant:"],echo=True,
# )
