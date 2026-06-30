from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.sequential import SequentialChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

# 创建大模型对象
model=ChatZhipuAI(
    api_key="1401d5d3c89a44c397bf3f3237a86db8.UxfDhvdIijZdfvzi",
    api_base="https://open.bigmodel.cn/api/paas/v4/chat/completions",
    model_name="glm-4-flash",
    temperature=0
)

# 调用大模型
def chat():
    result=model.invoke(
        [
            SystemMessage(content="你是一个助手"),
            HumanMessage(content="你好")
        ]
    )
    print(result)

# 多轮对话的案例
def multi_chat():
    # 创建一个内存对象
    memory = ConversationBufferMemory()
    #创建一个会话对象
    conversation = ConversationChain(
        llm=model,
        memory=memory
    )
    while True:
        # 获取用户输入
        user_input = input("你：")
        if user_input=="再见":
            break
        # 调用会话对象
        result = conversation.predict(input=user_input)
        print(f"大模型：{result}")

# 单节点的链（即工作流）调用
def test_chain():
    #定义一个提示词模板
    prompt_template=PromptTemplate(
        template="""
        你是一个职场文档专家，可以帮我生成一个{job}岗位的{type}文档，字数不超过500字
        """,
    )
    #定义一个链(工作流)
    chain=LLMChain(
        llm=model,
        prompt=prompt_template,
        verbose=True,
        output_key="content"
    )
    #执行链
    result=chain.invoke({"job":"数据科学家","type":"职位简介"})
    print(result)

#调用顺序链
def test_sequential_chain():
    #定义第一个节点的提示词模板
    prompt_template1=PromptTemplate(
        input_variabales=["city"],
        template="""
        你是一个美食专家，可以告诉用户{city}地区的特色菜是什么，最后输出一个菜名即可，不需要解释
        """,
    )
    #定义第一个节点
    chain1=LLMChain(llm=model,prompt=prompt_template1,verbose=True,output_key="city_food")
    #定义第二个节点的提示词模板
    prompt_template2=PromptTemplate(
        input_variabales=["city_food"],
        template="""
        你是一个厨师，可以输出一个{city_food}的菜谱，直接输出即可，不需要其他解释
        """,
    )
    #定义第二个节点
    chain2=LLMChain(llm=model,prompt=prompt_template2,verbose=True,output_key="info")
    #创建一个顺序链，把两个节点按照顺序组装
    all_chain=SequentialChain(
        chains=[chain1,chain2],
        input_variables=["city"],
        output_variables=["info"],
        verbose=True
    )
    #运行整个链
    result=all_chain.invoke({"city":"苏州"})
    print(result)
if __name__ == '__main__':
    # chat()
    # multi_chat()
    # test_chain()
    test_sequential_chain()