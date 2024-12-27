from utils.LLM import openai, spark


def generate_answer(question, reference="", rules_string = "", ReferenceTemplate_path="", model = "spark"):
    """
    输入：用户的问题，推理出的意图和目的实体
    处理过程：完善query，交给大模型
    输出：最终答案
    """
    #读取参考输出模板
    with open(ReferenceTemplate_path, "r", encoding="utf-8") as file:
        ReferenceTemplate = file.read()

    query = f"""
        I am working on a knowledge-based question-answering task.
        The user has input the following question:
        "{question}"

        I can provide you with some reference content, where each set of content consists of two parts: conditions and objectives.
        - Conditions: The known information from the question.
        - Objectives: The goals that the question seeks to address.
        The abstract paths are as follows:
        {rules_string}
        Below is the reference content:
        {reference}

        Please strictly follow the reference content to answer the question, applying logical reasoning as needed to generate the final answer.
        **Note**: The generated answer must not mention or disclose the existence of the reference content.
        \n\tThe output can refer to the following format.
        {ReferenceTemplate}
        The answer should be a paragraph without line breaks and in order.
    """

    # OpenAI System role content
    system_content = """
    You are a helpful and knowledgeable assistant. Your task is to provide precise answers by performing logical reasoning based on the user's input and additional reference content. 
    Remember: You must not disclose or mention the existence of the reference content provided to you in your response.
    """

    # OpenAI User role content
    user_content = f"""
    I am working on a knowledge-based question-answering task. 
    The user has input the following question:
    "{question}"

    I will provide you with some reference content. Each reference contains two parts: *conditions* and *goals*. 
    - Conditions: Information known from the user's question.
    - Goals: The specific objectives the question aims to answer.

    Here is the reference content:
    {reference}

    Please strictly follow the reference content to answer the question. Use logical reasoning if necessary to generate the final answer. 
    **Note**: The generated answer must not reveal or mention the existence of the reference content.
    \n\tThe output can refer to the following format.
    {ReferenceTemplate}
    """

    if model == "spark":
        print(f"query:\n{query}")
        return spark.spark_4_0_company(query)
    elif model == "gpt-4o-mini":
        print(f"system_content:\n{system_content}")
        print(f"user_content:\n{user_content}")
        return openai.gpt_4o_mini(system_content, user_content)
    else:
        return "没有这个模型。现在可选的模型有: spark, gpt-4o-mini"