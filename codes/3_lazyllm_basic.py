import os
import lazyllm

DOC_PATH = os.path.abspath("docs")


prompt = (
    'You will act as an AI question-answering assistant and complete a dialogue task.'  # noqa E501
    'In this task, you need to provide your answers based on the given context and questions.'  # noqa E501
)
documents = lazyllm.Document(dataset_path=os.path.join(DOC_PATH, "cmrc2018"))

with lazyllm.pipeline() as ppl:
    ppl.retriever = lazyllm.Retriever(
        doc=documents, group_name="CoarseChunk", similarity="bm25_chinese",
        topk=3, output_format="content", join='')
    ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | lazyllm.bind(query=ppl.input)  # noqa E501
    ppl.llm = lazyllm.OnlineChatModule().prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))  # noqa E501

query = input("请输入您的问题\n")
res = ppl(query)
print(f"With RAG Answer:\n{res}")
