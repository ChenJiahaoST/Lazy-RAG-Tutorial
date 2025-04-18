import os
import lazyllm

DOC_PATH = os.path.abspath("docs")


def online_chat():
    chat = lazyllm.OnlineChatModule()
    lazyllm.WebModule(chat, port=range(12345, 22311)).start().wait()


def retriever():
    from lazyllm import Document, Retriever
    doc = Document(dataset_path=DOC_PATH)
    seperator = '\n' + '='*200 + '\n'
    retriever = Retriever(
        doc=doc,
        group_name="FineChunk",
        similarity="bm25_chinese",
        topk=3,
        output_format="content",
        join=seperator
    )
    print(retriever("跑跑卡丁车"))


def prompt_chat():
    p = "你是懒懒猫，每次回答的开头都要叫三声。"
    llm = lazyllm.OnlineChatModule().prompt(p)
    lazyllm.WebModule(llm, port=range(12345, 22311)).start().wait()


def get_dataset():
    from datasets import load_dataset
    dataset = load_dataset("cmrc2018")
    context = list(set([i['context'] for i in dataset['test']]))
    chunk_size = 10
    total_files = (len(context) + chunk_size - 1)
    dataset_path = os.path.join(DOC_PATH, "cmrc2018")
    os.makedirs(dataset_path, exist_ok=True)

    for i in range(total_files):
        chunk = context[i*chunk_size: (i+1)*chunk_size]
        file_name = f"{dataset_path}/part_{i+1}.txt"
        with open(file_name, 'w', encoding="utf-8") as f:
            f.write("\n".join(chunk))


def my_first_rag():
    dataset_path = os.path.join(DOC_PATH, "cmrc2018")
    documents = lazyllm.Document(dataset_path=dataset_path)
    retriever = lazyllm.Retriever(
        doc=documents,
        group_name="CoarseChunk",
        similarity="bm25_chinese",
        topk=3,
        output_format="content",
        join=''
    )
    retriever.start()
    prompt = (
        'You will act as an AI question-answering assistant and complete a dialogue task.'  # noqa E501
        'In this task, you need to provide your answers based on the given context and questions.'  # noqa E501
    )
    llm = lazyllm.OnlineChatModule().prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))  # noqa E501
    query = input("请输入您的问题\n")
    res = llm({"query": query, "context_str": retriever(query=query)})
    print(f"With RAG Answer:\n{res}")


if __name__ == "__main__":
    online_chat()
    retriever()
    prompt_chat()
    get_dataset()
    my_first_rag()
