from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv  

load_dotenv() 

# initialize the embedding model
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# initialize the llm model
llm = ChatOpenAI(model="gpt-5-nano")

perist_directory = "db/chroma_basic_db"

#initalize the empty chat history
chat_history = []


def question_regeneration(query):
    """Based on the chat history rewrite the vague user question"""
    print(f"You asked the question: {query}")
    if chat_history:
        # we will use AI to generate standalone questions
        messages = [SystemMessage(content="You are a helpful prompt generator assistant")] + chat_history + [HumanMessage(content=f"Based on the chat history rewrite the user question {query}")]
        response = llm.invoke(messages)
        rewritten_query = response.content
        
    else:
        rewritten_query = query
    print(f"The rewritten user question is: {rewritten_query}")
    return rewritten_query

def retrieve_context(query,perist_directory="db/chroma_basic_db"):
    # load the vector store
    vector_store = Chroma(embedding_function=embedding, persist_directory=perist_directory, collection_metadata={"hsnw:space": "cosine"})
    # define the retriever 
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)

    print(f"User Query: {query}\n")

    print("---Context---")

    print(f"No.of documents retrieved {len(relevant_docs)}\n")

    for i,doc in enumerate(relevant_docs, 1):
        print(f"Chunk{i}:\n {doc.page_content}")
        print("--------------------------------")

    # Combine the user query with the retrieved chunks for summarization

    combined_chunks = '\n'.join(doc.page_content for doc in relevant_docs)

    combined_input = f""" Based on the following documents answer the user question {query}

    Documents:
    {combined_chunks}

    Please provide a clear and helpful answer using only  the information from these documents.If you can't answer the question, say I don't have the information.
    """

    # create the message
    messages = [
        SystemMessage(content= "You are a helpful AI assistant"),
        HumanMessage(content=combined_input)
    ]
    # add each message object seperately 
    chat_history.extend(messages)
    print(chat_history)
    
    # invoke the llm model
    response = llm.invoke(messages)
    # print(f"AI Assistant: {response.content}")
    chat_history.append(AIMessage(content=response.content))

    return response.content

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("GoodBye")
        break
    rewritten_query = question_regeneration(user_input)
    output = retrieve_context(rewritten_query)
    print(f"AI Assistant: {output}")



