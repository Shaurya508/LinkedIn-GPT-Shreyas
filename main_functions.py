from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama import ChatOllama

# I SHOULD add "source" parameter here and make it a common function for all possible source selected! (Check Claude's original code ----> "Filtering FAISS Search Results by Source" Chat)
def user_input(user_question):
    db_name = "Link_Stack_DB"
    prompt_template = """
    You have been given a question and a relevant context. Using the context, explain the answer to the question or discuss the topic given in the question in detail. Never return a blank response.\n
    If the context has anything relevant to the question, you are always supposed to answer something.
    Context:\n{context}?\n
    Question:\n{question}. + Explain in detail.\n
    Answer:
    """

    # Use Ollama to load the Llama 3.1:8b model (assumes Ollama is running locally)
    model = ChatOllama(model="llama3.1:latest", max_tokens=1000)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db1 = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)

    # Create a filtered retriever based on the source
    filtered_retriever = new_db1.as_retriever(
        search_kwargs={
            'k': 1,
            'filter': lambda x: x['source'] == "LinkedIn"
        }
    )

    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=filtered_retriever,
        llm=model
    )

    docs = mq_retriever.invoke(input=user_question.lower())

    print(f"Query = {user_question}\nSource: LinkedIn\nRelevant Documents Extracted:\n")
    print(docs)
    
    #print([doc for doc in docs])
    page_content = docs[0].page_content
    urls = re.findall(r'https?://\S+', page_content)

    image_address = urls[-1] if urls else None
    post_link = urls[0] if urls else None

    print("Moving on to generate response.")
    response = chain({
        "input_documents": docs,
        "question": user_question
    }, return_only_outputs=True)
    print("Response has been generated")

    print("\nThis Query has been completed\n==================================================================================")
    return response, image_address, post_link


def user_input1(user_question):
    db_name = "Link_Stack_DB"
    
    prompt_template = """
    You have been given a question and a relevant context. Using the context, explain the answer to the question or discuss the topic given in the question in detail. Never return a blank response.\n
    If the context has anything relevant to the question, you are always supposed to answer something.
    Context:\n{context}?\n
    Question:\n{question}. + Explain in detail.\n
    Answer:
    """

    # Use Ollama to load the Llama 3.1:8b model (assumes Ollama is running locally)
    model = ChatOllama(model="llama3.1:8b", max_tokens=1000)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)

    # Create a filtered retriever based on the source
    filtered_retriever = new_db.as_retriever(
        search_kwargs={
            'k': 1,
            'filter': lambda x: x['source'] == "StackExchange"
        }
    )

    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=filtered_retriever,
        llm=model
    )

    docs = mq_retriever.invoke(input=user_question.lower())
    print(f"Query = {user_question}\nSource: Stack Exchange\nRelevant Documents Extracted:\n")
    print(docs)
    
    #print([doc for doc in docs])
    page_content = docs[0].page_content
    urls = re.findall(r'https?://\S+', page_content)

    image_address = urls[-1] if urls else None
    post_link = urls[0] if urls else None

    print("Moving on to generate response.")
    response = chain({
        "input_documents": docs,
        "question": user_question
    }, return_only_outputs=True)
    print("Response has been generated")

    print("\nThis Query has been completed\n==================================================================================")
    return response, image_address, post_link

def user_input2(user_question):
    db_name = "Wiki_DB"
    
    prompt_template = """
    You have been given a question and a relevant context. Using the context, explain the answer to the question or discuss the topic given in the question in detail. Never return a blank response.\n
    If the context has anything relevant to the question, you are always supposed to answer something.
    Context:\n{context}?\n
    Question:\n{question}. + Explain in detail.\n
    Answer:
    """

    # Use Ollama to load the Llama 3.1:8b model (assumes Ollama is running locally)
    model = ChatOllama(model="llama3.1:8b", max_tokens=1000)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)

    # Create a filtered retriever based on the source
    filtered_retriever = new_db.as_retriever(
        search_kwargs={
            'k': 1,
            'filter': lambda x: x['source'] == "Wiki"
        }
    )

    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=filtered_retriever,
        llm=model
    )

    docs = mq_retriever.invoke(input=user_question.lower())
    print(f"Query = {user_question}\nSource: Wiki\nRelevant Documents Extracted:\n")
    print(docs)
    
    #print([doc for doc in docs])
    page_content = docs[0].page_content
    urls = re.findall(r'https?://\S+', page_content)

    image_address = urls[-1] if urls else None
    post_link = urls[0] if urls else None

    print("Moving on to generate response.")
    response = chain({
        "input_documents": docs,
        "question": user_question
    }, return_only_outputs=True)

    print("Response has been generated")

    print("\nThis Query has been completed\n==================================================================================")
    return response, image_address, post_link


#all_chunks = []

#links = extract_links('List of my best posts -2021.pdf') + extract_links('List of my best posts 2022.pdf') + extract_links('List of my best posts 2023.pdf')
#print(f"# links: {len(links)}\n==============================================")
#driver = webdriver.Chrome()

#credentials = get_credentials()
#if credentials:
#    linkedin_email = credentials.get('email')
#    linkedin_password = credentials.get('password')

# Log in to LinkedIn
#linkedin_login(linkedin_email, linkedin_password , driver)

#for linkedin_post_url in links:
#    post_text,image_address = scrape_linkedin_post(linkedin_post_url , driver)
#    text_chunks = get_text_chunks_with_metadata(post_text, "LinkedIn", linkedin_post_url)
#    for chunk in text_chunks:
#        all_chunks.append(chunk)

# Writing to a file
#with open('all_chunks.json', 'w') as file:
#    json.dump(all_chunks, file, indent=2)
# Reading from the file and recreating the list
#with open('all_chunks.json', 'r') as file:
#    all_chunks = json.load(file)

#print("Data has been read from all_chunks.txt to the variable all_chunks")
#print(all_chunks)
#create_faiss_db(all_chunks)

#print(f"\n\n\n\n\n\n\n\n\n\nThe length of all_chunks is: {len(all_chunks)} | The # links was: {len(links)}")
print("=============================================DONE=========================================")