# Import the necessary classes from langchain modules.
import os
from src.prompt import *
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain



# Import the load_dotenv function from the dotenv module to load environment 
# variables from a .env file.
from dotenv import load_dotenv

# Load environment variables from the .env file.
load_dotenv()

# Get the OpenAI API key from the loaded environment variables.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key in the environment variables.
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



# Define a function to process the PDF file and generate document chunks.
def file_processing(file_path):
    # Load data from the PDF file.
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Initialize an empty string to accumulate the content from all pages of the PDF.
    question_gen = ""

    # Iterate over each page in the loaded data and concatenate the text content
    #  to the question_gen string.
    for page in data:
        question_gen += page.page_content

    # Create an instance of the TokenTextSplitter class for splitting the content into larger chunks.
    # The 'model_name' parameter specifies the model to use for tokenization.
    # The 'chunk_size' parameter sets the maximum size of each text chunk.
    # The 'chunk_overlap' parameter specifies the number of overlapping tokens between chunks.
    splitter_ques_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=10000,
        chunk_overlap=200
    )

    # Use the split_text method of the splitter_ques_gen instance to divide the 
    # concatenated text into chunks.
    chunk_ques_gen = splitter_ques_gen.split_text(question_gen)

    # Create a list of Document objects from the text chunks.
    # For each chunk of text in chunk_ques_gen, a new Document object is created 
    # with 'page_content' set to the chunk text.
    document_ques_gen = [
        Document(page_content=chunk) for chunk in chunk_ques_gen
    ]

    # Create an instance of the TokenTextSplitter class for splitting the documents
    #  into smaller chunks.
    # The 'model_name' parameter specifies the model to use for tokenization.
    # The 'chunk_size' parameter sets the maximum size of each text chunk.
    # The 'chunk_overlap' parameter specifies the number of overlapping tokens between chunks.
    splitter_ans_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=1000,
        chunk_overlap=100
    )

    # Use the split_documents method of the splitter_ans_gen instance to divide 
    # the Document objects into smaller chunks.
    document_ans_gen = splitter_ans_gen.split_documents(document_ques_gen)

    # Return the generated document chunks.
    return document_ques_gen, document_ans_gen




def llm_pipeline(file_path):

    document_ques_gen, document_ans_gen = file_processing(file_path)

    # Create an instance of the ChatOpenAI class with specific parameters.
    # The 'model' parameter specifies the model to use for generating questions, 
    # in this case, "gpt-3.5-turbo".
    # The 'temperature' parameter controls the randomness of the model's output, 
    # set to 0.3 for balanced creativity and coherence.
    llm_ques_gen_pipeline = ChatOpenAI( 
        model='gpt-3.5-turbo',
        temperature=0.5
    )


    # Create a PromptTemplate object for generating questions, using the defined template.
    # The input_variables parameter specifies that the template expects a variable named "text".
    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template, 
        input_variables=["text"]
        )
    
    # Create a PromptTemplate object for refining questions, using the defined template.
    # The input_variables parameter specifies that the template expects variables named 
    # "existing_answer" and "text".
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    # Load a summarization chain that will generate and refine questions using 
    # the provided model and prompt templates.
    # The 'llm' parameter specifies the language model to use.
    # The 'chain_type' parameter specifies the type of chain to use, in this case, "refine".
    # The 'verbose' parameter is set to True to enable detailed logging of the process.
    # The 'question_prompt' parameter specifies the initial prompt template for generating questions.
    # The 'refine_prompt' parameter specifies the template for refining the generated questions.
    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine", 
        verbose=True, 
        question_prompt=PROMPT_QUESTIONS, 
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )

    # Run the question generation chain on the document chunks.
    # The resulting questions are stored in the 'ques' variable.
    ques = ques_gen_chain.run(document_ques_gen)

    # Create an instance of OpenAIEmbeddings to generate embeddings for the documents.
    embeddings = OpenAIEmbeddings()

    # Create a FAISS vector store from the documents with generated embeddings.
    # This will allow for efficient retrieval of relevant documents based on the generated questions.
    vector_store = FAISS.from_documents(
        documents=document_ans_gen, 
        embedding=embeddings
        )

    # Split the generated questions string into a list of individual questions.
    ques_list = ques.split("\n")

    # Create a RetrievalQA chain for generating answers using the language model and the 
    # vector store as a retriever.
    # The 'chain_type' parameter is set to "stuff", indicating the type of retrieval 
    # and processing method to use.
    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_ques_gen_pipeline, 
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    return answer_generation_chain, ques_list

