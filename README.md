End-to-End GenAI Solution using AWS Bedrock with LangChain
url: http://localhost:8501

Sample questions:
1. What home health services does Medicare cover?
2. What doesn’t Medicare cover for home health care?
3. How long can I get home health services?

Description:
Led the development of a sophisticated GenAI project aimed at extracting answers from PDF documents based on user-input questions.

Technical Architecture:
1. PDF's processed using RecursiveCharacterTextSplitter for chunking.
2. Chunk embeddings generated using Amazon-Titan model.
3. FAISS vector database utilized for storage and retrieval.

Functionality:
Upon receiving a user question, the system:

Converts the question into embeddings.
Conducts a similarity search in the vector store.
Retrieves relevant chunks and given to the Language Model (LLM) along with the prompts for final answer generation.

Technologies Used:
1. AWS Bedrock
2. RecursiveCharacterTextSplitter
3. Amazon-Titan Model
4. FAISS Vector Database
