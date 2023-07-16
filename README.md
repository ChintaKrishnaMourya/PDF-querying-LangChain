# PDF-querying-LangChain

used langchain framework.
pdf file as input=> pypdf2 to extract text => preprocess text data => converted to embeddings (openai model) => FAISS Vectordb.
user query => similarity search from db => output
