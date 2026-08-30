# RAG Real Estate — AI-Powered Property Search & Advisory

## Project Name
RAG Real Estate

## Problem Solved
Real estate buyers often struggle to find properties that match nuanced preferences (e.g., "quiet neighborhood near good schools with a large backyard"). Traditional property search filters (beds, baths, price) miss contextual details buried in listing descriptions. RAG Real Estate uses Retrieval-Augmented Generation to let buyers search properties using natural language queries, with an AI assistant that understands context from property descriptions, neighborhood data, and market trends.

## Tech Stack
- **AI/ML**: LangChain for RAG pipeline, OpenAI GPT-4 for generation
- **Vector Database**: Pinecone for property embedding storage and similarity search
- **Backend**: Python, FastAPI
- **Frontend**: Streamlit for rapid prototyping and demo interface
- **Data Processing**: pandas, BeautifulSoup for property data scraping and cleaning
- **Embeddings**: OpenAI text-embedding-ada-002 for document vectorization
- **Deployment**: Docker on AWS EC2

## Key Features
- **Natural Language Property Search**: Ask "Find me a 3-bedroom house under $400k near parks in a quiet suburb" and get semantically matched listings
- **RAG Pipeline**: Property listings chunked, embedded, and stored in Pinecone; retrieved based on query similarity; fed to GPT-4 for contextual answers
- **Property Comparison**: Side-by-side AI-generated summaries comparing shortlisted properties on user-defined criteria
- **Market Trend Analysis**: RAG-powered Q&A over scraped market reports — ask "How has pricing trended in this area over the last year?"
- **Conversational Memory**: Multi-turn chat that remembers previous queries and refines recommendations
- **Source Attribution**: Every AI response cites the specific property listings or data sources it drew from

## Results / Impact
- Indexed 10,000+ property listings from public MLS data for the demo
- Semantic search returned 85% more relevant results compared to keyword-based filters in user testing
- Demonstrated end-to-end RAG architecture: data ingestion → chunking → embedding → retrieval → generation
- Showcased expertise in vector databases, LLM integration, and building AI-powered search systems
- This project directly demonstrates RAG pipeline design skills applicable to any domain (legal, medical, enterprise knowledge bases)
