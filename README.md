# AI Chatbot with Milvus Integration

This project implements an advanced AI chatbot using FastAPI, Together AI's language models, and Milvus vector database for similarity search.

## Features

- Conversational AI powered by Together AI's language models
- Context-aware responses using Milvus vector database for similarity search
- FastAPI backend for efficient API handling

## Tech Stack

- FastAPI: Web framework for building APIs
- Together AI: Provides the language model for generating responses
- Milvus: Vector database for storing and retrieving similar conversations

## Project Structure

```bash
chatbot/
app/
├── api
│   └── chat.py
├── config.py
├── main.py
├── models
│   ├── chat.py
│   └── documents.py
├── services
│   ├── llm_service.py
│   └── milvus_service.py
└── utils
    └── helpers.py
```


## Setup and Installation

1. Clone the repository:
- git clone <https://github.com/storagerepo/AI_POC.git> 
- cd dir-name

2. Set up a virtual environment:
- python -m venv venv
- source venv/bin/activate # On Windows use venv\Scripts\activate

3. Set up environment variables:
Create a `.env` file in the root directory and add the following:

```txt
TOGETHER_API_KEY=your_together_ai_api_key
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

## Usage

1. Send a POST request to `/chat` endpoint with the following JSON body:
```json
{
  "message": "What's the latest news about AI?",
  "history": []
}
```

## Development Process
- Initial setup with FastAPI and Together AI integration
- Added Milvus vector database for similarity search
- Implemented context-aware responses using past conversations
- Enhanced context management to combine historical data
- Continuous improvements and optimizations


## Future Enhancements
- Implement user authentication and personalized conversations
- Add support for multi-modal inputs (images, voice)
- Implement conversation summarization for long chats
- Integrate external APIs for diverse information sources
- Develop a web-based user interface