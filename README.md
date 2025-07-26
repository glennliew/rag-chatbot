# 🤖📚 RAG Chatbot for Kids

A friendly AI chatbot that answers questions using Retrieval-Augmented Generation (RAG), specifically designed for primary school students (ages 8-12). The chatbot uses your PDF documents as a knowledge base and provides simple, educational answers.

## ✨ Features

- 📄 **PDF Knowledge Base**: Load any PDF document as the chatbot's knowledge source
- 🤖 **Kid-Friendly Responses**: Answers tailored for primary school students with simple language
- 🌍 **Multilingual Support**: Supports English, Chinese, Malay, and other languages
- 🛡️ **Out-of-Scope Detection**: Politely handles questions outside the knowledge base
- 📊 **RAGAS Evaluation**: Comprehensive evaluation with 80%+ target scores
- 🎨 **User-Friendly Interface**: Both Gradio and Streamlit interfaces available
- ⚡ **Fast Retrieval**: Uses vector embeddings for quick and accurate information retrieval

## 🏗️ Project Structure

```
ragchatbot/
├── app.py                 # Main entry point
├── load_kb.py            # PDF loading and embedding creation
├── rag_pipeline.py       # RAG pipeline with LLM integration
├── interface.py          # Gradio/Streamlit chat interfaces
├── evaluate.py           # RAGAS evaluation system
├── utils.py              # Utilities (translation, out-of-scope detection)
├── test_data.json        # Sample Q&A pairs for evaluation
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── README.md           # This file
└── chroma_db/          # Vector database (auto-created)
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or download the project
cd ragchatbot

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Prepare Your Knowledge Base

Place a PDF document in the project directory. You can name it:
- `sample_knowledge_base.pdf`
- `knowledge_base.pdf`
- `docs.pdf`

Or specify any PDF path when setting up.

### 3. Set Up Knowledge Base

```bash
# Automatic setup (looks for default PDF files)
python app.py --setup

# Or specify a PDF file
python app.py --setup path/to/your/document.pdf

# Force rebuild if needed
python app.py --setup --force-rebuild
```

### 4. Run the Chat Interface

```bash
# Gradio interface (default)
python app.py --chat

# Or Streamlit interface
python app.py --chat streamlit
```

### 5. Evaluate Performance

```bash
# Run RAGAS evaluation
python app.py --evaluate

# Run simple test
python app.py --test
```

## 📱 Usage Examples

### Chat Interface

Once the interface is running, you can ask questions like:

**English:**
- "What is photosynthesis?"
- "How do plants get water?"
- "Why are leaves green?"

**Chinese:**
- "什么是光合作用？"
- "植物如何获得水分？"

**Malay:**
- "Apakah fotosintesis?"
- "Bagaimana tumbuhan mendapat air?"

### Command Line Interface

```bash
# Show detailed help
python app.py --help-detailed

# Set up with specific PDF
python app.py --setup science_textbook.pdf

# Run evaluation and get scores
python app.py --evaluate

# Quick functionality test
python app.py --test
```

## 🧪 Evaluation System

The chatbot is evaluated using **RAGAS** (Retrieval-Augmented Generation Assessment) with four key metrics:

| Metric | Target Score | Description |
|--------|--------------|-------------|
| **Faithfulness** | ≥80% | Answers stick to the provided context |
| **Answer Relevancy** | ≥80% | Answers directly address the questions |
| **Context Precision** | ≥80% | Retrieved context is relevant to the question |
| **Context Recall** | ≥80% | All relevant information is retrieved |

### Sample Evaluation Output

```
📊 RAGAS EVALUATION SCORES
------------------------------
Faithfulness        : 0.850 (Target: 0.8) ✅ PASS
Answer Relevancy     : 0.823 (Target: 0.8) ✅ PASS
Context Precision    : 0.891 (Target: 0.8) ✅ PASS
Context Recall       : 0.776 (Target: 0.8) ❌ FAIL

Average Score: 0.835 ✅ OVERALL PASS
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Customization Options

You can customize the chatbot by modifying these files:

- **`rag_pipeline.py`**: Adjust LLM parameters, prompts, similarity threshold
- **`load_kb.py`**: Change chunk size, overlap, embedding model
- **`utils.py`**: Modify translation languages, out-of-scope patterns
- **`test_data.json`**: Add your own test cases

### Key Parameters

```python
# In rag_pipeline.py
RAGChatbot(
    model_name="gpt-4-1106-preview",     # LLM model
    temperature=0.1,                     # Response creativity (0-1)
    max_tokens=500,                      # Max response length
    similarity_threshold=0.7             # Min relevance score
)

# In load_kb.py
KnowledgeBaseLoader(
    chunk_size=1000,                     # Text chunk size
    chunk_overlap=200,                   # Overlap between chunks
    embedding_model="text-embedding-ada-002"  # Embedding model
)
```

## 🌟 Advanced Features

### Multilingual Support

The chatbot automatically detects input language and can respond in:
- English
- Chinese (Simplified)
- Malay
- Tamil
- Hindi
- Spanish
- French
- German
- Japanese
- Korean

### Out-of-Scope Detection

The system intelligently detects and handles:
- Time-sensitive questions ("What's the weather today?")
- Personal questions ("Tell me about myself")
- Math calculations ("What's 2+2?")
- Inappropriate content

### Kid-Friendly Features

- Simple, clear language suitable for ages 8-12
- Encouraging phrases ("Great question!", "Let me help you!")
- Visual emoji indicators in the interface
- Comic Sans font for a friendly appearance
- Colorful, engaging UI design

## 🐛 Troubleshooting

### Common Issues

**1. "No knowledge base found"**
```bash
solution: python app.py --setup your_document.pdf
```

**2. "OpenAI API key not found"**
```bash
solution: Add OPENAI_API_KEY to your .env file
```

**3. "RAGAS evaluation failed"**
```bash
solution: Ensure you have a stable internet connection and valid API key
```

**4. "Import errors"**
```bash
solution: pip install -r requirements.txt
```

### Performance Issues

- **Slow responses**: Try reducing chunk size or using a different embedding model
- **Irrelevant answers**: Lower the similarity threshold or improve PDF quality
- **High API costs**: Use smaller models or reduce max_tokens

### Debug Mode

```bash
# Run with detailed logging
python app.py --chat --debug

# Check system status
python app.py --test
```

## 📊 Sample Test Data

The project includes comprehensive test cases in `test_data.json`:

- 10 science questions with expected answers
- Kid-friendly explanations
- Multilingual test cases
- Ground truth answers for evaluation

Topics covered:
- Photosynthesis
- Plant biology
- Water cycle
- Animal breathing
- Seasons
- Sky color
- Magnets
- Gravity

## 🚀 Deployment Options

### Local Development
```bash
python app.py --chat gradio
# Access at http://127.0.0.1:7860
```

### HuggingFace Spaces
1. Create a new Space on HuggingFace
2. Upload all project files
3. Set `OPENAI_API_KEY` in Space settings
4. The app will auto-deploy

### Streamlit Cloud
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Add secrets for API keys
4. Deploy with `streamlit run app.py -- --mode streamlit`

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py", "--chat", "gradio"]
```

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure RAGAS scores remain above 80%

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Support

- 📧 Email: [glennliew1@gmail.com]
- 🐛 Issues: [GitHub Issues](https://github.com/glennliew/ragchatbot/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/glennliew/ragchatbot/discussions)

## 🏆 Acknowledgments

- **LangChain** for the RAG framework
- **OpenAI** for GPT-4 and embeddings
- **RAGAS** for evaluation metrics
- **Gradio & Streamlit** for user interfaces
- **Chroma** for vector storage

---

Made with ❤️ for young learners everywhere! 🌟

**Happy Learning! 🎓📚**