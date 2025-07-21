# ReAct Agent Implementation - Medical AI Assistant

A Vietnamese medical pre-diagnosis assistant built using the ReAct (Reasoning and Acting) framework with MedGemma-4B model. This project implements an intelligent agent that can analyze symptoms and provide preliminary medical guidance using retrieval-augmented generation.

## 🌟 Features

- **ReAct Framework**: Implements the Reasoning and Acting pattern for structured AI decision-making
- **Medical Knowledge Base**: Uses Vietnamese medical datasets for symptom-disease mapping
- **Multi-Model Support**: Compatible with MedGemma-4B and OpenAI models
- **Interactive Chat Interface**: Command-line interface for real-time medical consultations
- **Tool Integration**: Equipped with medical information search capabilities
- **Conversation Logging**: Automatic logging of chat sessions for analysis
- **DPO Training**: Direct Preference Optimization for model fine-tuning

## 📋 Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU (recommended for local model inference)
- 8GB+ RAM for model loading
- Internet connection for external API calls

## 🚀 Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/nguyenit67/ReAct-agent-implementation.git
cd ReAct-agent-implementation
```

### 2. Install Dependencies

This project uses [uv](https://docs.astral.sh/uv/) for fast dependency management:

```bash
# Install uv if you haven't already
pip install uv

# Install project dependencies
uv sync
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
# Model Configuration
MODEL_ID=nguyenit67/medgemma-4b-it-medical-agent-dpo
# MODEL_ID=google/medgemma-4b-it

# Hugging Face Token (for acessing privated models and uploaded trained models)
HF_TOKEN=your_huggingface_token_here

# Tavily API (for web search tool)
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Prepare Medical Knowledge Base

```bash
# Activate the environment
uv run prepare_index.py
```

This will create the TF-IDF index for medical information retrieval.

## 🏃‍♂️ Running the Application

### Main Interactive Assistant

```bash
$ uv run main.py
Nhập triệu chứng của bạn, 'clear' để reset lịch sử trò chuyện hoặc 'exit' để thoát: ...
```

The assistant will start in Vietnamese and prompt you to:

- Enter symptoms for medical consultation
- Type `clear` to reset conversation history
- Type `exit` to quit the application

### Example Interaction

```
Chào mừng đến với Trợ lý AI Y tế!
Nhập triệu chứng của bạn, 'clear' để reset lịch sử trò chuyện hoặc 'exit' để thoát: Tôi bị đau đầu, sốt nhẹ và đau họng

[Agent Analysis]
Thought: Người dùng mô tả các triệu chứng đau đầu, sốt và đau họng...
Action: Search[đau đầu, sốt, đau họng, cảm cúm]
...
Finish[Dựa trên các triệu chứng, có thể bạn đang bị cảm cúm hoặc viêm đường hô hấp trên...]
```

## 📁 Project Structure

```shell
ReAct-agent-implementation
├── data/                     # Medical datasets and indices
│   ├── dpo_train.json        # Data for DPO training on MedGemma-4B-IT
│   ├── symptoms.csv          # Data of disease names with symptom list
│   ├── tfidf_index.npz       # TF-IDF trained index
│   └── tfidf_vectorizer.pkl  # TF_IDF trained vectorizer
├── logs/                     # Agent chat logging sessions for each query
│   ├── gpt-4.1-mini-chat-2025_07_17-2_30.txt    
│   ├── medgemma-4b-it-2025_07_21-03_50.txt
│   └── ...                   
├── main.py                   # Entry point for the application
├── chat.py                   # Chat interface and model interaction
├── agent.py                  # Main ReAct agent implementation
├── model.py                  # Model loading and inference utilities
├── tools.py                  # Medical search tools and functions
├── logger.py                 # System & agent logging configuration
├── prepare_index.py          # TF-IDF index preparation
├── pyproject.toml            # Project dependencies and configuration
└── medgemma-4b-dpo.ipynb     # Notebook to run Direct Preference Optimization on MedGemma-4B-IT
```

## 🛠️ Key Components

### ReAct Agent (`agent.py`)

- Implements the ReAct framework with Thought-Action-Observation loops
- Manages conversation history and context
- Integrates medical search tools

### Medical Tools (`tools.py`)

- `search_disease_information()`: TF-IDF-based medical knowledge retrieval
- `search_disease_information_tavily()`: Web-based medical search

### Model Interface (`model.py`)

- Supports local Hugging Face models
- Handles tokenization and text generation
- Memory-efficient model loading

## 🔧 Configuration Options

### Model Selection

Edit the `MODEL_ID` in your `.env` file:

```env
# Local models
MODEL_ID=google/medgemma-4b-it
MODEL_ID=path/to/your/fine-tuned-model

# OpenAI models (requires API key)
MODEL_ID=gpt-4
MODEL_ID=gpt-3.5-turbo
```

### Memory Management

For systems with limited GPU memory, modify `model.py`:

```python
# Enable 8-bit quantization
load_in_8bit=True

# Enable 4-bit quantization (more aggressive)
load_in_4bit=True
```

## 🎯 Training and Fine-tuning

### Direct Preference Optimization (DPO)

Jupyter Notebooks:

Explore training processes:

## 📊 Data Sources

The project uses several Vietnamese medical datasets:

- **intent_train.json**: Disease symptoms and predictions from [Vietnamese-medical-chatbot-based](https://github.com/XuanHien304/Vietnamese-medical-chatbot-based)
- **symptoms.csv**: Comprehensive symptom database, transformed from **intent_train.json**
- **Disease-Scenario-SymptomDescription.csv**: Symptom-disease testing scenarios curated from LLM
- **dpo_train.json**: Training data for preference optimization, generated from LLM based on chat logs of agent

# 

### Performance Optimization

- Use GPU acceleration when available
- Enable model quantization for memory efficiency
- Adjust batch sizes based on available hardware

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MedGemma**: Google's medical language model
- **Vietnamese Medical Datasets**: Community-contributed medical knowledge
- **ReAct Framework**: Reasoning and Acting paradigm for LLM agents
- **Hugging Face Transformers**: Model infrastructure and utilities

## 📧 Contact

For questions or support, please open an issue in the repository or contact the development team.

---

_Built with ❤️ for Vietnamese healthcare accessibility_
