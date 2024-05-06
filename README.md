# Human-AI-Generated Text Corpus
The dataset contains human-generated, AI-generated and AI-rephrased texts from the educational domain in English, French, German, and Spanish and English texts
from the news domain.

The educational human-generated texts consist of 100 Wikipedia texts from the following categories:
- Biology
- Chemistry
- Geography
- History
- IT
- Music
- Politics
- Religion
- Sports
- Visual arts

The news human-generated texts consist of 100 news articles from the following categories:
- Crime
- Entertainment
- Politics
- Science
- Sports

For the generation of the AI-written texts, GPT-3.5 was used. For each human-generated text, 4 AI-written texts were generated using the following prompts:

### Basic AI-generated
Generate a text on the following topic: <Title>

### Advanced AI-generated
Generate a text on the following topic in a way a human would do it: <Title>

### Basic AI-rephrased
Rephrase a text on the following topic: <Text>

### Advanced AI-rephrased
Generate a text on the following topic in a way a human would do it: <Text>

For the other languages, the prompts were translated.
