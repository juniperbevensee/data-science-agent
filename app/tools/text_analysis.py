import os
import logging
import pandas as pd
import numpy as np
from collections import Counter
from app.sandbox import resolve_path

# Suppress verbose logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# NLTK imports and downloads
import nltk
import ssl

def _download_nltk_data():
    """Download required NLTK data with SSL fallback."""
    required = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]

    for path, name in required:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                # Fallback: disable SSL verification
                try:
                    ssl._create_default_https_context = ssl._create_unverified_context
                    nltk.download(name, quiet=True)
                except Exception as e:
                    logging.warning(f"Could not download NLTK data '{name}': {e}")

_download_nltk_data()

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# TextBlob for sentiment
from textblob import TextBlob

# Wordcloud
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Scikit-learn for topic modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def _get_unique_filename(base_path: str) -> str:
    """Generate a unique filename to avoid overwriting existing files."""
    if not os.path.exists(base_path):
        return base_path

    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)

    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_path = os.path.join(directory, new_filename) if directory else new_filename
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def preprocess_text(
    path: str,
    column: str,
    output_path: str = None,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    lemmatize: bool = False
) -> dict:
    """
    Preprocess text data: tokenize, clean, and optionally remove stopwords/lemmatize.
    """
    df = pd.read_csv(resolve_path(path))

    if column not in df.columns:
        return {"success": False, "error": f"Column '{column}' not found"}

    stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    lemmatizer = WordNetLemmatizer() if lemmatize else None

    def process_text(text):
        if pd.isna(text):
            return ""

        text = str(text)
        if lowercase:
            text = text.lower()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove punctuation and stopwords
        tokens = [t for t in tokens if t.isalnum()]
        if remove_stopwords:
            tokens = [t for t in tokens if t not in stop_words]

        # Lemmatize
        if lemmatize and lemmatizer:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]

        return ' '.join(tokens)

    df[f'{column}_processed'] = df[column].apply(process_text)

    result = {
        "success": True,
        "processed_column": f'{column}_processed',
        "sample": df[f'{column}_processed'].head(3).tolist()
    }

    if output_path:
        full_path = resolve_path(output_path)
        df.to_csv(full_path, index=False)
        result["output"] = output_path

    return result


def word_frequency(
    path: str,
    column: str,
    top_n: int = 20,
    remove_stopwords: bool = True,
    output_path: str = None,
    create_plot: bool = False,
    plot_path: str = "word_frequency.png",
    overwrite: bool = False
) -> dict:
    """
    Analyze word frequency in text data.
    """
    df = pd.read_csv(resolve_path(path))

    if column not in df.columns:
        return {"success": False, "error": f"Column '{column}' not found"}

    # Combine all text
    all_text = ' '.join(df[column].dropna().astype(str))

    # Tokenize and clean
    tokens = word_tokenize(all_text.lower())
    tokens = [t for t in tokens if t.isalnum()]

    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]

    # Count frequency
    word_freq = Counter(tokens)
    top_words = dict(word_freq.most_common(top_n))

    result = {
        "success": True,
        "total_words": len(tokens),
        "unique_words": len(word_freq),
        "top_words": top_words
    }

    # Save to CSV if requested
    if output_path:
        freq_df = pd.DataFrame(list(top_words.items()), columns=['word', 'frequency'])
        freq_df.to_csv(resolve_path(output_path), index=False)
        result["output"] = output_path

    # Create plot if requested
    if create_plot:
        plt.figure(figsize=(12, 6))
        words = list(top_words.keys())
        counts = list(top_words.values())
        plt.barh(words[::-1], counts[::-1])
        plt.xlabel('Frequency')
        plt.title(f'Top {top_n} Most Frequent Words')
        plt.tight_layout()

        full_plot_path = resolve_path(plot_path)
        if not overwrite:
            full_plot_path = _get_unique_filename(full_plot_path)

        plt.savefig(full_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        relative_path = os.path.relpath(full_plot_path, os.path.dirname(resolve_path("")))
        result["plot"] = relative_path

    return result


def sentiment_analysis(
    path: str,
    column: str,
    output_path: str = None
) -> dict:
    """
    Perform sentiment analysis on text data using TextBlob.
    Returns polarity (-1 to 1) and subjectivity (0 to 1) scores.
    """
    df = pd.read_csv(resolve_path(path))

    if column not in df.columns:
        return {"success": False, "error": f"Column '{column}' not found"}

    def analyze_sentiment(text):
        if pd.isna(text):
            return 0, 0, 'neutral'

        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return polarity, subjectivity, sentiment

    # Apply sentiment analysis
    sentiments = df[column].apply(analyze_sentiment)
    df['sentiment_polarity'] = sentiments.apply(lambda x: x[0])
    df['sentiment_subjectivity'] = sentiments.apply(lambda x: x[1])
    df['sentiment_label'] = sentiments.apply(lambda x: x[2])

    # Calculate statistics
    avg_polarity = df['sentiment_polarity'].mean()
    sentiment_counts = df['sentiment_label'].value_counts().to_dict()

    result = {
        "success": True,
        "average_polarity": round(avg_polarity, 3),
        "sentiment_distribution": sentiment_counts,
        "sample": df[[column, 'sentiment_polarity', 'sentiment_label']].head(5).to_dict(orient='records')
    }

    if output_path:
        full_path = resolve_path(output_path)
        df.to_csv(full_path, index=False)
        result["output"] = output_path

    return result


def create_word_cloud(
    path: str,
    column: str,
    output_path: str = "wordcloud.png",
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
    remove_stopwords: bool = True,
    max_words: int = 100,
    overwrite: bool = False
) -> dict:
    """
    Generate a word cloud visualization from text data.
    """
    df = pd.read_csv(resolve_path(path))

    if column not in df.columns:
        return {"success": False, "error": f"Column '{column}' not found"}

    # Combine all text
    text = ' '.join(df[column].dropna().astype(str))

    if not text.strip():
        return {"success": False, "error": "No text data found"}

    # Configure stopwords
    stopwords_set = set(stopwords.words('english')) if remove_stopwords else None

    # Generate word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        stopwords=stopwords_set,
        max_words=max_words
    ).generate(text)

    # Create plot
    plt.figure(figsize=(width/100, height/100))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)

    # Save
    full_path = resolve_path(output_path)
    if not overwrite:
        full_path = _get_unique_filename(full_path)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_path, os.path.dirname(resolve_path("")))

    return {
        "success": True,
        "output": relative_path,
        "words_displayed": len(wordcloud.words_)
    }


def topic_extraction(
    path: str,
    column: str,
    num_topics: int = 5,
    num_words: int = 10,
    output_path: str = None
) -> dict:
    """
    Extract topics from text data using Latent Dirichlet Allocation (LDA).
    """
    df = pd.read_csv(resolve_path(path))

    if column not in df.columns:
        return {"success": False, "error": f"Column '{column}' not found"}

    # Get text data
    texts = df[column].dropna().astype(str).tolist()

    if len(texts) < num_topics:
        return {"success": False, "error": f"Not enough documents. Need at least {num_topics}, got {len(texts)}"}

    # Create document-term matrix using CountVectorizer
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words='english',
        token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with 3+ letters
    )

    try:
        doc_term_matrix = vectorizer.fit_transform(texts)
    except ValueError as e:
        return {"success": False, "error": f"Unable to create document-term matrix: {str(e)}"}

    # Train LDA model
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=10,
        learning_method='online',
        n_jobs=-1
    )

    lda_model.fit(doc_term_matrix)

    # Extract topics
    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    for topic_idx, topic in enumerate(lda_model.components_):
        # Get top words for this topic
        top_indices = topic.argsort()[-num_words:][::-1]
        top_words = [(feature_names[i], topic[i]) for i in top_indices]

        # Normalize weights to sum to 1 for better interpretability
        total_weight = sum(weight for _, weight in top_words)
        topics[f"topic_{topic_idx+1}"] = [
            {"word": word, "weight": round(weight/total_weight, 4)}
            for word, weight in top_words
        ]

    result = {
        "success": True,
        "num_topics": num_topics,
        "topics": topics,
        "num_documents": len(texts),
        "vocabulary_size": len(feature_names)
    }

    # Save to file if requested
    if output_path:
        topics_df = []
        for topic_name, words in topics.items():
            for item in words:
                topics_df.append({
                    "topic": topic_name,
                    "word": item["word"],
                    "weight": item["weight"]
                })
        pd.DataFrame(topics_df).to_csv(resolve_path(output_path), index=False)
        result["output"] = output_path

    return result


# Tool schemas for LLM
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "preprocess_text",
            "description": "Preprocess text data: tokenize, clean, remove stopwords, and optionally lemmatize",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to CSV file"},
                    "column": {"type": "string", "description": "Column containing text data"},
                    "output_path": {"type": "string", "description": "Path to save processed data"},
                    "lowercase": {"type": "boolean", "default": True},
                    "remove_stopwords": {"type": "boolean", "default": True},
                    "lemmatize": {"type": "boolean", "default": False}
                },
                "required": ["path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "word_frequency",
            "description": "Analyze word frequency in text data, optionally create visualization",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "column": {"type": "string"},
                    "top_n": {"type": "integer", "default": 20},
                    "remove_stopwords": {"type": "boolean", "default": True},
                    "output_path": {"type": "string"},
                    "create_plot": {"type": "boolean", "default": False},
                    "plot_path": {"type": "string", "default": "word_frequency.png"},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sentiment_analysis",
            "description": "Perform sentiment analysis on text data, returns polarity and sentiment labels",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "column": {"type": "string"},
                    "output_path": {"type": "string"}
                },
                "required": ["path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_word_cloud",
            "description": "Generate a word cloud visualization from text data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "column": {"type": "string"},
                    "output_path": {"type": "string", "default": "wordcloud.png"},
                    "width": {"type": "integer", "default": 800},
                    "height": {"type": "integer", "default": 400},
                    "background_color": {"type": "string", "default": "white"},
                    "remove_stopwords": {"type": "boolean", "default": True},
                    "max_words": {"type": "integer", "default": 100},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "topic_extraction",
            "description": "Extract topics from text data using LDA topic modeling",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "column": {"type": "string"},
                    "num_topics": {"type": "integer", "default": 5},
                    "num_words": {"type": "integer", "default": 10},
                    "output_path": {"type": "string"}
                },
                "required": ["path", "column"]
            }
        }
    }
]

# Map function names to implementations
TOOLS = {
    "preprocess_text": preprocess_text,
    "word_frequency": word_frequency,
    "sentiment_analysis": sentiment_analysis,
    "create_word_cloud": create_word_cloud,
    "topic_extraction": topic_extraction,
}


if __name__ == "__main__":
    from app.tools.file_ops import write_csv

    # Create test data
    test_data = [
        {"id": 1, "review": "This product is amazing! I love it so much. Best purchase ever!"},
        {"id": 2, "review": "Terrible quality. Very disappointed. Would not recommend."},
        {"id": 3, "review": "It's okay. Nothing special but does the job."},
        {"id": 4, "review": "Excellent service and great product. Highly recommend!"},
        {"id": 5, "review": "Worst experience ever. Product broke after one day."},
    ]

    write_csv("test_reviews.csv", test_data)

    print("Testing text analysis tools...")

    # Test word frequency
    print("\n1. Word Frequency:")
    result = word_frequency("test_reviews.csv", "review", top_n=10, create_plot=True)
    print(result)

    # Test sentiment analysis
    print("\n2. Sentiment Analysis:")
    result = sentiment_analysis("test_reviews.csv", "review", output_path="sentiment_results.csv")
    print(result)

    # Test word cloud
    print("\n3. Word Cloud:")
    result = create_word_cloud("test_reviews.csv", "review")
    print(result)

    print("\nAll text analysis tools tested successfully!")
