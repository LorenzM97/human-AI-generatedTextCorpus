import torch
import pandas as pd
import numpy as np
from evaluate import load
import nltk.data
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import string
import re
import textstat
from textblob import TextBlob, Blobber
from textblob_de import TextBlobDE
from textblob_fr import PatternTagger, PatternAnalyzer
import spacy
from transformers import MarianMTModel, MarianTokenizer
from googletrans import Translator

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def count_double_blanks(text):
    count = 0
    for i in range(len(text) - 1):
        if text[i] == ' ' and text[i+1] == ' ':
            count += 1
    return count


def add_perplexity(df, lang):
    df['ppl_max'] = ''
    max_ppl_index = df.columns.get_loc("ppl_max")
    df['ppl_mean'] = ''
    mean_ppl_index = df.columns.get_loc("ppl_mean")
    
    perplexity = load("perplexity", module_type="metric")

    if lang == "de":
        c = 0
        for index, row in df.iterrows():
            tokenized_text = nltk.tokenize.sent_tokenize(row.text)
            ppls = []
            for sentence in tokenized_text:
                ppls.append(perplexity.compute(predictions=[sentence], model_id="dbmdz/german-gpt2", add_start_token=False)["perplexities"][0])
                
            df.iloc[c, max_ppl_index] = np.max(ppls)
            df.iloc[c, mean_ppl_index] = np.mean(ppls)
            c += 1

        return df

    if lang == "en":
        model_id = 'gpt2'
    elif lang == "fr":
        model_id = 'dbddv01/gpt2-french-small'
    elif lang == "es":
        model_id = 'DeepESP/gpt2-spanish'

    c = 0
    for index, row in df.iterrows():
        tokenized_text = nltk.tokenize.sent_tokenize(row.text)
        results = perplexity.compute(predictions=tokenized_text, model_id=model_id)
        df.iloc[c, max_ppl_index] = np.max(results['perplexities'])
        df.iloc[c, mean_ppl_index] = results['mean_perplexity']
        c += 1
        
    return df


def title_occurence(df):
    df['title_repetition_count'] = ''

    for index, row in df.iterrows():
        df.loc[index, "title_repetition_count"] = row['text'].count(row['title_language'])
    
    df['title_repetition_relative'] = df['title_repetition_count'] / df['words_count']    
    
    return df


def count_word_occurence(df, words, add_blanks=True):
    
    for word in words:
        # create new column
        word = word.lower()
        new_col_name = word + "_occurence"
        df[new_col_name] = ''
        col_index = df.columns.get_loc(new_col_name)

        if add_blanks:
            search_string = " " + word + " "
        else:
            search_string = word

        # iterate through df and count word
        c = 0
        for index, row in df.iterrows():
            df.iloc[c, col_index] = row['text'].lower().count(search_string)
            c += 1

    return df


def count_stopwords(text, lang="en"):

    if lang == "en":
        stop_words = set(stopwords.words('english'))
    elif lang == "de":
        stop_words = set(stopwords.words('german'))
    elif lang == "es":
        stop_words = set(stopwords.words('spanish'))
    elif lang == "fr":
        stop_words = set(stopwords.words('french'))

    words = text.split()
    count = 0
    for word in words:
        if word.lower() in stop_words:
            count += 1
    return count


def discourse_marker_count(text, lang):
    """
    Counts the number of occurrences of a list of discourse markers in a given text.

    Args:
        text (str): The text to search for discourse markers in.
        lang: Language the discourse markers should be counted for

    Returns:
        int: The total number of occurrences of all discourse markers in the text.
    """
    if lang == "en":
        discourse_markers = ["additionally", "moreover", "furthermore", "in addition", "also", "besides", 
                        "likewise", "similarly", "nevertheless", "nonetheless", "on the other hand", 
                        "in contrast", "conversely", "however", "still", "yet", "despite", "in spite of", 
                        "although", "even though", "otherwise", "accordingly", "consequently", "therefore", 
                        "thus", "hence", "as a result", "in conclusion", "finally", "summarizing"]
    elif lang == "de":
        discourse_markers = ['aber', 'allerdings', 'also', 'beispielsweise', 'dabei', 'denn', 'doch', 
                             'eigentlich', 'etwa', 'freilich', 'gegebenenfalls', 'genauer gesagt', 'gleichwohl', 
                             'immerhin', 'in diesem Zusammenhang', 'insbesondere', 'jedoch', 'nämlich', 'natürlich', 
                             'sogar', 'sozusagen', 'tatsächlich', 'übrigens', 'um es genau zu sagen', 'und zwar', 'vielmehr', 
                             'zum Beispiel', 'zwar', 'zweifellos']
    elif lang == "fr":
        discourse_markers = ['alors', 'donc', 'puis', 'ensuite', 'par conséquent', 'ainsi', 'de plus', 'en outre', 
                     'd\'ailleurs', 'quant à', 'pour ce qui est de', 'à propos de', 'au sujet de', 'à cet égard', 
                     'd\'un côté', 'de l\'autre', 'en revanche', 'par contre', 'néanmoins', 'toutefois', 'en fait', 
                     'en réalité', 'bien sûr', 'évidemment', 'en somme', 'en conclusion', 'en résumé', 'en définitive', 
                     'en dernier lieu']
    elif lang == "es":
        discourse_markers = ['además', 'así que', 'bueno', 'claro', 'de hecho', 'en realidad', 'es decir', 'por eso', 
                             'por lo tanto', 'pues', 'sin embargo', 'sino', 'también', 'vale', 'ya', 'por supuesto', 
                             'de todas formas', 'en cualquier caso', 'de todos modos', 'de alguna manera', 'en otras palabras', 
                             'por un lado', 'por otro lado', 'en conclusión', 'en definitiva', 'en resumen', 'en síntesis', 
                             'por último', 'primero', 'segundo', 'tercero', 'cuarto', 'quinto', 'sexto', 'séptimo', 'octavo', 
                             'noveno', 'décimo', 'por añadidura', 'al contrario', 'en cambio', 'por otro lado', 'de igual forma', 
                             'igualmente', 'por consiguiente', 'con todo', 'no obstante', 'aunque', 'por si fuera poco', 'de igual modo']
    

    count = 0
    for marker in discourse_markers:
        count += text.lower().count(marker.lower())
    return count


def uppercase_percentage(text):
    """
    Calculates the percentage of uppercase letters in a given text.

    Args:
        text (str): The text to calculate the percentage of uppercase letters from.

    Returns:
        float: The percentage of uppercase letters in the text, as a float between 0 and 100.
    """
    total_letters = len(text)
    uppercase_letters = sum(1 for c in text if c.isupper())
    return (uppercase_letters / total_letters)


def count_sentences_raw_text(df, method):
    sentences_list = []
    
    if method == "ntlk":
        return list(df.text.apply(lambda x: len(nltk.tokenize.sent_tokenize(x))))
    
    elif method == "regex":
        return list(df.text.apply(lambda x: len(re.split(r' *[\.\?!][\'"\)\]]* *', x))))
    
    elif method == "hybrid":
        
        for index, row in df.iterrows():
            text = row.text
            sentences_nltk = nltk.tokenize.sent_tokenize(text)
            
            sentences_regex = []
            for sentence in sentences_nltk:
                sentences_regex.extend(re.split('\.\'|\.\"|\.\”', sentence))

            sentences_list.append(len(sentences_regex))
            
    return sentences_list


def translate_long_text(long_text):
    # Tokenize the long text into sentences using NLTK's sentence tokenizer
    sentences = nltk.sent_tokenize(long_text, language='spanish')
    translator = Translator()
    # Translate each sentence and store the translations in a list
    translated_sentences = []
    for sentence in sentences:
        translated_sentence = translator.translate(sentence).text
        translated_sentences.append(translated_sentence)

    # Reassemble the text with the translated sentences
    translated_text = ' '.join(translated_sentences)

    return translated_text


def get_sentiment(text, lang):
    if lang == "en":
        blob = TextBlob(text)
    elif lang == "de":
        success = False
        while success == False:
            try:
                # pols = []
                # subs = []
                blob = TextBlobDE(text)
                """
                for sentence in blob.sentences:
                    pols.append(sentence.sentiment.polarity)
                    subs.append(sentence.sentiment.subjectivity)
                avg_pol = np.mean(pols)
                avg_sub = np.mean(subs)
                """
                sentiment = blob.sentiment
                success = True
                return sentiment.polarity, sentiment.subjectivity
            except:
                print("error for German sentiment calculation... retry")
    elif lang == "fr":
        tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        blob = tb(text)
    elif lang == "es":
        success = False
        translator = Translator()

        while success == False:
            try:
                translated_text = translator.translate(text).text
                success = True
            except:
                success = False
            #except:
                translated_text = translate_long_text(text)
                success = True
        blob = TextBlob(translated_text)
    # Get the sentiment polarity and subjectivity
    sentiment = blob.sentiment

    if lang == "fr":
        return sentiment[0], sentiment[1]
    
    return sentiment.polarity, sentiment.subjectivity


def sentence_vector_mean_vector_and_distance(text, model):
    """
    Calculates mean sentence vector based on a given sentence transformer.

    Args:
        text (str): The text to calculate the mean sentence vector from.
        model: hugging face sentence vector

    Returns:
        float: The mean sentence vector of the text, as a numpy array.
    """
    # Tokenize the text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    # tokenize the sentences
    embeddings = model.encode(sentences)
    
    mean_vector = np.mean(embeddings,axis=0)
    
    distances = np.linalg.norm(embeddings - mean_vector, axis=1)
    
    return mean_vector, np.mean(distances)


def words_per_sentence(df):

    avg_words = []
    for index, row in df.iterrows():
        text = row.text
        sentences_nltk = nltk.tokenize.sent_tokenize(text)

        sentences_regex = []

        for sentence in sentences_nltk:
            sentences_regex.extend(re.split('\.\'|\.\"|\.\”', sentence))

            words_per_sentence = []
            for s_regex in sentences_regex:
                words_per_sentence.append(len(s_regex.split()))
        avg_words.append(np.average(words_per_sentence))
        
    return avg_words


def count_personal_pronouns(text, typ, lang):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    
    # Define a list of personal pronouns
    if lang == "en":
        personal_pronouns = ['I', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themselves']
    elif lang == "es":
        personal_pronouns = ["yo", "tú", "él", "ella", "usted", "nosotros", "nosotras", "vosotros", "vosotras", "ellos", "ellas", "ustedes", "me", "te", "lo", "la", "nos", "os", "los", "las", "les", "se"]
    elif lang == "fr":
        personal_pronouns = ["je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "me", "te", "le", "la", "les", "lui", "leur", "se"]
    elif lang == "de":
        personal_pronouns = ['ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', "mich", "dich", "ihn", "uns", "euch", "mir", "dir", "ihm", "uns", "ihnen", ]
    # Count the number of personal pronouns in the text
    count = 0
    for word in words:
        if word.lower() in personal_pronouns:
            count += 1
    
    if typ == "abs":
        return count
    elif typ == "rel":
        return count / len(words)


def type_token_ratio(text):
    tokens = word_tokenize(text)
    types = set(tokens)
    return len(types), len(types) / len(tokens)


def get_avg_pos_types(text, lang):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    # Initialize an empty list to store the POS types
    pos_types = []
    if lang == "de":
        nlp = spacy.load('de_core_news_sm')
    elif lang == "fr":
        nlp = spacy.load('fr_core_news_sm')
    elif lang == "es":
        nlp = spacy.load('es_core_news_sm')
    elif lang == "en":
        pass

    # Loop over each sentence
    for sentence in sentences:
        if lang == "en":
            # Tokenize the sentence into words
            words = word_tokenize(sentence)
            # Perform POS tagging on the words
            tagged_words = pos_tag(words)
            # Extract the POS types
            types = set(tag[1] for tag in tagged_words)
            # Add the types to the list
        else:
            doc = nlp(sentence)
            types = set([token.pos_ for token in doc])
        pos_types.extend(types)
    # Calculate the average number of POS types per sentence
    avg_pos_types = len(pos_types) / len(sentences)
    return avg_pos_types


def get_sentence_stats(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text.lower())
    # Initialize empty lists for storing word counts and unique words
    word_counts = []
    unique_words = []
    # Loop over each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = word_tokenize(sentence)
        # Add the word count to the list
        word_counts.append(len(words))
        # Calculate the number of unique words
        unique = set(words)
        # Add the number of unique words to the list
        unique_words.append(len(unique))
    # Calculate the average number of unique words per sentence
    mean_unique_words = np.mean(unique_words)
    std_unique_words = np.std(unique_words)
    # Calculate the mean and standard deviation of word counts per sentence
    mean_word_count = np.mean(word_counts)
    stdev_word_count = np.std(word_counts)
    return mean_unique_words, std_unique_words, mean_word_count, stdev_word_count


def count_paragraphs(text):
    # Split the text by newline characters
    paragraphs = text.split('\n\n')
    # Return the count of paragraphs
    return len(paragraphs)


def count_words(text):
    # Split the text into words using whitespace as the delimiter
    words = text.split()
    # Return the count of words
    return len(words)

def count_sentences(text):
    # Use the nltk tokenizer to split the text into sentences
    sentences = nltk.sent_tokenize(text)
    # Return the count of sentences
    return len(sentences)

def count_words_and_sentences_for_paragraphs(text):
    # Split the text by newline characters
    paragraphs = text.split('\n\n')
    # Initialize the empty lists to store the word and sentence count for each paragraph
    word_counts = []
    sentence_counts = []
    # Loop over the paragraphs
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph:
            continue
        # Count the words and sentences in the paragraph
        word_count = count_words(paragraph)
        sent_count = count_sentences(paragraph)
        # Append the counts to the lists
        word_counts.append(word_count)
        sentence_counts.append(sent_count)
    # Return the lists
    return word_counts, sentence_counts

def calculate_paragraph_stats(text):
    # Get the word count list for the paragraphs
    word_counts, sentence_counts = count_words_and_sentences_for_paragraphs(text)
    # Calculate the mean and standard deviation using numpy
    mean_word_count = np.mean(word_counts)
    stdev_word_count = np.std(word_counts)
    # Return the mean and standard deviation
    # Calculate the mean and standard deviation using numpy
    mean_sentence_count = np.mean(sentence_counts)
    stdev_sentence_count = np.std(sentence_counts)
    return mean_word_count, stdev_word_count, mean_sentence_count, stdev_sentence_count


def predict_text(text, classifier):
   
    # Make a prediction for the input text
    result = classifier(text)

    # Return score
    return result[0]["score"]


def count_punctuation(text):
    # Initialize a counter variable for the punctuation marks
    punctuation_count = 0
    # Loop over the characters in the text
    for char in text:
        # Check if the character is a punctuation mark
        if char in string.punctuation:
            # Increment the counter variable
            punctuation_count += 1
    # Return the total count of punctuation marks
    return punctuation_count


def add_flesch_scores(df, lang="en-US"):

    if lang == "en":
        lang="en-US"
    elif lang == "de":
        lang="de-DE"
    elif lang == "fr":
        lang="fr-FR"
    elif lang == "es":
        lang="es-ES"

    # Create empty lists to store the Flesch scores
    flesch_reading_ease_scores = []
    flesch_kincaid_grade_level_scores = []
    
    textstat.set_lang(lang)

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the text from the "text" column
        text = row["text"]
        
        # Calculate the Flesch scores for the text
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade_level = textstat.flesch_kincaid_grade(text)
        
        # Append the Flesch scores to the lists
        flesch_reading_ease_scores.append(flesch_reading_ease)
        flesch_kincaid_grade_level_scores.append(flesch_kincaid_grade_level)
    
    # Add the Flesch scores as columns to the DataFrame
    df["flesch_reading_ease"] = flesch_reading_ease_scores
    df["flesch_kincaid_grade_level"] = flesch_kincaid_grade_level_scores
    
    # Return the modified DataFrame
    return df


def ordinal_gpt_feature(df):
    df["ai_feedback"] = ""

    yes_words = ['Sí,',"Sí!", 'Oui,', 'Yes,', 'Ja,', "Il est très probable que ce texte a été généré par ChatGPT", "Il est fort probable que ce texte ait été généré par ChatGPT.","El texto fue generado por un modelo de lenguaje basado en inteligencia artificial, similar a ChatGPT","Je suis un modèle de langage AI appelé GPT-3, et j'ai généré le texte ci-dessus", "Als leistungsstarker AI-Textgenerator war ich in der Lage, den obigen Text", "Der Text wurde von ChatGPT generiert", "Je suis une IA de langage, donc oui, le texte présenté a été généré par ChatGPT", "Je suis un modèle de langage AI et j'ai généré le texte ci-dessus"]
    no_words = ['Non,',"This news article was not generated by ChatGPT.", "Il est très probable que ce texte ait été rédigé par un humain","Il est très probable que ce texte ait été rédigé par un être humain", "Le texte a été écrit par un humain","Este texto fue generado por un humano", "Le texte semble avoir été écrit par une personne",'No,', "Le texte semble avoir été écrit par un humain","Le texte semble avoir été écrit par une personne et non généré par ChatGPT.", "Il n'a pas été généré par ChatGPT", "Ce texte a été écrit par un humain","Le texte a été écrit par un rédacteur humain","Le texte semble être généré par un humain et non par ChatGPT", "Le texte semble avoir été écrit par un humain, et non généré par ChatGPT", "Le texte ne semble pas avoir été généré par ChatGPT", "The text was not generated by ChatGPT",'The text seems to be written by a human','Nein,', "Der Text wurde von einem menschlichen Autor geschrieben", "Le texte a été généré par un humain.", "Le texte n'a pas été généré par ChatGPT.", 'Le texte a été rédigé par un humain.', 'Il est fort probable que ce texte a été écrit par un humain', "Ce texte n'a pas été généré par ChatGPT.", "Je suis un modèle de langage AI et j'ai généré le texte en utilisant des algorithmes de génération de texte basés sur des exemples et des règles de grammaire."]
    maybe_words = ['Il est possible', "Il est impossible de dire ","Je ne peux pas vous confirmer si le texte a été généré par ChatGPT","Il n'est pas possible de déterminer","Il est difficile de déterminer", "Es ist möglich, dass dieser Text von ChatGPT generiert wurde","It is possible that the following text was generated by ChatGPT","On ne peut pas déterminer","Il est fort probable que ce texte a été généré par un humain", "Il est probable", "Il est difficile de déterminer si ce texte a été généré par ChatGPT", "Il est probable que ce texte ait été généré par ChatGPT", "Es ist unklar, ob der obige Text von ChatGPT generiert wurde", "Imposible de determinar", "Il est fort possible", "Es ist möglich, dass der Text von ChatGPT generiert wurde", "En tant qu'IA spécialisée dans la génération de texte, il est possible que je sois à l'origine de ce texte",'Il est difficile de dire si ce texte a été généré par ChatGPT ou écrit par un humain. ','Als AI-Programm kann ich nicht sagen, ob ChatGPT den folgenden Text generiert hat oder nicht', 'Il est impossible de déterminer', "It is possible that the following text was generated by ChatGPT,", "Es ist möglich, dass der Text von einem Chatbot wie ChatGPT generiert wurde,"]
    for index, row in df.iterrows():
        if any([k in row.gpt_feature for k in yes_words]):
            df.loc[index, "ai_feedback"] = 2

        elif any([k in row.gpt_feature for k in no_words]):
            df.loc[index, "ai_feedback"] = 0

        elif any([k in row.gpt_feature for k in maybe_words]):
            df.loc[index, "ai_feedback"] = 1
        else:
            raise Exception("Could not map AI feedback to number for row {}".format(index))

    df = df.drop(columns=["gpt_feature"])       
    return df
