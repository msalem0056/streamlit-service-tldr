"""
A Streamlit App that allows the user to search for a topic and summarize articles
Idea for app: https://medium.com/analytics-vidhya/text-summarization-in-python-using-extractive-method-including-end-to-end-implementation-2688b3fd1c8c
"""

# -- Standard Imports
from importlib import import_module
import os
import subprocess
import time

# -- Third-party Imports
from mpire import WorkerPool
import networkx as nx
import newspaper
import numpy as np
import pandas as pd
from GoogleNews import GoogleNews
import spacy
from spacy.lang.en import English
import streamlit as st

# -- Local Imports

__author__ = "msalem0056"
__created__ = "October 2022"


@st.cache()
def download_and_init_nlp(model_name: str):
    """Downloads the pretrained language model"""
    subprocess.run(['python', '-m', 'spacy', 'download', f'{model_name}'])
    return spacy.load(model_name)


@st.cache()
def get_news(search_text: str) -> pd.DataFrame:
    """
    Gets the list of articles from a Google Search

        Parameters:
                search_text (str): Topic to search

        Returns:
                dataframe (pd.DataFrame): Information collected from the search
    """
    if search_text == "":
        return None
    gn = GoogleNews()
    # Note the when can be changed if desired
    gn.search(search_text)
    return gn.results(sort=True)


def read_article(text: str) -> list[str]:
    """
    Reads and article and cleans up sentences

        Parameters:
                text (str): An article

        Returns:
                list (list[str]): sentences in list format
    """
    sentences = []
    sentences = tokenizer(text)
    for sentence in sentences:
        sentence = str(sentence).replace("[^a-zA-Z0-9]", " ")
    return sentences


def sentence_similarity(s1: str, s2: str, nlp) -> float:
    """
    Calculates the similarity between two sentences

        Parameters:
                s1 (str): A sentence
                s2 (str): A sentence
                nlp : Spacy nlp model

        Returns:
                float (float): sentence similarity score [0,1]
    """
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    return doc1.similarity(doc2)


def build_similarity_matrix(sentences: list[str], nlp) -> np.matrix:
    """
    Build similarity matrix between sentences

        Parameters:
                sentences (list[str]): A list of sentences
                nlp : Spacy nlp model

        Returns:
                matrix (np.matrix): matrix of similarity scores
    """
    # create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(
                    sentences[idx1].text, sentences[idx2].text, nlp)
    return similarity_matrix


def tl_dr(title, media, date, datetime, desc, link, img):
    """Grabs the article description from the URL. Summarizes it and collects some stats."""
    try:
        article = newspaper.Article(url=link, language='en')
        article.download()
        time.sleep(.01)
        article.parse()
        article = {
            "title": str(article.title),
            "text": str(article.text),
            'link': link,
            "authors": article.authors,
            "published_date": str(article.publish_date),
            "top_image": str(article.top_image),
            "videos": article.movies,
            "keywords": article.keywords,
            "summary": str(article.summary),
            "text_len": None,
            "summary_len": None,
        }

        summary = ""
        if len(article["text"]):
            summarize_text = []
            num_sent = 5

            # Create a Tokenizer with the default settings for English
            # including punctuation rules and exceptions
            tokenizer = nlp.tokenizer

            # Build sentence similarity matrix
            sentences = list(nlp(article['text']).sents)
            sentence_sim_matrix = build_similarity_matrix(sentences, nlp)

            # Pagerank gives the most important references
            sentence_sim_graph = nx.from_numpy_array(sentence_sim_matrix)
            scores = nx.pagerank(sentence_sim_graph)

            # Sort by highest score and create summary
            ranked_sentences = sorted(
                ((scores[i], s.text) for i, s in enumerate(sentences)), reverse=True)
            for i in range(num_sent):
                summarize_text.append(ranked_sentences[i][1])
            summary, _ = " ".join(summarize_text), len(sentences)

            article['summary'] = summary
            article['text_len'] = len(str(article['text']))
            article['summary_len'] = len(summary)
    except BaseException:
        article = {
            "title": None,
            "text": None,
            'link': None,
            "authors": None,
            "published_date": None,
            "top_image": None,
            "videos": None,
            "keywords": None,
            "summary": None,
            'summary': None,
            'text_len': None,
            'summary_len': None
        }
    return article


# -- Main
if __name__ == "__main__":

    nlp = download_and_init_nlp("en_core_web_md")
    st.sidebar.write("# Controls and Stats")
    search_text = st.sidebar.text_area('Items to search here:')

    st.markdown("# tl;dr News")
    st.markdown("## News for busy people")
    st.markdown("""Disclaimer: This is not intended to be a commercial product, solution, or tool. Any likeness to any commercial or personal products are purely coincidental.""")

    # Get news and limit to 10 items for speed / processing
    df = pd.DataFrame(get_news(search_text))

    if df is not None and df.shape[0]:
        df = df.head(10)
        with WorkerPool(n_jobs=os.cpu_count() - 1) as pool:
            results = pool.map(
                tl_dr, list(
                    df.itertuples(
                        index=False, name=None)))
        stats_news = []
        print(results)
        for result in results:
            if result['title'] is None:
                continue
            try:
                st.markdown(
                    '### <a href="{}" rel="noopener noreferrer" target="_blank">{}</a>'.format(
                        result['link'], result['title']), unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                col1.markdown(
                    result['link'].replace(
                        "https://",
                        "").replace(
                        "www.",
                        "").split("/")[0])
                col2.markdown(
                    result['published_date'] if result['published_date'] != "None" else "")
                try:
                    st.image(result['top_image'])
                except BaseException:
                    pass
                st.markdown("### Summary")
                st.markdown(result['summary'])
                st.markdown("***")
                stats_news.append(result)
            except BaseException:
                pass

        stats_df = pd.DataFrame(stats_news)
        stats_df = stats_df.set_index("link")
        fig = stats_df['text_len'].hist()
        st.sidebar.write("Document Character Length")
        st.sidebar.bar_chart(stats_df.text_len)

        fig = stats_df['summary_len'].hist()
        st.sidebar.write("Summary Character Length")
        st.sidebar.bar_chart(stats_df.summary_len)
