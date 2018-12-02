from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path
from gensim.models import KeyedVectors
import pdb
import numpy as np
import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request
from IPython import embed
from qanta import util
from qanta.dataset import QuizBowlDataset
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')
MODEL_PATH = 'glove_tfidf.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3

def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    return guesses[0][0], buzz


def batch_guess_and_buzz(model, questions) -> List[Tuple[str, bool]]:
    question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)
    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        outputs.append((guesses[0][0], buzz))
    return outputs


class TfidfGuesser:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.tag_matrix = None
        self.glove_weights = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        questions = training_data[0]
        answers = training_data[1]
        # questions = training_data[0][:2000]
        # answers = training_data[1][:2000]

        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        #print("Questions: " + str(len(x_array)))
        #print("Answers: " + str(len(y_array)))

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=.9
        ).fit(x_array)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)
        print(" number of words: {}".format(len(self.tfidf_vectorizer.vocabulary_) ))

        emb_dim = 300
        file = "./data/glove.6B.300d.txt"

        sd = 1 / np.sqrt(emb_dim)
        self.glove_weights = np.random.normal(0, scale=sd, size=[len(self.tfidf_vectorizer.vocabulary_), emb_dim])
        self.glove_weights = self.glove_weights.astype(np.float32)
        #
        # with open(file, encoding="utf-8", mode="r") as textFile:
        #     for line in textFile:
        #         line = line.split()
        #         word = line[0]
        #
        #         id = self.tfidf_vectorizer.vocabulary_.get(word)
        #         if id is not None:
        #             self.glove_weights[id] = np.array(line[1:], dtype=np.float32)

        weight_path = './data/GoogleNews-vectors-negative300.bin'
        model = KeyedVectors.load_word2vec_format(weight_path, binary=True)
        for k,v in self.tfidf_vectorizer.vocabulary_.items():
            if k in model.vocab:
                self.glove_weights[v] = model[k]
            else:
                print("unfound word in word2vec: {}".format(k))

        # pdb.set_trace()
        self.tag_matrix = self.tfidf_matrix.dot(self.glove_weights)
        # self.tag_matrix = self.tag_transform(self.tfidf_matrix, x_array)


    def tag_transform(self, tfidf_representations, questions):
        result = []
        for tfidf_weights, question in zip(tfidf_representations.toarray(), questions):
            sum_weights = sum(tfidf_weights)
            weighted_glove_sum = 0
            for word in word_tokenize(question):
                i = self.tfidf_vectorizer.vocabulary_.get(word)
                if i is not None:
                    weighted_glove_sum += self.glove_weights[i]*tfidf_weights[i]
            #weighted_glove_sum = sum([ for i in range(len(tfidf_weights))])
            result.append(np.array(weighted_glove_sum/sum_weights))
        return result

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        tfidf_representations = self.tfidf_vectorizer.transform(questions)
        pdb.set_trace()
        # tag_representations = np.array(self.tag_transform(tfidf_representations, questions))
        tag_representations = tfidf_representations.dot(self.glove_weights)
        guess_matrix = np.array(self.tag_matrix).dot(tag_representations.T).T
        guess_indices = np.argsort(-guess_matrix, axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])
        return guesses

    def save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'tag_matrix': self.tag_matrix,
                'glove_weights': self.glove_weights
            }, f)

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            guesser.tag_matrix = params['tag_matrix']
            guesser.glove_weights = params['glove_weights']
            return guesser


def create_app(enable_batch=True):
    tfidf_guesser = TfidfGuesser.load()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(tfidf_guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(tfidf_guesser, questions)
        ])


    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.train(dataset.training_data())
    tfidf_guesser.save()


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
def download(local_qanta_prefix):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix)


if __name__ == '__main__':
    cli()