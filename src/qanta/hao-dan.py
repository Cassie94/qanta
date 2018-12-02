import numpy as np
import os
import pickle
import string
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from IPython import embed
from torch.utils.data import Dataset

import click
from flask import Flask, jsonify, request

from qanta.dataset import QuizBowlDataset


DAN_MODEL_DIR = 'dan'
BUZZ_THRESHOLD = 0.93
DAN_DEFAULT_CONFIG = {
    'ebd_dim': 300,
    'n_hidden_units': 150,
    #########
    'n_epochs': 1,
    'batch_size': 64,
    'lr': 1e-3,
    'log': 500,
    'cuda': False,
    'pretrained_word_ebd': True
}
STOPWORDS = set()
with open('stopwords.txt', 'r') as f:
    data = f.readlines()
    for x in data:
        x = x.strip()
        STOPWORDS.add(x)


def guess_and_buzz(guesser, question):
    guesses, probs = guesser.guess([question])
    buzz = probs[0] >= BUZZ_THRESHOLD
    return guesses[0], buzz


def batch_guess_and_buzz(model, questions):
    guesses, probs = model.guess(questions)
    outputs = []

    for guess, prob in zip(guesses, probs):
        buzz = prob >= BUZZ_THRESHOLD
        outputs.append((guess, buzz))

    return outputs


class QuestionDataset(Dataset):
    def __init__(self, questions, answers, word_to_idx, ans_to_idx):
        self.answers = answers
        self.word_to_idx = word_to_idx
        self.ans_to_idx = ans_to_idx

        questions_ = []
        for q in questions:
            q_text = ' '.join(q)
            q_text = preprocess_question(q_text)
            q_text = q_text.split()
            questions_.append(q_text)
        self.questions = questions_


    def __getitem__(self, index):
        return vectorize_question(self.questions[index], self.word_to_idx), \
               vectorize_answer(self.answers[index], self.ans_to_idx)


    def __len__(self):
        return len(self.questions)

class DanModel(nn.Module):
    def __init__(self, vocab_size, n_classes, ebd_dim=300, n_hidden_units=50):
        super(DanModel, self).__init__()

        print('Initializing DanModel...', end='')
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.ebd_dim = ebd_dim
        self.ans_ebd_matrix = None

        self.embedding = nn.Embedding(vocab_size, ebd_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.40)

        self.encoder = nn.Sequential(
            nn.Linear(ebd_dim, ebd_dim),
            nn.BatchNorm1d(ebd_dim),
            nn.ReLU(),
            # nn.Linear(n_hidden_units, ebd_dim),
            # nn.BatchNorm1d(ebd_dim),
            # nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(n_hidden_units, n_classes)

        print('Done!')


    def forward(self, text, text_length, is_prob=False):
        # Get embedding of text by taking average of embedding of words.
        word_embedding = self.embedding(text)
        text_embedding = word_embedding.sum(1)
        text_embedding /= text_length.view(text_length.size(0), -1)
        text_embedding = self.dropout(text_embedding)

        # Encode text for classification.
        text_encoded = self.encoder(text_embedding)

        # Get logits.
        # logits = self.classifier(text_encoded)
        logits = torch.mm(text_encoded, torch.t(self.ans_ebd_matrix))

        if is_prob:
            return F.softmax(logits, dim=1)
        else:
            return logits


    def inference(self, question):
        question = torch.from_numpy(
            np.array(question)).type(torch.LongTensor).reshape(1, -1)
        length = torch.from_numpy(
            np.array([len(question)])).type(torch.FloatTensor)
        model.eval()

        return model(question, length).data.numpy()

    def load_pretrained(self, word_to_idx):
        print('Loading pretrained word embedding...', end='')
        pretrained_weights = np.zeros((self.vocab_size, self.ebd_dim))
        with open("data/enwiki_20180420_300d.txt", mode='r') as f:
            for line in f:
                line = line.split(' ')
                word = line[0].replace('ENTITY/','')
                if word in word_to_idx:
                    word_idx = word_to_idx[word]
                    pretrained_weights[word_idx,:] = np.array(line[1:], dtype=np.float32)
        self.embedding.weight.data = torch.Tensor(pretrained_weights)
        print('Done!')

class DanGuesser:
    def __init__(self):
        print('Initializing DanGuesser...', end='')
        print('Done!')

    def add_answers_to_vocab(self, answer_set):
        for ans in answer_set:
            idx = len(self.vocab)
            self.word_to_idx[ans] = idx
            self.idx_to_word[idx] = ans
            self.vocab.add(ans)

    # def build_ans_embed_matrix(self, answer_set):


    def train(self, training_data, dev_data, cfg=DAN_DEFAULT_CONFIG,
              resume=False, hist=[], ckpt_file=''):
        print('--------DanGuess.train--------')
        questions, answers = training_data[0], training_data[1]

        # Build vocabulary set.
        print('Building vocabulary set + answer set...', end='')
        self.vocab, self.word_to_idx, self.idx_to_word = build_vocab(questions)

        self.answers_set, self.ans_to_idx, self.idx_to_ans = \
            build_answers_set(answers)
        self.n_answers = len(self.answers_set)
        self.add_answers_to_vocab(self.answers_set)
        self.vocab_size = len(self.vocab)
        print('Done!')

        self.model = DanModel(vocab_size=self.vocab_size,
                              n_classes=self.n_answers,
                              ebd_dim=cfg['ebd_dim'],
                              n_hidden_units=cfg['n_hidden_units'])
        if cfg['pretrained_word_ebd']:
            # self.model.load_pretrained(self.word_to_idx)
            # self.model.ans_ebd_matrix = torch.zeros(self.n_answers, cfg['ebd_dim'])
            # for ans in self.answers_set:
            #     self.model.ans_ebd_matrix[self.ans_to_idx[ans],:] = \
            #         self.model.embedding.weight.data[self.word_to_idx[ans]]
            with open('wiki2vec_300.pkl','rb') as f:
                self.model.embedding.weight.data = pickle.load(f)
            with open('ans_ebd_matrix.pkl', 'rb') as f:
                self.model.ans_ebd_matrix = pickle.load(f)

        if cfg['cuda']:
            self.model.cuda()

        train_dataset = QuestionDataset(questions, answers,
                                        self.word_to_idx, self.ans_to_idx)
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        dev_dataset = QuestionDataset(dev_data[0], dev_data[1],
                                      self.word_to_idx, self.ans_to_idx)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=cfg['batch_size'],
            sampler=dev_sampler, collate_fn=batchify)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        best_dev_accuracy = 0
        last_epoch = 0

        if resume:
            print('Resume training from file %s' % ckpt_file)
            print('Loading from checkpoint...', end='')
            ckpt = torch.load(os.path.join(DAN_MODEL_DIR, ckpt_file))
            self.model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            hist = ckpt['hist']
            best_dev_accuracy = max([x['dev_accuracy'] for x in hist])
            last_epoch = ckpt['last_epoch']
            print('Done!')

        criterion = nn.CrossEntropyLoss()
        if cfg['cuda']:
            criterion.cuda()

        print('Start training')
        for epoch in range(cfg['n_epochs']):
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg['batch_size'],
                sampler=train_sampler, collate_fn=batchify)
            print_loss_avg = 0

            for idx, batch in enumerate(train_loader):
                self.model.train()

                question_text = batch['text']
                question_len = batch['len']
                labels = batch['labels']
                if cfg['cuda']:
                    question_text = question_text.cuda()
                    question_len = question_len.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                logits = self.model(question_text, question_len, is_prob=False)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                print_loss_avg += loss

                if (idx + 1) % cfg['log'] == 0:
                    print_loss_avg /= cfg['log']
                    print('epoch %d/%d, iter %d/%d: ~train_loss = %.5f' % \
                          (epoch + 1 + last_epoch, cfg['n_epochs'] + last_epoch,
                          idx + 1, len(train_loader), print_loss_avg),
                          flush=True)
                    print_loss_avg = 0

            train_accuracy = self.evaluate(train_loader, cfg['cuda'])
            dev_accuracy = self.evaluate(dev_loader, cfg['cuda'])

            hist.append({
                'train_accuracy': train_accuracy,
                'dev_accuracy': dev_accuracy
            })

            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
            print('end of epoch %d: train_accuracy = %.5f, dev_accuracy = %.5f'%
                  (epoch + 1 + last_epoch, train_accuracy, dev_accuracy),
                  flush=True)

            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hist': hist,
                'last_epoch': epoch + 1 + last_epoch
            }
            checkpoint_filename = 'dan_epoch%d.ckpt' % (epoch + 1 + last_epoch)
            torch.save(checkpoint,
                       os.path.join(DAN_MODEL_DIR, checkpoint_filename))


    def evaluate(self, dataloader, cuda=False):
        self.model.eval()

        num_examples = 0
        error = 0
        for idx, batch in enumerate(dataloader):
            question_text = batch['text']
            question_len = batch['len']
            labels = batch['labels']
            if cuda:
                question_text = question_text.cuda()
                question_len = question_len.cuda()
                labels = labels.cuda()

            logits = self.model(question_text, question_len, is_prob=False)

            top_n, top_i = logits.topk(1)
            num_examples += question_text.size(0)
            error += torch.nonzero(top_i.squeeze() - labels).size(0)

        accuracy = 1 - error / num_examples

        return accuracy


    def guess(self, questions):
        self.model.eval()

        all_questions = []
        # print(questions, file=sys.stderr)
        for i in range(len(questions)):
            # q = ' '.join(questions[i])
            # print(q, file=sys.stderr)
            q = vectorize_question(preprocess_question(questions[i]).split(),
                                   self.word_to_idx)
            # print(q, file=sys.stderr)
            all_questions.append(q)

        questions, length = batchify_question(all_questions)
        ans_prob = self.model.forward(questions, length, is_prob=True).data.numpy()
        ans = np.argmax(ans_prob, axis=1)

        guesses = []
        guesses_prob = []
        for i in range(len(questions)):
            guesses.append(self.idx_to_ans[ans[i]])
            guesses_prob.append(ans_prob[i, ans[i]])

        return guesses, guesses_prob


    def load(self, cfg=DAN_DEFAULT_CONFIG, from_cuda=False):
        with open('vocab.p', 'rb') as f:
            data = pickle.load(f)
            vocab_size = data['vocab_size']
            n_answers = data['n_answers']
            word_to_idx = data['word_to_idx']
            idx_to_ans = data['idx_to_ans']

        if from_cuda:
            ckpt = torch.load('dan_exp8.ckpt', map_location='cpu')
        else:
            ckpt = torch.load('dan_exp8.ckpt')
        model = DanModel(vocab_size=vocab_size, n_classes=n_answers,
                         ebd_dim=cfg['ebd_dim'],
                         n_hidden_units=cfg['n_hidden_units'])

        model.load_state_dict(ckpt['model'])

        self.word_to_idx = word_to_idx
        self.idx_to_ans = idx_to_ans
        self.model = model


def vectorize_question(question, word_to_idx):
    vec_text = [0] * len(question)

    for i, w in enumerate(question):
        if w in word_to_idx:
            vec_text[i] = word_to_idx[w]
        else:
            vec_text[i] = word_to_idx['<unk>']

    return vec_text


def vectorize_answer(answer, ans_to_idx):
    answer_idx = 0
    if answer in ans_to_idx:
        answer_idx = ans_to_idx[answer]

    return answer_idx

def build_vocab(questions):
    # Unknown token.
    UNK = '<unk>'
    # Padding token.
    PAD = '<pad>'

    word_to_idx = {PAD: 0, UNK: 1}
    idx_to_word = {0: PAD, 1: UNK}

    words = set()
    words.add(UNK)
    words.add(PAD)

    for q in questions:
        q_text = ' '.join(q).lower()
        q_text = q_text.translate(str.maketrans('', '', string.punctuation))
        q_text = q_text.split()
        for w in q_text:
            if w not in words:
                idx = len(words)
                word_to_idx[w] = idx
                idx_to_word[idx] = w
                words.add(w)

    return words, word_to_idx, idx_to_word

def build_answers_set(answers):
    answers_set = set()
    ans_to_idx = {}
    idx_to_ans = {}

    for ans in answers:
        if ans not in answers_set:
            idx = len(answers_set)
            ans_to_idx[ans] = idx
            idx_to_ans[idx] = ans
            answers_set.add(ans)

    return answers_set, ans_to_idx, idx_to_ans


def preprocess_question(question):
    question = question.lower()
    question = question.translate(str.maketrans('', '', string.punctuation))

    words = question.split(' ')
    q = ''
    for w in words:
        if w not in STOPWORDS:
            q += w + ' '

    return q


def batchify(batch):
    question_len = list()
    label_list = list()

    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])

    target_labels = torch.LongTensor(label_list)
    x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch


def batchify_question(questions):
    length = []
    for q in questions:
        length.append(len(q))

    x1 = torch.LongTensor(len(questions), max(length)).zero_()
    for i in range(len(questions)):
        vec = torch.LongTensor(questions[i])
        x1[i, :length[i]].copy_(vec)

    return x1, torch.FloatTensor(length)


def create_app(enable_batch=True):
    dan_guesser = DanGuesser()
    dan_guesser.load(from_cuda=True)
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(dan_guesser, question)
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
            for guess, buzz in batch_guess_and_buzz(dan_guesser, questions)
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
    dataset = QuizBowlDataset(guesser_train=True)
    dan_guesser = DanGuesser()
    dan_guesser.train(dataset.training_data(), dataset.dev_data())


if __name__ == '__main__':
    cli()
