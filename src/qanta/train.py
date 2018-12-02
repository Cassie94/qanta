import argparse

from dataset import QuizBowlDataset
from dan import DanGuesser

DAN_DEFAULT_CONFIG = {
    'ebd_dim': 300,
    'n_hidden_units': 40,
    #########
    'n_epochs': 1,
    'batch_size': 64,
    'lr': 1e-3,
    'log': 500,
    'cuda': False,
    'pretrained_word_ebd': True
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DAN model")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size for training (default: 64)")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="number of epochs to train (default: 3)")
    parser.add_argument("--log", type=int, default=500,
                        help="log frequency (default: 500 iterations)")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="filename of checkpoint")
    parser.add_argument("--cuda", type=bool, default=False,
                        help="whether to use CUDA")
    args = parser.parse_args()

    dataset = QuizBowlDataset(guesser_train=True)

    dan_guesser = DanGuesser()

    train_cfg = DAN_DEFAULT_CONFIG
    train_cfg['batch_size'] = args.batch_size
    train_cfg['n_epochs'] = args.n_epochs
    train_cfg['log'] = args.log
    train_cfg['cuda'] = args.cuda

    if len(args.checkpoint) == 0:
        dan_guesser.train(dataset.training_data(), dataset.dev_data(),
                          cfg=train_cfg)
    else:
        dan_guesser.train(dataset.training_data(), dataset.dev_data(),
                          cfg=train_cfg, resume=True, ckpt_file=args.checkpoint)