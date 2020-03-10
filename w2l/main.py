from .utils.data import read_data_config
from .utils.errors import letter_error_rate_corpus, word_error_rate_corpus
from .utils.vocab import parse_vocab
from .model import W2L
from .input import w2l_input_fn_npy


def run_asr(mode, data_config, model_dir, data_format="channels_first",
            cpu=False, reg=(None, 0.),
            adam_params=(1e-4, 0.9, 0.9, 1e-8), batch_size=16, clipping=500,
            fix_lr=False, normalize=False, steps=300000, threshold=0.,
            which_sets=None):
    """
    All of these parameters can be passed from w2l_cli. Please check
    that one for docs on what they are.

    Returns:
        Depends on mode!
        If train, eval-current or eval-all: Nothing is returned.
        If predict: Returns a generator over predictions for the requested set.
        If return: Return the model object. Use this if you want access to
                   the variables or their values, for example, or if you want
                   to use the forward etc. functions yourself.

    """
    data_config_dict = read_data_config(data_config)
    csv_path, array_dir, vocab_path, mel_freqs = (
        data_config_dict["csv_path"], data_config_dict["array_dir"],
        data_config_dict["vocab_path"], data_config_dict["n_freqs"])

    ch_to_ind, ind_to_ch = parse_vocab(vocab_path)
    ind_to_ch[-1] = "<PAD>"
    ind_to_ch[0] = "<BL>"

    try:
        reg = (reg[0], float(reg[1]))  # tuples are immutable...
    except (ValueError, TypeError):
        raise ValueError("Could not convert regularization coefficient '{}' "
                         "to float.".format(reg[1]))

    model = W2L(model_dir, len(ch_to_ind), mel_freqs, data_format)

    if mode == "return":
        return model

    dataset = w2l_input_fn_npy(csv_path, array_dir, which_sets, mode == "train",
                               ch_to_ind, mel_freqs, batch_size, threshold,
                               normalize)

    if mode == "train":
        model.train_full(dataset, steps, adam_params, not cpu)

    elif mode == "predict" or mode == "errors":
        def gen():
            for features, labels in dataset:
                layer_list = model.forward(
                    features["audio"], return_all=True)

                for ind in range(layer_list[0].shape[0]):
                    predictions_repacked = dict()
                    predictions_repacked["input_length"] = features["length"][ind]

                    predictions_repacked["all_layers"] = [layer[ind] for
                                                          layer in layer_list]
                    predictions_repacked["input"] = features["audio"][ind]

                    yield predictions_repacked

    if mode == "predict":
        return gen()

    if mode == "errors":
        true = []
        predicted = []
        for sent_ind, p in enumerate(gen(), start=1):
            true.append(p["true"])
            predicted.append(p["decoding"])
            if not sent_ind % 1000:
                print("Went through {}...".format(sent_ind))
        ler = letter_error_rate_corpus(true, predicted)
        wer = word_error_rate_corpus(true, predicted)
        print("LER: {}\nWER: {}".format(ler, wer))
