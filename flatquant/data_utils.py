import os
import pickle
import datasets
import random
import transformers

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        traindata = traindata.filter(lambda x: len(x) > 0)
        traindata = traindata.map(lambda x : {'text': x['text'].strip()})
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        valdata = datasets.load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', download_mode="force_redownload")
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', download_mode="force_redownload")
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_pile(nsamples, seqlen, tokenizer):
    traindata = datasets.load_dataset("pile-val-backup", split="validation")
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def _format_openbookqa_example(example):
    choices = example["choices"]["text"]
    labels = example["choices"]["label"]
    choice_map = {label.strip(): text for label, text in zip(labels, choices)}
    answer_label = example["answerKey"].strip()
    answer_text = choice_map.get(answer_label, "")
    choices_str = " ".join([f"{label}) {text}" for label, text in zip(labels, choices)])
    return f"Question: {example['question_stem']} Choices: {choices_str} Answer: {answer_text}"


def get_openbookqa(nsamples, seqlen, tokenizer, eval_mode=False):
    split = "test" if eval_mode else "train"
    data = datasets.load_dataset("openbookqa", "main", split=split)
    texts = [_format_openbookqa_example(ex) for ex in data]
    enc = tokenizer("\n\n".join(texts), return_tensors="pt")
    if eval_mode:
        return enc
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def _format_winogrande_example(example):
    answer = example["answer"].strip()
    option = example["option1"] if answer == "1" else example["option2"]
    return example["sentence"].replace("_", option)


def get_winogrande(nsamples, seqlen, tokenizer, eval_mode=False):
    split = "validation" if eval_mode else "train"
    data = datasets.load_dataset("winogrande", "winogrande_xl", split=split)
    texts = [_format_winogrande_example(ex) for ex in data]
    enc = tokenizer("\n\n".join(texts), return_tensors="pt")
    if eval_mode:
        return enc
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_loaders(
    args, name, tokenizer, nsamples=128, seqlen=2048, eval_mode=False
):
    if 'wikitext2' in name:
        dataset = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode)
    elif 'ptb' in name:
        dataset = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'c4' in name:
        dataset = get_c4_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'pile' in name:
        dataset = get_pile(nsamples, seqlen, tokenizer)
    elif 'openbookqa' in name:
        dataset = get_openbookqa(nsamples, seqlen, tokenizer, eval_mode)
    elif 'winogrande' in name:
        dataset = get_winogrande(nsamples, seqlen, tokenizer, eval_mode)

    if 'c4' in name and eval_mode:
        dataset = dataset.input_ids
        dataset = TokenizerWrapper(dataset)
    return dataset
