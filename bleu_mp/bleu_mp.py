'''
改自 huggingface evaluate 的 bleu 评分工具
https://github.com/huggingface/evaluate/blob/main/metrics/bleu/bleu.py

改为多进程计算版本
支持字符串和数字列表计算。

多线程实现使用共享内存和spawn进程。
不使用python内置多进程实现，改为自行实现的多进程通讯，可以让子计算进程非常节省内存，对Windows和Linux高度兼容性。

注意：不要传入pytorch的tensor变量，会极大的降低性能。

示例：
```
from bleu_mp import compute_bleu

pred_data = ['床前明月光，疑是地上霜', '举头望明月，低头思故乡'] * 1000
tgt_data = [['床前明月光，疑是地上霜'], ['举头望明月，低头思故乡', '静夜思']] * 1000
result = compute_bleu(pred_data, tgt_data)
print('bleu score', result[0])

pred_data = [[1, 2, 3, 4], [2, 3, 4, 5]] * 1000
tgt_data = [[[1, 2, 3, 4]], [[2, 3, 4, 5], [4, 5, 6]]] * 1000
result = compute_bleu(pred_data, tgt_data)
print('bleu score', result[0])

```

'''

import os
import math
import uuid
import argparse
import pickle
import collections
import subprocess
from multiprocessing.spawn import get_executable
from multiprocessing.shared_memory import SharedMemory


# -------------------------------------------------------------------------------------
_py_exe = get_executable()
_py_script = os.path.abspath(__file__)
_shm_min_size = 1024


# -------------------------------------------------------------------------------------
class _BleuMiniServer:
    # 使用共享内存传递数据
    def __init__(self, shm_name):
        shm = SharedMemory(shm_name, create=False)
        assert shm.size >= _shm_min_size, f'Error! Shm size must be bigger than {_shm_min_size} bytes.'

        data = pickle.loads(shm.buf)
        result = self.compute(data)
        self.send(shm, result)
        shm.close()
        # 忽略任何unlink导致的错误
        try:
            shm.unlink()
        except BaseException:
            pass

    @staticmethod
    def compute(data):
        pred_corpus, target_corpus, max_order = data
        result = _stat_corpus(pred_corpus, target_corpus, max_order)
        return result

    @staticmethod
    def send(shm, result):
        pack_result = pickle.dumps(result)
        pack_len = len(pack_result)
        assert pack_len <= shm.size, 'Error! Pack result is too big.'
        shm.buf[:pack_len] = pack_result


# -------------------------------------------------------------------------------------

def _list_split_by_group(self: list, n_group: int):
    # 顺序分组
    assert n_group > 0
    self = list(self)
    sizes = [int(len(self) / n_group)] * n_group
    for i in range(len(self) % n_group):
        sizes[i] += 1
    g = []
    i = 0
    for s in sizes:
        l = self[i: i+s]
        g.append(l)
        i += s
    return g


def _get_ngrams(segment, max_order):
    '''
    Extracts all n-grams upto a given maximum order from an input segment.
    :param segment:     text segment from which n-grams will be extracted.
    :param max_order:   maximum length in tokens of the n-grams returned by this methods.
    :return:            The Counter containing all n-grams upto max_order in segment
                            with a count of how many times each n-gram occurred.
    '''
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i: i+order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _stat_corpus(pred_corpus, target_corpus, max_order):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    tgt_len = 0
    pred_len = 0

    # for tgt_seqs, pred_seq in tqdm.tqdm(zip(target_corpus, pred_corpus)):
    for tgt_seqs, pred_seq in zip(target_corpus, pred_corpus):
        tgt_len += min(len(r) for r in tgt_seqs)
        pred_len += len(pred_seq)

        merged_ref_ngram_counts = collections.Counter()
        for tgt_seq in tgt_seqs:
            merged_ref_ngram_counts |= _get_ngrams(tgt_seq, max_order)

        pred_ngram_counts = _get_ngrams(pred_seq, max_order)
        overlap = pred_ngram_counts & merged_ref_ngram_counts

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]

        for order in range(1, max_order + 1):
            possible_matches = len(pred_seq) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    return matches_by_order, possible_matches_by_order, tgt_len, pred_len


def compute_bleu(pred_corpus, target_corpus, max_order=4, smooth=False, n_worker=-1, enable_multiprocee_boundary=500):
    '''
    Computes BLEU score of translated segments against one or more references.
    :param pred_corpus:     list of translations to score. Each translation should be tokenized into a list of tokens.
    :param target_corpus:   list of lists of references for each translation. Each reference should be tokenized into a list of tokens.
    :param max_order:       Maximum n-gram order to use when computing BLEU score.
    :param smooth:          Whether or not to apply Lin et al. 2004 smoothing.
    :param n_worker:        number of worker.
    :param enable_multiprocee_boundary: Use multi-process compute when the number of data is greater than this parameter.
    :return:    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram precisions and brevity penalty.
    '''
    assert len(pred_corpus) == len(target_corpus), 'Error! The length of pred_corpus and target_corpus must be equal.'

    if n_worker == -1:
        n_worker = os.cpu_count()
        if n_worker is None:
            n_worker = 0
    n_worker = int(n_worker)

    # 获得bleu计算所需数据
    results = []

    if n_worker <= 1 or len(pred_corpus) <= enable_multiprocee_boundary:
        # 单进程路径
        results.append(_stat_corpus(pred_corpus, target_corpus, max_order))

    else:
        # 多进程路径
        # 分组
        pred_corpus_groups = _list_split_by_group(pred_corpus, n_worker)
        tgt_corpus_groups = _list_split_by_group(target_corpus, n_worker)

        shm_list = []
        proc_list = []
        for s1, s2 in zip(pred_corpus_groups, tgt_corpus_groups):
            if len(s1) == 0:
                break

            data = pickle.dumps([s1, s2, max_order])
            data_len = len(data)

            shm_name = str(uuid.uuid1())
            shm = SharedMemory(shm_name, create=True, size=max(_shm_min_size, data_len))
            shm.buf[:data_len] = data
            shm_list.append(shm)

            proc = subprocess.Popen([_py_exe, _py_script, '--bleu_mini_server', '--shm', shm_name])
            proc_list.append(proc)

        for proc, shm in zip(proc_list, shm_list):
            proc.wait()
            assert proc.returncode == 0, 'Error! The child process of calculating bleu ended abnormally.'
            r = pickle.loads(shm.buf)
            results.append(r)
            shm.close()
            # 忽略任何unlink导致的错误
            try:
                shm.unlink()
            except BaseException:
                pass

    # 进行统计
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    tgt_len = 0
    pred_len = 0

    for sub_matches_by_order, sub_possible_matches_by_order, sub_tgt_len, sub_pred_len in results:

        for i in range(len(matches_by_order)):
            matches_by_order[i] += sub_matches_by_order[i]

        for i in range(len(possible_matches_by_order)):
            possible_matches_by_order[i] += sub_possible_matches_by_order[i]

        tgt_len += sub_tgt_len
        pred_len += sub_pred_len

    # 统计结束
    precisions = [0] * max_order

    for i in range(0, max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.)
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    length_ratio = float(pred_len) / tgt_len

    if length_ratio > 1.0:
        brevity_penalty = 1.
    else:
        brevity_penalty = math.exp(1 - 1. / length_ratio)

    bleu = geo_mean * brevity_penalty

    return bleu, precisions, brevity_penalty, length_ratio, pred_len, tgt_len


# Bleu Mini Server Check
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bleu_mini_server', action='store_true')
    parser.add_argument('--shm', default='', type=str)

    args = parser.parse_args()

    if args.bleu_mini_server:
        _BleuMiniServer(args.shm)
        exit(0)
