import unittest
try:
    from bleu_mp import compute_bleu
except (ModuleNotFoundError, ImportError):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__)+'/..')
    from bleu_mp import compute_bleu


def _test_speed(pred_data, tgt_data):
    import time
    t1 = time.time()
    score1 = compute_bleu(pred_data, tgt_data, n_worker=0)
    t2 = time.time()
    score2 = compute_bleu(pred_data, tgt_data, n_worker=10)
    t3 = time.time()
    print('score', score1, score2)
    print('1  process cost time', t2 - t1)
    print('10 process cost time', t3 - t2)
    print()


class MyTestCase(unittest.TestCase):
    def test_speed_1(self):
        pred_a1 = ['床前明月光，疑是地上霜', '举头望明月，低头思故乡'] * 100000
        label_a1 = [['床前明月光，疑是地上霜'], ['举头望明月，低头思故乡']] * 100000
        _test_speed(pred_a1, label_a1)

        pred_a1 = ['床前明月光，疑是地上霜'*10, '举头望明月，低头思故乡'*10] * 100000
        label_a1 = [['床前明月光，疑是地上霜'*10], ['举头望明月，低头思故乡'*10]] * 100000
        _test_speed(pred_a1, label_a1)

        pred_a1 = [[1, 2, 3, 4], [2, 3, 4, 5]] * 100000
        label_a1 = [[[1, 2, 3, 4]], [[2, 3, 4, 5]]] * 100000
        _test_speed(pred_a1, label_a1)

        pred_a1 = [[1, 2, 3, 4]*20, [2, 3, 4, 5]*20] * 100000
        label_a1 = [[[1, 2, 3, 4]*20], [[2, 3, 4, 5]*20]] * 100000
        _test_speed(pred_a1, label_a1)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
