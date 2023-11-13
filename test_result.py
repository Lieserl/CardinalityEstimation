import pytest
import json

class Test_simple():
    def test_card(self):
        json_file = 'lab1/eval/results.json'
        with open(json_file, 'r') as f:
            data = json.load(f)
        act = data.get('act')
        avi = data.get('avi')
        ebo = data.get('ebo')
        min_sel = data.get('min_sel')
        AI1 = data.get('your_model_1')
        AI2 = data.get('your_model_2')

        assert(len(act) > 0)
        assert(len(act) == len(avi))
        assert(len(act) == len(ebo))
        assert(len(act) == len(AI1))
        assert(len(act) == len(AI2))
        assert(len(act) == len(min_sel))
        assert(sum(avi) > 0)
        assert(sum(ebo) > 0)
        assert(sum(min_sel) > 0)
        assert(sum(AI1) > 0)
        assert(sum(AI2) > 0)


if __name__ == '__main__':
    pytest.main(['test_result.py'])