#!/bin/bash
python3 hw1-1.py ./testdata/0a.png  ./result/0a_y.png
python3 hw1-1.py ./testdata/0b.png  ./result/0b_y.png
python3 hw1-1.py ./testdata/0c.png  ./result/0c_y.png
python3 hw1-2.py ./testdata/0a.png 3 0.2
python3 hw1-2.py ./testdata/0b.png 3 0.2
python3 hw1-2.py ./testdata/0c.png 3 0.2
python3 hw1-3.py ./testdata/0a.png ./result
python3 hw1-3.py ./testdata/0b.png ./result
python3 hw1-3.py ./testdata/0c.png ./result