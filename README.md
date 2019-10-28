platform: mac OS X

python version: Python 3.7


```sh
usage: dinner_party.py [-h]
                       [--solver {a-star,random-restart,hill-climbing,local-beam-search}]
                       [--instance {hw1-inst2.txt,hw1-inst3.txt,hw1-inst1.txt}]
                       [--restart RESTART] [--beam BEAM] [--halt HALT]
                       [--stochastic]

dinner party problem

optional arguments:
  -h, --help            show this help message and exit
  --solver {a-star,random-restart,hill-climbing,local-beam-search}, -s {a-star,random-restart,hill-climbing,local-beam-search}
                        solve algorithm
  --instance {hw1-inst2.txt,hw1-inst3.txt,hw1-inst1.txt}, -i {hw1-inst2.txt,hw1-inst3.txt,hw1-inst1.txt}
                        instance file
  --restart RESTART, -r RESTART
                        random restart time
  --beam BEAM, -b BEAM  beam width of local beam search
  --halt HALT, -ha HALT
                        maximum time that cannot breakthrough current maximum
                        score, halt algorithm
  --stochastic          using stochastic beam search instead of local beam
                        search
```