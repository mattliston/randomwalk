This repository contains the code used to generate the results described in the paper [Applying Deep Neural Networks to the Random Walk Theory](https://github.com/mattliston/randomwalk/blob/master/paper.pdf) by Matt Liston

1.   install Unbuntu 14.04 on a machine with a GPU
2.   install cuda version 7.5, run "sudo nvidia-modprobe -c 0 -u" to generate /dev/nvidia-uvm
3.   install docker
4.   install git
5.   git clone https://github.com/mattliston/randomwalk.git
6.   cd randomwalk
7.   docker build -t ml .
8.   ./launch\_ml
9.   python fetch.py --start 2016-06-02 --end 2006-06-02
10.  h5ls nyse\_nasdaq.hdf5
11.  python window.py --window 100 --test\_split 0.2
12.  h5ls train\*.hdf5 test\*.hdf5
13.  ls train.flist test.flist
14.  caffe train -solver=solver.prototxt |& tee train.log
15.  /tmp/caffe/tools/extra/parse\_log.py train.log .
16.  python cluster.py --window 100 --test\_split 0.2 --k 50 --minit points
17.  ls cluster\_daily.csv cluster\_summary.csv
