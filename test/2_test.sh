python 1_wavmark.py --input=data/1_librispeech_8s.wav --output=out.wav
python 1_wavmark.py --input=data/2_valle_1min.wav --output=out.wav
python 1_wavmark.py --mode=decode --input=out.wav
python 1_wavmark.py --mode=decode --input=data/1_librispeech_8s.wav
python 1_wavmark.py --mode=decode --input=data/2_valle_1min.wav
python 1_wavmark.py --mode=decode --input=data/8_speartts.wav