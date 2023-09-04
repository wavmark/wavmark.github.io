# WavMark Demo

Our model enables encoding 32 bits of information into 1-second audio while maintaining high inaudibility and robustness. Here we provide a demo to showcase the most basic application.


## Add Watermark

add 16 bit watermark into the in.wav file: 

`python wavmark.py --input=in.wav --output=out.wav --watermark=0010101010100111`



The `--watermark`   is a string of zeros and ones of length 16. You can generate a random watermark using the following code：

```python
import numpy as np
array = np.random.choice([0, 1], size=16)
array_str = "".join([str(i) for i in array])
# 0101101110111100
```



## Decode Watermark

`python wavmark.py --mode=decode --input=<the watermarked audio>`


## Install
1.clone this project

2.pip install -r requirements.txt

