# WavMark



## Demo

Our model encodes 32 bits information into a 1-second audio.

In this demo， the first 16 bits are used as the pattern to determine the presence of the watermark（Probability for error detection：1/65535=0.0015%). The remaining 16 bits are used as the payload. 



**add watermark**

`python wavmark.py --input=<Host Audio Path> --output=<Output Path> --watermark=0010101010100111`

for example：

`python wavmark.py --input=data/1_librispeech_8s.wav --output=out.wav --watermark=0010101010100111`



**Tips:**

The "--watermark"  value is a string of zeros and ones of length 16, and you can generate a random watermark string with the following code：

```python
import numpy as np
array = np.random.choice([0, 1], size=16)
array_str = "".join([str(i) for i in array])
# 0101101110111100
```



**decode watermark**

`python wavmark.py --mode=extract --input=<Your Audio Path>`


