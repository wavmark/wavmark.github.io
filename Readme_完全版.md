# WavMark Demo

Our model enables encoding 32 bits information into a 1-second audio while maintain high invisibility and  robustness. Here we provide a simple demo to show the most basic application.



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

`python wavmark.py --mode=extract --input=<the watermarked audio>`





## Theory



The first 16 bits are used as pattern to determine the presence of the watermark（probability of error detection：1/(2^16)=0.0015%).  The remaining 16 bits are used as the payload ( defined by `--watermark= ` )







## Application 2: Verification

如果你只想来判断一段声音是否包含水印，那么这是一个更有效的工具。在这个demo中，你需要指定全部的32bit信息，随后在解码时，工具会提供是水印的概率值











## Citation

```
```

