import numpy as np


def hexChar2binStr(v):
    assert len(v) == 1
    # e => '1110'
    return '{0:04b}'.format(int(v, 16))


def hexStr2BinStr(hex_str):
    output = [hexChar2binStr(c) for c in hex_str]
    # ['1110', '1100', ....]
    return "".join(output)


def hexStr2BinArray(hex_str):
    # 十六进制字符串==> 0,1g构成的数组
    tmp = hexStr2BinStr(hex_str)
    return np.array([int(i) for i in tmp])


def binStr2HexStr(binary_str):
    return hex(int(binary_str, 2))[2:]


def binArray2HexStr(bin_array):
    tmp = "".join(["%d" % i for i in bin_array])
    return binStr2HexStr(tmp)


# 判断是否为合法的16进制字符串
def is_hex_str(s):
    hex_chars = "0123456789abcdefABCDEF"
    return all(c in hex_chars for c in s)




def flip_bytearray(input_bytearray, num_bits_to_flip):
    tmp = bytearray_to_binary_list(input_bytearray)
    tmp = flip_array(tmp,num_bits_to_flip)
    return binary_list_to_bytearray(tmp)

def flip_array(input_bits, num_bits_to_flip):

    # 随机选择要翻转的位的索引
    flip_indices = np.random.choice(len(input_bits), num_bits_to_flip, replace=False)

    # 创建一个全零的掩码数组
    mask = np.zeros_like(input_bits)

    # 将选定的索引设置为 1
    mask[flip_indices] = 1

    # 将输入位数组与掩码进行逐元素异或运算，实现翻转位
    flipped_bits = input_bits ^ mask
    return flipped_bits



def bytearray_to_binary_list(byte_array):
    binary_list = []
    for byte in byte_array:
        binary_str = format(byte, '08b')  # 将字节转换为 8 位二进制字符串
        binary_digits = [int(bit) for bit in binary_str]  # 将二进制字符串转换为整数列表
        binary_list.extend(binary_digits)  # 将整数列表添加到结果列表中
    return binary_list


def binary_list_to_bytearray(binary_list):
    # 这个函数假设输入列表的长度是 8 的倍数，否则将引发异常。
    byte_list = []
    for i in range(0, len(binary_list), 8):
        binary_str = ''.join(str(bit) for bit in binary_list[i:i + 8])  # 将 8 个位连接为一个二进制字符串
        byte_value = int(binary_str, 2)  # 将二进制字符串转换为整数
        byte_list.append(byte_value)  # 将整数添加到字节列表中
    return bytearray(byte_list)




if __name__ == "__main__":
    # hex_str = "ecd057f0d1fbb25d6430b338b5d72eb2"
    # arr = hexStr2BinArray(hex_str)
    # out = binArray2HexStr(arr)
    # print(out==hex_str)
    # bin_str = "".join()
    # assert bin2hex_str(bin_str) == hex_str
    # print(bin_str, len(bin_str))
    #
    watermark = np.random.randint(2, size=44)
    res = binArray2HexStr(watermark)
    print(res)

    test_str1 = "3ad30c748a2"
    test_str2 = "3ad30Z748a2"

    print(is_hex_str(test_str1))  # 输出 True
    print(is_hex_str(test_str2))  # 输出 False


    # encode_file("1.wav", watermark)
    # out = decode_file("tmp_output.wav")
    # assert np.all(watermark == out)
