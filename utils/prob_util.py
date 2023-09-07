import math


def calculate_probability(bit_length, not_equal_count):
    total_cases = 2 ** (bit_length * 2)

    combinations = math.comb(bit_length, not_equal_count)

    equal_cases = combinations * (2 ** not_equal_count) * (2 ** (bit_length - not_equal_count))

    probability = equal_cases / total_cases

    # print("BER:", (bit_length - not_equal_count) / bit_length)
    # print(f"bit_length:{bit_length}  not_equal_count:{not_equal_count} 概率值为: {probability * 100}%")
    return probability
