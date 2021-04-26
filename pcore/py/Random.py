import random


def divide_chunks(data, start, split_len):
    for i in range(start, start + split_len, 1):
        yield data[i: i+1]


def shuffle_split(data, split_proportion=0.1, seed_num=47):
    random.Random(seed_num).shuffle(data)
    total_size = len(data)
    split_size = int(total_size * split_proportion)
    part1_size = total_size - split_size
    part1_data = [elem[0] for elem in divide_chunks(data, 0, part1_size)]
    part2_data = [elem[0] for elem in divide_chunks(data, part1_size, split_size)]
    return part1_data, part2_data
