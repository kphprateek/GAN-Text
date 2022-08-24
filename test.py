import tensorflow as tf
from models.generator import Generator
from settings import *
from preprocessing import load_tokenizer
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)


def load_generator():
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH)
    generator.load(pretrained_generator_file)
    return generator

def decode_to_sentences(sentences):
    return tokenizer.sequences_to_texts(sentences)


if __name__ == "__main__":
    generator = load_generator()
    tokenizer = load_tokenizer()
    File_data = np.loadtxt("C:/Users/nbtc068/Desktop/seqgan-text-generation-tf2/dataset/negatives.txt", dtype=int)
    generated_sentences = generator.generate_one_batch().numpy()
    sentences1=tokenizer.sequences_to_texts(File_data)
    sentences = tokenizer.sequences_to_texts(generated_sentences)
    print(*sentences, sep='\n')
    print("abcd")
    print(*sentences1, sep='\n')
