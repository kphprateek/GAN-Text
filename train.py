import tensorflow as tf

import os
from utils.dataloader import generator_dataloader, discriminator_dataloader
from models.generator import Generator
from models.discriminator import Discriminator
from models.rollout import ROLLOUT
from preprocessing import *
from settings import *
import nltk

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH)
    generator_tag = Generator(vocab_size_tag, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size,
                                  embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  dropout_keep_prob=dis_dropout_keep_prob,
                                  l2_reg_lambda=dis_l2_reg_lambda)
    discriminator_tag = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size_tag,
                                  embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  dropout_keep_prob=dis_dropout_keep_prob,
                                  l2_reg_lambda=dis_l2_reg_lambda)

    gen_dataset = generator_dataloader(positive_file, BATCH_SIZE)
    gen_dataset_tag = generator_dataloader(positive_file_tag, BATCH_SIZE)

    if not os.path.exists("pretrained_models"):
        os.makedirs("pretrained_models")

    if not os.path.exists(pretrained_generator_file):
        print('Start pre-training generator')
        generator.pretrain(gen_dataset, PRE_EPOCH_NUM, generated_num // BATCH_SIZE)
        generator.save(pretrained_generator_file)
        print('Finished pre-training generator...')
    else:
        generator.load(pretrained_generator_file)
    if not os.path.exists(pretrained_generator_file_tag):
        print('Start pre-training generator')
        generator.pretrain(gen_dataset_tag, PRE_EPOCH_NUM, generated_num // BATCH_SIZE)
        generator.save(pretrained_generator_file_tag)
        print('Finished pre-training generator...')
    else:
        generator.load(pretrained_generator_file_tag)


    if not os.path.exists(pretrained_discriminator_file):
        print('Start pre-training discriminator...')
        for _ in range(5):
            print("Dataset", _)
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file)
            dis_dataset = discriminator_dataloader(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
        discriminator.save(pretrained_discriminator_file)
        print('Finished pre-training discriminator...')
    else:
        discriminator.load(pretrained_discriminator_file)

    if not os.path.exists(pretrained_discriminator_file_tag):
        print('Start pre-training discriminator...')
        for _ in range(5):
            print("Dataset_tag", _)
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file_tag)
            dis_dataset_tag = discriminator_dataloader(positive_file_tag, negative_file_tag, BATCH_SIZE)
            discriminator.train(dis_dataset_tag, 3, (generated_num // BATCH_SIZE) * 2)
        discriminator.save(pretrained_discriminator_file_tag)
        print('Finished pre-training discriminator...')
    else:
        discriminator.load(pretrained_discriminator_file_tag)

    rollout = ROLLOUT(generator, 0.8)
    rollout_tag=ROLLOUT(generator_tag,0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')

    for epoch in range(EPOCH_NUM):
        print("Generator", epoch)
        for it in range(1):
            samples = generator.generate_one_batch().numpy()
            # rewards = rollout.get_reward(samples, 16, discriminator)
            print(samples,"ppppp",samples.shape)
            if mixed_flag == 0:
                rewards = rollout.get_reward(samples, 16, discriminator)
            else:
                tokenizer = load_tokenizer()
                sentences = tokenizer.sequences_to_texts(samples)
                # print(sentences,"iiii",len(sentences))
                for i in sentences:
                    h = open("outfile.txt", "a")
                    text = nltk.word_tokenize(i)
                    tagging = nltk.pos_tag(text)
                    temp = []
                    # print(tagging)
                    # coun=count+1
                    for i in tagging:
                        temp.append(i[1])
                    h.write(' '.join(temp))
                    h.write('\n')
                    h.close()
                h = open("outfile.txt", "r")
                print(h)
                tokenizer = load_tokenizer_tag()
                sentences_tag = tokenizer.texts_to_sequences(h)
                open("outfile.txt", "w").close()
                # print(sentences_tag)
                # text = []
                # temp1 = [i.lower() for i in temp]
                # print(d_tag,"abc")
                # for i in range(len(temp1)):
                #     try:
                #         text.append(int(d_tag[temp1[i]]))
                #     except:
                #         text.append(vocab_size_tag - 1)
                # pad = len(max(sentences_tag, key=len))+1
                pad=20
                samples_tag=np.array([i + [0] * (pad - len(i)) for i in sentences_tag])
                # samples_tag=np.asarray(sentences_tag)
                # print(samples_tag)
                # print(samples_tag.shape)
                # samples_tag = text_code2tag_code(samples, d, tokenizer_file, d_tag, tokenizer_file_tag, VOCAB_SIZE,
                #                                  VOCAB_SIZE_TAG)
                # calculate the reward
                # print(samples.shape,samples_tag.shape,"abcd")
                rewards_tag = rollout_tag.get_reward(samples_tag, 16, discriminator_tag)
                # print(rewards_tag,"ooooo")
                rewards_text = rollout.get_reward(samples, 16, discriminator)
                rewards = rewards_tag + rewards_text
            generator.train_step(samples, rewards)

        rollout.update_params()

        print("Discriminator", epoch)
        for _ in range(5):
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file)
            dis_dataset = discriminator_dataloader(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
    generator.save(generator_file)
    discriminator.save(discriminator_file)

    generator.generate_samples(generated_num // BATCH_SIZE, generated_file)
