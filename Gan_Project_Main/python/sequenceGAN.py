from wordsegment import load, segment
load()
import sys
import numpy as np
import random
from nltk.translate.bleu_score import SmoothingFunction
import nltk
import os
from multiprocessing import Pool
from datetime import datetime
import torch
from config import TOTAL_BATCH, DIS_NUM_EPOCH, DEVICE, PATH, NrGPU, openLog
from data_processing import gen_label,decode
from lstmCore import read_sampleFile, pretrain_LSTMCore
from discriminator import train_discriminator
from generator import Generator, train_generator
from rollout import Rollout, getReward


def convert_textcode2tagcode(sample, iw_dict, wi_dict_tag,VOCAB_SIZE_TAG):
    text = []
    for i in range(len(sample)):
        try:
            text.append(iw_dict[str(sample[i])])
        except:
            text.append(" ")

    tagging = nltk.pos_tag(text)
    temp = []
    for i in tagging:
        temp.append(i[1])
    text = []
    temp1 = [i.lower() for i in temp]
    for i in range(len(temp1)):
        try:
            text.append(int(wi_dict_tag[temp1[i]]))
        except:
            text.append(VOCAB_SIZE_TAG - 1)
    return np.array(text)

def postag(samples,iw_dict, wi_dict_tag,vocab_size):
    samples = samples.cpu().numpy()
    output = []
    for i in samples:
        output.append(
            convert_textcode2tagcode(i,iw_dict, wi_dict_tag,vocab_size))
    output = torch.from_numpy(np.array(output))
    return output.cuda()


def pretrain_generator(x,start_token,end_token,ignored_tokens=None,
                       sentence_lengths=None,batch_size=1,vocab_size=10):
    # print("------------------------------------")
    # print(x)
    pretrain_result = pretrain_LSTMCore(train_x=x,
                            sentence_lengths=sentence_lengths, 
                            batch_size=batch_size, end_token=end_token,
                            vocab_size=vocab_size)
    generator = Generator(pretrain_model=pretrain_result[0],
                start_token=start_token, ignored_tokens=ignored_tokens)
    # generator is not DataParallel. the lstmCore inside is. 
    # if generator is also DataParallel, when it calls lstmCore it invokes the
    #   error message "RuntimeError: all tensors must be on devices[0]"
    #   because the generator instance may not be on devices[0].
    generator.to(DEVICE)
    return generator

def train_discriminator_wrapper(x, x_gen, batch_size=1, vocab_size=10):
    y = gen_label(len(x),fixed_value=1)
    y_gen = gen_label(len(x_gen),fixed_value=0)
    # print(y,y_gen,"abcd")
    x_train = torch.cat([x.int(),x_gen.int()], dim=0)
    y_train = torch.cat([y,y_gen], dim=0)
    discriminator = train_discriminator(x_train, y_train, batch_size, vocab_size)
    return discriminator

def scale(rewards, metric):
    # Returning metric value in scale of reward
    mean1 = np.mean(rewards)
    temp = metric
    while temp < mean1:
        temp = temp * 10
    temp = temp/10
    return temp


def eval_epoch_bleu(EVAL_FILE, data_loc):
    # Evaluate the generated samples with original or real data
    # return the similarity measure i.e, Bleu score

    f = open(data_loc)
    length = len(f.read().split('\n'))
    # code2text(EVAL_FILE,wi_dict, iw_dict,write_file+str(flag))
    return get_bleu_fast(EVAL_FILE, 2, data_loc, int(length / 10))


def get_bleu_fast(test_data1, gram, real_data, sample_size):
    # Calculate the BLEU Score fast
    reference = get_reference_tokens(real_data)
    random.shuffle(reference)
    reference = reference[0:sample_size]
    return get_bleu_parallel(test_data1, gram=gram, reference=reference)


def get_reference_tokens(real_data):
    # Getting the tokens for text and return the list of list of tokens for each line in text file
    reference = list()
    with open(real_data) as real_data:
        for text in real_data:
            text = nltk.word_tokenize(text)
            reference.append(text)
    return reference


def get_bleu_parallel(test_data1, gram, reference=None, ):
    # Getting the BLEU Score using parallel methodology
    ngram = gram
    weight = tuple((1. / ngram for _ in range(ngram)))
    pool = Pool(os.cpu_count())
    result = []
    with open("genspam.txt",'r') as test_data:
        for hypothesis in test_data:
            hypothesis = nltk.word_tokenize(hypothesis)

            result.append(pool.apply_async(calc_bleu, args=(reference, hypothesis, weight)))
    score = 0.0
    cnt = 0

    for i in result:
        score += i.get()
        cnt += 1


    pool.close()
    pool.join()

    return score / cnt


def calc_bleu(reference, hypothesis, weight):
    # Function to calculate the BLEU Score value
    return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                   smoothing_function=SmoothingFunction().method1)


def main(batch_size, num=None):
    if batch_size is None:
        batch_size = 1
    x, vocabulary, reverse_vocab, sentence_lengths = read_sampleFile(num=num)
    # print(x,"rrrr12345",vocabulary)

    if batch_size > len(x):
        batch_size = len(x)
    start_token = vocabulary['START']
    end_token = vocabulary['END']
    pad_token = vocabulary['PAD']
    ignored_tokens = [start_token, end_token, pad_token]
    vocab_size = len(vocabulary)
    # print(vocab_size,"abcdef")
    
    log = openLog()
    log.write("###### start to pretrain generator: {}\n".format(datetime.now()))
    log.close()
    generator = pretrain_generator(x, start_token=start_token,
                    end_token=end_token,ignored_tokens=ignored_tokens,
                    sentence_lengths=torch.tensor(sentence_lengths,device=DEVICE).long(),
                    batch_size=batch_size,vocab_size=vocab_size)
    x_gen = generator.generate(start_token=start_token, ignored_tokens=ignored_tokens,
                               batch_size=len(x))
    # print(x_gen,"8888")
    log = openLog()
    log.write("###### start to pretrain discriminator: {}\n".format(datetime.now()))
    log.close()
    # print("##############")
    # print(x,x_gen)
    discriminator = train_discriminator_wrapper(x, x_gen, batch_size, vocab_size)
    rollout = Rollout(generator, r_update_rate=0.8)
    rollout = torch.nn.DataParallel(rollout)#, device_ids=[0])
    rollout.to(DEVICE)

    log = openLog()
    log.write("###### start to train adversarial net: {}\n".format(datetime.now()))
    log.close()
    for total_batch in range(TOTAL_BATCH):
        log = openLog()
        log.write('batch: {} : {}\n'.format(total_batch, datetime.now()))
        print('batch: {} : {}\n'.format(total_batch, datetime.now()))
        log.close()
        for it in range(1):
            # print(batch_size,"ppppp")
            samples = generator.generate(start_token=start_token,
                    ignored_tokens=ignored_tokens, batch_size=batch_size)
            # Take average of ROLLOUT_ITER times of rewards.
            #   The more times a [0,1] class (positive, real data)
            #   is returned, the higher the reward.
            # print("11111111111111111111111111111")
            # print(samples,"000000000")
            log = openLog('genTxt_predict1.txt')
            result = decode(x, reverse_vocab, log)
            log.close()
            # lines = []
            with open(PATH+'genTxt_predict1.txt',"r",encoding="utf-8") as f:
                # print(f.readlines(),"1111111")
                # print(f.read(),"abcd")
                # f.seek(0)
                lines = f.readlines()
            for line in lines:
                temp = segment(line)
                str1 = ""
                for j in range(len(temp)):
                    str1 += temp[j] + " "
                with open("C:/Users/nbtc068/Desktop/GAN_Project_Main/python/gennotspam.txt", "a",encoding="utf-8") as file:
                    file.write(str1 + "\n")
            log = openLog('genTxt_predict2.txt')
            result = decode(samples, reverse_vocab, log)
            log.close()
            lines1 = []
            with open(PATH+'genTxt_predict2.txt','r',encoding="utf-8") as p:
                # p.seek(0)
                lines1 = p.readlines()
                # print(l)

            for line in lines1:
                temp = segment(line)
                str1 = ""
                for j in range(len(temp)):
                    str1 += temp[j] + " "
                with open("C:/Users/nbtc068/Desktop/GAN_Project_Main/python/genspam.txt", "a",encoding="utf-8") as file:
                    file.write(str1 + "\n")
            similarity_score= eval_epoch_bleu('genspam.txt','gennotspam.txt')
            rewards = getReward(samples, rollout, discriminator)
            os.remove("genspam.txt")
            os.remove("gennotspam.txt")
            # bleu = scale(rewards, similarity_score)
            # print(type(rewards))
            # print(rewards,"abc",bleu)
            # print(rewards,type(rewards),"iiii")
            # print(similarity_score,"1234")
            # vocab_size_tag="genspam.txt"
            # samples_tag=postag(samples,vocabulary, reverse_vocab,vocab_size_tag)
            # rewards_tag = rollout.get_reward(samples_tag, 16, discriminator)
            # rewards=rewards-similarity_score
            # rewards=rewards+rewards_tag
            # print(rewards,"aaaaaaaaaaaaaaaaaaaaaaaaa")
            (generator, y_prob_all, y_output_all) = train_generator(model=generator, x=samples,
                    reward=rewards, iter_n_gen=1, batch_size=batch_size, sentence_lengths=sentence_lengths)

        rollout.module.update_params(generator)

        for iter_n_dis in range(DIS_NUM_EPOCH):
            log = openLog()
            log.write('  iter_n_dis: {} : {}\n'.format(iter_n_dis, datetime.now()))
            log.close()
            x_gen = generator.generate(start_token=start_token, ignored_tokens=ignored_tokens,
                               batch_size=len(x))

            # print("##############")
            # print(len(x),type(x),x)
            log = openLog('genTxt_predict1.txt')
            result = decode(x, reverse_vocab, log)
            # print("++++++++++++++++")
            # print(x_gen)
            log = openLog('genTxt_predict2.txt')
            result = decode(x_gen, reverse_vocab, log)
            discriminator = train_discriminator_wrapper(x, x_gen, batch_size,vocab_size)

    log = openLog()
    log.write('###### training done: {}\n'.format(datetime.now()))
    log.close()
    
    torch.save(reverse_vocab, PATH+'reverse_vocab.pkl')
    try:
        torch.save(generator, PATH+'generator.pkl')
        print('successfully saved generator model.')
    except:
        print('error: model saving failed!!!!!!')

    log = openLog('genTxt.txt')
    num = generator.generate(batch_size=batch_size)
    log.close()
#    words_all = decode(num, reverse_vocab, log)
#    print(words_all)

#%%
if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    try:
        num = int(sys.argv[2])
    except IndexError:
        num=10
    if batch_size<NrGPU:
        batch_size = NrGPU
        
    main(batch_size,num)


