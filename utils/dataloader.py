import tensorflow as tf


def generator_dataloader(data_file, batch_size):
    token_stream = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]

            if len(parse_line) == 20:
                token_stream.append(parse_line)

    return tf.data.Dataset.from_tensor_slices(token_stream).shuffle(len(token_stream)).batch(batch_size)


def discriminator_dataloader(positive_file, negative_file, batch_size):
    examples = []
    labels = []
    with open(positive_file) as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            if len(parse_line) == 20:
                examples.append(parse_line)
                labels.append([0, 1])

    with open(negative_file) as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            if len(parse_line) == 20:
                examples.append(parse_line)
                labels.append([1, 0])
    return tf.data.Dataset.from_tensor_slices((examples, labels)).shuffle(len(examples)).batch(batch_size)


def convert_textcode2tagcode(sample, wi_dict, iw_dict, wi_dict_tag, iw_dict_tag, VOCAB_SIZE, VOCAB_SIZE_TAG):
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


def text_code2tag_code(samples, wi_dict, iw_dict, wi_dict_tag, iw_dict_tag, VOCAB_SIZE, VOCAB_SIZE_TAG):
    samples = samples.cpu().numpy()
    output = []
    for i in samples:
        output.append(
            convert_textcode2tagcode(i, wi_dict, iw_dict, wi_dict_tag, iw_dict_tag, VOCAB_SIZE, VOCAB_SIZE_TAG))
    output = torch.from_numpy(np.array(output))
    return output.cuda()