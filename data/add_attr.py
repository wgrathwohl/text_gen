import json
from json.encoder import JSONEncoder
JSONENCODER = JSONEncoder()
import nltk


def lazy_read_json(file_object):
    """lazily read json file line by line"""
    while True:
        data = file_object.readline()
        if not data:
            break
        yield json.loads(data)



def lazy_read_json(file_object, vocab):
    """lazily read json file line by line"""
    while True:
        data = file_object.readline()

        if not data:
            break
        yield json.loads(data)



def fill_business_dict(file, b_dict):
    """lazily read json file line by line"""
    while True:
        data = file.readline()
        if not data:
            b_file.close()
            yield False
        x = json.loads(data)
        b_dict[x['business_id']] = x
        yield True


def _preprocess_sentence(s):
    s = s.lower()
    s_list = nltk.tokenize.word_tokenize(s)
    s = " ".join(s_list)
    return s


def add_attribute(in_file, out_file, attr_names, b_dict):
    """ adds attribute for attr in attr_names """
    while True:
        data = json.loads(in_file.readline())
        if not data:
            in_file.close()
            out_file.close()
            yield False
        data['text'] = _preprocess_sentence(data['text'])
        key = data['business_id']
        for attr in attr_names:
            try:
                data[attr] = b_dict[key][attr]
            except KeyError:
                pass
        out_file.write(JSONENCODER.encode(data))
        out_file.write('\n')
        yield True


if __name__=='__main__':
    attr_list = ['categories', 'city', 'attributes']
    VOCAB_FILE= None
    with open(VOCAB_FILE, 'r') as v:
        words = v.readlines()
        vocab = dict(enumerate(words))

    b_dict = dict()
    b_file = open('yelp_academic_dataset_business.json')
    lazy_read_and_fill = fill_business_dict(b_file, b_dict)
    while next(lazy_read_and_fill): pass
    print "Read in business dictionary"

    rev_file = open('yelp_academic_dataset_review.json')
    out_file = open('NEW_yelp_academic_dataset_review.json','w')
    lazy_add_and_write = add_attribute(rev_file, out_file, attr_list, b_dict)

    print "About to write new JSON file"
    while next(lazy_add_and_write): pass
    print "Finished writing new JSON file"
