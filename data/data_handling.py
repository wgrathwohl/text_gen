"""
Stolen from https://github.com/alfredolainez/yelp-dataset/blob/master/data_handling.py

Functions to read and extract review data from the Yelp Dataset Challenge dataset.
Data is partitioned so that there are no memory problems when processing the data.
However creating the partitions requires a lot of memory.
Works with Yelp Challenge Dataset 7
"""

import json
import pickle
import random
import sys
import numpy as np
import collections
import nltk
from tqdm import tqdm

DEFAULT_REVIEWS_FILE = "./data/yelp_data/int_idx_yelp_academic_dataset_review.json"
DEFAULT_BUSINESS_FILE = "./data/yelp_data/yelp_academic_dataset_business.json"
DEFAULT_REVIEWS_PICKLE = "./data/pickles/reviews.pickle"

def pickles_from_json(num_partitions=200, accepted=None,
                      json_file=DEFAULT_REVIEWS_FILE, pickle_name=DEFAULT_REVIEWS_PICKLE):
    """
    Dumps a json into a number of smaller pickle partitions, which contain a list of python objects
    read from the json. This allows easier posterior data processing.
    Pickle files are saved as pickle_name1, ..., pickle_name
    accepted is a generic function that returns true or false for a single json object, specifying whether or not
    the object should be added to the pickle. It allows preprocessing of the data so as not to save
    unnecessary elements
    """

    print "Reading json file..."
    object = []
    num_not_accepted = 0
    total_processed = 0
    with open(json_file) as json_data:
        for line in json_data:
            if accepted != None:
                element = json.loads(line)
                if accepted(element):
                    object.append(element)
                else:
                    num_not_accepted += 1
                    sys.stdout.write('Not accepted objects: %d / %d \r' % (num_not_accepted, total_processed))
                    sys.stdout.flush()
            else:
                object.append(json.loads(line))
            total_processed += 1

    print "Shuffling resulting python objects"
    random.shuffle(object)

    length_partition = len(object)/num_partitions
    remaining_to_process = len(object)
    current_partition = 1
    while remaining_to_process > 0:
        sys.stdout.write('Exporting pickle %d out of %d \r' % (current_partition, num_partitions))
        sys.stdout.flush()

        # All the remaining elements go to the last partition
        if current_partition == num_partitions:
            stop = None
            num_in_partition = remaining_to_process
        else:
            stop = -remaining_to_process + length_partition
            num_in_partition = length_partition

        pickle.dump(object[-remaining_to_process:stop],
                    open(pickle_name + '.' + str(current_partition), "wb"),
                    pickle.HIGHEST_PROTOCOL)

        current_partition += 1
        remaining_to_process -= num_in_partition

def load_partitions(partition_list, pickle_base_name=DEFAULT_REVIEWS_PICKLE + '.'):
    """
    Returns a python object being a list of dictionaries.
    It reads the data from a sequence of files starting with the given base name. For instance:
    partition_list = [2,4,6], pickle_base_name = "pickle." will read files pickle.2, pickle.4, pickle.6
    """

    num_partition = 1
    result = []
    for partition in partition_list:
        print 'Reading partition %d of %d' % (num_partition, len(partition_list))
        with open(pickle_base_name + str(partition)) as file:
            loaded_element = pickle.load(file)
            result.extend(loaded_element)

        num_partition += 1

    print "Read a total of %d partitions for a total of %d objects" % (num_partition - 1, len(result))
    return result

def get_business_data(json_file=DEFAULT_BUSINESS_FILE):
    """
    Reads business file and saves all data in a hash indexed by business id
    """

    business_hash = {}
    with open(json_file) as json_data:
        for line in json_data:
            business = json.loads(line)
            business_hash[business["business_id"]] = business
    return business_hash

def get_reviews_data(partitions_to_use, business_data, not_include_states=["EDH", "QC", "BW"],
                     pickle_base_name=DEFAULT_REVIEWS_PICKLE + '.'):
    """
    Gets loaded json data in pickles and returns fields of interest
    Filters by state: list of 'states' (they can be outside of the US) not to include in the reviews
    """
    data = load_partitions(partitions_to_use, pickle_base_name)
    review_texts = []
    useful_votes = []
    funny_votes = []
    cool_votes = []
    review_stars = []

    for review in data:
        if business_data[review["business_id"]]["state"] not in not_include_states:
            review_texts.append(review['text'])
            useful_votes.append(review['votes']['useful'])
            cool_votes.append(review['votes']['cool'])
            funny_votes.append(review['votes']['funny'])
            review_stars.append(review['stars'])

    return review_texts, useful_votes, funny_votes, cool_votes, review_stars

def give_balanced_classes(reviews, votes, votes_threshold):
    """
    From all the reviews and votes given for a given category, partitions the data
    into two classes.
    Reviews with 0 votes are considered "negative".
    Reviews with votes_threshold or more are considered "positive"
    All the positive reviews found are returned. The method is assuming majority of negative reviews.
    The same number of negative reviews is returned, randomly selected.
    Returned data is a shuffled balanced set of negative and positive reviews.
    """
    if votes_threshold <= 0:
        print "Needs positive threshold"
        return

    negative_reviews_indices = []

    # Find all the funny reviews we can
    final_reviews = []
    final_labels = []
    for i, review in enumerate(reviews):
        if votes[i] >= votes_threshold:
            final_reviews.append(review)
            final_labels.append(1)
        elif votes[i] == 0:
            negative_reviews_indices.append(i)

    # We want balanced classes so take same number
    np.random.shuffle(negative_reviews_indices)
    num_positive_reviews = len(final_reviews)
    for i in range(num_positive_reviews):
        final_reviews.append(reviews[negative_reviews_indices[i]])
        final_labels.append(0)

    # Shuffle final reviews and labels
    combined_lists = zip(final_reviews, final_labels)
    np.random.shuffle(combined_lists)
    final_reviews[:], final_labels[:] = zip(*combined_lists)

    print "Returning %d positive reviews and a total of %d reviews" % (num_positive_reviews, len(final_reviews))

    return (final_reviews, final_labels)


tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

def _process_sentence(s):
    s = s.lower()
    s_list = nltk.tokenize.word_tokenize(s)
    s = " ".join(s_list)
    return s

def dump_reviews_text(output_filename, pickle_base_name=DEFAULT_REVIEWS_PICKLE + '.'):
    """
    writes all review text to a file for vocabulary-building purposes
    """
    #business_data = get_business_data()

    for idx in range(100):
        with open('./data/rawtext/part_' + str(idx), 'w') as file:
            # iterate over
            partition = idx + 1
            reviews = load_partitions([partition])
            for review in tqdm(reviews):
                line = _process_sentence(review['text'])
                file.write((line+'\n').encode('utf-8'))
            print "Wrote partition " + str(partition) + "to file"


def create_data_sets(reviews, labels, write_to_pickle=True, problem=""):
    """
    Creates a 50% - 25% - 25% train/validation/test (default values) dataset
    for a binary balanced problem with labels 1/0
    All sets are automatically balanced assuming the input is balanced
    Data comes shuffled
    If write to pickles is equal to true, the datasets will be saved in files, labeled by problem
    """
    def sanity_check(labels):
        print str(len(labels)) + " total labels. " + str(sum(labels)) + " positive labels. " \
              + str(len(labels) - sum(labels)) + " negative labels. "

    train_reviews = []
    train_labels = []
    dev_reviews = []
    dev_labels = []
    test_reviews = []
    test_labels = []

    total_train = int(len(reviews) * 0.5 / 2) # divided by 2 because of 2 classes
    total_dev = int(len(reviews) * 0.25 / 2)

    current_pos_training = 0
    current_neg_train = 0
    current_pos_dev = 0
    current_neg_dev = 0

    for (review, vote) in zip(reviews, labels):
        if vote == 1:
            if current_pos_training < total_train:
                train_reviews.append(review)
                train_labels.append(vote)
                current_pos_training += 1
            elif current_pos_dev < total_dev:
                dev_reviews.append(review)
                dev_labels.append(vote)
                current_pos_dev += 1
            else:
                test_reviews.append(review)
                test_labels.append(vote)

        # Negative review
        else:
            if current_neg_train < total_train:
                train_reviews.append(review)
                train_labels.append(vote)
                current_neg_train += 1
            elif current_neg_dev < total_dev:
                dev_reviews.append(review)
                dev_labels.append(vote)
                current_neg_dev += 1
            else:
                test_reviews.append(review)
                test_labels.append(vote)

    # Shuffle data for every dataset
    combined_lists = zip(train_reviews, train_labels)
    np.random.shuffle(combined_lists)
    train_reviews, train_labels = zip(*combined_lists)

    combined_lists = zip(dev_reviews, dev_labels)
    np.random.shuffle(combined_lists)
    dev_reviews, dev_labels = zip(*combined_lists)

    combined_lists = zip(test_reviews, test_labels)
    np.random.shuffle(combined_lists)
    test_reviews, test_labels = zip(*combined_lists)

    # Sanity checks
    print "Total reviews: " + str(len(reviews))
    print "Original distribution: "
    sanity_check(labels)
    print "========================"
    print "Train labels"
    sanity_check(train_labels)
    print "========================"
    print "Dev labels"
    sanity_check(dev_labels)
    print "========================"
    print "Train labels"
    sanity_check(test_labels)

    # Write to pickles
    N = len(reviews)
    if write_to_pickle:
        print "Writing to pickle..."
        pickle.dump([train_reviews, train_labels],
                    open("TrainSet_" + problem + '_' + str(N), "wb"), pickle.HIGHEST_PROTOCOL)

        pickle.dump([dev_reviews, dev_labels],
                    open("DevSet_" + problem + '_' + str(N), "wb"), pickle.HIGHEST_PROTOCOL)

        pickle.dump([test_reviews, test_labels],
                    open("TestSet_" + problem + '_' + str(N), "wb"), pickle.HIGHEST_PROTOCOL)
        print "Done."

    return train_reviews, train_labels, dev_reviews, dev_labels, test_reviews, test_labels
