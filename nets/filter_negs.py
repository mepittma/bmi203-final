# This script compares the negative and postive Rap1 binding files and removes
# any positive sequences accidentally included in the negative file. The output
# is filt-negative.txt, a text file of same format as rap1-lieb-positives.txt.

from itertools import groupby

# Function to read in text file as a list of sequences
def read_text(filename):

    with open(filename, 'r') as pf:
        pos_list = pf.read().splitlines()

    return pos_list

# Function to read in fasta file as a list of sequences
def read_fasta(filename):

    seqs = []

    with open(filename, "r") as nf:
        headers = (x[1] for x in groupby(nf, lambda line: line[0] == ">"))

        for header in headers:
            headerStr = header.__next__()[1:].strip()
            seq = "".join(s.strip() for s in headers.__next__())
            seqs.append(seq)

    return seqs

# Function to remove an element from a list if any other element from another list
# is a substring of that element.
def remove_strings_w_subs(big_list, substring_list):

    save_list = []
    overlaps = 0

    for big_string in big_list:

        matched = False
        for sub_string in substring_list:
            if sub_string in big_string:
                matched = True
                overlaps += 1
                break

        if not matched:
            save_list.append(big_string)

    print("Number of sequences removed: ", overlaps)
    return save_list


# # # # # # # # # # COMMANDS # # # # # # # # # #

# files:
base_dir = "/Users/student/Documents/BMI206/bmi203-final/seqs"
neg_file = "{}/yeast-upstream-1k-negative.fa".format(base_dir)
pos_file = "{}/rap1-lieb-positives.txt".format(base_dir)
out_file = "{}/filt-negative.txt".format(base_dir)

# Read in the lists from the files
pos_list = read_text(pos_file)
neg_list = read_fasta(neg_file)
keep_list = remove_strings_w_subs(neg_list, pos_list)

# Save out the negative sequences as seq\nseq\n etc.
with open(out_file, 'w') as fh:
    fh.write("\n".join(str(i) for i in keep_list))
