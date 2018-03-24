# This script compares the negative and postive Rap1 binding files and removes
# any positive sequences accidentally included in the negative file. The output
# is filt-negative.txt, a text file of same format as rap1-lieb-positives.txt.

from itertools import groupby

# Function to return the reverse complement of a sequence
def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    bases = list(seq)
    bases = reversed([complement.get(base,base) for base in bases])
    return ''.join(bases)

# Function to read in text file as a list of sequences
def read_text(filename):

    full_list = []
    with open(filename, 'r') as pf:
        fwd_list = pf.read().splitlines()

        # Also return reverse complements of sequences
        for seq in fwd_list:
            full_list.append(seq)
            full_list.append(reverse_complement(seq))

    return full_list

# Function to read in fasta file as a list of sequences
def read_fasta(filename):

    seqs = []

    with open(filename, "r") as nf:
        headers = (x[1] for x in groupby(nf, lambda line: line[0] == ">"))

        for header in headers:
            headerStr = header.__next__()[1:].strip()
            seq = "".join(s.strip() for s in headers.__next__())
            seqs.append(seq)

            # Include the reverse complement of the sequence, too
            seqs.append(reverse_complement(seq))

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
neg_file = "seqs/yeast-upstream-1k-negative.fa"
pos_file = "seqs/rap1-lieb-positives.txt"
out_nfile = "seqs/filt-negative.txt"
out_pfile = "seqs/filt-positive.txt"

# Read in the lists from the files
pos_list = read_text(pos_file)
neg_list = read_fasta(neg_file)
keep_list = remove_strings_w_subs(neg_list, pos_list)

# Save out all the positive sequences (including reverse complements)
with open(out_pfile, 'w') as fh:
    rev_list = []
    for i in pos_list:
        rev_list.append(reverse_complement(i))
    full_list = pos_list + rev_list
    fh.write("\n".join(str(i) for i in full_list))

# Save out the negative sequences as seq\nseq\n etc.
with open(out_nfile, 'w') as fh:
    fh.write("\n".join(str(i) for i in keep_list))
