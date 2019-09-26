from os.path import join


def get_bn2wn(resources_path):
    """
      Returns a dictionary with a mapping between
        BabelNet synsets and WordNet synsets
      """
    bn2wn, wn2bn = dict(), dict()
    file_path = join(resources_path, 'babelnet2wordnet.tsv')
    with open(file_path) as f:
        for line in f:
            bn, wn = line.strip().split('\t')[:2]
            bn2wn[bn] = wn
            wn2bn[wn] = bn

    return bn2wn, wn2bn


def get_bn2lex(resources_path):
    bn2lex, lex2bn = dict(), dict()
    file_path = join(resources_path, 'babelnet2lexnames.tsv')
    with open(file_path) as f:
        for line in f:
            bn, lex = line.strip().split('\t')[:2]
            bn2lex[bn] = lex
            lex2bn[lex] = bn

    return bn2lex, lex2bn


def get_bn2dom(resources_path):
    bn2dom, dom2bn = dict(), dict()
    file_path = join(resources_path, 'babelnet2wndomains.tsv')

    with open(file_path) as f:
        for line in f:
            bn, dom = line.strip().split('\t')[:2]
            bn2dom[bn] = dom
            dom2bn[dom] = bn

    return bn2dom, dom2bn
