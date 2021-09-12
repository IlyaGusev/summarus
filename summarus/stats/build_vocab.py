import argparse
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from summarus.util.spacy import spacy_deserialize, normalize


def build_idf_vocabulary(docs_lemmas, max_df=1.0, min_df=2):
    texts = [" ".join(lemmas) for lemmas in docs_lemmas]

    print("Building TfidfVectorizer...")
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    vectorizer.fit(texts)
    idf_vector = vectorizer.idf_.tolist()

    print("{} words in vocabulary".format(len(idf_vector)))
    idfs = list()
    for word, idx in vectorizer.vocabulary_.items():
        idfs.append((word, idf_vector[idx]))

    idfs.sort(key=lambda x: (x[1], x[0]))
    return idfs


def count_words(docs_lemmas):
    cnt = Counter()
    for lemmas in tqdm(docs_lemmas):
        cnt.update(lemmas)
    return cnt


def main(files, output_path):
    docs = []
    for f in files:
        docs += spacy_deserialize(f)

    lemmas = [normalize(doc) for doc in docs]
    idfs = build_idf_vocabulary(lemmas)
    cnt = count_words(lemmas)
    cnt_list = cnt.most_common()
    cnt_list.sort(key=lambda x: (x[1], -len(x[0])), reverse=True)
    ranks = {word: rank for rank, (word, cnt) in enumerate(cnt_list)}

    print("Saving vocabulary...")
    with open(output_path, "w") as w:
        w.write("{}\t{}\t{}\t{}\n".format("word", "idf", "count", "rank"))
        for word, idf in idfs:
            count = cnt[word]
            rank = ranks[word]
            w.write("{}\t{}\t{}\t{}\n".format(word, idf, count, rank))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+')
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
