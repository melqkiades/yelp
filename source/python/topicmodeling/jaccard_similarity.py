import numpy

from topicmodeling import hungarian


class JaccardBinary:
    """
    Simple binary Jaccard-based ranking comparison, which does not take into account rank positions.
    """

    def similarity(self, gold_ranking, test_ranking):
        sx = set(gold_ranking)
        sy = set(test_ranking)
        numer = len(sx.intersection(sy))
        if numer == 0:
            return 0.0
        denom = len(sx.union(sy))
        if denom == 0:
            return 0.0
        return float(numer) / denom

    def __str__(self):
        return "%s" % (self.__class__.__name__)


class AverageJaccard(JaccardBinary):
    """
    A top-weighted version of Jaccard, which takes into account rank positions.
    This is based on Fagin's Average Overlap Intersection Metric.
    """

    def similarity(self, gold_ranking, test_ranking):
        k = min(len(gold_ranking), len(test_ranking))
        total = 0.0
        for i in range(1, k + 1):
            total += JaccardBinary.similarity(self, gold_ranking[0:i],
                                              test_ranking[0:i])
        return total / k


# --------------------------------------------------------------
# Ranking Set Agreement
# --------------------------------------------------------------

class RankingSetAgreement:
    """
    Calculates the agreement between pairs of ranking sets, using a specified measure of
    similarity between rankings.
    """

    def __init__(self, metric=AverageJaccard()):
        self.metric = metric

    def similarity(self, rankings1, rankings2):
        """
        Calculate the overall agreement between two different ranking sets. This is given by the
        mean similarity values for all matched pairs.
        """
        self.results = None
        self.S = self.build_matrix(rankings1, rankings2)
        score, self.results = self.hungarian_matching()
        return score

    def build_matrix(self, rankings1, rankings2):
        """
        Construct the similarity matrix between the pairs of rankings in two
        different ranking sets.
        """
        rows = len(rankings1)
        cols = len(rankings2)
        S = numpy.zeros((rows, cols))
        for row in range(rows):
            for col in range(cols):
                S[row, col] = self.metric.similarity(rankings1[row], rankings2[col])
        return S

    def hungarian_matching(self):
        """
        Solve the Hungarian matching problem to find the best matches between columns and rows based on
        values in the specified similarity matrix.
        """
        # apply hungarian matching
        h = hungarian.Hungarian()
        C = h.make_cost_matrix(self.S)
        h.calculate(C)
        results = h.get_results()
        # compute score based on similarities
        score = 0.0
        for (row, col) in results:
            score += self.S[row, col]
        score /= len(results)
        return (score, results)


def main():
    list1 = ['album', 'music', 'best', 'award', 'win']
    list2 = ['sport', 'best', 'win', 'medal', 'award']

    R1 = [
        ['sport', 'win', 'award'],
        ['bank', 'finance', 'money'],
        ['music', 'album', 'band']
    ]

    R2 = [
        ['finance', 'bank', 'economy'],
        ['music', 'band', 'award'],
        ['win', 'sport', 'money']
    ]

    average_jaccard = AverageJaccard()
    jaccard_binary = JaccardBinary()

    print('Jaccard similarity :%f' % jaccard_binary.similarity(list1, list2))
    print('average Jaccard similarity :%f' %
          average_jaccard.similarity(list1, list2))

    print('Jaccard similarity :%f' % jaccard_binary.similarity(list1, list2))

    # First argument was the reference term ranking
    all_term_rankings = [R1, R2]

    reference_term_ranking = all_term_rankings[0]
    all_term_rankings = all_term_rankings[1:]
    r = len(all_term_rankings)
    print("Loaded %d non-reference term rankings" % r)

    # Perform the evaluation
    metric = AverageJaccard()
    matcher = RankingSetAgreement(metric)
    print("Performing reference comparisons with %s ..." % str(metric))
    all_scores = []
    for i in range(r):
        score = matcher.similarity(reference_term_ranking, all_term_rankings[i])
        all_scores.append(score)

    # Get overall score across all candidates
    all_scores = numpy.array(all_scores)

    print("Stability=%.4f [%.4f,%.4f]" % (
        all_scores.mean(), all_scores.min(), all_scores.max()))


# main()
