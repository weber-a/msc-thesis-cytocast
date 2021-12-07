from scipy.stats import fisher_exact
from collections import Counter

def enrichment_analysis(spotlight_terms, spotlight_complexes, background_terms, background_complexes):
    termcount_of_spotlight = dict(Counter(spotlight_terms))
    termcount_of_background = dict(Counter(background_terms))
    
    total_no_in_spotlight = len(spotlight_complexes)
    total_no_in_background = len(background_complexes)

    pvalues = dict()
    
    for association, count_in_spotlight in termcount_of_spotlight.items():
        count_in_background = termcount_of_background[association]
        contingency_table = [
            [count_in_spotlight, total_no_in_spotlight-count_in_spotlight],
            [count_in_background, total_no_in_background-count_in_background]
        ]
        pvalue = fisher_exact(contingency_table, alternative="greater")[1]
        pvalues[association] = pvalue
        
    return pvalues