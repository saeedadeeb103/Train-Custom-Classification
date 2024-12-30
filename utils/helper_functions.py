def normalize_ratios(ratios):
    total = sum(ratios)
    return [r / total for r in ratios]