from Utils.TimeLogger import log

def early_stopping(log_val, bst_val, es, expected_order='HR', patience=5):
    assert expected_order in ['HR', 'NDCG']
    should_stop = False
    if log_val[expected_order] > bst_val[expected_order]:
        es = 0
        bst_val = log_val
    else:
        es += 1
        if es >= patience:
            should_stop = True
    return bst_val, es, should_stop