def feature_selection(training_set, validation_set):
    k = 0  # counter number of feature to select
    subset = set()  # empty set ("null set") so that the k = 0 (where k is the size of the subset)
    header = list(training_set[0].columns.values)
    total_features = set(header)
    go_on = True

    while k < 5 and go_on:

        best_score = 1
        last_score = 1
        best_feature = ""
        feature_set = subset.copy()

        for f in (total_features - subset):
            feature_set.add(f)
            score = evaluate_score(training_set, list(feature_set), validation_set, header)
            feature_set.remove(f)
            if score < best_score:
                best_score = score
                best_feature = f

        subset.add(best_feature)
        k += 1
        print("best feature:" + str(best_feature) + "best score:" + str(best_score))
        print(subset)

        if best_score > last_score:
            go_on = False
        else:
            go_on = True

        last_score = best_score

    return subset, best_score
