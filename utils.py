def return_prediction(model, target_headers, X1, X2):
    print("predicting...")

    # get predictions
    predictions = model.predict(x=[X1, X2])
    one_hot = [[1 if max(prediction)==x else 0 for x in prediction] for prediction in predictions]
    processed_to_names = [target_headers[vec.index(1)] for vec in one_hot]

    scores_names = {processed_to_names.count(name)/len(processed_to_names):name for name in processed_to_names}

    # return results in easy to consume format for js
    scores = list(scores_names.keys())
    scores.sort(reverse=True)
    results = dict()
    for i, score in enumerate(scores[:5]):
        results[f"{i}"] = {
            "username": scores_names[score],
            "score": score
        }
    
    return results