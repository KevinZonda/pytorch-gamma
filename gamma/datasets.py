from sklearn.model_selection import train_test_split

def split_dataset(X, y, sizes=[0.2], rand_state=42):
    if isinstance(sizes, float):
        sizes = [sizes]
    elif isinstance(sizes, int):
        sizes = [sizes]
    elif isinstance(sizes, list):
        pass
    else:
        sizes = list(sizes)
    
    results = []
    current_X, current_y = X, y
    
    remaining = 1.0
    for size in sizes[:-1]:
        adj_size = size / remaining
        X_train, X_split, y_train, y_split = train_test_split(
            current_X, current_y, 
            test_size=adj_size,
            random_state=rand_state
        )
        results.append((X_split, y_split))
        current_X, current_y = X_train, y_train
        remaining -= size
    
    results.append((current_X, current_y))
    
    return tuple(results)
