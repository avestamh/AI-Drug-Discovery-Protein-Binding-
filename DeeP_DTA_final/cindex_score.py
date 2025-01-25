# cindex_score.py

def cindex_score(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i != j:  # Fixed comparison from 'is not' to '!='
                if Y[i] > Y[j]:
                    pair += 1
                    summ +=  1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    return summ / pair if pair > 0 else 0
