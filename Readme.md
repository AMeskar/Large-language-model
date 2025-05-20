# Bigram Name Generator (with a Tiny Neural Net)

> A step-by-step walk through of the notebook.

---

## 1  Dataset  
`names.txt` is a plain text list one name per line. We read it, grab some quick stats (min/max length, first 10 names) and keep going.

## 2  Raw Bigram Counts  
1. Add start `<S>` and end `<E>` tokens around every name.  
2. Walk through each adjacent pair (`zip(new_n, new_n[1:])`) and count it in a `dict`.  
3. Sort counts to see which pairs dominate. *(Spoiler: “n → <E>” wins.)*

## 3  28 × 28 Count Matrix  
* Map chars to integers (`a→0 … z→25, <S>→26, <E>→27`).  
* Fill a tensor `N[x1, x2] += 1` for every observed bigram.

## 4  Visual Check  
Plot `N` as a heat-map—row = first char, column = second. It exposes garbage pairs like `<E><S>`.

## 5  Cleaner Boundaries  
Switch from `<S>/<E>` to a single dot `.`. Index 0 becomes “boundary”, A–Z shift to 1-26. This drops illegal pairs and shrinks the matrix to 27 × 27.

## 6  Probabilities & Sampling  
* Add-one smoothing (`N+1`) ⇒ `P`.  
* Row-normalise so each row sums to 1.  
* Sample new names by walking the Markov chain until you hit “.” again.

## 7  Model Quality  
Average **negative log-likelihood** (≈ 2.42 bits) tells us how surprised the model is, on average, per character. Lower = better.

## 8  Toy Neural Net  
* One-hot encode inputs.  
* Single linear layer (`27×27` weights).  
* Softmax manually (`exp` + row-normalise).  
* Use cross-entropy a.k.a. NLL as the loss.  
* Backprop + dumb SGD step shows the net quickly reaches the same 2.4-ish NLL—no magic yet, just rediscovering the count table.

## 9  Name Generation (NN)  
After training, feed the network its last prediction, sample the next char from the softmax row, loop until “.”, print. Same idea, but the probabilities now come from learned weights rather than raw counts.

---

### Running It

```bash
pip install torch matplotlib
run the notebook

