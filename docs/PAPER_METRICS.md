# Mathematical definitions

This appendix defines the quantities used to compare a base model, `m0`, with
its fine-tuned version, `m1`. For a given input, `P` and `Q` are their
next-token probability distributions. All logarithms use base `e`, and
divergences are measured in nats.

The implementation rules and result fields are described in
[`MEASUREMENT_PROTOCOL.md`](MEASUREMENT_PROTOCOL.md).

## Metrics

| Quantity | Meaning | Value and theoretical range |
|---|---|---|
| `B` | divergence between the base and fine-tuned next-token distributions over the complete vocabulary | one value per input, from 0 to `ln(2)` nats |
| `B_alpha` | the same divergence after conditioning both distributions on the union of their top-p token sets | one value per input, from 0 to `ln(2)` nats |
| `B_k` | the same divergence after conditioning both distributions on the union of their top-k token sets | one value per input, from 0 to `ln(2)` nats |
| `P(T)`, `Q(T)` | total probability assigned to a declared target-token set `T` | one value per model and input, in `(0, 1]` |
| `B_T` | divergence between the two distributions after conditioning them on `T` | one value per input, from 0 to `ln(2)` nats |
| `B_between` | component of `B_T` produced by changes in the total probability assigned to each group | between-group component, from 0 to `B_T` |
| `B_within` | component of `B_T` produced by changes in the relative probabilities of tokens within each group | within-group component, from 0 to `B_T` |
| `SI` | difference between the average `G_m`-versus-`G_f` gap in `C_L` and `C_S` | one value per model, from -2 to 2 |
| `Delta SI` | fine-tuned `SI` minus base-model `SI` | one value per run, from -4 to 4 |
| `G_l` | similarity of contextual concept representations at layer `l` | one value per layer, from -1 to 1 |
| `1-G_l` | representational change measured at layer `l` | one value per layer, from 0 to 2 |
| depth centroid | position of the `1-G_l` curve along model depth | one value from 0 to 1; undefined for a zero curve |

## Complete-vocabulary divergence

Let `P` be the base-model next-token distribution and `Q` the fine-tuned
distribution for the same input. Their midpoint is:
~~~text
M = (P + Q) / 2
~~~
ERA uses Lin's K-divergence:
~~~text
B(P, Q) = sum_i P_i log(P_i / M_i)
~~~
For any two distributions `a` and `b`, the KL divergence used below is:
~~~text
KL(a || b) = sum_i a_i log(a_i / b_i)
~~~
The measure is directional because `P` is the reference distribution.
Exchanging `P` and `Q` can change the result. Its range and zero point are:
~~~text
0 <= B <= ln(2)
B = 0 if and only if P = Q
~~~

## Restricted top-p and top-k views

`B_alpha` builds a top-p set for each model. A top-p set is the smallest set
of high-probability tokens whose total mass reaches `alpha`. `B_k` selects the
`k` highest-probability tokens from each model. Both `alpha` and `k` are probe
parameters.

For both metrics, ERA takes the exact union of the sets selected by `m0` and
`m1`. It reads the original probability of every token in that union from both
full distributions, conditions each distribution on the union, and applies
the same K-divergence.

## Target-conditioned divergence and its decomposition

Let `T` be a finite set of target tokens, partitioned into disjoint groups.
First condition both distributions on `T`:
~~~text
P(T) = sum of P_i for tokens i in T
Q(T) = sum of Q_i for tokens i in T
p_i = P_i / P(T)
q_i = Q_i / Q(T)
m_i = (p_i + q_i) / 2
~~~
The conditioned distributions `p` and `q` each have total mass one over `T`.
Their divergence is:
~~~text
B_T = sum over i in T of p_i log(p_i / m_i)
~~~
For each group `g`, its total mass and its conditional distributions are:
~~~text
p_g = sum over i in g of p_i
m_g = sum over i in g of m_i
p(i|g) = p_i / p_g
m(i|g) = m_i / m_g
~~~
This gives the decomposition:
~~~text
B_between = sum_g p_g log(p_g / m_g)
B_within  = sum_g p_g KL(p(.|g) || m(.|g))
B_T       = B_between + B_within
~~~
`B_between` measures changes in the total probability assigned to the groups.
`B_within` measures changes in the relative probabilities of tokens inside
each group. Together they account for the complete target-conditioned
divergence. A group with `p_g = 0` contributes zero to `B_within`. The same
components may also be written as `B_B` and `B_W`.

## Stereotype Index

Let `G_m` and `G_f` be the two target groups, and let `C_L` and `C_S` be the
two context families being compared. For a model `m` and context `c`, the two
group probabilities are conditioned on their union. Their difference is:
~~~text
gap_m(c) = P_m(G_m | c) - P_m(G_f | c)
~~~
The mean gaps in the two context families are:
~~~text
LB_m = mean of gap_m(c) over c in C_L
SB_m = mean of gap_m(c) over c in C_S
~~~
The index and its fine-tuning change are:
~~~text
SI_m = LB_m - SB_m
Delta SI = SI_m1 - SI_m0
~~~
Each gap lies in `[-1, 1]`, so `SI` lies in `[-2, 2]` and `Delta SI` lies in
`[-4, 4]`. Positive `Delta SI` means that the declared contrast increased
after fine-tuning. Negative values mean that it decreased.

## Internal representations across layers

This measurement uses the hidden states produced for the declared contexts and
concepts. Let `N` be the number of contexts. For every context `c` and every
concept token `k` in `K`, ERA creates a separate sequence. It tokenizes the
context without adding special tokens and appends `k` as the final token. At
each hidden-state level `l`, `a_m(c, k, l)` is the vector produced by model `m`
at that final position.

For every model, concept, and level, ERA averages these vectors across the `N`
contexts:
~~~text
h_m(k, l) = (1 / N) sum_c a_m(c, k, l)
~~~
It then compares the base and fine-tuned average vectors for each concept and
averages their cosine similarities. Here, `|K|` is the number of concepts:
~~~text
G_l = (1 / |K|) sum_{k in K} cosine(h_m0(k, l), h_m1(k, l))
d_l = 1 - G_l
~~~
`G_l` is the average representational similarity at level `l`; `d_l` is the
corresponding drift. Their ranges are:
~~~text
-1 <= G_l <= 1
0 <= d_l <= 2
~~~
Every context-averaged vector used in the cosine must have positive norm.

The comparison assumes that each vector coordinate in the base model
corresponds to the same coordinate in the fine-tuned model. This correspondence
comes from fine-tuning the descendant directly from the base. The two models
must therefore have matching hidden-state levels and matching vector shapes at
each corresponding level. Different levels may have different widths.

For `L` hidden-state levels indexed from `0` to `L - 1`, the normalized depth
centroid summarizes where the drift values `d_l` are concentrated:
~~~text
depth centroid = sum_l l * d_l / ((L - 1) * sum_l d_l)
~~~
The factor `L - 1` maps the level indices onto normalized depth. Level zero has
depth 0 and the final level has depth 1. When every `d_l` is zero, there is no
drift location to summarize and the centroid is undefined. A one-level curve
uses depth 0.

## Aggregation rules

One comparison contains a base model, its fine-tuned version, and `N`
evaluation inputs.

The probability quantities `B`, `B_alpha`, `B_k`, `B_T`, `B_between`,
`B_within`, `P(T)`, and `Q(T)` are calculated separately for every input.
Each quantity is summarized across the `N` inputs by its arithmetic mean,
sample standard deviation with denominator `N - 1`, minimum, and maximum. The
sample standard deviation is undefined for a single input.

The between and within shares describe how the total target-conditioned
divergence is divided across the complete set of inputs:
~~~text
between share = sum_c B_between(c) / sum_c B_T(c)
within share  = sum_c B_within(c)  / sum_c B_T(c)
~~~
Both shares are undefined when the denominator is zero. When it is positive,
the two shares sum to one.

`SI` uses two separate context-family means. For each model, ERA first
averages the conditional group gaps within `C_L` and within `C_S`, then
subtracts the second mean from the first. `Delta SI` is the fine-tuned value
of `SI` minus the base value.

The layer calculation follows a different order. For every concept and layer,
ERA first averages its activation vectors across the `N` inputs. It calculates
the cosine similarity between the corresponding base and fine-tuned average
vectors, then averages those cosine values across concepts to obtain `G_l`.
The mean geometric drift is the arithmetic mean of `1-G_l` across layers. The
depth centroid uses the same complete layer-drift curve.

## Mathematical domain and undefined cases

- `P` and `Q` are finite probability distributions with non-negative entries
  and total mass one.
- Top-p and top-k ties follow one fixed order, so the selected sets are unique.
- `P(T)` and `Q(T)` must be positive because conditioning on a zero-mass set is
  undefined.
- Between and within fractions are undefined when the total target divergence
  is zero, because their denominator is zero.
- Cosine similarity is undefined for a zero-norm representation vector.
