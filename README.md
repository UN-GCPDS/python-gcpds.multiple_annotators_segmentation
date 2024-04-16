# Chained deep learning using generalized cross entropy for multiple annotators segmentation

Given a $k$ class multiple annotators segmentation problem with a dataset like the following:

$$
X ∈ ℝ^{W × H}, { Y_r ∈ {0,1}^{W × H × K} }_{r=1}^R; Ŷ ∈ [0,1]^{W×H×K} = f(X)
$$

The segmentation mask function will map input output as follows:

$$
f: \mathbb{R}^{W\times H} \to [0,1]^{W\times H\times K}
$$

$\mathbf Y$ will satisfy the following condition for being a softmax-like representation:

$$
Y_r[w,h,:] 1^⊤_k = 1; w ∈ W, h ∈ H
$$

Now, let's suppose the existence of an annotators reliability map estimation $𝛌_r; r ∈ R$:

$$
Y_r[w,h,:] 1^⊤_k = 1; w ∈ W, h ∈ H
$$

Then, our $TGCE_{SS}$ is defined as:

$$
TGCE_SS(Y_r, f(X; θ) | 𝛌_r(X; θ)) = E_r { E_{w,h} { 𝛌_r(X; θ) ◦ E_k { Y_r ◦ ((1_{W×H×K} - f(X; θ)^{∘ q }) / q); k ∈ K } + (1_{W×H} - 𝛌_r(X; θ)) ◦ ((1_{W×H} - ((1/k) 1_{W×H})^{∘ q}) / q); w ∈ W, h ∈ H }; r ∈ R }
$$

Where $q \in (0,1)$.

The total loss for a given batch holding $N$ samples is:

$$
\mathscr{L}\left(\mathbf{Y}_r[n],f(\mathbf X[n];\theta) | \mathbf{\Lambda}_r (\mathbf X[n];\theta)\right)  = \frac{1}{N} \sum_{n}^NTGCE_{SS}(\mathbf{Y}_r[n],f(\mathbf X[n];\theta) | \mathbf{\Lambda}_r (\mathbf X[n];\theta))
$$
