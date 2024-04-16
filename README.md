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
\bigg\{ \Lambda_r (\mathbf X; \theta ) \in [0,1] ^{W\times H} \bigg\}_{r=1}^R
$$

Then, our $TGCE_{SS}$ is defined as:

$$
\begin{align*}
TGCE_{SS}(\mathbf{Y}_r,f(\mathbf X;\theta) | \mathbf{\Lambda}_r (\mathbf X;\theta)) &= \mathbb{E}_{r} \left\{ \mathbb{E}_{w,h} \left\{ \Lambda_r (\mathbf X; \theta) \circ \mathbb{E}_k \left\{    \mathbf Y_r \circ \left( \frac{\mathbf 1 _{W\times H \times K} - f(\mathbf X;\theta) ^{\circ q }}{q} \right); k \in K  \right\}  \right. \right. \\ 
& \left. \left. + \left(\mathbf 1 _{W \times H } - \Lambda _r (\mathbf X;\theta)\right) \circ \left(   \frac{\mathbf 1_{W\times H} - \left(\frac {1}{k} \mathbf 1_{W\times H}\right)^{\circ q}}{q} \right); w \in W, h \in H \right\};r\in R\right\}
\end{align*}
$$

Where $q \in (0,1)$.

The total loss for a given batch holding $N$ samples is:

$$
\mathscr{L}\left(\mathbf{Y}_r[n],f(\mathbf X[n];\theta) | \mathbf{\Lambda}_r (\mathbf X[n];\theta)\right)  = \frac{1}{N} \sum_{n}^NTGCE_{SS}(\mathbf{Y}_r[n],f(\mathbf X[n];\theta) | \mathbf{\Lambda}_r (\mathbf X[n];\theta))
$$
