# Chained deep learning using generalized cross entropy for multiple annotators segmentation

Given a $k$ class multiple annotators segmentation problem with a dataset like the following:

$$
\mathbf X \in \mathbb{R}^{W \times H}, \{ \mathbf Y_r \in \{0,1\}^{W \times H \times K} \}_{r=1}^R; \mathbf {\hat Y} \in [0,1]^{W\times H \times K} = f(\mathbf X)
$$

The segmentation mask function will map input output as follows:

$$
f: \mathbb  R ^{W\times H} \to [0,1]^{W\times H\times K}
$$

$\mathbf Y$ will satisfy the following condition for being a softmax-like representation:

$$
\mathbf Y_r[w,h,:] \mathbf{1} ^ \top _ k = 1; w \in W, h \in H
$$

Now, let's suppose the existence of an annotators reliability map estimation $\Lambda_r; r \in R$;

$$
\{{ \Lambda_r (\mathbf X; \theta ) \in [0,1] ^{W\times H} \}}_{r=1}^R
$$

Then, our $TGCE_{SS}$:

$$
TGCE_{SS}(\mathbf{Y}_r,f(\mathbf X;\theta) | \mathbf{\Lambda}_r (\mathbf X;\theta)) =\mathbb E_{r} \left\{ \mathbb E_{w,h} \left\{ \Lambda_r (\mathbf X; \theta) \circ \mathbb E_k \bigg\{    \mathbf Y_r \circ \bigg( \frac{\mathbf 1 _{W\times H \times K} - f(\mathbf X;\theta) ^{\circ q }}{q} \bigg); k \in K  \bigg\}  + \right. \right.
$$

$$
\left. \left. \left(\mathbf 1 _{W \times H } - \Lambda _r (\mathbf X;\theta)\right) \circ \bigg(   \frac{\mathbf 1_{W\times H} - (\frac {1}{k} \mathbf 1_{W\times H})^{\circ q}}{q} \bigg); w \in W, h \in H \right\};r\in R\right\}
$$

Where $q \in (0,1)$

Total Loss for a given batch holding $N$ samples:

$$
\mathscr{L}\left(\mathbf{Y}_r[n],f(\mathbf X[n];\theta) | \mathbf{\Lambda}_r (\mathbf X[n];\theta)\right)  = \frac{1}{N} \sum_{n}^NTGCE_{SS}(\mathbf{Y}_r[n],f(\mathbf X[n];\theta) | \mathbf{\Lambda}_r (\mathbf X[n];\theta))
$$
