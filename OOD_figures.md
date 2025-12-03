# Additional figures for Reviewer Pu3Xf
We provide here additional figures for our official comment to Reviewer Pu3Xf. Thank you for taking the time to review them.

## Comparison of XPlore and RSGG: t-SNE projection of Wavelet Characteristic embeddings (Tree-Cycle dataset)
We can compare behaviours of XPlore against the second best explainer over the Tree-Cycle dataset. Depending on the original instance, we can see distinct behaviours of the two explainers taking place, let's analyze them.

### XPlore unable to land on Tree distribution for a Cycle CF
These figures show some instances in which original samples belonging to the Tree class are explained poorly by XPlore, highlighting a strong exploiting of the Out-of-distribution effect by XPlore, leading the oracle to missclassify such instances as Tree.

<img src="imgs\XPlore_RSGG\16.png" alt="Photo" width="650" />
<img src="imgs\XPlore_RSGG\56.png" alt="Photo" width="650" />
<!-- ![img](imgs\XPlore_RSGG\16.png) -->
<!-- ![img](imgs\XPlore_RSGG\56.png) -->

The following error is **remarking**, due to the propension of XPlore to **add edges** during the counterfactual search:

<img src="imgs\XPlore_RSGG\53.png" alt="Photo" width="650" />
<!-- ![img](imgs\XPlore_RSGG\53.png) -->

Sometimes, however, **also RSGG is unable to correctly land on the Tree distribution**:

<img src="imgs\XPlore_RSGG\20.png" alt="Photo" width="650" />
<!-- ![img](imgs\XPlore_RSGG\20.png) -->


### XPlore strenghts, RSGG weaknesses
Looking at counterfactuals for the Tree class the results show an **opposite trend**, XPlore is able to **correctly land CFs** on the Cycle distribution, while RSGG struggles to generate in-distribution CFs, not always landing on the Cycle distribution:

<img src="imgs\XPlore_RSGG\9.png" alt="Photo" width="650" />
<img src="imgs\XPlore_RSGG\15.png" alt="Photo" width="650" />
<img src="imgs\XPlore_RSGG\19.png" alt="Photo" width="650" />
<img src="imgs\XPlore_RSGG\35.png" alt="Photo" width="650" />
<!-- ![img](imgs\XPlore_RSGG\9.png)
![img](imgs\XPlore_RSGG\15.png)
![img](imgs\XPlore_RSGG\19.png)
![img](imgs\XPlore_RSGG\35.png) -->


Yet sometimes **both succeed**:

<img src="imgs\XPlore_RSGG\32.png" alt="Photo" width="650" />
<img src="imgs\XPlore_RSGG\48.png" alt="Photo" width="650" />
<img src="imgs\XPlore_RSGG\54.png" alt="Photo" width="650" />

<!-- ![img](imgs\XPlore_RSGG\32.png)
![img](imgs\XPlore_RSGG\48.png)
![img](imgs\XPlore_RSGG\54.png) -->


## Comparison of XPlore, CFGNNE and RSGG: t-SNE projection of Wavelet Characteristic embeddings (Tree-Cycle dataset)
We now compare togheter CFs for XPlore, CFGNNE and RSGG. Here we show only CFs for the Tree class, as we plot only instances where all 3 explainers were able to find a counterfactual, and CFGNNE has success only in finding CFs for this class.

### CFGNNE does not land on Cycle distribution
This is expected as **CFGNNE only removes edges**, hence it cannot produce Cycles. Note how **XPlore and RSGG behave similarly**.

<img src="imgs\CFGNNE_XPlore_RSGG\3.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\6.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\8.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\11.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\14.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\44.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\48.png" alt="Photo" width="400" />

Sometimes, **XPlore is the only explainer finding a solution**

<img src="imgs\CFGNNE_XPlore_RSGG\9.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\15.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\19.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\29.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\35.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\36.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\39.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\54.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\57.png" alt="Photo" width="400" />

Rarely, the embedder gets tricked (we know CFGNNE cannot produce cycles):

<img src="imgs\CFGNNE_XPlore_RSGG\63.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\90.png" alt="Photo" width="400" />
<img src="imgs\CFGNNE_XPlore_RSGG\114.png" alt="Photo" width="400" />

In this scenario, we were not able to find a case where XPlore landed on the wrong distribution.

---

We also invite the reader to take a look at all images in [XPLORE_RSGG](imgs\XPlore_RSGG) and [CFGNNE_XPLORE_RSGG](imgs\CFGNNE_XPLORE_RSGG).
