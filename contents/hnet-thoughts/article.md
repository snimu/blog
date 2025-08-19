# Various thoughts on the H-Net

- would be amazing embedding model
  - with late interaction, it could be hybrid search at different levels of abstraction
- the chunking itself leads to very interesting artifacts
  - if we have a DNA model (which benefits greatly from the H-Net, as shown in the paper), grouping of base pairs into higher-level abstractions (ideally at multiple levels of abstraction) might be highly informative for the field of biology, especially if multiple models trained on similar data but with different numbers of abstraction levels, different initializations, different data order, etc. come to similar groupings of base pairs
- when reducing dim, use the cutoff part as x0
- when increasing dim, use input from another path chunked like your main input as the additional vector (you could make that smaller than the missing dim, and make the rest up with a learned vector if that helps)
- Linear in residual? Or another Mamba layer / transformer layer
  - actually, trafo more likely to work because Mamba is specifically good for chunking, which we don't have to do with this
  - would preserve a residual stream through all paths in the network -> cleaner gradient
  - would also increase expressiveness of the residual connection across M
- Value embeddings?
- Latent looping of model
  - Scaling the number of loops in one of two ways:
    1. if p is close to 1.0, increase the number of loops
        - Increasing compute to this token is already taken care of by the smoothing module, by making the next token also consider this token
        - In other words, the higher the probability of a token boundary, the closer to the capacity limit the model *should be* for that token
          - &rarr; needs more compute for tokens with very certain boundaries than uncertain ones
          - &rarr; doing it this way would also allow the model to make the tokens larger because it's a second mechanism for keeping compute per bit of entropy constant
    2. use the number of loops to make up for the amount of compression in the model
        - the model will have varying memory and compute requirements depending on how much a sequence is compressed, which is bad
        - you can simply adjust the amount of latent looping to make up for compute variance, and the number of steps you propagate back with truncated BPTT to make up for memory requirements if they differ
        - this would additionally allow the model to compress more agressively
  - Both of these ways of scaling compute through latent looping are compatible
  - The encoder and decoder act like the Prelude and Coda in Geiping et al. so the techniques from that paper could simply be used on an H-Net
