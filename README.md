# Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data

Submitted to UAI (Uncertainty in Artificial Intelligence) 2025

## Abstract

Causality in time series can be difficult to determine, especially in the presence of non-linear dependencies. The concept of Granger causality helps analyze potential relationships between variables, thereby offering a method to determine whether one time series can predict—Granger cause—future values of another. Although successful, Granger causal methods still struggle with capturing long-range relations between variables. To this end, we leverage the recently successful Extended Long Short-Term Memory (xLSTM) architecture and propose Granger causal xLSTMs (GC-xLSTM). It first enforces sparsity between the time series components by using a novel dynamic lass penalty on the initial projection. Specifically, we adaptively improve the model and identify sparsity candidates. Our joint optimization procedure then ensures that the Granger causal relations are recovered in a robust fashion. Our experimental evaluations on three datasets demonstrate the overall efficacy of our proposed GC-xLSTM model.


