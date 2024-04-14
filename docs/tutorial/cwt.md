# Time-series spectral analysis using wavelets

In this tutorial, we will walk through each step in order to use `pycwt' to perform the wavelet analysis of a given time-series.

In this example we will follow the approach suggested by Torrence and Compo (1998)[^1], using the NINO3 sea surface temperature anomaly dataset between 1871 and 1996. This and other sample data files are kindly provided by C. Torrence and G. Compo [here](http://paos.colorado.edu/research/wavelets/software.html).

We begin by importing the relevant libraries. Please make sure that PyCWT is properly installed in your system.

```python
--8<-- "src/pycwt/sample/simple_sample.py:10:15"
```

Then, we load the dataset and define some data related parameters. In this case, the first 19 lines of the data file contain meta-data, that we ignore, since we set them manually (*i.e.* title, units).

```python
--8<-- "src/pycwt/sample/simple_sample.py:20:26"
```

We also create a time array in years.

```python
--8<-- "src/pycwt/sample/simple_sample.py:28:29"
```

We write the following code to detrend and normalize the input data by its standard deviation. Sometimes detrending is not necessary and simply removing the mean value is good enough. However, if your dataset has a well defined trend, such as the Mauna Loa CO<sub>2</sub> dataset available in the above mentioned website, it is strongly advised to perform detrending. Here, we fit a one-degree polynomial function and then subtract it from the original data.

```python
--8<-- "src/pycwt/sample/simple_sample.py:38:42"
```

The next step is to define some parameters of our wavelet analysis. We select the mother wavelet, in this case the Morlet wavelet with $\omega_0=6$.

```python
--8<-- "src/pycwt/sample/simple_sample.py:47:51"
```

The following routines perform the wavelet transform and inverse wavelet transform using the parameters defined above. Since we have normalized our input time-series, we multiply the inverse transform by the standard deviation.

```python
--8<-- "src/pycwt/sample/simple_sample.py:57:58"
```

We calculate the normalized wavelet and Fourier power spectra, as well as the Fourier equivalent periods for each wavelet scale.

```python
--8<-- "src/pycwt/sample/simple_sample.py:62:65"
```

Optionally, we could also rectify the power spectrum according to the suggestions proposed by Liu et al. (2007)[^2]

```python
power /= scales[:, None]
```

We could stop at this point and plot our results. However we are also interested in the power spectra significance test. The power is significant where the ratio ``power / sig95 > 1``.

```python
--8<-- "src/pycwt/sample/simple_sample.py:69:73"
```

Then, we calculate the global wavelet spectrum and determine its significance level.

```python
--8<-- "src/pycwt/sample/simple_sample.py:77:82"
```

We also calculate the scale average between 2 years and 8 years, and its significance level.

```python
--8<-- "src/pycwt/sample/simple_sample.py:86:99"
```

Finally, we plot our results in four different subplots containing the (i) original series anomaly and the inverse wavelet transform; (ii) the wavelet power spectrum (iii) the global wavelet and Fourier spectra ; and (iv) the range averaged wavelet spectrum. In all sub-plots the significance levels are either included as dotted lines or as filled contour lines.

```python
--8<-- "src/pycwt/sample/simple_sample.py:107:188"
```

Running this sequence of commands you should be able to generate the following figure:

<figure markdown="span">
  ![Image](../img/sample_NINO3.png){ width="100%" }
  <figcaption>Wavelet analysis of the NINO3 Sea Surface Temperature record: (a) Time- series (solid black line) and inverse wavelet transform (solid grey line), (b) Normalized wavelet power spectrum of the NINO3 SST using the Morlet wavelet ($\omega_0=6$) as a function of time and of Fourier equivalent wave period (in years). The black solid contour lines enclose regions of more than 95% confidence relative to a red-noise random process ($\alpha=0.77$). The cross-hatched and shaded area indicates the affected by the cone of influence of the mother wavelet. (iii) Global wavelet power spectrum (solid black line) and Fourier power spectrum (solid grey line). The dotted line indicates the 95% confidence level. (iv) Scale-averaged wavelet power over the 2--8 year band (solid black line), power trend (solid grey line) and the 95% confidence level (black dotted line).</figcaption>
</figure>


[^1]: Torrence, C. and Compo, G. P.. A Practical Guide to Wavelet Analysis. Bulletin of the American Meteorological Society, *American Meteorological Society*, **1998**, 79, 61-78. DOI [10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2](http://dx.doi.org/10.1175/1520-0477(1998)079%3C0061:APGTWA%3E2.0.CO;2).

[^2]: Liu, Y., Liang, X. S. and Weisberg, R. H. Rectification of the bias in the wavelet power spectrum. *Journal of Atmospheric and Oceanic Technology*, **2007**, 24, 2093-2102. DOI [10.1175/2007JTECHO511.1](http://dx.doi.org/10.1175/2007JTECHO511.1).
