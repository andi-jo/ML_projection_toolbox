# ML_projection_toolbox
Simple Python wrapper for machine learning models in the context of lead-lag projection modelling.

Questions, comments and bug reports can be sent to andreas.joseph@bankofengland.co.uk.


MACHINE LEARNING WRAPPER for projection analyses
------------------------------------------------

of Bank of England SWP 674: Machine learning at central banks (September 2017).

Authors: Chiranjit Chakraborty & Andreas Joseph.

Disclaimer: licence.txt and SWP 674 disclaimer apply.


General package features:
-------------------------

	- simplified use of sophisticated ML models (wrapper based on scikit-learn package (http://scikit-learn.org/stable/))
	- automated bootstrapped training-testing framework (bagging)
	- time series training, test and projection framework
	- cross-sectional training & testing
	- conditional model predictions
	- set of diagnostic tools, including several specialised plots like prediction interval (fan) charts
	- general data handling capabilities
	- I/O options for results, model instances and plots

Program structure:
------------------

	1. load packages configuration file (allow for flexibility and arbitrary new features can be added)
		 two cases implemented for demonstration: 
		    - UK inflation forecasting: macro time series (sources: BoE, ONS, Worldbank, BIS)
		    - BJ air quality modelling: hourly dataset (Jan 2010 - Dec 2014, source: Song Xi Chen, csx'@'gsm.pku.edu.cn,
						Guanghua School of Management, Center for Statistical Science, Peking University
	2. load & transform data (__A__ML_load_data.py)
	3. time-series training-testing (__B__ML_projections.py)
	4. lead-lag shift analysis (__C__ML_shift_analysis.py; model evaluation for different forecast horizons)
	5. diagnostic plots (__D__ML_diagnostics.py; data series, projections, conditional model output (heatmap), 
                         				conditional fan chart, feature importance by horizon)
                         
Dependency structure:
---------------------

	- dependencies run from bottom to top, i.e. serial execution is recommended
	- everything depends on the config-part
	- __B__ and __C__ are independent
	- __D__ depends on A-C, conditions on options set in config file

Directory structure (can be changed in config files):
-----------------------------------------------------

	- code includes the package
	- results stores all non-graphical output
	- figures stores all graphical output
	- data holds the input data

Data sources and acknowledgement
--------------------------------

We would like to kindly thank the below persons and institutions which made this work possible: 

    1. Scikit-learn: Machine Learning in Python
	      Paper: Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion,
	             Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas,
	             Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay; 
               Journal of Machine Learning Research, 12 (Oct):2825-2830, 2011.

    2.  UK CPI projection case study:
	    - Regression problem: Model UK CPI inflation on a quarterly basis
	    - All data have been last retrieved on 23 Jan 2017.
	    - All used data have been publicly available from (sources and data IDs):
		  - Bank of England (http://www.bankofengland.co.uk/boeapps/iadb/NewInterMed.asp?Travel=NIxAZx): LPQAUYN, IUQLBEDR, IUQASIZC, XUQLBK82, 
		  - UK Office for National Statistics (ONS, https://www.ons.gov.uk/): D7BT, LF24, MGSX, ABMI, A4YM, QWND.
		  - Bank for International Settlements (credit to non-financial sector: http://www.bis.org/statistics/totcredit.htm): Q:GB:P:A:M:XDC:A.
		  - World Bank, commodities prices: http://www.worldbank.org/en/research/commodity-markets (Pink Sheet, averge of energy and and non-energy price indices, excluding precious metals.)
	    - Please refer to the Terms of Use of the above institutions for the usage and eventual redistributions of data.

    3. BJ air quality modelling (additional example application to test portability and document program options):
	    - Classification problem: Model air pollution in Beijing on an hourly basis
	    - The air pollution level column "pm2.5: PM2.5 concentration (ug/m^3)" has been exchanged with (1/0) values:
		      1: level above 100 (ug/m^3).
		      0: level below 100 (ug/m^3).
	    - Source: Song Xi Chen, csx'@'gsm.pku.edu.cn, Guanghua School of Management, Center for Statistical Science, Peking University.
	    - Downloaded from https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data (last access: 9 July 2017).
	    - Publication reference: Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H. and Chen, S. X.,
			  	Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating. 
				  Proceedings of the Royal Society A, 471, 20150257, 2015.
