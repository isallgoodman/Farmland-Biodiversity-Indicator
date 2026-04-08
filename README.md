# Farmland-Biodiversity-Indicator
Developing a Farmland habitat Biodiversity Indicator using DLR CropTpe Maps , Copernicus LMS data for Bayern District Germany 
with OECD Guidelines 

Part for Master Thesis supervised by 
Dr. Maninder Singh Dhillon (JMU)
Dr Ursula Gessner (DLR)

BAckground: 
This indicator was formulated under IECD Guidelines by Bayr et al (2023) which we have modified and enhanced.


Habitats defined 
Cropland, Perennial, Grassland and Small Woody Features (including Hedgerows)

Data to be downloaded 
    Croptype maps, Grassland Mowing Frequency , Grassland First Cut and Hedgerows : https://geoservice.dlr.de/ (2017-2024)
    Small Woody Features:  https://land.copernicus.eu/ ( 2015,2018,2021)
    Hexagon shapefiles https://github.com/uber/h3


    
 Preliminary steps needed to be done 
 1. Downloading the necessary data for the years requried for the case
 2. Select the geographical subset thats required


The codes uploaded are Chronolgically named which follows the methodology  
1= Pre-processing (Clipping, reprojecting and resamplign ) 
2=Generating Intermidiates( Crop Count, S/L rotation, Legumes)
3=Assigning Habitat Qiuality Scores /resampling ( editable )
4=Final mosaicing and hexagonal representation


if the names have alphanumeric numbers such as 1a, 2a they denote a sub process of the main task 
"FHBI_hexagons" file is the final Indicator that shows on Bayern that runs
FHBI = (% Very low × 0) + (% Low × 0.25) + (% Moderate × 0.5) + (% High × 0.75) + (% Very high × 1.0) 






 We proprocess the products of each habitat
1. Small Woody features and Hedgerows are combined in Or and buffered within 100 m of a valid farmland Pixel. Presence is considered Very HIgh and absence is No data
2. Perennial 
Only Crop classes that are 4 or more years consequitively  (Vineyards, hops and and Woody trees /Fruits were considered. 
3. Grasslands 
a 12 class Mowing Frequency and first cut date matrix was created and each class were given a value by Reinermann et al 2023
4. Cropland
It has 3 sub metrics ( 2a,b,c) Crop Count , SL rotation by Jännicke et al (2022)  and Share of legume/ season
Note:Cropland Median takes the median of these 3 sub metrics to have one represntative cropland habitat value 

















Acknowledgements

: Data provided and Framework methodology was devised by Dr Ursula Gessner, Defining objectives evaluating the metrics and by Dr Maninder Singh Dhillon . Habitat Qulaity values  Grassland habitat Quality values expert opinion were given by Dr Sophia Reinermann)


References 


Jänicke, C., Kuemmerle, T., Hostert, P., & Pflugmacher, D. (2022). Field-level land-use data reveal heterogeneous crop sequences with distinct regional differences in Germany. European Journal of Agronomy, 141, 126632. https://doi.org/10.1016/j.eja.2022.126632


Reinermann, S., Asam, S., Gessner, U., Ullmann, T., & Kuenzer, C. (2023). Multi-annual grassland mowing dynamics in Germany: Spatio-temporal patterns and the influence of climate, topographic and socio-political conditions. Frontiers in Environmental Science, 11, 1040551. https://doi.org/10.3389/fenvs.2023.1040551



