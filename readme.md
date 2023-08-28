# Smarty4covid

## Overview
Harnessing the power of Artificial Intelligence (AI) and m-health towards detecting new bio-markers indicative of the onset and progress of respiratory abnormalities/conditions has greatly attracted the scientific and research interest especially during COVID-19 pandemic. The smarty4covid dataset contains audio signals of cough (4,676), regular breathing (4,665), deep breathing (4,695) and voice (4,291) as recorded by means of mobile devices following a crowd-sourcing approach supported by a responsive web-based application. 

The smarty4covid dataset is released in the form of a web-ontology language (OWL) knowledge base enabling data consolidation from other relevant datasets, complex queries and reasoning. The smarty4covid dataset has been utilized towards the development of models able to: (i) extract clinically informative respiratory indicators (e.g respiratory rate, Inhalation/Exhalation ratio, and fractional inspiration time) from regular breathing records, and (ii) identify cough, breath and voice segments in crowd-sourced audio recordings.    


## Data Access
The data is available in the public ZENODO repository (DOI: 10.5281/zenodo.7760170)


## How to Use
Create a new environment
```
conda env create -f environment.yml
```

### Create knowledge triples
After downloading the data, the knowledge triples are available in the smarty-triples.nt file. This file can be modified by running the script ```python create_knowledge_triples.py```.

### Audio Type Classifier
The weights for the short and long scale audio type classifiers (CNNs) are provided in the files ```audio_type_classifier_long.h5```, and ```audio_type_classifier_short.h5```, respectively. The proposed multiscale classifier is defined in the script ```audio_type_classifier_inference.py``` as the ```Multitimescale``` class. It can be loaded by running the following python code: 

```python
from audio_type_classifier_inference import Multitimescale

ms = Multitimescale(m_small_pth='audio_type_short.h5',m_large_pth='audio_type_long.h5')
pred = ms.predict('file.wav')
```

### Breathing Features
The script ```breathing_features.py``` includes functions for calculating the clinically relevant breath characteristics from audio breathing files that can be exctracted by running the script ```extract_breathing_features.py```.

